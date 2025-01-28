import time
import torch
import numpy as np
import scipy as sp

from .. import const
from .. import utils
from .. import backend as bkd
from .sources import Sources
from .mixture import Mixture
from .kinetics import Kinetics
from typing import Tuple


class ArgonCR(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    kin_dtb,
    rad_dtb=None,
    use_rad=False,
    use_proj=False,
    use_factorial=False,
    use_coll_int_fit=False,
    isothermal=False
  ):
    # Thermochemistry database
    # -------------
    # Mixture
    self.species_order = ("Ar", "Arp", "em")
    self.mix = Mixture(
      species,
      species_order=self.species_order,
      use_factorial=use_factorial
    )
    self.mix.build()
    # Kinetics
    self.kin = Kinetics(
      mixture=self.mix,
      reactions=kin_dtb,
      use_fit=use_coll_int_fit
    )
    # Radiation
    self.use_rad = bool(use_rad)
    self.rad = rad_dtb
    # Dimensions
    self.nb_comp = self.mix.nb_comp
    self.nb_temp = 2
    self.nb_eqs = self.nb_comp + self.nb_temp
    # Sources
    # -------------
    self.sources = Sources(
      mixture=self.mix,
      kinetics=self.kin,
      radiation=None
    )
    # ROM
    # -------------
    # Bases
    self.phi = None
    self.psi = None
    self.use_rom = False
    # Projector
    self.use_proj = bool(use_proj)
    self.P = None
    # Output
    # -------------
    self.output_lin = True
    self.C = None


    self.isothermal = bool(isothermal)

  # Equilibrium composition
  # ===================================
  def compute_eq_comp(self, p, T):
    # Solve this system of equations:
    # -------------
    # 1) Charge neutrality:
    #    x_em = x_Arp
    # 2) Mole conservation:
    #    x_em + x_Arp + x_Ar = 1
    # 3) Detailed balance:
    #    (n_Arp*n_em)/n_Ar = (Q_Arp*Q_em)/Q_Ar
    # -------------
    # Update thermo
    self.mix.update_species_thermo(T)
    # Compute number density
    n = p / (const.UKB*T)
    # Compute coefficient for quadratic system
    f = self.mix.species["Arp"].Q * self.mix.species["em"].Q
    f /= (self.mix.species["Ar"].Q * n)
    # Solve quadratic system for 'x_em'
    a = 1.0
    b = 2.0 * f
    c = -f
    x = (-b+torch.sqrt(b**2-4*a*c))/(2*a)
    x = torch.clip(x, const.XMIN, 1.0)
    # Set molar fractions
    s = self.mix.species["em"]
    s.x = x
    s = self.mix.species["Arp"]
    s.x = x * s.q / s.Q
    s = self.mix.species["Ar"]
    s.x = (1.0-2.0*x) * s.q / s.Q
    # Number densities
    n = n * torch.cat([self.mix.species[k].x for k in self.species_order])
    # # Density
    # rho = self.mix.get_rho(n)
    return n#, rho

  # Initial composition
  # ===================================
  def get_init_composition(self, p, T, noise=False, sigma=1e-2):
    # Compute equilibrium mass fractions
    n, rho = self.compute_eq_comp(p, T)
    # Add random noise
    if noise:
      # > Electron
      s = self.mix.species["em"]
      x = s.x * self._add_norm_noise(s, sigma, use_q=False)
      x = torch.clip(x, const.XMIN, 1.0)
      s.x = x
      # > Argon Ion
      s = self.mix.species["Arp"]
      s.x = x * self._add_norm_noise(s, sigma)
      # > Argon
      s = self.mix.species["Ar"]
      s.x = (1.0-2.0*x) * self._add_norm_noise(s, sigma)
      # Number densities
      n = torch.sum(n)
      n = n * torch.cat([self.mix.species[k].x for k in self.species_order])
    return n, rho

  def _add_norm_noise(self, species, sigma, use_q=True):
    f = 1.0 + sigma * torch.rand(species.nb_comp)
    if use_q:
      f *= species.q * f
      f /= torch.sum(f)
    return f

  # ROM Basis
  # ===================================
  def set_basis(self, phi, psi):
    self.phi = bkd.to_torch(phi)
    self.psi = bkd.to_torch(psi)
    # Biorthogonalize
    self.phi = self.phi @ torch.linalg.inv(self.psi.T @ self.phi)
    # Projector
    self.P = self.phi @ self.psi.T

  # Function/Jacobian
  # ===================================
  def set_fun_jac(self):
    self._jac = torch.func.jacrev(self._fun, argnums=1)
    self.fun_np = self.fun = bkd.make_fun_np(self._fun)
    self.jac_np = bkd.make_fun_np(self._jac)

  def jac(self, t, y):
    j = self.jac_np(t, y)
    j_not = utils.is_nan_inf(j)
    if j_not.any():
      # Finite difference Jacobian
      j_fd = sp.optimize.approx_fprime(
        xk=y,
        f=lambda z: self.fun(t, z),
        epsilon=bkd.epsilon()
      )
      j[j_not] = j_fd[j_not]
    return j

  def _fun(self, t, y):
    # ROM activated
    if self.use_rom:
      y = self._decode(y)
    # Extract primitive variables
    n, T, Te = self.get_prim(y)
    # Compute sources
    # > Conservative variables
    f_rho, f_et, f_ee = self.sources(n, T, Te)
    # > Primitive variables
    f_n = self.mix.m_inv_mat @ f_rho
    if self.isothermal:
      f_e = torch.zeros(2)
    else:
      f_e = torch.cat([
        self.omega_T(f_rho, f_et, f_ee),
        self.omega_pe(f_ee)]
      )
    # > Concatenate
    f = torch.cat([f_n/const.UNA, f_e])
    # ROM activated
    if self.use_rom:
      f = self._encode(f)
    return f

  def get_prim(self, y):
    # Unpacking
    c, T, Tpe = y[:-2], y[-2], y[-1]
    # Get number densities
    n = const.UNA * c
    # Get electron temperature
    Te = Tpe if self.isothermal else self.mix.get_Te(pe=Tpe, ne=n[-1])
    # Clip temperatures
    T, Te = [self.clip_temp(z) for z in (T, Te)]
    return n, T, Te

  def clip_temp(self, T):
    return torch.clip(T, const.TMIN, const.TMAX)

  def omega_T(self, f_rho, f_e, f_ee):
    # Translational temperature
    f_T = f_e - (f_ee + self.mix._e_h(f_rho))
    f_T = f_T / (self.mix.rho * self.mix.cv_h)
    return f_T.reshape(1)

  def omega_pe(self, f_ee):
    # Electron pressure
    gamma = self.mix.species["em"].gamma
    f_pe = (gamma - 1.0) * f_ee
    return f_pe.reshape(1)

  # Output
  # ===================================
  def output_fun(self, x):
    y = self.C @ x
    return y if self.output_lin else np.log(y)

  def output_jac(self, x):
    if self.output_lin:
      return self.C
    else:
      y = self.C @ x
      return np.diag(1.0/y) @ self.C

  def set_output(self, max_mom=2, linear=True):
    # Linear or log-scaled output
    self.output_lin = bool(linear)
    # Compose C matrix
    self.C = np.eye(self.nb_eqs)
    if (max_mom > 0):
      self.C[::] = 0.0
      # > Species
      si, ei = 0, 0
      for k in self.species_order:
        sk = self.mix.species[k]
        mm = max_mom if (sk.nb_comp > 1) else 1
        ei += mm
        self.C[si:ei,sk.indices] = sk.compute_mom_basis(mm)
        si = ei
      # > Translational temperature
      self.C[ei,-2] = 1.0
      # > Electron pressure
      self.C[ei+1,-1] = 1.0
      # > Remove zeros rows
      self.C = self.C[:ei+2]

  # Solving
  # ===================================
  def _solve(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float
  ) -> Tuple[np.ndarray]:
    # Setting up
    self.mix.set_rho(rho)
    self.set_fun_jac()
    # Solving
    runtime = time.time()
    y = sp.integrate.solve_ivp(
      fun=self.fun,
      t_span=[0.0,t[-1]],
      y0=y0,
      method="LSODA",
      t_eval=t,
      first_step=1e-14,
      rtol=1e-6,
      atol=0.0,
      jac=self.jac
    ).y
    runtime = time.time()-runtime
    runtime = np.array(runtime).reshape(1)
    return y, runtime

  def solve_fom(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float
  ) -> Tuple[np.ndarray]:
    """Solve FOM."""
    self.use_rom = False
    return self._solve(t, y0, rho)

  def solve_rom(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float
  ) -> Tuple[np.ndarray]:
    """Solve ROM."""
    self.use_rom = True
    # Encode initial conditions
    z0 = self._encode(y0)
    # Solve
    z, runtime = self._solve(t, z0, rho)
    # Decode solution
    y = self._decode(z.T).T
    return y, runtime

  def _encode(self, y):
    # Split variables
    w, T_pe = y[...,:-2], y[...,-2:]
    # Encode
    z = w @ self.P.T if self.use_proj else w @ self.psi
    # Concatenate
    return torch.cat([z, T_pe], dim=-1)

  def _decode(self, y):
    # Split variables
    z, T_pe = y[...,:-2], y[...,-2:]
    # Decode
    w = z @ self.P.T if self.use_proj else z @ self.phi.T
    # Concatenate
    return torch.cat([w, T_pe], dim=-1)

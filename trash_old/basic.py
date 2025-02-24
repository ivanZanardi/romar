import abc
import time
import torch
import numpy as np
import scipy as sp
from typing import Tuple

from .. import utils
from .. import backend as bkd
from .thermochemistry import Sources
from .thermochemistry import Mixture
from .thermochemistry import Kinetics
from .thermochemistry import Equilibrium


class Basic(object):

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
    use_coll_int_fit=False
  ):
    # Thermochemistry
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
    # Sources
    # -------------
    self.sources = Sources(
      mixture=self.mix,
      kinetics=self.kin,
      radiation=self.rad
    )
    # Equilibrium
    # -------------
    self.equil = Equilibrium(
      solve=self.solve_fom,
      mixture=self.mix,
      clipping=False
    )
    # ROM
    # -------------
    # Basis
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
    # Solving
    # -------------
    # Dimensions
    self.nb_comp = self.mix.nb_comp
    self.nb_temp = 2
    self.nb_eqs = self.nb_comp + self.nb_temp
    # Setting up function
    self.set_up = bkd.make_fun_np(self._set_up)

  # # Initial composition
  # # ===================================
  # def get_init_composition(self, p, T, noise=False, sigma=1e-2):
  #   # Compute equilibrium mass fractions
  #   n, rho = self.compute_eq_comp(p, T)
  #   # Add random noise
  #   if noise:
  #     # > Electron
  #     s = self.mix.species["em"]
  #     x = s.x * self._add_norm_noise(s, sigma, use_q=False)
  #     x = torch.clip(x, const.XMIN, 1.0)
  #     s.x = x
  #     # > Argon Ion
  #     s = self.mix.species["Arp"]
  #     s.x = x * self._add_norm_noise(s, sigma)
  #     # > Argon
  #     s = self.mix.species["Ar"]
  #     s.x = (1.0-2.0*x) * self._add_norm_noise(s, sigma)
  #     # Number densities
  #     n = torch.sum(n)
  #     n = n * torch.cat([self.mix.species[k].x for k in self.species_order])
  #   return n, rho

  # def _add_norm_noise(self, species, sigma, use_q=True):
  #   f = 1.0 + sigma * torch.rand(species.nb_comp)
  #   if use_q:
  #     f *= species.q * f
  #     f /= torch.sum(f)
  #   return f

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

  @abc.abstractmethod
  def _fun(self, t, y):
    pass

  # Output
  # ===================================
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
      # > Remove zeros rows
      self.C = self.C[:ei+1]

  def output_fun(self, x):
    y = self.C @ x
    return y if self.output_lin else np.log(y)

  def output_jac(self, x):
    if self.output_lin:
      return self.C
    else:
      y = self.C @ x
      return np.diag(1.0/y) @ self.C

  # Solving
  # ===================================
  @abc.abstractmethod
  def _set_up(self, y0, rho):
    pass

  def _solve(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float
  ) -> Tuple[np.ndarray]:
    # Setting up
    y0 = self.set_up(y0, rho)
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

  @abc.abstractmethod
  def _encode(self, y):
    pass

  @abc.abstractmethod
  def _decode(self, y):
    pass

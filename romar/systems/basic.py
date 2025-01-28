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
      mixture=self.mix,
      clipping=True
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
    # Solving
    # -------------
    # Dimensions
    self.nb_comp = self.mix.nb_comp
    self.nb_temp = 2
    self.nb_eqs = self.nb_temp + self.nb_comp
    # Class methods
    # -------------
    self.encode = bkd.make_fun_np(self._encode)
    self.decode = bkd.make_fun_np(self._decode)
    self.set_up = bkd.make_fun_np(self._set_up)
    self.get_init_sol = self.equil.get_init_sol

  # Properties
  # ===================================
  # Linear Model
  @property
  def A(self):
    return self._A

  @A.setter
  def A(self, value):
    self._A = value

  @property
  def b(self):
    return self._b

  @b.setter
  def b(self, value):
    self._b = value

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

  # Linear Model
  # ===================================
  def fun_lin(self, t, y):
    return self.A @ y + self.b

  def jac_lin(self, t, y):
    return self.A

  def compute_lin_fom_ops(
    self,
    y: np.ndarray
  ) -> None:
    """
    Compute the linearized full-order model (FOM) operators.

    This function computes and stores the Jacobian matrix `A` and the residual
    vector `b` for the system evaluated at the given state `y`.

    :param y: The state vector at which the Jacobian and residual are computed.
    :type y: np.ndarray
    """
    # Compute Jacobian matrix
    self.A = self.jac(0.0, y)
    # Compute residual vector
    self.b = self.fun(0.0, y)

  def compute_timescale(
    self,
    y: np.ndarray,
    rho: float,
    use_rom: bool = False
  ) -> float:
    """
    Compute the characteristic timescale of a given species.

    This function calculates the timescale by linearizing the system,
    extracting the sub-Jacobian corresponding to the specified species,
    and evaluating the eigenvalues of the sub-Jacobian.

    :param y: The state vector at which the timescale is computed.
    :type y: np.ndarray
    :param species: The species for which the timescale is calculated.
                    Defaults to "Ar".
    :type species: str, optional
    :param index: The index of the timescale to return (sorted by magnitude).
                  Defaults to -2.
    :type index: int, optional
    :return: The computed timescale for the given species.
    :rtype: float
    """
    # Setting up
    self.use_rom = bool(use_rom)
    y = self.set_up(y, rho)
    # Compute linearized operators
    self.compute_lin_fom_ops(y)
    # Compute eigenvalues of the Jacobian
    l = sp.linalg.eigvals(self.A)
    # Compute and return the smallest timescale
    t = np.amin(np.abs(1.0/l.real))
    return float(t)

  def compute_lin_tmax(
    self,
    t: np.ndarray,
    y: np.ndarray,
    rho: float,
    err_max: float = 30.0
  ) -> float:
    """
    Compute the maximum time validity for the linearized model.

    This function determines the time limit up to which the linear model
    remains valid, either by using eigenvalues of the Jacobian or by
    comparing the nonlinear with the linearized solution.

    :param t: Time array over which the model is evaluated.
    :type t: np.ndarray
    :param y: Solution of the nonlinear model, used as the reference for
              validation.
    :type y: np.ndarray
    :param use_eig: Flag to determine whether to use eigenvalue-based
                    timescale computation. If False, the function will
                    use error-based validation. Defaults to True.
    :type use_eig: bool, optional
    :param err_max: Maximum allowed percentage error between the nonlinear and
                    the linearized model for validity. Only used if
                    `use_eig` is False. Defaults to 30.0.
    :type err_max: float, optional
    :return: The maximum time (tmax) up to which the linearized model
             remains valid.
    :rtype: float
    """
    # Check solution matrix shape
    if (len(t.reshape(-1)) != len(y)):
      y = y.T
    # Compute the linearized solution
    ylin = self.solve_fom(t, y[0], rho, linear=True)[0].T
    # Number of time instants actually solved
    nt = len(ylin)
    # Compute the error between nonlinear and linear solutions
    err = utils.mape(y[:nt], ylin, eps=0.0, axis=-1)
    # Find the last index where the error is within the threshold
    idx = np.argmin(np.abs(err - err_max))
    # Return the corresponding time value
    return t[:nt][idx]

  # ROM Model
  # ===================================
  def set_basis(self, phi, psi):
    # Biorthogonalize
    phi = phi @ sp.linalg.inv(psi.T @ phi)
    # Projector
    P = phi @ psi.T
    # Convert
    self.phi, self.psi, self.P = [bkd.to_torch(z) for z in (phi, psi, P)]

  # Output
  # ===================================
  def compute_c_mat(
    self,
    max_mom: int = 1,
    state_specs: bool = False
  ) -> None:
    """
    Compute the observation matrix for a linear output model.

    This function constructs the `C` matrix that maps the state vector to
    the output vector. It includes species contributions and their moments,
    up to a specified maximum moment order.

    :param max_mom: The maximum number of moments to include for each species.
    :type max_mom: int
    """
    max_mom = max(int(max_mom), 1)
    # Compose C matrix for a linear output
    self.C = np.zeros((self.nb_comp*max_mom, self.nb_eqs))
    # Variables to track row indices in C
    si, ei = 0, 0
    # Loop over species in the defined order
    for k in self.species_order:
      if (k != "em"):
        # Get species object
        s = self.mix.species[k]
        # Compute the moment basis for the species and populate C
        basis = s.compute_mom_basis(max_mom)
        for b in basis:
          ei += s.nb_comp if state_specs else 1
          self.C[np.arange(si,ei),s.indices] = b
          si = ei
    # Remove not used rows from the C matrix
    self.C = self.C[:ei]

  # Solving
  # ===================================
  @abc.abstractmethod
  def _set_up(self, y0, rho):
    pass

  def _solve(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    linear: bool = False
  ) -> Tuple[np.ndarray]:
    # Linear model
    if linear:
      self.compute_lin_fom_ops(y0)
    # Solving
    runtime = time.time()
    y = sp.integrate.solve_ivp(
      fun=self.fun_lin if linear else self.fun,
      t_span=[0.0,t[-1]],
      y0=np.zeros_like(y0) if linear else y0,
      method="BDF",
      t_eval=t,
      first_step=1e-14,
      rtol=1e-6,
      atol=1e-15,
      jac=self.jac_lin if linear else self.jac,
    ).y
    # Linear model
    if linear:
      y += y0.reshape(-1,1)
    runtime = time.time()-runtime
    runtime = np.array(runtime).reshape(1)
    return y, runtime

  def solve_fom(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float,
    linear: bool = False
  ) -> Tuple[np.ndarray]:
    """Solve FOM."""
    # Setting up
    self.use_rom = False
    y0 = self.set_up(y0, rho)
    # Solving
    return self._solve(t, y0, linear)

  def solve_rom(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float,
    linear: bool = False
  ) -> Tuple[np.ndarray]:
    """Solve ROM."""
    # Setting up
    self.use_rom = True
    y0 = self.set_up(y0, rho)
    # Encode initial conditions
    z0 = self.encode(y0)
    # Solving
    z, runtime = self._solve(t, z0, linear)
    # Decode solution
    y = self.decode(z.T).T
    return y, runtime

  @abc.abstractmethod
  def _encode(self, y):
    pass

  @abc.abstractmethod
  def _decode(self, y):
    pass

  def get_tgrid(
    self,
    start: float,
    stop: float,
    num: int
  ) -> np.ndarray:
    t = np.geomspace(start, stop, num=num-1)
    t = np.insert(t, 0, 0.0)
    return t

import abc
import sys
import time
import torch
import numpy as np
import scipy as sp

import joblib as jl
from tqdm import tqdm
from typing import Tuple

import tensorflow as tf


import abc
import time
import numpy as np
import scipy as sp
import pandas as pd

from pyDOE import lhs


from .. import env
from .. import utils
from .. import backend as bkd
from .thermochemistry import *
from .thermochemistry.equilibrium import MU_VARS
from typing import Dict, List, Optional, Tuple


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
    use_factorial=True,
    use_tables=False
  ):
    # Thermochemistry
    # -------------
    # Mixture
    self.species_order = ("Ar", "Arp", "em")
    self.mix = Mixture(
      species,
      species_order=self.species_order,
      use_factorial=bool(use_factorial)
    )
    self.mix.build()
    # Kinetics
    self.kin = Kinetics(
      mixture=self.mix,
      processes=kin_dtb,
      use_tables=use_tables
    )
    # Radiation
    self.rad = Radiation(
      processes=rad_dtb,
      use_tables=use_tables,
      active=use_rad
    )
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
      mixture=self.mix
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
    self.set_methods()

  def set_methods(self):
    self.encode = bkd.make_fun_np(self._encode)
    self.decode = bkd.make_fun_np(self._decode)
    self.set_up = bkd.make_fun_np(self._set_up)
    self.get_init_sol = self.equil.get_init_sol

  # Properties
  # ===================================
  # Linear model operators
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
  def set_rom(self, phi, psi, mask):
    # Biorthogonalize
    phi = phi @ sp.linalg.inv(psi.T @ phi)
    # Projector
    P = phi @ psi.T
    # Mask
    if isinstance(mask, str):
      mask = np.loadtxt(mask)
    # Convert
    self.phi, self.psi, self.P, self.mask = [
      bkd.to_torch(z) for z in (phi, psi, P, mask)
    ]
    self.mask = self.mask.bool()
    # Dimension
    shape = list(self.phi.shape)
    if self.use_proj:
      self.rom_dim = shape[0]
    else:
      self.rom_dim = shape[1]

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
      atol=1e-20,
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

  def _encode(self, y):
    # Split variables
    yhat = y[..., self.mask]
    ynot = y[...,~self.mask]
    # Encode
    z = yhat @ self.P.T if self.use_proj else yhat @ self.psi
    # Concatenate
    return torch.cat([z, ynot], dim=-1)

  def _decode(self, z):
    # Split variables
    z, ynot = z[...,:self.rom_dim], z[...,self.rom_dim:]
    # Decode
    yhat = z @ self.P.T if self.use_proj else z @ self.phi.T
    # Fill decoded tensor
    shape = list(z.shape)[:-1]+[self.nb_eqs]
    y = torch.zeros(shape)
    y[..., self.mask] = yhat
    y[...,~self.mask] = ynot
    return y

  def get_tgrid(
    self,
    start: float,
    stop: float,
    num: int
  ) -> np.ndarray:
    t = np.geomspace(start, stop, num=num-1)
    t = np.insert(t, 0, 0.0)
    return t

  # Data generation and testing
  # ===================================
  def construct_design_mat_mu(
    self,
    limits: Dict[str, List[float]],
    nb_samples: int,
    log_vars: Tuple[str] = ("Te", "rho"),
    eps: float = 1e-8
  ) -> Tuple[pd.DataFrame]:
    # Sample remaining parameters
    design_space = [np.sort(limits[k]) for k in MU_VARS]
    design_space = np.array(design_space).T
    # Log-scale
    ilog = [i for (i, k) in enumerate(MU_VARS) if (k in log_vars)]
    design_space[:,ilog] = np.log(design_space[:,ilog] + eps)
    # Construct
    ddim = design_space.shape[1]
    dmat = lhs(ddim, int(nb_samples))
    # Rescale
    amin, amax = design_space
    mu = dmat * (amax - amin) + amin
    mu[:,ilog] = np.exp(mu[:,ilog]) - eps
    # Convert to dataframe
    mu = pd.DataFrame(data=mu, columns=MU_VARS)
    return mu

  def compute_sol_fom(
    self,
    t: np.ndarray,
    mu: np.ndarray,
    noise: bool = False,
    path: Optional[str] = None,
    index: Optional[int] = None,
    shift: int = 0,
    filename: Optional[str] = None
  ) -> Optional[np.ndarray]:
    mui = mu[index] if (index is not None) else mu
    y0, rho = self.get_init_sol(mui, noise)
    y, runtime = self.solve_fom(t, y0, rho)
    if (y.shape[1] == len(t)):
      # Converged
      data = {"index": index, "mu": mui, "t": t, "y0": y0, "rho": rho, "y": y}
      if (index is not None):
        index += shift
      utils.save_case(path=path, index=index, data=data, filename=filename)
    else:
      # Not converged
      runtime = None
    return runtime

  def compute_sol_rom(
    self,
    path: Optional[str] = None,
    index: Optional[int] = None,
    filename: Optional[str] = None,
    eval_err: bool = False,
    eps: float = 1e-8
  ) -> Tuple[Optional[np.ndarray]]:
    try:
      # Load test case
      icase = utils.load_case(path=path, index=index, filename=filename)
      t, y0, rho, y_fom = [icase[k] for k in ("t", "y0", "rho", "y")]
      # Solve ROM
      y_rom, runtime = self.solve_rom(t, y0, rho)
      if (y_rom.shape[1] == len(t)):
        # Converged
        if eval_err:
          error = {
            "t": t,
            "dist": self.compute_err_dist(y_fom, y_rom, eps),
            "temp": self.compute_err_temp(y_fom, y_rom, rho, eps),
            "mom": self.compute_err_mom(y_fom, y_rom, rho, eps)
          }
          return error, runtime
        else:
          sol = {
            "t": t,
            "y_fom": y_fom,
            "y_rom": y_rom,
            "rho": rho
          }
          return sol, runtime
      else:
        return None, None
    except:
      return None, None

  def compute_err(
    self,
    path: str,
    irange: List[int],
    nb_workers: int = 1,
    eps: float = 1e-8
  ) -> Tuple[Optional[np.ndarray]]:
    irange = np.sort(irange)
    nb_samples = irange[1]-irange[0]
    iterable = tqdm(
      iterable=range(*irange),
      ncols=80,
      desc="  Cases",
      file=sys.stdout
    )
    kwargs = dict(
      path=path,
      eval_err=True,
      eps=eps
    )
    if (nb_workers > 1):
      sols = jl.Parallel(nb_workers)(
        jl.delayed(
          env.make_fun_parallel(self.compute_sol_rom)
        )(index=i, **kwargs) for i in iterable
      )
    else:
      sols = [self.compute_sol_rom(index=i, **kwargs) for i in iterable]
    # Split error values and running times
    error, runtime = list(zip(*sols))
    error = [x for x in error if (x is not None)]
    runtime = [x for x in runtime if (x is not None)]
    converged = len(runtime)/nb_samples
    print(f"  Total converged cases: {len(runtime)}/{nb_samples}")
    if (converged >= 0.8):
      # Stack error values
      t = error[0]["t"]
      _error = error[0]
      for ierror in error[1:]:
        _error = tf.nest.map_structure(
          lambda e1, e2: np.vstack([e1, e2]), _error, ierror
        )
      # Compute statistics
      error = tf.nest.map_structure(
        lambda e: {
          "mean": np.mean(e, axis=0),
          "std": np.std(e, axis=0)
        },
        _error
      )
      error["t"] = t
      runtime = {
        "mean": float(np.mean(runtime, 0)),
        "std": float(np.std(runtime, 0))
      }
      return error, runtime
    else:
      return None, None

  def compute_err_dist(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8
  ) -> np.ndarray:
    return utils.absolute_percentage_error(
      y_true=y_true[:-2],
      y_pred=y_pred[:-2],
      eps=eps
    )

  def compute_err_temp(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rho: float,
    eps: float = 1e-8
  ) -> Dict[str, np.ndarray]:
    # Extract T
    Th_true = y_true[-2]
    Th_pred = y_pred[-2]
    # Compute Te
    self.mix.set_rho(rho)
    n_true = self.mix.get_n(w=y_true[:-2])
    n_pred = self.mix.get_n(w=y_pred[:-2])
    Te_true = self.mix.get_Te(pe=y_true[-1], ne=n_true[-1])
    Te_pred = self.mix.get_Te(pe=y_pred[-1], ne=n_pred[-1])
    # Compute error
    return {
      "Th": utils.absolute_percentage_error(Th_true, Th_pred, eps=eps),
      "Te": utils.absolute_percentage_error(Te_true, Te_pred, eps=eps)
    }

  def compute_err_mom(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rho: float,
    eps: float = 1e-8
  ) -> Dict[str, Dict[str, np.ndarray]]:
    self.mix.set_rho(rho)
    n_true = self.mix.get_n(w=y_true[:-2])
    n_pred = self.mix.get_n(w=y_pred[:-2])
    error = {}
    for (name, s) in self.mix.species.items():
      error[name] = self._compute_err_mom(
        species=s,
        n_true=n_true[s.indices],
        n_pred=n_pred[s.indices],
        eps=eps
      )
    return error

  def _compute_err_mom(
    self,
    species: Species,
    n_true: np.ndarray,
    n_pred: np.ndarray,
    eps: float = 1e-8
  ) -> Dict[str, np.ndarray]:
    error = {}
    for m in range(2):
      m_true = species.compute_mom(n=n_true, m=m)
      m_pred = species.compute_mom(n=n_pred, m=m)
      if (m == 0):
        m0_true = m_true
        m0_pred = m_pred
      else:
        m_true /= m0_true
        m_pred /= m0_pred
      error[f"m{m}"] = utils.absolute_percentage_error(m_true, m_pred, eps)
    return error

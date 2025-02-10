import abc
import sys
import time
import torch
import numpy as np
import scipy as sp
import joblib as jl
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from pyDOE2 import lhs
from typing import *

from .. import env
from .. import utils
from .. import backend as bkd
from .thermochemistry import *
from .thermochemistry.equilibrium import MU_VARS


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
    use_tables=False,
    species_order=("em", "Ar", "Arp")
  ):
    # Thermochemistry
    # -------------
    # Mixture
    self.species_order = tuple(species_order)
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
    self.C = None
    # Solving
    # -------------
    # Dimensions
    self.nb_comp = self.mix.nb_comp
    self.nb_temp = 2
    self.nb_eqs = self.nb_temp + self.nb_comp
    self.set_methods()

  def set_methods(self):
    self.get_init_sol = self.equil.get_init_sol
    self.encode = bkd.make_fun_np(self._encode)
    self.decode = bkd.make_fun_np(self._decode)
    self.set_up = bkd.make_fun_np(self._set_up)
    self.get_prim = bkd.make_fun_np(self._get_prim)

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

  def _fun(self, t, y):
    y = self._decode(y) if self.use_rom else y
    f = self._fun_fom(t, y)
    f = self._encode(f) if self.use_rom else f
    return f

  @abc.abstractmethod
  def _fun_fom(self, t, y):
    pass

  @abc.abstractmethod
  def _get_prim(self, y, clip=True):
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
  def set_rom(
    self,
    phi: np.ndarray,
    psi: np.ndarray,
    mask: Union[str, np.ndarray],
    xref: Optional[np.ndarray] = None,
    xscale: Optional[np.ndarray] = None
  ) -> None:
    # Biorthogonalize
    phi = phi @ sp.linalg.inv(psi.T @ phi)
    # Projector
    P = phi @ psi.T
    # Mask
    if isinstance(mask, str):
      mask = np.loadtxt(mask)
    mask = mask.astype(bool)
    # Normalization
    # > Reference value
    xref = np.zeros(self.nb_eqs)[mask] if (xref is None) else xref
    xref = xref.squeeze()
    # > Scaling value
    xscale = np.ones(self.nb_eqs)[mask] if (xscale is None) else xscale
    xscale = xscale.squeeze()
    ov_xscale = np.diag(1.0/xscale)
    xscale = np.diag(xscale)
    # Dimension
    shape = list(phi.shape)
    self.rom_dim = shape[0] if self.use_proj else shape[1]
    # Convert
    for (k, v) in (
      ("phi", phi),
      ("psi", psi),
      ("P", P),
      ("mask", mask),
      ("xref", xref),
      ("xscale", xscale),
      ("ov_xscale", ov_xscale)
    ):
      setattr(self, k, bkd.to_torch(v))
    self.mask = self.mask.bool()

  # Output
  # ===================================
  def compute_c_mat(
    self,
    max_mom: int = 1,
    state_specs: bool = False,
    include_em: bool = True,
    include_temp: bool = False
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
      if ((k == "em") and (not include_em)):
        continue
      # Get species object
      s = self.mix.species[k]
      # Compute the moment basis for the species and populate C
      m = max_mom if (k != "em") else 1
      basis = s.compute_mom_basis(m)
      for b in basis:
        ei += s.nb_comp if state_specs else 1
        self.C[np.arange(si,ei),s.indices] = b
        si = ei
    # Temperatures
    if include_temp:
      for i in range(2):
        ei += 1
        self.C[np.arange(si,ei),-2+i] = 1.0
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
      method="LSODA",
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

  def _encode(self, y):
    # Split variables
    yhat = y[..., self.mask]
    ynot = y[...,~self.mask]
    # Normalize
    yhat = (yhat - self.xref) @ self.ov_xscale
    # Encode
    z = yhat @ self.P.T if self.use_proj else yhat @ self.psi
    # Concatenate
    return torch.cat([z, ynot], dim=-1)

  def _decode(self, z):
    # Split variables
    z, ynot = z[...,:self.rom_dim], z[...,self.rom_dim:]
    # Decode
    yhat = z @ self.P.T if self.use_proj else z @ self.phi.T
    # Denormalize
    yhat = yhat @ self.xscale + self.xref
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
    dmat = lhs(ddim, samples=int(nb_samples))
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
        prim_fom = self.get_prim(y_fom, clip=False)
        prim_rom = self.get_prim(y_rom, clip=False)
        data = {
          "t": t,
          "FOM": self.postproc_sol(*prim_fom),
          "ROM": self.postproc_sol(*prim_rom),
          "err": {
            "mom": self.compute_err_mom(prim_fom[0], prim_rom[0], eps),
            "dist": self.compute_err_dist(prim_fom[0], prim_rom[0], eps),
            "temp": self.compute_err_temp(prim_fom[1:], prim_rom[1:], eps)
          }
        }
        return data, runtime
      else:
        return None, None
    except:
      return None, None

  # Postprocessing
  # -----------------------------------
  def postproc_sol(self, n, Th, Te):
    return {
      "mom": self.compute_mom(n),
      "dist": self.compute_dist(n),
      "temp": {"Th": Th, "Te": Te}
    }

  def compute_mom(
    self,
    n: np.ndarray
  ) -> Dict[str, Dict[str, np.ndarray]]:
    moms = {}
    for (name, s) in self.mix.species.items():
      moms[name] = self._compute_mom(n=n[s.indices], species=s)
    return moms

  def _compute_mom(
    self,
    n: np.ndarray,
    species: Species
  ) -> Dict[str, np.ndarray]:
    moms = {}
    for m in range(2):
      mom = species.compute_mom(n=n, m=m)
      if (m == 0):
        mom0 = mom
      else:
        mom /= mom0
      moms[f"m{m}"] = mom
    return moms

  def compute_dist(
    self,
    n: np.ndarray
  ) -> Dict[str, np.ndarray]:
    dist = {}
    for (name, s) in self.mix.species.items():
      gi = bkd.to_numpy(s.lev["g"])
      ni = n[s.indices]
      dist[name] = (ni.T/gi).T
    return dist

  # Error computation
  # -----------------------------------
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
    if (nb_workers > 1):
      sols = jl.Parallel(nb_workers)(
        jl.delayed(
          env.make_fun_parallel(self.compute_sol_rom)
        )(index=i, path=path, eps=eps) for i in iterable
      )
    else:
      sols = [
        self.compute_sol_rom(index=i, path=path, eps=eps) for i in iterable
      ]
    # Split error values and running times
    error, runtime = list(zip(*sols))
    not_converged  = [i for i in range(*irange) if (runtime[i-irange[0]] is None)]
    error = [x for x in error if (x is not None)]
    runtime = [x for x in runtime if (x is not None)]
    converged = len(runtime)/nb_samples
    print(f"  Total converged cases: {len(runtime)}/{nb_samples}")
    if (converged >= 0.5):
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
      return error, runtime, not_converged
    else:
      return None, None, not_converged

  def compute_err_dist(
    self,
    n_true: np.ndarray,
    n_pred: np.ndarray,
    eps: float = 1e-8
  ) -> np.ndarray:
    return utils.absolute_percentage_error(
      y_true=bkd.to_numpy(self.mix.get_w(bkd.to_torch(n_true))),
      y_pred=bkd.to_numpy(self.mix.get_w(bkd.to_torch(n_pred))),
      eps=eps
    )

  def compute_err_temp(
    self,
    Ti_true: np.ndarray,
    Ti_pred: np.ndarray,
    eps: float = 1e-8
  ) -> Dict[str, np.ndarray]:
    return {
      "Th": utils.absolute_percentage_error(Ti_true[0], Ti_pred[0], eps),
      "Te": utils.absolute_percentage_error(Ti_true[1], Ti_pred[1], eps)
    }

  def compute_err_mom(
    self,
    n_true: np.ndarray,
    n_pred: np.ndarray,
    eps: float = 1e-8
  ) -> Dict[str, Dict[str, np.ndarray]]:
    return tf.nest.map_structure(
      lambda m1, m2: utils.absolute_percentage_error(m1, m2, eps),
      self.compute_mom(n_true),
      self.compute_mom(n_pred)
    )

import abc
import time
import numpy as np
import scipy as sp
import pandas as pd

from pyDOE import lhs

from .. import const
from .. import utils
from .mixture import Mixture
from .kinetics import Kinetics
from typing import Dict, List, Optional, Tuple


class BasicSystem(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species: Dict[str, str],
    rates_coeff: str,
    use_einsum: bool = False,
    use_factorial: bool = False,
    use_arrhenius: bool = False
  ) -> None:
    # Thermochemistry database
    # -------------
    self.T = 1.0
    # Mixture
    self.mix = Mixture(species, use_factorial)
    self.nb_eqs = self.mix.nb_eqs
    # > Kinetics
    self.kin = Kinetics(self.mix.species, rates_coeff, use_arrhenius)
    # FOM
    # -------------
    # Solving
    self.use_einsum = use_einsum
    self.fun = None
    self.jac = None
    # ROM
    # -------------
    self.rom_ops = None
    # Basis
    self.phi = None
    self.psi = None
    self.runtime = 0.0

  def _check_rom_ops(self) -> None:
    if (self.rom_ops is None):
      raise ValueError("Update ROM operators.")

  def _is_einsum_used(self, identifier: str) -> None:
    if self.use_einsum:
      raise NotImplementedError(
        "This functionality is not supported " \
        f"when using 'einsum': '{identifier}'."
      )

  # Operators
  # ===================================
  # FOM
  # -----------------------------------
  def update_fom_ops(self, T) -> None:
    self.T = float(T)
    # Update mixture
    self.mix.update(self.T)
    # Update kinetics
    self.kin.update(self.T)
    # Compose operators
    if self.use_einsum:
      self.fom_ops = self.kin.rates
    else:
      self.fom_ops = self._update_fom_ops(self.kin.rates)
    self.fom_ops["m_ratio"] = self.mix.m_ratio

  @abc.abstractmethod
  def _update_fom_ops(self, rates: dict) -> dict:
    pass

  # Linearized FOM
  # -----------------------------------
  def compute_lin_fom_ops(
    self,
    mu: np.ndarray,
    rho: float,
    max_mom: int = 10
  ) -> Dict[str, np.ndarray]:
    self._is_einsum_used("compute_lin_fom_ops")
    # Equilibrium
    n_eq = self.mix.compute_eq_comp(rho)
    w_eq = self.mix.get_w(n_eq, rho)
    # A operator
    A = self._compute_lin_fom_ops_a_full(n_eq[0])
    # C operator
    C = self._compute_lin_fom_ops_c(max_mom)
    # Initial solutions
    M = self._compute_lin_init_sols(mu, w_eq)
    # Return data
    return {"A": A, "C": C, "M": M, "x_eq": w_eq}

  def _compute_lin_fom_ops_a_full(
    self,
    n_a_eq: np.ndarray,
    phi: Optional[np.ndarray] = None,
    psi: Optional[np.ndarray] = None,
    by_mass: bool = True
  ) -> np.ndarray:
    A = self._compute_lin_fom_ops_a(n_a_eq)
    b = self._compute_lin_fom_ops_b(n_a_eq)
    m = self.mix.m_ratio
    if (phi is not None):
      A = psi.T @ A @ phi
      b = psi.T @ b
      m = self.mix.m_ratio @ phi
    A = np.hstack([b.reshape(-1,1), A])
    a = - m @ A
    A = np.vstack([a.reshape(1,-1), A])
    if by_mass:
      A = self.mix.M @ A @ self.mix.Minv
    return A

  @abc.abstractmethod
  def _compute_lin_fom_ops_a(
    self,
    n_a_eq: np.ndarray
  ) -> np.ndarray:
    pass

  @abc.abstractmethod
  def _compute_lin_fom_ops_b(
    self,
    n_a_eq: np.ndarray
  ) -> np.ndarray:
    pass

  def _compute_lin_fom_ops_c(
    self,
    max_mom: int
  ) -> np.ndarray:
    if (max_mom > 0):
      C = np.zeros((max_mom,self.nb_eqs))
      C[:,1:] = self.mix.species["molecule"].compute_mom_basis(max_mom)
    else:
      C = np.eye(self.nb_eqs)
      C[0,0] = 0.0
    return C

  def _compute_lin_init_sols(
    self,
    mu: np.ndarray,
    w_eq: np.ndarray,
    noise: bool = True,
    sigma: float = 1e-2
  ) -> np.ndarray:
    M = []
    for mui in mu:
      w0 = self.mix.get_init_sol(*mui, noise=noise, sigma=sigma)
      M.append(w0 - w_eq)
    M = np.vstack(M).T
    return M

  # ROM
  # -----------------------------------
  def update_rom_ops(
    self,
    phi: Optional[np.ndarray] = None,
    psi: Optional[np.ndarray] = None,
  ) -> None:
    self._is_einsum_used("update_rom_ops")
    # Set basis
    if (phi is not None):
      self.set_basis(phi, psi)
    # Compose operators
    self.rom_ops = self._update_rom_ops()
    self.rom_ops["m_ratio"] = self.mix.m_ratio @ self.phi

  def set_basis(
    self,
    phi: np.ndarray,
    psi: np.ndarray
  ) -> None:
    self.phi, self.psi = phi, psi
    # Biorthogonalize
    self.phi = self.phi @ sp.linalg.inv(self.psi.T @ self.phi)

  @abc.abstractmethod
  def _update_rom_ops(self) -> None:
    pass

  # Solving
  # ===================================
  def _solve(
    self,
    t: np.ndarray,
    n0: np.ndarray,
    ops: dict
  ) -> np.ndarray:
    sol = sp.integrate.solve_ivp(
      fun=self.fun,
      t_span=[0.0,t[-1]],
      y0=n0/const.UNA,
      method="LSODA",
      t_eval=t,
      args=(ops,),
      first_step=1e-14,
      rtol=1e-6,
      atol=0.0,
      jac=self.jac
    )
    n = sol.y * const.UNA
    nb_n = n.shape[1]
    nb_t = len(t.reshape(-1))
    if (nb_n != nb_t):
      raise ValueError("Solution not converged!")
    return n

  def solve_fom(
    self,
    t: np.ndarray,
    n0: np.ndarray
  ) -> Tuple[np.ndarray]:
    """Solve FOM."""
    runtime = time.time()
    n = self._solve(t=t, n0=n0, ops=self.fom_ops)
    runtime = time.time()-runtime
    runtime = np.array(runtime).reshape(1)
    return n[:1], n[1:], runtime

  def solve_lin_fom(
    self,
    t: np.ndarray,
    n0: np.ndarray,
    A: Optional[np.ndarray] = None
  ) -> Tuple[np.ndarray]:
    # Equilibrium composition
    rho = self.mix.get_rho(n0)
    n_eq = self.mix.compute_eq_comp(rho)
    # Linear operator
    if (A is None):
      A = self._compute_lin_fom_ops_a_full(n_eq[0], by_mass=False)
    # Eigendecomposition
    l, v = [x.real for x in sp.linalg.eig(A)]
    # Solution
    n = sp.linalg.solve(v, n0-n_eq)
    n = [n_eq + v @ (np.exp(l*ti) * n) for ti in t]
    n = np.vstack(n).T
    return n[:1], n[1:]

  def solve_rom(
    self,
    t: np.ndarray,
    n0: np.ndarray
  ) -> Tuple[np.ndarray]:
    """Solve ROM."""
    self._check_rom_ops()
    self._is_einsum_used("solve_rom")
    # Encode initial condition
    z_m = self._encode(n0[1:])
    z0 = np.concatenate([n0[:1], z_m])
    # Solve
    runtime = time.time()
    z = self._solve(t=t, n0=z0, ops=self.rom_ops)
    runtime = time.time()-runtime
    runtime = np.array(runtime).reshape(1)
    # Decode solution
    n_m = self._decode(z[1:].T).T
    return z[:1], n_m, runtime

  def _encode(self, x: np.ndarray) -> np.ndarray:
    return x @ self.psi

  def _decode(self, z: np.ndarray) -> np.ndarray:
    return z @ self.phi.T

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
  def construct_design_mat_temp(
    self,
    limits: List[float],
    nb_samples: int
  ) -> Tuple[pd.DataFrame]:
    # Construct
    dmat = lhs(1, int(nb_samples))
    # Rescale
    amin, amax = np.sort(limits)
    T = dmat * (amax - amin) + amin
    # Convert to dataframe
    T = pd.DataFrame(data=np.sort(T.reshape(-1)), columns=["T"])
    return T

  def construct_design_mat_mu(
    self,
    limits: Dict[str, List[float]],
    nb_samples: int,
    log_vars: List[str] = ["T0", "rho"],
    eps: float = 1e-7
  ) -> Tuple[pd.DataFrame]:
    _mu_keys = ("T0", "w0_a", "rho")
    # Sample remaining parameters
    design_space = [np.sort(limits[k]) for k in _mu_keys]
    design_space = np.array(design_space).T
    # Log-scale
    ilog = [i for (i, k) in enumerate(_mu_keys) if (k in log_vars)]
    design_space[:,ilog] = np.log(design_space[:,ilog] + eps)
    # Construct
    ddim = design_space.shape[1]
    dmat = lhs(ddim, int(nb_samples))
    # Rescale
    amin, amax = design_space
    mu = dmat * (amax - amin) + amin
    mu[:,ilog] = np.exp(mu[:,ilog]) - eps
    # Convert to dataframe
    mu = pd.DataFrame(data=mu, columns=_mu_keys)
    return mu

  def compute_fom_sol(
    self,
    T: float,
    t: np.ndarray,
    mu: np.ndarray,
    mu_type: str = "mass",
    update: bool = False,
    path: Optional[str] = None,
    index: Optional[int] = None,
    shift: int = 0,
    filename: Optional[str] = None
  ) -> np.ndarray:
    try:
      if update:
        self.update_fom_ops(T)
      mui = mu[index] if (index is not None) else mu
      n0 = self.mix.get_init_sol(*mui, mu_type=mu_type)
      *n, runtime = self.solve_fom(t, n0)
      data = {"index": index, "mu": mui, "T": T, "t": t, "n0": n0, "n": n}
      if (index is not None):
        index += shift
      utils.save_case(path=path, index=index, data=data, filename=filename)
    except:
      runtime = None
    return runtime

  def compute_rom_sol(
    self,
    update: bool = False,
    path: Optional[str] = None,
    index: Optional[int] = None,
    filename: Optional[str] = None,
    eval_err: Optional[str] = None,
    eps: float = 1e-7
  ) -> Tuple[np.ndarray]:
    try:
      # Load test case
      icase = utils.load_case(path=path, index=index, filename=filename)
      T, t, n0, n_fom = [icase[k] for k in ("T", "t", "n0", "n")]
      if update:
        self.update_fom_ops(T)
        self.update_rom_ops()
      # Solve ROM
      *n_rom, runtime = self.solve_rom(t, n0)
      # Evaluate error
      if (eval_err == "mom"):
        # > Moments
        return self.compute_mom_err(n_fom[1], n_rom[1], eps), runtime
      elif (eval_err == "dist"):
        # > Distribution
        rho = self.mix.get_rho(n0)
        return self.compute_dist_err(n_fom[1], n_rom[1], rho, eps), runtime
      else:
        # > None: return the solution
        return t, n_fom, n_rom, runtime
    except:
      nb_none = 2 if (eval_err is not None) else 4
      return [None for _ in range(nb_none)]

  def compute_mom_err(
    self,
    n_true: np.ndarray,
    n_pred: np.ndarray,
    eps: float = 1e-7
  ) -> np.ndarray:
    error = []
    for m in range(2):
      m_true = self.mix.species["molecule"].compute_mom(n=n_true, m=m)
      m_pred = self.mix.species["molecule"].compute_mom(n=n_pred, m=m)
      if (m == 0):
        m0_true = m_true
        m0_pred = m_pred
      else:
        m_true /= m0_true
        m_pred /= m0_pred
      error.append(utils.absolute_percentage_error(m_true, m_pred, eps))
    return np.vstack(error)

  def compute_dist_err(
    self,
    n_true: np.ndarray,
    n_pred: np.ndarray,
    rho: float,
    eps: float = 1e-7
  ) -> np.ndarray:
    y_true = n_true * self.mix.species["molecule"].m / rho
    y_pred = n_pred * self.mix.species["molecule"].m / rho
    return utils.absolute_percentage_error(y_true, y_pred, eps)

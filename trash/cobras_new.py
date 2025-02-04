import os
import sys
import torch
import numpy as np
import scipy as sp
import joblib as jl
import dill as pickle
import multiprocessing

from tqdm import tqdm
from typing import *

from .. import env
from .. import ops
from .. import backend as bkd
from ..systems import SYS_TYPES


class CoBRAS(object):

  """
  CoBRAS: Model Reduction for Nonlinear Systems by Balanced Truncation of State
  and Gradient Covariance.

  This module implements the CoBRAS method, a model reduction technique
  designed for nonlinear systems by leveraging the balanced truncation of
  covariance matrices for states and gradients. The approach is particularly
  useful for reducing the computational complexity of high-dimensional
  nonlinear systems while preserving key dynamics.

  For further details, refer to the publication:
  https://doi.org/10.1137/22M1513228
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    system: SYS_TYPES,
    tgrid: Dict[str, float],
    quad_mu: Dict[str, np.ndarray],
    xref: Optional[Union[str, np.ndarray]] = None,
    xscale: Optional[Union[str, np.ndarray]] = None,
    path_to_saving: str = "./",
    saving: bool = True
  ) -> None:
    """
    Initialize the CoBRAS class with the specified system, quadrature points,
    time grid, and saving configurations.

    :param system: Instance of the system to be reduced. Can be either a
                   `BoxAd` or `BoxIso` system representing a nonlinear system.
    :type system: BoxAd
    :param tgrid: Dictionary specifying the time grid. Must include keys:
                  - "start": Start time of the simulation.
                  - "stop": End time of the simulation.
                  - "num": Number of time points in the grid.
    :type tgrid: Dict[str, float]
    :param quad_mu: Dictionary containing quadrature points and weights
                    for initial conditions. Must include:
                    - "x": A 1D numpy array of quadrature points.
                    - "w": A 1D numpy array of corresponding weights.
    :type quad_mu: Dict[str, np.ndarray]
    :param path_to_saving: Directory path where the computed data and modes
                           will be saved. Defaults to "./".
    :type path_to_saving: str, optional
    :param saving: Flag indicating whether to enable saving of results.
                   Defaults to True.
    :type saving: bool, optional
    """
    # Store attributes
    self.system = system
    self.tgrid = tgrid
    self.quad_mu = quad_mu
    # Normalization procedure
    self.set_norm(xref, xscale)
    # Configure saving options
    self.saving = saving
    self.path_to_saving = path_to_saving
    os.makedirs(self.path_to_saving, exist_ok=True)

  # Normalization
  # ===================================
  def set_norm(self, xref, xscale):
    # Reference value
    if (xref is None):
      xref = np.zeros(self.system.nb_eqs)
    elif isinstance(xref, str):
      xref = np.loadtxt(xref)
    self.xref = xref.squeeze()
    # Scaling value
    if (xscale is None):
      xscale = np.ones(self.system.nb_eqs)
    elif isinstance(xscale, str):
      xscale = np.loadtxt(xscale)
    xscale = xscale.squeeze()
    self.xscale = np.diag(xscale)
    self.ov_xscale = np.diag(1.0/xscale)

  def normalize(self, x):
    return self.ov_xscale @ (x - self.xref)

  # Covariance matrices
  # ===================================
  def compute_cov_mats(
    self,
    nb_meas: int = 5,
    noise: bool = False,
    err_max: float = 25.0,
    nb_workers: int = 1
  ) -> Tuple[np.ndarray]:
    """
    Compute state and gradient covariance matrices based on quadrature
    points and system dynamics.

    :param nb_meas: Number of output measurements to use for adjoint
                    simulations. Defaults to 10.
    :type nb_meas: int
    :param use_eig: Whether to use eigenvalue-based analysis to determine
                    the maximum time (`tmax`) up to which the linear model
                    is valid. If True, eigenvalues are used to calculate
                    maximum valid timescales.
    :type use_eig: bool
    :param err_max: Maximum percentage error (in %) allowed between linear and
                    nonlinear models for determining the maximum time (`tmax`)
                    up to which the linear model is valid.
                    Defaults to 30.0.
    :type err_max: float

    :return: Tuple containing:
             - `X` (np.ndarray): State covariance matrix.
             - `Y` (np.ndarray): Gradient covariance matrix.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Unpack initial conditions quadrature points and weights
    mu, w_mu = self.quad_mu["x"], self.quad_mu["w"]
    # Loop over initial conditions quadrature points
    iterable = tqdm(
      iterable=range(len(mu)),
      ncols=80,
      desc="Trajectories",
      file=sys.stdout
    )
    with multiprocessing.Manager() as manager:
      # Define input arguments for covariance matrices calculation
      kwargs = dict(
        X=manager.list(),
        Y=manager.list(),
        nb_meas=nb_meas,
        noise=noise,
        err_max=err_max
      )
      if (nb_workers > 1):
        # Run parallel jobs
        jl.Parallel(nb_workers)(
          jl.delayed(env.make_fun_parallel(self._compute_cov_mats))(
            mu=mu[i],
            w_mu=w_mu[i],
            **kwargs
          ) for i in iterable
        )
      else:
        # Run jobs in series
        for i in iterable:
          self._compute_cov_mats(
            mu=mu[i],
            w_mu=w_mu[i],
            **kwargs
          )
      # Stack covariance matrices
      X = np.vstack(list(kwargs["X"])).T
      Y = np.vstack(list(kwargs["Y"])).T
    return X, Y

  def _compute_cov_mats(
    self,
    X: List[np.ndarray],
    Y: List[np.ndarray],
    mu: np.ndarray,
    w_mu: float,
    nb_meas: int = 5,
    noise: bool = False,
    err_max: float = 25.0
  ) -> Tuple[np.ndarray]:
    """
    Compute state and gradient covariance matrices based on quadrature
    points and system dynamics.

    :param nb_meas: Number of output measurements to use for adjoint
                    simulations. Defaults to 10.
    :type nb_meas: int
    :param use_eig: Whether to use eigenvalue-based analysis to determine
                    the maximum time (`tmax`) up to which the linear model
                    is valid. If True, eigenvalues are used to calculate
                    maximum valid timescales.
    :type use_eig: bool
    :param err_max: Maximum percentage error (in %) allowed between linear and
                    nonlinear models for determining the maximum time (`tmax`)
                    up to which the linear model is valid.
                    Defaults to 30.0.
    :type err_max: float

    :return: Tuple containing:
             - `X` (np.ndarray): State covariance matrix.
             - `Y` (np.ndarray): Gradient covariance matrix.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Scaling factor for output measurements
    w_meas = 1.0 / np.sqrt(nb_meas)
    # Compute the initial solution for the system
    y0, rho = self.system.equil.get_init_sol(mu, noise=noise, sigma=1e-1)
    # Determine the smallest time scale for resolving system dynamics
    tmin = self.system.compute_timescale(y0, rho)
    # Generate a time quadrature grid and associated weights
    t, w_t = self._get_tquad(tmin)
    # Solve the nonlinear forward problem to compute the state evolution
    y = self.system.solve_fom(t, y0, rho)[0].T
    # Build an interpolator for the solution
    sol_interp = self._build_sol_interp(t, y)
    # Determine the maximum valid time for linear model approximation
    tlin = np.geomspace(tmin, t[-1], num=100)
    tmax = self.system.compute_lin_tmax(tlin, sol_interp(tlin), rho, err_max)
    # Generate a time grid for the linear adjoint simulation
    tadj = np.geomspace(tmin, 0.9*tmax, num=nb_meas)
    # Loop over each initial time for adjoint simulations
    for j in range(len(t)-1):
      # Solve the j-th linear adjoint model
      Yij = w_meas * self.solve_lin_adjoint(tadj, y[j], rho).T
      # Compute the combined quadrature weight (mu and t)
      wij = w_mu * w_t[j]
      # Store weighted samples for gradient covariance matrix
      Y.append(wij * Yij)
      # Store weighted samples for state covariance matrix
      X.append(wij * self.normalize(y[j]))

  # def _compute_cov_mats(
  #   self,
  #   X: List[np.ndarray],
  #   Y: List[np.ndarray],
  #   mu: np.ndarray,
  #   w_mu: float,
  #   nb_meas: int = 5,
  #   noise: bool = False,
  #   err_max: float = 25.0
  # ) -> Tuple[np.ndarray]:
  #   """
  #   Compute state and gradient covariance matrices based on quadrature
  #   points and system dynamics.

  #   :param nb_meas: Number of output measurements to use for adjoint
  #                   simulations. Defaults to 10.
  #   :type nb_meas: int
  #   :param use_eig: Whether to use eigenvalue-based analysis to determine
  #                   the maximum time (`tmax`) up to which the linear model
  #                   is valid. If True, eigenvalues are used to calculate
  #                   maximum valid timescales.
  #   :type use_eig: bool
  #   :param err_max: Maximum percentage error (in %) allowed between linear and
  #                   nonlinear models for determining the maximum time (`tmax`)
  #                   up to which the linear model is valid.
  #                   Defaults to 30.0.
  #   :type err_max: float

  #   :return: Tuple containing:
  #            - `X` (np.ndarray): State covariance matrix.
  #            - `Y` (np.ndarray): Gradient covariance matrix.
  #   :rtype: Tuple[np.ndarray, np.ndarray]
  #   """
  #   # Scaling factor for output measurements
  #   w_meas = 1.0 / np.sqrt(nb_meas)
  #   # Compute the initial solution for the system
  #   y0, rho = self.system.equil.get_init_sol(mu, noise=noise, sigma=1e-1)
  #   # Determine the smallest time scale for resolving system dynamics
  #   tmin = self.system.compute_timescale(y0, rho)
  #   # Generate a time quadrature grid and associated weights
  #   t, w_t = self._get_tquad(tmin)
  #   # Solve the nonlinear forward problem to compute the state evolution
  #   y = self.system.solve_fom(t, y0, rho)[0].T
  #   # Build an interpolator for the solution
  #   sol_interp = self._build_sol_interp(t, y)
  #   # Loop over each initial time for adjoint simulations
  #   for j in range(len(t)-1):
  #     # Generate a time grid for the j-th linear model
  #     tj = np.geomspace(t[j], t[-1], num=100)
  #     yj = sol_interp(tj)
  #     tj -= tj[0]
  #     # Determine the maximum valid time for linear model approximation
  #     tmax = self.system.compute_lin_tmax(tj, yj, rho, err_max)
  #     # Compute the combined quadrature weight (mu and t)
  #     wij = w_mu * w_t[j]
  #     if (tmax > tmin):
  #       # Generate a time grid for the j-th linear adjoint simulation
  #       tadj = np.geomspace(tmin, tmax, num=nb_meas)
  #       # Solve the j-th linear adjoint model
  #       Yij = w_meas * self.solve_lin_adjoint(tadj, y[j], rho).T
  #       # Store weighted samples for gradient covariance matrix
  #       Y.append(wij * Yij)
  #     # Store weighted samples for state covariance matrix
  #     X.append(wij * self.normalize(y[j]))

  def _get_tquad(
    self,
    tmin: float,
    deg: int = 2
  ) -> Tuple[np.ndarray]:
    """
    Generate time quadrature points and weights.

    :param tmin: Minimum time to resolve system dynamics.
    :type tmin: float
    :param deg: Degree of the quadrature rule. Defaults to 2.
    :type deg: int

    :return: Time quadrature points and corresponding weights.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    self.tgrid["start"] = tmin
    x, w = ops.get_quad_1d(
      x=self.system.get_tgrid(**self.tgrid),
      quad="gl",
      deg=deg,
      dist="uniform"
    )
    return x, np.sqrt(w)

  def _build_sol_interp(
    self,
    t: np.ndarray,
    x: np.ndarray
  ) -> sp.interpolate.interp1d:
    axis = 0 if (x.shape[0] == len(t)) else 1
    return sp.interpolate.interp1d(t, x, kind="cubic", axis=axis)

  # Linear adjoint model
  # -----------------------------------
  def solve_lin_adjoint(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float
  ) -> np.ndarray:
    """
    Solve the linear adjoint system for given time grid and initial condition.

    :param t: Array of time points for simulation.
    :type t: np.ndarray
    :param y0: Initial state for the linear adjoint simulation.
    :type y0: np.ndarray

    :return: Solution of the linear adjoint system.
    :rtype: np.ndarray
    """
    # Setting up
    self.system.use_rom = False
    y0 = self.system.set_up(y0, rho)
    # Compute linear operators
    self.system.compute_lin_fom_ops(y0)
    A, C = [getattr(self.system, k) for k in ("A", "C")]
    # Normalization procedure
    A = self.ov_xscale @ A @ self.xscale
    C = C @ self.xscale
    # Eigendecomposition
    l, V = sp.linalg.eig(A)
    Vinv = sp.linalg.inv(V)
    # Allocate memory
    shape = [len(t)] + list(C.T.shape)
    g = np.zeros(shape)
    # Compute solution
    VC = V.T @ C.T
    for (i, ti) in enumerate(t):
      L = np.diag(np.exp(ti*l))
      g[i] = Vinv.T @ (L @ VC)
    # Manipulate tensor
    g = np.transpose(g, axes=(1,2,0))
    g = np.reshape(g, (shape[1],-1))
    return g

  # Balanced modes
  # ===================================
  def compute_modes(
    self,
    X: np.ndarray,
    Y: np.ndarray,
    xnot: list = [],
    pod: bool = False,
    rank: int = 100,
    niter: int = 50
  ) -> None:
    """
    Compute balancing (and POD) modes based on input covariance matrices.

    :param X: State covariance matrix.
    :type X: np.ndarray
    :param Y: Gradient covariance matrix.
    :type Y: np.ndarray
    :param pod: Flag to compute POD modes instead of balancing modes.
                Defaults to False.
    :type pod: bool
    :param rank: Maximum rank for the reduced model. Defaults to 100.
    :type rank: int
    :param niter: Number of iterations for randomized SVD. Defaults to 30.
    :type niter: int
    """
    # Masking
    # -------------
    mask = self._make_mask(xnot)
    X, Y = X[mask], Y[mask]
    np.savetxt(self.path_to_saving+"/rom_mask.txt", mask, fmt='%d')
    # CoBRAS
    # -------------
    # Perform randomized SVD
    X, Y = [bkd.to_torch(z) for z in (X, Y)]
    U, s, V = ops.svd_lowrank(
      X=X,
      Y=Y,
      q=min(rank, X.shape[0]),
      niter=niter
    )
    # Compute balancing transformation
    sqrt_s = torch.diag(torch.sqrt(1.0/s))
    phi = X @ V @ sqrt_s
    psi = Y @ U @ sqrt_s
    # Save balancing modes
    s, phi, psi = [bkd.to_numpy(z) for z in (s, phi, psi)]
    data = {"s": s, "phi": phi, "psi": psi}
    filename = self.path_to_saving+"/cobras_bases.p"
    pickle.dump(data, open(filename, "wb"))
    # POD
    # -------------
    if pod:
      U, s, _ = torch.svd_lowrank(
        A=X,
        q=min(rank, X.shape[0]),
        niter=niter
      )
      s, phi = [bkd.to_numpy(z) for z in (s, U)]
      data = {"s": s, "phi": phi, "psi": phi}
      filename = self.path_to_saving+"/pod_bases.p"
      pickle.dump(data, open(filename, "wb"))

  def _make_mask(self, xnot: list) -> np.ndarray:
    """
    Generate a mask to exclude specific states from ROM computations.

    :param xnot: List of state indices to exclude.
    :type xnot: list

    :return: Boolean mask indicating included states.
    :rtype: np.ndarray
    """
    mask = np.ones(self.system.nb_eqs)
    xnot = np.array(xnot).astype(int).reshape(-1)
    mask[xnot] = 0
    return mask.astype(bool)

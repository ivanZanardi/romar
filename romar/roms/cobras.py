import sys
import torch
import numpy as np
import scipy as sp
import joblib as jl
import multiprocessing

from tqdm import tqdm
from typing import *

from .. import env
from .. import ops
from .. import utils
from .. import backend as bkd
from ..systems import SYS_TYPES
from .basic import Basic


class CoBRAS(Basic):

  """
  CoBRAS: Model Reduction for Nonlinear Systems by Balanced Truncation of
  State and Gradient Covariance.

  This class implements the CoBRAS method, a model reduction technique
  designed for nonlinear systems using balanced truncation of covariance
  matrices for states and gradients. It reduces computational complexity
  while preserving essential system dynamics.

  Reference:
  - https://doi.org/10.1137/22M1513228
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    system: SYS_TYPES,
    tgrid: Dict[str, float],
    quad_mu: Dict[str, np.ndarray],
    scale: bool = False,
    xref: Optional[Union[str, np.ndarray]] = None,
    xscale: Optional[Union[str, np.ndarray]] = None,
    path_to_saving: str = "./"
  ) -> None:
    """
    Initialize the CoBRAS class with the specified system, quadrature points,
    time grid, and saving configurations.

    :param system: Instance of the system to be reduced.
    :type system: SYS_TYPES
    :param tgrid: Dictionary specifying the time grid with required keys:
                  - "start": Start time of the simulation.
                  - "stop": End time of the simulation.
                  - "num": Number of time points.
    :type tgrid: Dict[str, float]
    :param quad_mu: Dictionary containing quadrature points and weights for
                    initial conditions. Must include:
                    - "x": A 1D numpy array of quadrature points.
                    - "w": A 1D numpy array of corresponding weights.
    :type quad_mu: Dict[str, np.ndarray]
    :param scale: Whether to apply scaling (default: False).
    :type scale: bool, optional
    :param xref: Mean reference values for scaling (array or file path).
    :type xref: Union[str, np.ndarray], optional
    :param xscale: Scaling factors (array or file path).
    :type xscale: Union[str, np.ndarray], optional
    :param path_to_saving: Directory path where the computed data and modes
                           will be saved. Defaults to "./".
    :type path_to_saving: str, optional

    :raises ValueError: If `tgrid` does not contain the required keys.
    """
    super(CoBRAS, self).__init__(path_to_saving)
    # Validate required `tgrid` keys
    required_keys = {"start", "stop", "num"}
    if (not required_keys.issubset(tgrid.keys())):
      raise ValueError(f"'tgrid' must contain keys: {required_keys}. " \
                       f"Received: {tgrid.keys()}")
    self.tgrid = tgrid
    self.tmin = self.tgrid["start"]
    # Store system and quadrature points
    self.system = system
    self.quad_mu = quad_mu
    # Set scaling if system equations are defined
    self._set_scaling(self.system.nb_eqs, xref, xscale, active=scale)

  # Compute covariance matrices
  # ===================================
  def compute_cov_mats(
    self,
    nb_meas: int = 5,
    noise: bool = False,
    err_max: float = 25.0,
    nb_workers: int = 1,
    fix_tmin: bool = False
  ) -> Tuple[np.ndarray]:
    """
    Compute state and gradient covariance matrices based on quadrature
    points and system dynamics.

    This method computes covariance matrices using quadrature points
    and system dynamics, with optional parallel execution.

    :param nb_meas: Number of output measurements for adjoint simulations.
                    Defaults to 5.
    :type nb_meas: int
    :param noise: Whether to add noise to the initial conditions.
    :type noise: bool, optional
    :param err_max: Maximum allowable percentage error (%) between linear
                    and nonlinear models for determining `tmax`.
                    Defaults to 25.0.
    :type err_max: float, optional
    :param nb_workers: Number of parallel workers for computation.
                       Defaults to 1 (sequential execution).
    :type nb_workers: int, optional
    :param fix_tmin: If True, ensures that `tmin` is set based on a fixed
                     global timescale instead of being dynamically computed.
    :type fix_tmin: bool, optional

    :return: Tuple containing:
            - `X` (np.ndarray): State covariance matrix.
            - `Y` (np.ndarray): Gradient covariance matrix.
            - `wx` (np.ndarray): Weights for state covariance matrix.
            - `wy` (np.ndarray): Weights for gradient covariance matrix.
    :rtype: Tuple[np.ndarray]
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
        wx=manager.list(),
        wy=manager.list(),
        nb_meas=nb_meas,
        noise=noise,
        err_max=err_max,
        fix_tmin=fix_tmin
      )
      if (nb_workers > 1):
        # Run jobs in parallel
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
      # Stack matrices
      X = np.vstack(list(kwargs["X"])).T
      Y = np.vstack(list(kwargs["Y"])).T
      wx = np.concatenate(list(kwargs["wx"])).reshape(-1)
      wy = np.concatenate(list(kwargs["wy"])).reshape(-1)
    return X, Y, wx, wy

  def _compute_cov_mats(
    self,
    X: List[np.ndarray],
    Y: List[np.ndarray],
    wx: List[np.ndarray],
    wy: List[np.ndarray],
    mu: np.ndarray,
    w_mu: float,
    nb_meas: int = 5,
    noise: bool = False,
    err_max: float = 25.0,
    fix_tmin: bool = False
  ) -> None:
    """
    Compute state and gradient covariance matrices based on quadrature
    points and system dynamics.

    :param X: List to store state covariance matrix contributions.
    :type X: List[np.ndarray]
    :param Y: List to store gradient covariance matrix contributions.
    :type Y: List[np.ndarray]
    :param wx: List to store weights corresponding to `X` matrix.
    :type wx: List[np.ndarray]
    :param wy: List to store weights corresponding to `y` matrix.
    :type wy: List[np.ndarray]
    :param mu: Initial condition quadrature point.
    :type mu: np.ndarray
    :param w_mu: Weight associated with the quadrature point `mu`.
    :type w_mu: float
    :param nb_meas: Number of output measurements for adjoint simulations.
    :type nb_meas: int, optional
    :param noise: Whether to include noise in the initial conditions.
    :type noise: bool, optional
    :param err_max: Maximum error percentage for linear model validity.
    :type err_max: float, optional
    :param fix_tmin: Whether to use a fixed `tmin` based on the global time
                     scale.
    :type fix_tmin: bool, optional

    :return: None (Results are stored in `X`, `Y`, `wx`, and `wy` lists).
    :rtype: None
    """
    # Scaling factor for output measurements
    w_meas = 1.0 / np.sqrt(nb_meas)
    # Compute the initial solution for the system
    y0, rho = self.system.equil.get_init_sol(mu, noise=noise, sigma=1e-1)
    # Determine the smallest time scale for resolving system dynamics
    tmin = self.system.compute_timescale(y0, rho)
    if fix_tmin:
      tmin = max(tmin, self.tmin)
    # Generate a time quadrature grid and associated weights
    t, w_t = self._get_tquad(tmin)
    # Solve the nonlinear forward problem to compute the state evolution
    y = self.system.solve_fom(t, y0, rho)[0].T
    # Build an interpolator for the solution
    sol_interp = self._build_sol_interp(t, y)
    # Loop over each initial time for adjoint simulations
    for j in range(len(t)-1):
      # Generate a time grid for the j-th linear model
      tj = np.geomspace(t[j], t[-1], num=100)
      yj = sol_interp(tj)
      tj -= tj[0]
      # Determine the maximum valid time for linear model approximation
      tmax = self.system.compute_lin_tmax(tj, yj, rho, err_max)
      # Compute the combined quadrature weight (mu and t)
      wij = w_mu * w_t[j]
      if (tmax > tmin):
        # Generate a time grid for the j-th linear adjoint simulation
        tadj = np.geomspace(tmin, tmax, num=nb_meas)
        # Solve the j-th linear adjoint model
        Yij = w_meas * self._solve_adjoint_lin(tadj, y[j], rho).T
        # Store weights and samples for gradient covariance matrix
        wy.append(np.repeat(wij, len(Yij)))
        Y.append(Yij)
      # Store weights and samples for state covariance matrix
      wx.append(np.repeat(wij, 1))
      X.append(self._apply_scaling(y[j]))

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

    :return: Tuple containing:
             - Time quadrature points (1D numpy array).
             - Corresponding weights (1D numpy array).
    :rtype: Tuple[np.ndarray]
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
    """
    Build a cubic interpolation function for the given solution.

    :param t: Time points.
    :type t: np.ndarray
    :param x: State trajectory or solution matrix.
    :type x: np.ndarray

    :return: Interpolation function.
    :rtype: sp.interpolate.interp1d
    """
    axis = 0 if (x.shape[0] == len(t)) else 1
    return sp.interpolate.interp1d(t, x, kind="cubic", axis=axis)

  def _solve_adjoint_lin(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float
  ) -> np.ndarray:
    """
    Solve the adjoint system of the linerized forward model for given time
    grid and initial conditions.

    :param t: Array of time points for simulation.
    :type t: np.ndarray
    :param y0: Initial state for the adjoint simulation.
    :type y0: np.ndarray

    :return: Solution of the adjoint system.
    :rtype: np.ndarray
    """
    # Setting up
    self.system.use_rom = False
    y0 = self.system.set_up(y0, rho)
    # Compute linear operators
    self.system.compute_lin_fom_ops(y0)
    A, C = [getattr(self.system, k) for k in ("A", "C")]
    # Scaling procedure
    A = self.ov_xscale_mat @ A @ self.xscale_mat
    C = C @ self.xscale_mat
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

  # Compute balanced modes
  # ===================================
  def compute_modes(
    self,
    X: np.ndarray,
    Y: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    xnot: Optional[List[int]] = None,
    rank: int = 100,
    niter: int = 50
  ) -> Dict[str, np.ndarray]:
    """
    Compute balancing modes based on input covariance matrices.

    :param X: State covariance matrix.
    :type X: np.ndarray
    :param Y: Gradient covariance matrix.
    :type Y: np.ndarray
    :param wx: Weights for state covariance matrix.
    :type wx: np.ndarray
    :param wy: Weights for gradient covariance matrix.
    :type wy: np.ndarray
    :param xnot: List of feature indices to exclude.
    :type xnot: List[int], optional
    :param rank: Maximum rank for randomized SVD.
    :type rank: int
    :param niter: Number of iterations for randomized SVD.
    :type niter: int

    :return: Dictionary containing computed PCA components.
    :rtype: Dict[str, np.ndarray]
    """
    # Mask covariance matrices
    mask = self._make_mask(X.shape[0], xnot)
    X, Y = X[mask], Y[mask]
    # Weight covariance matrices
    X *= wx
    Y *= wy
    # Balance covariance matrices
    rank = min(rank, X.shape[0])
    X, Y = map(bkd.to_torch, (X, Y))
    U, s, Vh = ops.svd_lowrank_xy(X=X, Y=Y, q=rank, niter=niter)
    # Compute balancing transformation
    sqrt_s = torch.diag(torch.sqrt(1.0/s))
    phi = X @ Vh @ sqrt_s
    psi = Y @ U @ sqrt_s
    # Save results
    data = {
      "s": s,
      "phi": phi,
      "psi": psi,
      "mask": mask,
      "xref": self.xref,
      "xscale": self.xscale
    }
    data = utils.map_nested_dict(bkd.to_numpy, data)
    self._save(data)
    return data

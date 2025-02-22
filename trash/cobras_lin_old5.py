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
from .basic import Basic


class CoBRASLin(Basic):

  """
  CoBRASLin: Model Reduction for Nonlinear Systems by Balanced Truncation of
  State and Gradient Covariance.

  This class implements the CoBRASLin method, a model reduction technique
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
    system: Any,
    path_to_data: str,
    scale: bool = False,
    xref: Optional[Union[str, np.ndarray]] = None,
    xscale: Optional[Union[str, np.ndarray]] = None,
    path_to_saving: str = "./"
  ) -> None:
    """
    Initialize the CoBRASLin class with the specified system, quadrature points,
    time grid, and saving configurations.

    :param system: Instance of the system to be reduced.
    :type system: Any
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
    super(CoBRASLin, self).__init__(
      system, path_to_data, scale, xref, xscale, path_to_saving
    )

  # Compute covariance matrices
  # ===================================
  def compute_cov_mats(
    self,
    irange: List[int],
    nb_meas: int = 5,
    err_max: float = 25.0,
    use_quad_w: bool = True,
    nb_workers: int = 1
  ) -> Tuple[np.ndarray]:
    """
    Compute state and gradient covariance matrices from system simulations.

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
            - `X` (np.ndarray): Unweighted state covariance matrix.
            - `X` (np.ndarray): Weighted state covariance matrix.
            - `Y` (np.ndarray): Weighted gradient covariance matrix.
    :rtype: Tuple[np.ndarray]
    """
    # Loop over computed solutions
    irange = np.sort(irange)
    iterable = tqdm(
      iterable=range(*irange),
      ncols=80,
      desc="Trajectories",
      file=sys.stdout
    )
    with multiprocessing.Manager() as manager:
      # Define input arguments for covariance matrices calculation
      kwargs = dict(
        X=manager.list(),
        Y=manager.list(),
        nb_mu=irange[1]-irange[0],
        nb_meas=nb_meas,
        err_max=err_max,
        use_quad_w=use_quad_w
      )
      if (nb_workers > 1):
        # Run jobs in parallel
        fun = env.make_fun_parallel(self._compute_cov_mats)
        jl.Parallel(nb_workers)(
          jl.delayed(fun)(index=i, **kwargs) for i in iterable
        )
      else:
        # Run jobs in series
        for i in iterable:
          self._compute_cov_mats(index=i, **kwargs)
      # Stack matrices
      X = np.vstack(list(kwargs["X"])).T
      Y = np.vstack(list(kwargs["Y"])).T
    return X, Y

  def _compute_cov_mats(
    self,
    index: int,
    X: List[np.ndarray],
    Y: List[np.ndarray],
    nb_mu: int,
    nb_meas: int = 5,
    err_max: float = 25.0,
    use_quad_w: bool = True
  ) -> None:
    """
    Compute state and gradient covariance matrices using quadrature points
    and system dynamics.

    This function evaluates state trajectories and their corresponding gradient
    adjoint solutions to construct weighted covariance matrices. The computed
    matrices are stored in the provided lists (`X`, `X`, and `Y`).

    :param X: List to store weighted state covariance matrix contributions.
    :type X: List[np.ndarray]
    :param Y: List to store weighted gradient covariance matrix contributions.
    :type Y: List[np.ndarray]
    :param nb_meas: Number of measurement points for adjoint simulations.
                    Default is 5.
    :type nb_meas: int, optional
    :param err_max: Maximum allowable error percentage between the linearized
                    and nonlinear models, used to determine the validity range
                    (`tmax`) of the linear approximation.
                    Default is 25.0.
    :type err_max: float, optional

    :return: None (results are appended to `X` and `Y`).
    :rtype: None
    """
    # Load solution
    data = utils.load_case(path=self.path_to_data, index=index)
    if (data is not None):
      # Unpack
      # > Time grid
      tmin = float(data["tmin"])
      t = data["t"].reshape(-1)
      nt = len(t)
      # > Solution
      y = data["y"].T
      rho = float(data["rho"])
      # Set weights
      w_meas = 1.0/np.sqrt(nb_meas)
      w_t, w_mu = [data[k] for k in ("w_t", "w_mu")]
      if (not use_quad_w):
        w_mu = 1.0/np.sqrt(nb_mu)
        w_t[:] = 1.0/np.sqrt(nt)
      # Build an interpolator for the solution
      sol_interp = self._build_sol_interp(t, y)
      # Loop over each initial time for adjoint simulations
      for j in range(nt-1):
        # Generate a time grid for the j-th linear model
        tj = np.geomspace(t[j]+1e-15, t[-1], num=100)
        yj = sol_interp(tj)
        tj -= tj[0]
        # Determine the maximum valid time for linear model approximation
        tmax = self.system.compute_lin_tmax(tj, yj, rho, err_max)
        # Compute the combined quadrature weight (mu and t)
        wij = w_mu*w_t[j]
        if (tmax > tmin):
          # Generate a time grid for the j-th linear adjoint simulation
          tadj = np.geomspace(tmin, tmax, num=nb_meas)
          # Solve the j-th linear adjoint model
          Yij = w_meas*self._solve_adjoint_lin(tadj, y[j], rho).T
          # Store samples for gradient covariance matrix
          Y.append(wij*Yij)
        # Store samples for state covariance matrix
        X.append(wij*self._apply_scaling(y[j]))

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
    xnot: Optional[List[int]] = None,
    rank: int = 100,
    niter: int = 50
  ) -> None:
    """
    Compute balancing modes using covariance matrices.

    :param X: Weighted state covariance matrix.
    :type X: np.ndarray
    :param Y: Weighted gradient covariance matrix.
    :type Y: np.ndarray
    :param xnot: List of feature indices to exclude (default: None).
    :type xnot: Optional[List[int]]
    :param rank: Maximum rank for randomized SVD (default: 100).
    :type rank: int
    :param niter: Number of iterations for randomized SVD (default: 50).
    :type niter: int

    :return: Dictionary containing computed ROM data including basis functions
             and singular values.
    :rtype: Dict[str, np.ndarray]
    """
    # Mask covariance matrices
    mask = self._make_mask(X.shape[0], xnot)
    X, Y = X[mask], Y[mask]
    # Balance covariance matrices
    rank = min(rank, X.shape[0])
    X, Y = map(bkd.to_torch, (X, Y))
    U, s, V = ops.svd_lowrank_xy(X=X, Y=Y, q=rank, niter=niter)
    # Compute balancing transformation
    sqrt_s = torch.diag(torch.sqrt(1.0/s))
    phi = X @ V @ sqrt_s
    psi = Y @ U @ sqrt_s
    # Save results
    data = utils.map_nested_dict(bkd.to_numpy, {
      "s": s,
      "phi": {r: phi[:,:r] for r in range(2,rank+1)},
      "psi": {r: psi[:,:r] for r in range(2,rank+1)},
      "mask": mask,
      "xref": self.xref,
      "xscale": self.xscale
    })
    self._save(data)

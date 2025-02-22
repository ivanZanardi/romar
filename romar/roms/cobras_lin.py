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
from .cobras import CoBRAS


class CoBRASLin(CoBRAS):

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
    err_max: float = 25.0,
    nb_meas: int = 5,
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
    :param nb_workers: Number of parallel workers for computation.
                       Defaults to 1 (sequential execution).
    :type nb_workers: int, optional

    :return: Tuple containing:
            - `X` (np.ndarray): Weighted state covariance matrix.
            - `Y` (np.ndarray): Weighted gradient covariance matrix.
    :rtype: Tuple[np.ndarray]
    """
    return self._compute_cov_mats_loop(
      kwargs=dict(
        err_max=err_max
      ),
      irange=irange,
      nb_meas=nb_meas,
      use_quad_w=use_quad_w,
      nb_workers=nb_workers
    )

  def _compute_cov_mats(
    self,
    index: int,
    X: List[np.ndarray],
    Y: List[np.ndarray],
    nb_mu: int,
    err_max: float = 25.0,
    nb_meas: int = 5,
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

    :return: None (results are appended to `X` and `Y`).
    :rtype: None
    """
    # Load solution
    data = utils.load_case(path=self.path_to_data, index=index)
    if (data is not None):
      # Setting up
      # -----------
      # Unpacking
      t = data["t"].reshape(-1)
      y = data["y"].T
      rho = float(data["rho"])
      tmin = float(data["tmin"])
      nt = len(t)
      # Set up system
      self.system.use_rom = False
      self.system.mix.set_rho(rho)
      # Build an interpolator for the solution
      ysol = self._build_sol_interp(t, y)
      # Set weights
      # -----------
      w_meas = 1.0/np.sqrt(nb_meas)
      w_t, w_mu = [data[k] for k in ("w_t", "w_mu")]
      if (not use_quad_w):
        w_t[:] = 1.0/np.sqrt(nt)
        w_mu = 1.0/np.sqrt(nb_mu)
      # State covariance matrix
      # -----------
      Xi = w_mu * w_t * self._apply_scaling(y).T
      X.append(Xi.T)
      # Gradient covariance matrix
      # -----------
      # Set time weights
      if (not use_quad_w):
        w_t[:] = 1.0/np.sqrt(nt-1)
      # Loop over each sampled initial time
      for i in range(nt-1):
        # > Generate a time grid for the i-th linear model
        t0 = max(t[i], tmin)
        ti = np.geomspace(t0, t[-1], num=100)
        yi = ysol(ti)
        ti = ti-t0
        # Determine the maximum valid time for linear model approximation
        tmax = self.system.compute_lin_tmax(ti, yi, rho, err_max)
        # Solve the adjoint problem and store samples
        if (tmax > 0.0):
          Yi = self._solve_adj(
            t0=t0,
            tf=t0+tmax,
            nb_meas=nb_meas,
            y0=y[i]
          )
          Yi = w_mu * w_t[i] * w_meas * Yi
          Y.append(Yi)

  def _solve_adj(
    self,
    t0: float,
    tf: float,
    nb_meas: int,
    y0: np.ndarray
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
    # Generate a time grid
    t = np.geomspace(t0, tf, num=nb_meas+1)[1:] - t0
    # LTI Jacobian operator
    A = self.system.jac(0.0, y0)
    A = self.ov_xscale_mat @ A @ self.xscale_mat
    # Eigendecomposition
    l, V = sp.linalg.eig(A)
    Vinv = sp.linalg.inv(V)
    # Allocate memory
    shape = [len(t)] + list(self.C.T.shape)
    g = np.zeros(shape)
    # Compute solution
    VC = V.T @ self.C.T
    for (i, ti) in enumerate(t):
      L = np.diag(np.exp(ti*l))
      g[i] = Vinv.T @ (L @ VC)
    # Manipulate tensor
    g = np.transpose(g, axes=(2,0,1))
    g = np.reshape(g, (-1,shape[1]))
    return g

import numpy as np
import scipy as sp

from typing import *

from .. import ops
from .. import utils
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
    system: callable,
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
    use_quad_w: bool = False,
    nb_workers: int = 1
  ) -> Dict[str, np.ndarray]:
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
    # Training case indices to be loaded
    indices_mu = np.arange(*irange)
    # Loop over training cases
    return self._compute_cov_mats_loop(
      kwargs=dict(
        err_max=err_max
      ),
      indices_mu=indices_mu,
      nb_meas=nb_meas,
      use_quad_w=use_quad_w,
      nb_workers=nb_workers
    )

  def _compute_cov_mats(
    self,
    index: int,
    X: List[np.ndarray],
    Y: List[np.ndarray],
    conv: List[int],
    nb_mu: int,
    nb_meas: int = 5,
    use_quad_w: bool = True,
    err_max: float = 25.0
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
      # Extract solution
      y = data["y"].T
      t = data["t"].reshape(-1)
      tmin = float(data["tmin"])
      nb_t = len(t)
      # Set density
      self.system.mix.set_rho(rho=data["rho"])
      # Build an interpolator for the solution
      ysol = self._build_sol_interp(t, y)
      # State covariance matrix
      if use_quad_w:
        w = data["w_mu"] * data["w_t"].reshape(-1,1)
      else:
        w = 1.0/np.sqrt(nb_mu*nb_t)
      X.append(w * self._apply_scaling(y))
      # Gradient covariance matrix
      Yi, ti = [], []
      for j in range(nb_t-1):
        # > Generate a time grid for the i-th linear model
        t0 = max(t[j], tmin)
        tj = np.geomspace(t0, t[-1], num=100)
        yj = ysol(tj)
        tj = tj-t0
        # > Determine the maximum valid time for linear model approximation
        tmax = self.system.compute_lin_tmax(tj, yj, rho, err_max)
        # > Solve the adjoint problem and store samples
        if (tmax > 0.0):
          gradj = self._solve_adj(
            t0=t0,
            tf=t0+tmax,
            nb_meas=nb_meas,
            y0=y[j]
          )
          Yi.append(gradj)
          ti.append(t0)
          conv.append(self.nb_out)
        else:
          conv.append(0)
      # > Weight and store adjoint solutions
      nb_ti = len(ti)
      if (nb_ti > 0):
        w_meas = 1.0/np.sqrt(nb_meas)
        if use_quad_w:
          _, w_t = ops.get_quad_1d(
            x=np.asarray(ti).reshape(-1),
            quad="trapz",
            dist="uniform"
          )
          w_t = np.sqrt(w_t)
          w = w_meas * data["w_mu"] * w_t
        else:
          w = np.full(nb_ti, w_meas/np.sqrt(nb_mu*nb_ti))
        Yi = [w[j]*Yij for (j, Yij) in enumerate(Yi)]
        Y.append(np.vstack(Yi))

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
    g = np.transpose(g, axes=(1,2,0))
    g = np.reshape(g, (shape[1],-1))
    return g.T

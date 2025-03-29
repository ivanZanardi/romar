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
    system: callable,
    path_to_data: str,
    scale: bool = False,
    xref: Optional[Union[str, np.ndarray]] = None,
    xscale: Optional[Union[str, np.ndarray]] = None,
    path_to_saving: str = "./"
  ) -> None:
    """
    Initialize the CoBRAS class with the specified system, quadrature points,
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
    super(CoBRAS, self).__init__(
      system, path_to_data, scale, xref, xscale, path_to_saving
    )
    # Setting up system
    self.system.use_rom = False
    self.system.set_fun_jac()
    # Setting output
    self.C = self.system.C @ self.xscale_mat
    self.nb_out = self.C.shape[0]

  # Compute covariance matrices
  # ===================================
  def compute_cov_mats(
    self,
    irange: List[int],
    dt_log: float = 2.0,
    tf_min: float = 1e-7,
    t0_perc: float = 1.0,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    tout: float = 30.0,
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
    # Loop over computed solutions
    indices_mu = np.arange(*irange)
    iterable = tqdm(
      iterable=indices_mu,
      ncols=80,
      desc="Trajectories",
      file=sys.stdout
    )
    with multiprocessing.Manager() as manager:
      # Define input arguments for covariance matrices calculation
      kwargs = dict(
        X=manager.list(),
        Y=manager.list(),
        conv=manager.list(),
        nb_mu=len(indices_mu),
        dt_log=dt_log,
        tf_min=tf_min,
        t0_perc=np.clip(t0_perc, 0.1, 1.0),
        rtol=rtol,
        atol=atol,
        tout=tout,
        nb_meas=nb_meas,
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
      # Converged adjoints
      conv = list(kwargs["conv"])
      nb_adj = len(conv) * self.nb_out
      print(f"Total converged adjoints: {sum(conv)}/{nb_adj}")
    return {"X": X, "Y": Y}

  def _compute_cov_mats(
    self,
    index: int,
    X: List[np.ndarray],
    Y: List[np.ndarray],
    conv: List[int],
    nb_mu: int,
    dt_log: float = 2.0,
    tf_min: float = 1e-7,
    t0_perc: float = 1.0,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    tout: float = 30.0,
    nb_meas: int = 5,
    use_quad_w: bool = False
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
    # Load data
    data = utils.load_case(path=self.path_to_data, index=index)
    if (data is not None):
      # State covariance matrix
      # -----------
      Xi = self._compute_state_cov(data, nb_mu, use_quad_w)
      X.append(Xi)
      # Gradient covariance matrix
      # -----------
      # > Set density
      self.system.mix.set_rho(rho=data["rho"])
      # > Build an interpolator for the solution
      y = data["y"].T
      t = data["t"].reshape(-1)
      ysol = self._build_sol_interp(t, y)
      # > Sample initial times uniformly
      ti, Yi = [], []
      t0_indices = self._sample_t0_indices(size=len(t), perc=t0_perc)
      for j in t0_indices:
        # > Set initial/final times
        t0 = max(t[j], data["tmin"])
        tf = t0 * np.power(1e1, dt_log)
        tf = min(max(tf, tf_min), t[-1])
        # > Solve the j-th adjoint problem and store samples
        gradj, convj = self._solve_adj(
          t0=t0,
          tf=tf,
          nb_meas=nb_meas,
          ysol=ysol,
          rtol=rtol,
          atol=atol,
          tout=tout
        )
        if (gradj is not None):
          Yi.append(gradj)
          ti.append(t0)
        conv.append(convj)
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
          w = w_meas * data["w_mu"] * np.sqrt(w_t)
        else:
          w = np.full(nb_ti, w_meas/np.sqrt(nb_mu*nb_ti))
        Yi = [w[j]*Yij for (j, Yij) in enumerate(Yi)]
        Y.append(np.vstack(Yi))

  def _sample_t0_indices(
    self,
    size: int,
    perc: float
  ) -> np.ndarray:
    # Indices vector
    i = np.arange(size)
    # Number of samples
    ns = np.round(perc * size)
    # Sample uniformly
    ij = np.array_split(i, ns)
    return np.asarray([j[j.size//2] for j in ij])

  def _solve_adj(
    self,
    t0: float,
    tf: float,
    nb_meas: int,
    ysol: sp.interpolate.interp1d,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    tout: float = 30.0
  ) -> np.ndarray:
    """Solve the adjoint problem"""
    # Generate a time grid
    t = np.geomspace(t0, tf, num=nb_meas+1)
    t = tf - np.flip(t)
    # Make solve function with timeout control
    solve_ivp = utils.make_solve_ivp(tout)
    # Compute gradients
    grad = []
    conv = 0
    for g0 in self.C:
      sol = solve_ivp(
        fun=self._fun_adj,
        t_span=[t[0],t[-1]],
        y0=g0,
        method="BDF",
        t_eval=t[1:],
        args=(tf, ysol),
        first_step=1e-10,
        rtol=rtol,
        atol=atol,
        jac=self._jac_adj
      )
      if ((sol is not None) and sol.success):
        grad.append(sol.y.T)
        conv += 1
    grad = np.vstack(grad) if (conv > 0) else None
    return grad, conv

  def _fun_adj(
    self,
    t: np.ndarray,
    g: np.ndarray,
    tf: float,
    ysol: sp.interpolate.interp1d
  ) -> np.ndarray:
    return self._jac_adj(t, g, tf, ysol) @ g

  def _jac_adj(
    self,
    t: np.ndarray,
    g: np.ndarray,
    tf: float,
    ysol: sp.interpolate.interp1d
  ) -> np.ndarray:
    tau = tf-t
    j = self.system.jac(t=tau, y=ysol(tau))
    j = self.ov_xscale_mat @ j @ self.xscale_mat
    return j.T

  # Utils
  # -----------------------------------
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

  # Compute balanced modes
  # ===================================
  def compute_modes(
    self,
    X: np.ndarray,
    Y: np.ndarray,
    rotation: Optional[str] = None,
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
    mask = self._make_mask(
      nb_feat=X.shape[0],
      xnot=xnot
    )
    X, Y = X[mask], Y[mask]
    # Balance covariance matrices
    rank = min(rank, X.shape[0])
    X, Y = map(bkd.to_torch, (X, Y))
    U, s, V = ops.svd_lowrank_xy(X=X, Y=Y, q=rank, niter=niter)
    # Compute balancing transformation
    sqrt_s = torch.diag(torch.sqrt(1.0/s))
    phi = bkd.to_numpy(X @ V @ sqrt_s)
    psi = bkd.to_numpy(Y @ U @ sqrt_s)
    # Vanilla model
    # -------------
    phi = {r: phi[:,:r] for r in range(2,rank+1)}
    psi = {r: psi[:,:r] for r in range(2,rank+1)}
    data = {
      "s": s,
      "phi": phi,
      "psi": psi,
      "mask": mask,
      "xref": self.xref,
      "xscale": self.xscale
    }
    self._save(data)
    # Rotated model
    # -------------
    if (rotation is not None):
      rot = self.get_rotator(rotation)
      data["psi"] = {r: rot.fit_transform(basis) for (r, basis) in psi.items()}
      self._save(data, identifier=rotation)

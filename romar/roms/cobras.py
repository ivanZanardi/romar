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
  - Otto et al., "Model Reduction for Nonlinear Systems by Balanced Truncation
    of State and Gradient Covariance", SIAM J. Sci. Comput. (2023)
    DOI: https://doi.org/10.1137/22M1513228
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
    Initialize the base class.

    :param system: System object containing the governing equations. Must
                   provide `nb_eqs`, `C`, `jac()`, and `set_fun_jac()` methods.
    :type system: callable
    :type system: callable
    :param path_to_data: Path to the dataset or simulation results used for
                         model reduction.
    :type path_to_data: str
    :param scale: Whether to apply scaling to the system variables.
                  Default is ``False``.
    :type scale: bool, optional
    :param xref: Reference values for centering (shape: (nb_feat,)).
                 Can be a string identifier or an array.
    :type xref: str or np.ndarray, optional
    :param xscale: Scaling factors (shape: (nb_feat,)).
                   Can be a string identifier or an array.
    :type xscale: str or np.ndarray, optional
    :param path_to_saving: Directory where results will be saved.
                           Default is ``"./"``.
    :type path_to_saving: str, optional
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
    Compute state and gradient covariance matrices over a set of simulations.

    :param irange: List specifying start and end indices [start, stop) of
                   simulations.
    :type irange: List[int]
    :param dt_log: Logarithmic time expansion factor for adjoint simulations.
    :type dt_log: float, optional
    :param tf_min: Minimum final time for adjoint simulations.
    :type tf_min: float, optional
    :param t0_perc: Percentage of time points to sample for adjoint
                    problems (0.1 to 1.0).
    :type t0_perc: float, optional
    :param rtol: Relative tolerance for ODE integrator.
    :type rtol: float, optional
    :param atol: Absolute tolerance for ODE integrator.
    :type atol: float, optional
    :param tout: Timeout (in seconds) for each adjoint simulation.
    :type tout: float, optional
    :param nb_meas: Number of time measurements per adjoint simulation.
    :type nb_meas: int, optional
    :param use_quad_w: Whether to apply quadrature weights over time and
                       parameter samples.
    :type use_quad_w: bool, optional
    :param nb_workers: Number of parallel workers. Set to 1 for serial execution.
    :type nb_workers: int, optional

    :return: Dictionary with:
             - ``X``: state covariance matrix (np.ndarray)
             - ``Y``: gradient covariance matrix (np.ndarray)
    :rtype: Dict[str, np.ndarray]
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
    matrices are stored in the provided lists (`X` and `Y`).

    :param index: Index of the simulation case to process.
    :type index: int
    :param X: List to collect state covariance matrix contributions.
    :type X: List[np.ndarray]
    :param Y: List to collect gradient covariance matrix contributions.
    :type Y: List[np.ndarray]
    :param conv: List to record number of successfully converged adjoint solutions.
    :type conv: List[int]
    :param nb_mu: Total number of parameter samples.
    :type nb_mu: int
    :param dt_log: Logarithmic time factor for adjoint integration.
    :type dt_log: float, optional
    :param tf_min: Minimum final time for adjoint simulation.
    :type tf_min: float, optional
    :param t0_perc: Percentage of time points to sample for adjoints.
    :type t0_perc: float, optional
    :param rtol: Relative tolerance for ODE integrator.
    :type rtol: float, optional
    :param atol: Absolute tolerance for ODE integrator.
    :type atol: float, optional
    :param tout: Timeout (in seconds) for each adjoint simulation.
    :type tout: float, optional
    :param nb_meas: Number of adjoint output samples per simulation.
    :type nb_meas: int, optional
    :param use_quad_w: Whether to apply quadrature weights.
    :type use_quad_w: bool, optional

    :return: None (results are appended to shared lists).
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
    """
    Uniformly sample a subset of time indices.

    :param size: Total number of time steps in the solution.
    :type size: int
    :param perc: Fraction (0 < perc ≤ 1) of time points to sample.
    :type perc: float

    :return: Array of sampled time indices.
    :rtype: np.ndarray
    """
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
  ) -> Tuple[Optional[np.ndarray], int]:
    """
    Solve the adjoint system from tf to t0.

    :param t0: Initial physical time.
    :type t0: float
    :param tf: Final physical time.
    :type tf: float
    :param nb_meas: Number of time points at which to evaluate gradients.
    :type nb_meas: int
    :param ysol: Interpolator of the primal solution.
    :type ysol: sp.interpolate.interp1d
    :param rtol: Relative tolerance for ODE solver.
    :type rtol: float, optional
    :param atol: Absolute tolerance for ODE solver.
    :type atol: float, optional
    :param tout: Timeout in seconds for solver.
    :type tout: float, optional

    :return: Tuple of:
             - grad: gradient trajectory (np.ndarray) or None if failed.
             - conv: number of successfully converged adjoint solves.
    :rtype: Tuple[Optional[np.ndarray], int]
    """
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
    """
    Evaluate the adjoint RHS function.

    :param t: Current time.
    :type t: np.ndarray
    :param g: Adjoint vector at time t.
    :type g: np.ndarray
    :param tf: Final physical time.
    :type tf: float
    :param ysol: Primal solution interpolator.
    :type ysol: sp.interpolate.interp1d

    :return: Adjoint RHS vector.
    :rtype: np.ndarray
    """
    return self._jac_adj(t, g, tf, ysol) @ g

  def _jac_adj(
    self,
    t: np.ndarray,
    g: np.ndarray,
    tf: float,
    ysol: sp.interpolate.interp1d
) -> np.ndarray:
    """
    Evaluate the adjoint Jacobian function.

    :param t: Current time.
    :type t: np.ndarray
    :param g: Adjoint vector at time t.
    :type g: np.ndarray
    :param tf: Final physical time.
    :type tf: float
    :param ysol: Primal solution interpolator.
    :type ysol: sp.interpolate.interp1d

    :return: Adjoint Jacobian matrix.
    :rtype: np.ndarray
    """
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
    Construct a cubic interpolator for the given solution.

    :param t: Array of time points.
    :type t: np.ndarray
    :param x: Solution array (either shape [T, D] or [D, T]).
    :type x: np.ndarray

    :return: Cubic interpolation function over time.
    :rtype: sp.interpolate.interp1d
    """
    axis = 0 if (x.shape[0] == len(t)) else 1
    return sp.interpolate.interp1d(t, x, kind="cubic", axis=axis)

  # Compute basis
  # ===================================
  def compute_basis(
    self,
    X: np.ndarray,
    Y: np.ndarray,
    rotation: Optional[str] = None,
    xnot: Optional[List[int]] = None,
    max_y_norm: Optional[float] = None,
    rank: int = 100,
    niter: int = 50
  ) -> None:
    """
    Compute balancing modes from state and gradient covariance matrices.

    :param X: State covariance matrix (shape: [nb_features, nb_snapshots]).
    :type X: np.ndarray
    :param Y: Gradient covariance matrix (shape: [nb_features, nb_snapshots]).
    :type Y: np.ndarray
    :param rotation: Optional rotation method to apply to the test basis.
    :type rotation: str, optional
    :param xnot: List of feature indices to exclude from the basis.
    :type xnot: List[int], optional
    :param max_y_norm: Optional clipping threshold for adjoint snapshot norm.
    :type max_y_norm: float, optional
    :param rank: Maximum number of basis vectors to compute. Default is 100.
    :type rank: int
    :param niter: Number of power iterations for randomized SVD. Default is 50.
    :type niter: int

    :return: None. Saves computed basis data to disk.
    :rtype: None

    :notes:
      - Saves the following data to disk:
        - ``s``: Singular values
        - ``phi``: Trial basis (state space).
        - ``psi``: Test basis (adjoint space).
        - ``mask``: Feature inclusion mask
        - ``xref``, ``xscale``: Scaling parameters
    """
    # Mask covariance matrices
    mask = self._make_mask(
      nb_feat=X.shape[0],
      xnot=xnot
    )
    X, Y = X[mask], Y[mask]
    # Remove outliers adjoint snaphots with too high norm
    # > Numerical instabilites in adjoint computation
    if (max_y_norm is not None):
      y_norm = np.linalg.norm(Y, axis=0)
      i = np.where(y_norm <= max_y_norm)[0]
      Y = Y[:,i]
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
      rot = self._get_rotator(rotation)
      data["psi"] = {r: rot.fit_transform(basis) for (r, basis) in psi.items()}
      self._save(data, identifier=rotation)

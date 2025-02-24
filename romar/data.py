import os
import sys
import numpy as np
import joblib as jl
import pandas as pd
import dill as pickle

from tqdm import tqdm
from pyDOE2 import lhs
from typing import *

from . import env
from . import ops
from . import utils
from .systems.thermochemistry.equilibrium import MU_VARS
from .roms.utils import compute_scaling, POSSIBLE_SCALINGS


class Data(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    system: Any,
    grids: dict,
    path_to_saving: str = "./"
  ) -> None:
    # Store system
    self.system = system
    # Grids
    self.grids = grids
    # > Time grid
    self.tmin = self.grids["t"]["start"]
    self.tvec = np.geomspace(**self.grids["t"])
    # Saving options
    self.path_to_saving = path_to_saving
    os.makedirs(self.path_to_saving, exist_ok=True)

  # Training data
  # ===================================
  def generate_data_train(
    self,
    init_sols: Optional[Union[str,np.ndarray]] = None,
    noise: bool = False,
    sigma: float = 1e-1,
    fix_tmin: bool = False,
    nb_workers: int = 1
  ) -> Tuple[np.ndarray]:
    if (init_sols is None):
      # Get initial conditions quadrature points and weights
      mu, w_mu = self._get_quad_mu()
      # Get initial solutions
      init_sols = self._get_init_sols(mu, w_mu, noise, sigma)
    else:
      if isinstance(init_sols, str):
        init_sols = pd.read_csv(init_sols).values
    # Compute solutions
    return self.compute_sols(
      y0=init_sols[:,:-1],
      w_mu=init_sols[:,-1],
      fix_tmin=fix_tmin,
      nb_workers=nb_workers
    )

  def _get_quad_mu(
    self,
    deg: int = 2
  ) -> Tuple[np.ndarray]:
    # Compute quadrature points/weights for initial conditions
    x, dist = [], []
    for k in MU_VARS:
      kfun = np.linspace if (k == "T") else np.geomspace
      x.append(kfun(**self.grids["mu"][k]))
      kdist = "loguniform" if (k == "Te") else "uniform"
      dist.append(kdist)
    x, w = ops.get_quad_nd(
      x=x,
      deg=deg,
      dist=dist
    )
    w = np.sqrt(w)
    # Save quadrature points/weights
    pd.DataFrame(
      data=np.vstack([x.T, w]).T,
      columns=list(MU_VARS)+["w"]
    ).to_csv(
      path_or_buf=self.path_to_saving + "/quad_mu.csv",
      float_format="%.10e",
      index=False
    )
    return x, w

  # Testing data
  # ===================================
  def generate_data_test(
    self,
    init_sols: Optional[Union[str,np.ndarray]] = None,
    limits: Optional[Dict[str, List[float]]] = None,
    nb_samples: Optional[int] = None,
    log_vars: Tuple[str] = ("Te", "rho"),
    eps: float = 1e-8,
    nb_workers: int = 1
  ) -> Tuple[np.ndarray]:
    if (init_sols is None):
      # Get sampled initial conditions
      mu = self._get_samples_mu(
        limits=limits,
        nb_samples=nb_samples,
        log_vars=log_vars,
        eps=eps
      )
      # Get initial solutions
      init_sols = self._get_init_sols(mu)
    else:
      if isinstance(init_sols, str):
        init_sols = pd.read_csv(init_sols).values
    # Compute solutions
    return self.compute_sols(
      y0=init_sols,
      t=self.tvec,
      nb_workers=nb_workers
    )

  def _get_samples_mu(
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
    # Save
    pd.DataFrame(data=mu, columns=MU_VARS).to_csv(
      path_or_buf=self.path_to_saving + "/samples_mu.csv",
      float_format="%.10e",
      index=False
    )
    return mu

  # Compute solutions
  # ===================================
  def compute_sols(
    self,
    y0: np.ndarray,
    w_mu: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
    fix_tmin: bool = False,
    nb_workers: int = 1
  ) -> List[Optional[float]]:
    # Loop over initial conditions quadrature points
    nb_samples = len(y0)
    iterable = tqdm(
      iterable=range(nb_samples),
      ncols=80,
      desc="Trajectories",
      file=sys.stdout
    )
    # Define input arguments
    kwargs = dict(
      t=t,
      fix_tmin=fix_tmin
    )
    if (nb_workers > 1):
      # Run jobs in parallel
      runtime = jl.Parallel(nb_workers)(
        jl.delayed(env.make_fun_parallel(self.compute_sol))(
          index=i,
          y0=y0[i],
          w_mu=w_mu[i] if (w_mu is not None) else None,
          **kwargs
        ) for i in iterable
      )
    else:
      # Run jobs in series
      runtime = [self.compute_sol(
        index=i,
        y0=y0[i],
        w_mu=w_mu[i] if (w_mu is not None) else None,
        **kwargs
      ) for i in iterable]
    # Compute scalings
    self._compute_scalings(nb_samples=nb_samples, nb_workers=nb_workers)
    return runtime

  def compute_sol(
    self,
    mu: Optional[np.ndarray] = None,
    y0: Optional[np.ndarray] = None,
    index: Optional[int] = None,
    w_mu: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    fix_tmin: bool = False,
    filename: Optional[str] = None
  ) -> Optional[float]:
    # Unpack initial solution for the system
    if (y0 is not None):
      y0, rho = y0[:-1], y0[-1]
    else:
      y0, rho = self.system.get_init_sol(mu)
    # Set time grid
    if (t is None):
      # > Determine the smallest time scale for resolving system dynamics
      tmin = self.system.compute_timescale(y0, rho)
      if fix_tmin:
        tmin = self.tmin
      # > Generate a time quadrature grid and associated weights
      t, w_t = self._get_quad_t(tmin)
    else:
      tmin = np.amin(t[t>0.0])
      w_t = None
    # Solve the nonlinear forward problem to compute the state evolution
    y, runtime = self.system.solve_fom(t, y0, rho)
    # Check convergence
    if ((y is not None) and (y.shape[1] == len(t))):
      sol = {
        "index": index,
        "w_mu": w_mu,
        "t": t,
        "w_t": w_t,
        "tmin": tmin,
        "y0": y0,
        "rho": rho,
        "y": y,
        "runtime": runtime
      }
      utils.save_case(
        path=self.path_to_saving,
        index=index,
        data=sol,
        filename=filename
      )
    else:
      runtime = None
    return runtime

  def _get_quad_t(
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
    self.grids["t"]["start"] = tmin
    x = np.geomspace(**self.grids["t"])
    x = np.insert(x, 0, 0.0)
    x, w = ops.get_quad_1d(x=x, quad="gl", deg=deg, dist="uniform")
    return x, np.sqrt(w)

  def _compute_scalings(
    self,
    nb_samples: int,
    nb_workers: int = 1
  ) -> None:
    X = np.hstack(utils.load_case_parallel(
      path=self.path_to_saving,
      irange=[0,nb_samples],
      key="y",
      nb_workers=nb_workers,
      verbose=False
    ))
    scalings = {s: compute_scaling(scaling=s, X=X) for s in POSSIBLE_SCALINGS}
    # Save scalings
    with open(self.path_to_saving+"/scalings.p", "wb") as file:
      pickle.dump(scalings, file)

  def _get_init_sols(
    self,
    mu: np.ndarray,
    w_mu: Optional[np.ndarray] = None,
    noise: bool = False,
    sigma: float = 1e-1
  ) -> np.ndarray:
    sols = []
    for i in range(len(mu)):
      sol = self.system.get_init_sol(mu[i], noise=noise, sigma=sigma)
      print("SOL:", sol[0])
      sol = list(sol)
      if (w_mu is not None):
        sol.append(w_mu[i])
      sol = np.concatenate(list(map(np.atleast_1d, sol)))
      sols.append(sol)
    sols = np.vstack(sols)
    # Save
    columns = self.system.var_names + ["rho"]
    if (w_mu is not None):
      columns.append("weight")
    pd.DataFrame(data=sols, columns=columns).to_csv(
      path_or_buf=self.path_to_saving + "/init_sols.csv",
      float_format="%.10e",
      index=False
    )
    return sols

import os
import time
import torch
import numpy as np
import scipy as sp

from .. import utils
from .. import backend as bkd
from romar.ops import svd_lowrank
from silx.io.dictdump import dicttoh5
from typing import Tuple, Optional


class CoBRAS(object):
  """
  Model Reduction for Nonlinear Systems by Balanced
  Truncation of State and Gradient Covariance (CoBRAS)

  See:
    https://doi.org/10.1137/22M1513228
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    system=None,
    quadrature=None,
    path_to_saving="./",
    saving=True,
    verbose=True
  ):
    self.verbose = verbose
    # System
    self.system = system
    self.quad = quadrature
    # Saving
    self.saving = saving
    self.path_to_saving = path_to_saving
    os.makedirs(self.path_to_saving, exist_ok=True)

  # Calling
  # ===================================
  def __call__(
    self,
    l: int,
    M: np.ndarray,
    xnot: Optional[list] = None,
    modes: bool = True,
    pod: bool = False
  ) -> None:
    X, Y = self.compute_cov_mats(l, M)
    if (xnot is not None):
      mask = self.make_mask(xnot)
      X, Y = X[mask], Y[mask]
    if modes:
      self.compute_modes(X, Y, pod)
    else:
      return X, Y

  def make_mask(self, xnot):
    xnot = np.array(xnot).astype(int).reshape(-1)
    mask = np.ones(self.system.nb_eqs)
    mask[xnot] = 0
    return mask.astype(bool)

  # Covariance matrices computation
  # -----------------------------------
  def compute_cov_mats(
    self,
    l: int,
    M: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute covariance matrices for a given number of measurements and 
    initial conditions.

    This function computes both the state and gradient covariance matrices by
    solving the forward and adjoint problems iteratively and returning the
    results based on the weighted samples.

    :param l: The number of measurements to consider.
    :type l: int
    :param M: Initial conditions matrix, where each row represents the initial
              state for the forward problem.
    :type M: np.ndarray

    :return: A tuple containing two covariance matrices:
             - X: State covariance matrix
             - Y: Gradient covariance matrix
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Set time grid
    t = self.quad["t"]["x"]
    nt = len(t)
    # Initialize dynamic arrays: state/gradient covariance matrices
    X, Y = [], []
    # Loop over 'forward problem' initial conditions (mu)
    # -------------
    for (i, mui) in enumerate(self.quad["mu"]["x"]):
      # > Compute initial condition for 'forward problem'
      y0 = self.system.get_init_sol(mui)
      # > Solve 'forward problem'
      y = self.system.solve_fom(t, y0).T
      # > Build interpolator for 'forward problem' solution
      self.build_sol_interp(t, y)
      # Loop over sampling initial times (t0)
      # -------------
      for j in range(nt-1):
        # > Define j-th backward time grid
        si, ei = j, np.minimum(j+l, nt-1)
        tj = np.flip(t[si:ei+1])
        fj = 1.0 / np.sqrt(len(tj))
        # Loop over 'adjoint problem' initial conditions
        # -------------
        # > Compute output Jacobian
        G = self.system.output_jac(y[ei])
        # > Solve 'adjoint problem'
        Yij = [self.solve_adjoint(tj, g0).T for g0 in G]
        Yij = fj * np.vstack(Yij)
        # Store samples
        # -------------
        # Quadrature weight
        wij = self.quad["mu"]["w"][i] * self.quad["t"]["w"][j]
        X.append(wij * y[j])
        Y.append(wij * Yij)
    # Return covariance matrices
    X = np.vstack(X).T
    Y = np.vstack(Y).T
    return X, Y

  # Adjoint model
  # -----------------------------------
  def solve_adjoint(
    self,
    t: np.ndarray,
    g0: np.ndarray
  ) -> np.ndarray:
    return sp.integrate.solve_ivp(
      fun=self.adjoint_fun,
      t_span=[t[0],t[-1]],
      y0=g0,
      method="BDF",
      # t_eval=t,
      first_step=1e-6,
      rtol=1e-6,
      atol=0.0,
      jac=self.adjoint_jac
    ).y

  def adjoint_fun(
    self,
    t: np.ndarray,
    g: np.ndarray
  ) -> np.ndarray:
    # return self.adjoint_jac(t, g) @ g
    dy = self.adjoint_jac(t, g) @ g
    print(t)
    print(np.abs(dy).max())
    return dy

  def adjoint_jac(
    self,
    t: np.ndarray,
    g: np.ndarray
  ) -> np.ndarray:
    x = self.sol_interp(np.abs(t))
    j = self.system.jac(t, x)
    return - j.T

  def build_sol_interp(
    self,
    t: np.ndarray,
    x: np.ndarray
  ) -> None:
    axis = 0 if (x.shape[0] == len(t)) else 1
    self.sol_interp = sp.interpolate.interp1d(t, x, kind="linear", axis=axis)

  # Balancing modes
  # -----------------------------------
  def compute_modes(
    self,
    X: np.ndarray,
    Y: np.ndarray,
    pod: bool = False,
    rank: int = 100,
    niter: int = 30
  ) -> None:
    if self.verbose:
      print("Computing CoBRAS modes ...")
    # Perform randomized SVD
    X, Y = [bkd.to_torch(z) for z in (X, Y)]
    U, s, V = svd_lowrank(
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
    dicttoh5(
      treedict={"s": s, "phi": phi, "psi": psi},
      h5file=self.path_to_saving+"/cobras_basis.hdf5",
      overwrite_data=True
    )
    if pod:
      if self.verbose:
        print("Computing POD modes ...")
      U, s, _ = torch.svd_lowrank(
        A=X,
        q=min(rank, X.shape[0]),
        niter=niter
      )
      s, phi = [bkd.to_numpy(z) for z in (s, U)]
      dicttoh5(
        treedict={"s": s, "phi": phi, "psi": phi},
        h5file=self.path_to_saving+"/pod_basis.hdf5",
        overwrite_data=True
      )

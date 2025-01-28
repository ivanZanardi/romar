import os
import time
import torch
import numpy as np
import scipy as sp

from .. import utils
from .. import backend as bkd
from romar.ops import svd_lowrank
from silx.io.dictdump import dicttoh5, h5todict


class LinCoBRAS(object):
  """
  Model Reduction for Nonlinear Systems by Balanced
  Truncation of State and Gradient Covariance (CoBRAS) - Linearized

  See:
    https://doi.org/10.1137/22M1513228
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    operators=None,
    quadrature=None,
    path_to_saving="./",
    saving=True,
    verbose=True
  ):
    self.verbose = verbose
    # Initialize operators (A, C, and M)
    # -------------
    self.ops = operators
    self.quad = quadrature
    # Nb. of equations
    self.nb_eqs = None if (self.ops is None) else self.ops["A"].shape[0]
    # Saving
    # -------------
    self.saving = saving
    self.path_to_saving = path_to_saving
    os.makedirs(self.path_to_saving, exist_ok=True)
    # Properties
    # -------------
    self.eiga = None
    self.runtime = {}

  # Properties
  # ===================================
  # Operators
  @property
  def ops(self):
    return self._ops

  @ops.setter
  def ops(self, value):
    self._ops = None
    if (value is not None):
      self._ops = {}
      for (k, op) in value.items():
        if (len(op.shape) == 1):
          op = op.reshape(-1,1)
        self._ops[k] = bkd.to_torch(op)

  # Quadrature points
  @property
  def quad(self):
    return self._quad

  @quad.setter
  def quad(self, value):
    self._quad = None
    if (value is not None):
      self._quad = utils.map_nested_dict(
        value, lambda x: bkd.to_torch(x.reshape(-1))
      )

  # Eigendecomposition of operator A
  @property
  def eiga(self):
    return self._eiga

  @eiga.setter
  def eiga(self, value):
    self._eiga = None
    if (value is not None):
      self._eiga = {k: bkd.to_torch(x) for (k, x) in value.items()}

  # Calling
  # ===================================
  def __call__(
    self,
    X=None,
    Y=None,
    xnot=None,
    modes=True,
    pod=False,
    runtime={}
  ):
    self.runtime = runtime
    if ((X is None) or (Y is None)):
      self.compute_eiga()
      X, Y = self.compute_cov_mats()
    if (xnot is not None):
      mask = self.make_mask(xnot)
      X, Y = X[mask], Y[mask]
    if modes:
      self.compute_modes(X, Y, pod)
    else:
      return X, Y

  def make_mask(self, xnot):
    xnot = np.array(xnot).astype(int).reshape(-1)
    mask = np.ones(self.nb_eqs)
    mask[xnot] = 0
    return mask.astype(bool)

  def compute_eiga(self, real_only=True):
    if (self.eiga is None):
      filename = self.path_to_saving + "/eiga.hdf5"
      if os.path.exists(filename):
        eiga = h5todict(filename)
      else:
        if self.verbose:
          print("Performing eigendecomposition of matrix A ...")
        a = bkd.to_numpy(self.ops["A"])
        runtime = time.time()
        l, v = sp.linalg.eig(a)
        vinv = sp.linalg.inv(v)
        self.runtime["eiga"] = time.time()-runtime
        eiga = {"l": l, "v": v, "vinv": vinv}
        # Save eigendecomposition
        if self.saving:
          dicttoh5(
            treedict=eiga,
            h5file=filename,
            overwrite_data=True
          )
      # Only real part
      if real_only:
        eiga = {k: x.real for (k, x) in eiga.items()}
      self.eiga = eiga

  # Covariance matrices computation
  # -----------------------------------
  def compute_cov_mats(self):
    if self.verbose:
      print("Computing covariance matrices ...")
    # Compute the empirical controllability Gramian
    op = self.ops["M"]
    X, runtime = self.compute_cov_mat(op)
    self.runtime["Ws_mean"] = runtime / op.shape[1]
    # Compute the empirical observability Gramian
    op = self.ops["C"].T
    Y, runtime = self.compute_cov_mat(op, adjoint=True)
    self.runtime["Wg_mean"] = runtime / op.shape[1]
    return [bkd.to_numpy(z) for z in (X, Y)]

  def compute_cov_mat(self, op, adjoint=False):
    runtime = time.time()
    # Allocate Gramian's memory
    shape = [len(self.quad["t"]["x"])] + list(op.shape)
    g = torch.zeros(shape, dtype=bkd.floatx(), device="cpu")
    # Compute tensor
    x = self.eiga["v"].T @ op if adjoint else self.eiga["vinv"] @ op
    for (i, ti) in enumerate(self.quad["t"]["x"]):
      xi = x * torch.exp(ti*self.eiga["l"]).reshape(-1,1)
      xi = self.eiga["vinv"].T @ xi if adjoint else self.eiga["v"] @ xi
      if (not adjoint):
        # Add steady-state equilibrium solution
        xi += self.ops["x_eq"].reshape(-1,1)
        # Scale by quadrature weights
        wi = self.quad["mu"]["w"] * self.quad["t"]["w"][i]
        xi *= wi.reshape(1,-1)
      g[i] = xi.cpu()
    # Manipulate tensor
    g = torch.permute(g, dims=(1,2,0))
    g = torch.reshape(g, (self.nb_eqs,-1))
    runtime = time.time()-runtime
    return g, runtime

  # Balancing modes
  # -----------------------------------
  def compute_modes(self, X, Y, pod=False, rank=100, niter=30):
    if self.verbose:
      print("Computing balanced CoBRAS modes ...")
    runtime = time.time()
    # Perform randomized SVD
    X, Y = [bkd.to_torch(z) for z in (X, Y)]
    U, s, V = svd_lowrank(
      X=X,
      Y=Y,
      q=min(rank, X.shape[0]),
      niter=niter
    )
    # Compute balancing transformation
    sqrt_s = torch.diag(torch.sqrt(1/s))
    phi = X @ V @ sqrt_s
    psi = Y @ U @ sqrt_s
    self.runtime["modes"] = time.time()-runtime
    # Save balancing modes
    s, phi, psi = [bkd.to_numpy(z) for z in (s, phi, psi)]
    dicttoh5(
      treedict={"s": s, "phi": phi, "psi": psi},
      h5file=self.path_to_saving+"/cobras_bases.hdf5",
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
        h5file=self.path_to_saving+"/pod_bases.hdf5",
        overwrite_data=True
      )

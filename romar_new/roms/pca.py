import sys
import torch
import numpy as np
import joblib as jl
import multiprocessing

from tqdm import tqdm
from typing import *

from .. import env
from .. import utils
from .basic import Basic
from .. import backend as bkd
from .utils import check_method
from typing import List, Optional, Union
from factor_analyzer.rotator import Rotator, POSSIBLE_ROTATIONS


class PCA(Basic):

  """
  Principal Component Analysis (PCA) with support for feature scaling,
  rotation, and randomized Singular Value Decomposition (SVD).
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
    Initialize the PCA class.

    :param scaling: Scaling method for normalization.
    :type scaling: str, optional
    :param rotation: Rotation method for principal components.
    :type rotation: str, optional
    :param path_to_saving: Directory path to save computed PCA results.
    :type path_to_saving: str

    :raises ValueError: If the specified rotation and scaling method are
                        invalid.
    """
    super(PCA, self).__init__(
      system, path_to_data, scale, xref, xscale, path_to_saving
    )

  # Compute covariance matrices
  # ===================================
  def compute_cov_mats(
    self,
    irange: List[int],
    use_quad_w: bool = True,
    nb_workers: int = 1
  ) -> Tuple[np.ndarray]:
    """
    Compute state and gradient covariance matrices from system simulations.

    This method computes covariance matrices using quadrature points
    and system dynamics, with optional parallel execution.

    :param nb_workers: Number of parallel workers for computation.
                       Defaults to 1 (sequential execution).
    :type nb_workers: int, optional

    :return: Tuple containing:
            - `X` (np.ndarray): Weighted state covariance matrix.
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
        nb_mu=irange[1]-irange[0],
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
    return X

  def _compute_cov_mats(
    self,
    index: int,
    X: List[np.ndarray],
    nb_mu: int,
    use_quad_w: bool = True
  ) -> None:
    """
    Compute state and gradient covariance matrices using quadrature points
    and system dynamics.

    This function evaluates state trajectories and their corresponding gradient
    adjoint solutions to construct weighted covariance matrices. The computed
    matrices are stored in the provided lists (`X`).

    :param X: List to store weighted state covariance matrix contributions.
    :type X: List[np.ndarray]

    :return: None (results are appended to `X`).
    :rtype: None
    """
    # Load solution
    data = utils.load_case(path=self.path_to_data, index=index)
    t, y, w_t, w_mu = [data[k] for k in ("t", "y", "w_t", "w_mu")]
    # Set weights
    if (not use_quad_w):
      w_mu = 1.0/np.sqrt(nb_mu)
      w_t[:] = 1.0/np.sqrt(len(t))
    w_t = w_t.reshape(-1,1)
    # Scale
    X.append(w_mu * w_t * self._apply_scaling(y.T))

  # Compute principal components
  # ===================================
  def compute_modes(
    self,
    X: np.ndarray,
    rotation: Optional[str] = None,
    xnot: Optional[List[int]] = None,
    rank: int = 100,
    niter: int = 50
  ) -> None:
    """
    Compute PCA modes for the given dataset using randomized SVD.

    :param X: Data matrix with shape (nb_features, nb_samples).
    :type X: np.ndarray
    :param scale: If True, applies scaling to the dataset. Default is True.
    :type scale: bool
    :param xref: Mean reference values for scaling (array or file path).
    :type xref: Optional[Union[str, np.ndarray]]
    :param xscale: Scaling factors (array or file path).
    :type xscale: Optional[Union[str, np.ndarray]]
    :param xnot: List of feature indices to exclude from PCA computation.
    :type xnot: Optional[List[int]]
    :param rank: Maximum rank for randomized SVD. Default is 100.
    :type rank: int
    :param niter: Number of iterations for randomized SVD. Default is 50.
    :type niter: int

    :return: Dictionary containing computed PCA components.
    :rtype: Dict[str, np.ndarray]
    """
    # Mask covariance matrices
    mask = self._make_mask(X.shape[0], xnot)
    X = X[mask]
    # Compute SVD
    rank = min(rank, X.shape[0])
    phi, s, _ = map(bkd.to_numpy, torch.svd_lowrank(
      A=bkd.to_torch(X),
      q=rank,
      niter=niter
    ))
    # Rotation method
    rotation = check_method(
      method="rotation",
      name=rotation,
      valid_names=POSSIBLE_ROTATIONS
    )
    # Apply rotation
    rotator = Rotator(method=rotation)
    phi = {r: rotator.fit_transform(phi[:,:r]) for r in range(2,rank+1)}
    # Save results
    data = {
      "s": np.ones_like(s),
      "phi": phi,
      "psi": phi,
      "mask": mask,
      "xref": self.xref,
      "xscale": self.xscale
    }
    self._save(data)

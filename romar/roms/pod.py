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
from typing import List, Optional, Union


class POD(Basic):

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
    Initialize the POD class.

    :param scaling: Scaling method for normalization.
    :type scaling: str, optional
    :param rotation: Rotation method for principal components.
    :type rotation: str, optional
    :param path_to_saving: Directory path to save computed POD results.
    :type path_to_saving: str

    :raises ValueError: If the specified rotation and scaling method are
                        invalid.
    """
    super(POD, self).__init__(
      system, path_to_data, scale, xref, xscale, path_to_saving
    )

  # Compute covariance matrices
  # ===================================
  def compute_cov_mats(
    self,
    irange: List[int],
    use_quad_w: bool = False,
    nb_workers: int = 1
  ) -> Dict[str, np.ndarray]:
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
    return {"X": X}

  def _compute_cov_mats(
    self,
    index: int,
    X: List[np.ndarray],
    nb_mu: int,
    use_quad_w: bool = False
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
    # Load data
    data = utils.load_case(path=self.path_to_data, index=index)
    if (data is not None):
      # State covariance matrix
      Xi = self._compute_state_cov(data, nb_mu, use_quad_w)
      X.append(Xi)

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
    Compute POD modes for the given dataset using randomized SVD.

    :param X: Data matrix with shape (nb_features, nb_samples).
    :type X: np.ndarray
    :param scale: If True, applies scaling to the dataset. Default is True.
    :type scale: bool
    :param xref: Mean reference values for scaling (array or file path).
    :type xref: Optional[Union[str, np.ndarray]]
    :param xscale: Scaling factors (array or file path).
    :type xscale: Optional[Union[str, np.ndarray]]
    :param xnot: List of feature indices to exclude from POD computation.
    :type xnot: Optional[List[int]]
    :param rank: Maximum rank for randomized SVD. Default is 100.
    :type rank: int
    :param niter: Number of iterations for randomized SVD. Default is 50.
    :type niter: int

    :return: Dictionary containing computed POD components.
    :rtype: Dict[str, np.ndarray]
    """
    # Mask covariance matrices
    mask = self._make_mask(
      nb_feat=X.shape[0],
      xnot=xnot
    )
    X = X[mask]
    # Compute SVD
    rank = min(rank, X.shape[0])
    phi, s, _ = map(bkd.to_numpy, torch.svd_lowrank(
      A=bkd.to_torch(X),
      q=rank,
      niter=niter
    ))
    # Vanilla model
    # -------------
    phi = {r: phi[:,:r] for r in range(2,rank+1)}
    data = {
      "s": s,
      "phi": phi,
      "psi": phi,
      "mask": mask,
      "xref": self.xref,
      "xscale": self.xscale
    }
    self._save(data)
    # Rotated model
    # -------------
    if (rotation is not None):
      rot = self.get_rotator(rotation)
      phi = {r: rot.fit_transform(basis) for (r, basis) in phi.items()}
      data["phi"] = phi
      data["psi"] = phi
      self._save(data, identifier=rotation)

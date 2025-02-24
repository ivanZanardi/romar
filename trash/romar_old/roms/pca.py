import torch
import numpy as np

from .utils import *
from .basic import Basic
from .. import backend as bkd
from typing import Dict, List, Optional, Union
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
    scaling: Optional[str] = None,
    rotation: Optional[str] = None,
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
    super(PCA, self).__init__(path_to_saving)
    # Scaling method
    self.scaling = check_method(
      method="scaling",
      name=scaling,
      valid_names=POSSIBLE_SCALINGS
    )
    # Rotation method
    self.rotation = check_method(
      method="rotation",
      name=rotation,
      valid_names=POSSIBLE_ROTATIONS
    )

  # Compute principal components
  # ===================================
  def compute_modes(
    self,
    X: np.ndarray,
    scale: bool = True,
    xref: Optional[Union[str, np.ndarray]] = None,
    xscale: Optional[Union[str, np.ndarray]] = None,
    xnot: Optional[List[int]] = None,
    rank: int = 100,
    niter: int = 50
  ) -> Dict[str, np.ndarray]:
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
    # Scale data if required
    X = self._scale(X=X, xref=xref, xscale=xscale, active=scale)
    # Mask data
    mask = self._make_mask(nb_feat=X.shape[0], xnot=xnot)
    X = X[mask]
    # Compute SVD
    rank = min(rank, X.shape[0])
    phi, s, _ = map(bkd.to_numpy, torch.svd_lowrank(
      A=bkd.to_torch(X),
      q=rank,
      niter=niter
    ))
    # Apply rotation
    rotator = Rotator(method=self.rotation)
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
    return data

  # Data scaling
  # ===================================
  def _scale(
    self,
    X: np.ndarray,
    xref: Optional[Union[str, np.ndarray]] = None,
    xscale: Optional[Union[str, np.ndarray]] = None,
    active: bool = True
  ) -> np.ndarray:
    """
    Apply scaling to the dataset if enabled.

    :param X: Input data matrix.
    :type X: np.ndarray
    :param xref: Mean reference values for scaling (array or file path).
    :type xref: Optional[Union[str, np.ndarray]]
    :param xscale: Scaling factors (array or file path).
    :type xscale: Optional[Union[str, np.ndarray]]
    :param active: If True, applies scaling. Default is True.
    :type active: bool

    :return: Scaled dataset.
    :rtype: np.ndarray

    :raises ValueError: If `xref` or `xscale` dimensions do not match `X`.
    """
    nb_feat = X.shape[0]
    if active:
      if ((xref is None) or (xscale is None)):
        data = compute_scaling(self.scaling, X)
        xref, xscale = [data[k] for k in ("xref", "xscale")]
      if ((xref.shape[0] != nb_feat) or (xscale.shape[0] != nb_feat)):
        raise ValueError(f"'xref' and 'xscale' must match ({nb_feat},) as " \
                         f"shape. Received {xref.shape} and {xscale.shape}.")
    self._set_scaling(nb_feat, xref, xscale, active)
    return self._apply_scaling(X.T).T

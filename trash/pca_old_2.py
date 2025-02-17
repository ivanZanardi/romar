import numpy as np
import rpy2.robjects as ro

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from .basic import Basic
from .utils import check_scaling, compute_scaling
from typing import Dict, List, Optional, Union

# Activate automatic conversion between R and NumPy
numpy2ri.activate()
# Import 'psych' R package
psych = importr("psych")


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
                    Options: "std", "pareto", or None.
    :type scaling: str, optional
    :param rotation: Rotation method for principal components.
                     Options: "varimax", "quartimax", or None.
    :type rotation: str, optional
    :param path_to_saving: Directory path to save computed PCA results.
    :type path_to_saving: str

    :raises ValueError: If the specified rotation and scaling method are
                        invalid.
    """
    super(PCA, self).__init__(path_to_saving)
    # Scaling method
    check_scaling(scaling)
    self.scaling = scaling
    # Rotation method
    if (rotation is None):
      rotation = "none"
    self.rotation = rotation.lower()

  # Compute principal components
  # ===================================
  def compute_modes(
    self,
    X: np.ndarray,
    scale: bool = True,
    xref: Optional[Union[str, np.ndarray]] = None,
    xscale: Optional[Union[str, np.ndarray]] = None,
    xnot: Optional[List[int]] = None,
    rank: int = 100
  ) -> Dict[str, np.ndarray]:
    """
    Compute PCA modes for the given dataset.

    :param X: Data matrix of shape (nb_features, nb_samples).
    :type X: np.ndarray
    :param scale: Whether to apply scaling (default: True).
    :type scale: bool, optional
    :param xref: Mean reference values for scaling (array or file path).
    :type xref: Union[str, np.ndarray], optional
    :param xscale: Scaling factors (array or file path).
    :type xscale: Union[str, np.ndarray], optional
    :param xnot: List of feature indices to exclude from PCA.
    :type xnot: List[int], optional
    :param rank: Maximum rank for randomized SVD.
    :type rank: int
    :param niter: Number of iterations for randomized SVD.
    :type niter: int

    :return: Dictionary containing computed PCA components.
    :rtype: Dict[str, np.ndarray]
    """
    # Scale data if required
    X = self._scale(X, xref=xref, xscale=xscale, active=scale)
    print(X.shape)
    # Mask data
    mask = self._make_mask(X.shape[0], xnot)
    X = X[mask]
    # Get data matrix shape
    nb_feat, nb_samples = X.shape
    # Convert to R matrix
    Xr = ro.r.matrix(X.T, nrow=nb_samples, ncol=nb_feat)
    # Perform PCA with Varimax rotation
    p = psych.principal(
      r=Xr,
      nfactors=min(rank, nb_feat),
      rotate=self.rotation
    )
    # Extract rotated loadings
    phi = np.array(p.rx2("loadings"))
    # Save results
    data = {
      "s": np.ones(phi.shape[1]),
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
    Apply scaling to the dataset.

    :param X: Data matrix.
    :type X: np.ndarray
    :param xref: Mean reference values (array or file path).
    :type xref: Union[str, np.ndarray], optional
    :param xscale: Scaling factors (array or file path).
    :type xscale: Union[str, np.ndarray], optional
    :param active: Whether to apply scaling (default: True).
    :type active: bool

    :return: Scaled dataset.
    :rtype: np.ndarray

    :raises ValueError: If `xref` or `xscale` has incorrect dimensions.
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

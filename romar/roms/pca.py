import numpy as np

from .. import ops
from .basic import Basic
from typing import Dict, List, Optional, Tuple, Union

_SCALINGS = {"std", "pareto", None}
_ROTATIONS = {"varimax", "quartimax", None}


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
    self.scaling = scaling
    if (self.scaling not in _SCALINGS):
      raise ValueError(f"Invalid scaling method: '{self.scaling}'. " \
                       f"Must be one of {_SCALINGS}.")
    # Rotation method
    self.rotation = rotation
    if (self.rotation not in _ROTATIONS):
      raise ValueError(f"Invalid rotation method: '{self.rotation}'. " \
                       f"Must be one of {_ROTATIONS}.")

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
    randomized: bool = False,
    niter: int = 50
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
    :param rank: Number of principal components to retain.
    :type rank: int
    :param randomized: Whether to use randomized SVD.
    :type randomized: bool
    :param niter: Number of power iterations for randomized SVD.
    :type niter: int

    :return: Dictionary containing computed PCA components.
    :rtype: Dict[str, np.ndarray]
    """
    # Scale data if required
    X = self._scale(X, xref=xref, xscale=xscale, active=scale)
    # Mask data
    mask = self._make_mask(X.shape[0], xnot)
    X = X[mask]
    # Compute SVD
    rank = min(X.shape[0], rank)
    if randomized:
      phi, s, _ = ops.svd_lowrank_x(A=X, q=rank, niter=niter)
    else:
      phi, s, _ = np.linalg.svd(X)
    phi = phi[:,:rank]
    # Apply rotation
    phi = self._rotate(phi)
    # Save results
    data = {
      "s": s,
      "phi": phi,
      "psi": phi,
      "mask": mask,
      "xref": self.xref,
      "xscale": np.diag(self.xscale)
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
        xref, xscale = self._compute_scaling(X)
      if ((xref.shape[0] != nb_feat) or (xscale.shape[0] != nb_feat)):
        raise ValueError(f"'xref' and 'xscale' must match ({nb_feat},) as " \
                         f"shape. Received {xref.shape} and {xscale.shape}.")
    self._set_scaling(nb_feat, xref, xscale, active)
    return self._apply_scaling(X.T).T

  def _compute_scaling(
    self,
    X: np.ndarray
  ) -> Tuple[np.ndarray]:
    """
    Compute scaling parameters based on the selected method.

    :param X: Data matrix of shape (nb_features, nb_samples).
    :type X: np.ndarray

    :return: Tuple containing:
             - Mean reference of shape (nb_features,).
             - Scaling factor of shape (nb_features,).
    :rtype: Tuple[np.ndarray]
    """
    nb_feat = X.shape[0]
    xref = np.mean(X, axis=-1)
    std = np.std(X, axis=-1)
    if (self.scaling == "std"):
      xscale = std
    elif (self.scaling == "pareto"):
      xscale = np.sqrt(std)
    else:
      xref = np.zeros(nb_feat)
      xscale = np.ones(nb_feat)
    return xref, xscale

  # Component rotation
  # ===================================
  def _rotate(
    self,
    phi: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 500
  ) -> np.ndarray:
    """
    Apply rotation to principal components.

    :param phi: Matrix of principal components (nb_features, nb_components).
    :type phi: np.ndarray
    :param tol: Convergence tolerance.
    :type tol: float
    :param max_iter: Maximum iterations.
    :type max_iter: int

    :return: Rotated principal components.
    :rtype: np.ndarray
    """
    if (self.rotation is None):
      return phi
    nb_feat, nb_comp = phi.shape
    # Initialize rotation matrix
    R = np.eye(nb_comp)
    # Initial variance measure
    var = 0
    # Iterate
    for _ in range(max_iter):
      comp_rot = phi @ R
      if (self.rotation == "varimax"):
        tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nb_feat)
      elif (self.rotation == "quartimax"):
        tmp = 0
      # Compute new rotation
      u, s, vh = np.linalg.svd(phi.T @ (comp_rot**3 - tmp))
      R = u @ vh
      var_new = np.sum(s)
      # Check convergence
      if ((var != 0) and (var_new < var * (1.0 + tol))):
        break
      var = var_new
    return phi @ R

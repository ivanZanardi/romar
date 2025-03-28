import os
import abc
import numpy as np
import dill as pickle

from typing import *
from .utils import init_scaling_param, check_method
from factor_analyzer.rotator import Rotator, POSSIBLE_ROTATIONS


class Basic(abc.ABC):

  """
  Base class for model reduction.
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

    :param path_to_saving: Directory path to save computed results.
    :type path_to_saving: str
    """
    # Class name
    self.name = self.__class__.__name__.lower()
    # Store system
    self.system = system
    # Path to solutions
    self.path_to_data = path_to_data
    # Set scaling if system equations are defined
    self._set_scaling(
      nb_feat=self.system.nb_eqs,
      xref=xref,
      xscale=xscale,
      active=scale)
    # Saving options
    self.path_to_saving = path_to_saving
    os.makedirs(self.path_to_saving, exist_ok=True)

  # Compute covariance matrices
  # ===================================
  @abc.abstractmethod
  def compute_cov_mats(self, *args, **kwargs) -> None:
    pass

  # Compute modes
  # ===================================
  @abc.abstractmethod
  def compute_modes(self, *args, **kwargs) -> None:
    """
    Abstract method for performing model reduction.

    This method must be implemented in a subclass.
    """
    pass

  # Masking features
  # ===================================
  def _make_mask(
    self,
    nb_feat: int,
    xnot: Optional[List[int]] = None
  ) -> np.ndarray:
    """
    Generate a boolean mask to exclude specified features from reduction.

    :param nb_feat: Total number of features.
    :type nb_feat: int
    :param xnot: List of feature indices to exclude.
    :type xnot: list[int]

    :return: Boolean mask of shape (nb_feat,).
    :rtype: np.ndarray
    """
    mask = np.ones(nb_feat, dtype=bool)
    if (xnot is not None):
      xnot = np.asarray(xnot, dtype=int).reshape(-1)
      mask[xnot] = False
    return mask

  # Data scaling
  # ===================================
  def _set_scaling(
    self,
    nb_feat: int,
    xref: Optional[Union[str, np.ndarray]] = None,
    xscale: Optional[Union[str, np.ndarray]] = None,
    active: bool = True
  ) -> None:
    """
    Set scaling parameters for feature normalization.

    :param nb_feat: Number of features.
    :type nb_feat: int
    :param xref: Mean reference values (shape: (nb_feat,)).
    :type xref: np.ndarray, optional
    :param xscale: Scaling factors (shape: (nb_feat,)).
    :type xscale: np.ndarray, optional
    :param active: Whether to apply scaling or use identity transformation.
    :type active: bool, optional

    :return: None
    :rtype: None

    :raises ValueError: If `xscale` has zero values or incorrect shape.
    """
    # Initialize scaling parameters
    xr = init_scaling_param(xref, nb_feat, ref_value=0.0)
    xs = init_scaling_param(xscale, nb_feat, ref_value=1.0)
    # If scaling is inactive, reset to default
    if (not active):
      xr.fill(0.0)
      xs.fill(1.0)
    # Ensure valid scaling factors
    if np.any(xs == 0.0):
      raise ValueError("Scaling factors must be nonzero " \
                       "to avoid division errors.")
    # Store parameters
    self.xref = xr
    self.xscale = xs
    self.xscale_mat = np.diag(xs)
    self.ov_xscale_mat = np.diag(1.0/xs)

  def _apply_scaling(
    self,
    x: np.ndarray
  ) -> np.ndarray:
    """
    Apply the stored scaling transformation.

    :param x: Input data of shape (nb_features,).
    :type x: np.ndarray

    :return: Scaled data.
    :rtype: np.ndarray

    :raises ValueError: If input dimensions do not match scaling dimensions.
    """
    if (x.shape[-1] != len(self.xref)):
      raise ValueError("Input data dimensions do not " \
                       "match the scaling dimensions.")
    return (x - self.xref) @ self.ov_xscale_mat

  # Rotation
  # ===================================
  def get_rotator(self, rotation):
    # Rotation method
    rotation = check_method(
      method="rotation",
      name=rotation,
      valid_names=POSSIBLE_ROTATIONS
    )
    # Build rotator
    return Rotator(method=rotation)

  # State covariance matrix
  # ===================================
  def _compute_state_cov(
    self,
    data: Dict[str, Any],
    nb_mu: int,
    use_quad_w: bool = False
  ) -> None:
    # Compute weights
    if use_quad_w:
      w = data["w_mu"] * data["w_t"].reshape(-1,1)
    else:
      nb_t = len(data["t"])
      w = 1.0/np.sqrt(nb_mu*nb_t)
    # Compute matrix
    y = data["y"].T
    return w * self._apply_scaling(y)

  # Saving Data
  # ===================================
  def _save(
    self,
    data: Dict[str, np.ndarray],
    identifier: Optional[str] = None
  ) -> None:
    """
    Save data to a file using pickle.

    :param data: Dictionary containing data arrays to save.
    :type data: Dict[str, np.ndarray]

    :return: None
    :rtype: None

    :raises OSError: If there is an issue saving the file.
    """
    if (identifier is None):
      identifier = "basis"
    else:
      identifier = f"basis_{identifier}"
    filename = self.path_to_saving + f"/{identifier}.p"
    try:
      with open(filename, "wb") as f:
        pickle.dump(data, f)
    except OSError as e:
      raise OSError(f"Error saving file {filename}: {e}")

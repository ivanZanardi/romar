import os
import abc
import numpy as np
import dill as pickle

from .utils import init_scaling_param
from typing import Dict, List, Optional, Union


class Basic(abc.ABC):

  """
  Base class for model reduction.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    path_to_saving: str = "./"
  ) -> None:
    """
    Initialize the base class.

    :param path_to_saving: Directory path to save computed results.
    :type path_to_saving: str
    """
    # Class name
    self.name = self.__class__.__name__.lower()
    # Saving options
    self.path_to_saving = path_to_saving
    os.makedirs(self.path_to_saving, exist_ok=True)

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

  # Saving Data
  # ===================================
  def _save(
    self,
    data: Dict[str, np.ndarray]
  ) -> None:
    """
    Save data to a file using pickle.

    :param data: Dictionary containing data arrays to save.
    :type data: Dict[str, np.ndarray]

    :return: None
    :rtype: None

    :raises OSError: If there is an issue saving the file.
    """
    filename = os.path.join(self.path_to_saving, f"{self.name}_basis.p")
    try:
      with open(filename, "wb") as f:
        pickle.dump(data, f)
    except OSError as e:
      raise OSError(f"Error saving file {filename}: {e}")

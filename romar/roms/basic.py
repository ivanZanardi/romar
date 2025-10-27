import os
import abc
import numpy as np
import dill as pickle

from typing import *
from .utils import init_scaling_param, check_method
from factor_analyzer.rotator import Rotator, POSSIBLE_ROTATIONS


class Basic(abc.ABC):
  """
  Abstract base class for model reduction methods.
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

    :param system: System object containing the governing equations. Must
                   define an attribute `nb_eqs` for the number of equations.
    :type system: callable
    :param path_to_data: Path to the dataset or simulation results used for
                         model reduction.
    :type path_to_data: str
    :param scale: Whether to apply scaling to the system variables.
                  Default is ``False``.
    :type scale: bool, optional
    :param xref: Reference values for centering (shape: (nb_feat,)).
                 Can be a string identifier or an array.
    :type xref: str or np.ndarray, optional
    :param xscale: Scaling factors (shape: (nb_feat,)).
                   Can be a string identifier or an array.
    :type xscale: str or np.ndarray, optional
    :param path_to_saving: Directory where results will be saved.
                           Default is ``"./"``.
    :type path_to_saving: str, optional
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
    """
    Abstract method to compute covariance matrices.

    This method must be implemented by subclasses. It should compute
    and store the covariance matrices required for model reduction.

    :raises NotImplementedError: If not implemented by a subclass.
    """
    pass

  # Compute basis
  # ===================================
  @abc.abstractmethod
  def compute_basis(self, *args, **kwargs) -> None:
    """
    Abstract method for performing model reduction.

    Subclasses must implement this method to compute the projection basis
    for the system.

    :raises NotImplementedError: If not implemented by a subclass.
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
                 Default is ``None`` (no exclusions).
    :type xnot: list[int], optional

    :return: Boolean mask of shape (nb_feat,), where excluded features
             are set to ``False``.
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

    :param nb_feat: Number of features (system variables).
    :type nb_feat: int
    :param xref: Reference values for centering (shape: (nb_feat,)).
                 Default is zeros.
    :type xref: str or np.ndarray, optional
    :param xscale: Scaling factors for normalization (shape: (nb_feat,)).
                   Default is ones.
    :type xscale: str or np.ndarray, optional
    :param active: Whether to apply scaling (True) or use identity scaling.
                   Default is ``True``.
    :type active: bool, optional

    :raises ValueError: If any scaling factor is zero, causing division errors.
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
    Apply the stored scaling transformation to the input data.

    :param x: Input data array (shape: (..., nb_features)).
    :type x: np.ndarray

    :return: Scaled data array of the same shape.
    :rtype: np.ndarray

    :raises ValueError: If input dimensions do not match the scaling dimensions.
    """
    if (x.shape[-1] != len(self.xref)):
      raise ValueError("Input data dimensions do not " \
                       "match the scaling dimensions.")
    return (x - self.xref) @ self.ov_xscale_mat

  # Rotation
  # ===================================
  def _get_rotator(self, rotation):
    """
    Instantiate a rotation object for basis rotation.

    :param rotation: Rotation method name. Must be one of
                     ``factor_analyzer.rotator.POSSIBLE_ROTATIONS``.
    :type rotation: str

    :return: A configured rotation object.
    :rtype: factor_analyzer.rotator.Rotator

    :raises ValueError: If an invalid rotation method name is provided.
    """
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
    """
    Compute the scaled state covariance matrix.

    :param data: Dictionary containing simulation data.
                 Must include keys ``"y"`` (state array),
                 ``"t"`` (time), and optionally ``"w_mu"``, ``"w_t"``.
    :type data: dict
    :param nb_mu: Number of parameter samples.
    :type nb_mu: int
    :param use_quad_w: Whether to use quadrature weights.
                       Default is ``False``.
    :type use_quad_w: bool, optional

    :return: Weighted and scaled state matrix.
    :rtype: np.ndarray
    """
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
    Save data to disk using pickle serialization.

    :param data: Dictionary containing arrays or data structures to save.
    :type data: dict
    :param identifier: Optional identifier appended to the saved filename.
                       If None, defaults to ``basis``.
    :type identifier: str, optional

    :raises OSError: If an error occurs while writing the file.

    :notes:
      The file is saved to ``{path_to_saving}/basis_{identifier}.p``.
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

import numpy as np

from .. import backend as bkd
from typing import Dict, List, Optional, Union

POSSIBLE_SCALINGS = {
  "std", "level", "range", "max", "pareto", "vast", "0to1", "-1to1"
}


def init_scaling_param(
  x: Optional[Union[str, np.ndarray]] = None,
  nb_feat: int = 1,
  ref_value: float = 1.0
) -> np.ndarray:
  """
  Build a scaling parameter.

  :param x: Input scaling parameter (array, filename, or None).
  :type x: np.ndarray, optional
  :param nb_feat: Number of features.
  :type nb_feat: int
  :param ref_value: Default value if `x` is None.
  :type ref_value: float, optional

  :return: Initialized scaling parameter as a NumPy array.
  :rtype: np.ndarray

  :raises ValueError: If the loaded file does not match expected dimensions.
  """
  if (x is None):
    return np.full(nb_feat, ref_value)
  if isinstance(x, str):
    try:
      x = np.loadtxt(x)
    except Exception as e:
      raise ValueError(f"Error loading file '{x}': {e}")
  x = np.asarray(x).reshape(-1)
  if (len(x) != nb_feat):
    raise ValueError(f"Expected input vector of length {nb_feat}, " \
                     f"but got {len(x)}.")
  return x

def compute_scaling(
  scaling: str,
  X: np.ndarray
) -> Dict[str, np.ndarray]:
  """
  Compute scaling parameters based on the selected method.

  :param scaling: Scaling method to apply.
  :type scaling: str
  :param X: Data matrix of shape (nb_features, nb_samples).
  :type X: np.ndarray

  :return: Dictionary containing:
           - 'xref': Mean reference of shape (nb_features,).
           - 'xscale': Scaling factor of shape (nb_features,).
  :rtype: Dict[str, np.ndarray]
  """
  # Scaling method
  scaling = check_method(
    method="scaling",
    name=scaling,
    valid_names=POSSIBLE_SCALINGS
  )
  # Dimension
  nb_feat = X.shape[0]
  # Common factors
  xref = np.mean(X, axis=-1)
  xstd = np.std(X, axis=-1)
  xmin = np.min(X, axis=-1)
  xmax = np.max(X, axis=-1)
  # Compute scaling
  if (scaling == "std"):
    xscale = xstd
  elif (scaling == "level"):
    xscale = xref
  elif (scaling == "range"):
    xscale = xmax - xmin
    xscale = np.where(xmax == xmin, 1.0, xmax - xmin)
  elif (scaling == "max"):
    xscale = np.where(xmax == 0.0, 1.0, xmax)
  elif (scaling == "pareto"):
    xscale = np.sqrt(xstd)
  elif (scaling == "pareto_nocenter"):
    xref[::] = 0.0
    xscale = np.sqrt(xref)
  elif (scaling == "vast"):
    xscale = xstd * xstd / (xref + bkd.epsilon()*np.sign(xref))
  elif (scaling == "0to1"):
    xref = xmin
    xscale = np.where(xmax == xmin, 1.0, xmax - xmin)
  elif (scaling == "-1to1"):
    xref = 0.5*(xmax + xmin)
    xscale = np.where(xmax == xmin, 1.0, 0.5*(xmax - xmin))
  else:
    xref = np.zeros(nb_feat)
    xscale = np.ones(nb_feat)
  return {"xref": xref, "xscale": xscale}

def check_method(
  method: Optional[str] = None,
  name: Optional[str] = None,
  valid_names: Optional[List[str]] = None
) -> Optional[str]:
  """
  Validate the method name against a list of valid methods.

  :param method: Method category (e.g., "scaling").
  :type method: Optional[str]
  :param name: Name of the specific method.
  :type name: Optional[str]
  :param valid_names: List of valid method names.
  :type valid_names: Optional[List[str]]

  :return: Validated method name.
  :rtype: Optional[str]

  :raises ValueError: If the method name is invalid.
  """
  if (name is not None):
    name = name.lower()
    if ((valid_names is not None) and (name not in valid_names)):
      raise ValueError(f"Invalid {method} method: '{name}'. " \
                       f"Must be one of {valid_names}.")
    else:
      return name

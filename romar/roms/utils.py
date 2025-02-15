import os
import numpy as np

from .. import backend as bkd
from typing import Optional, Tuple, Union

SCALINGS = {"std", "level", "range", "max", "pareto", "vast", "0to1", "-1to1"}


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
  # Scaling method
  scaling = scaling.lower()
  check_scaling(scaling)
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
  elif (scaling == "max"):
    xscale = xmax
  elif (scaling == "pareto"):
    xscale = np.sqrt(xstd)
  elif (scaling == "vast"):
    xscale = xstd * xstd / (xref + bkd.epsilon())
  elif (scaling == "0to1"):
    xref = xmin
    xscale = xmax - xmin
  elif (scaling == "-1to1"):
    xref = 0.5*(xmax + xmin)
    xscale = 0.5*(xmax - xmin)
  else:
    xref = np.zeros(nb_feat)
    xscale = np.ones(nb_feat)
  return {"xref": xref, "xscale": xscale}

def check_scaling(
  scaling: Optional[str] = None
) -> None:
  if ((scaling is not None) and (scaling not in SCALINGS)):
    raise ValueError(f"Invalid scaling method: '{scaling}'. " \
                     f"Must be one of {SCALINGS}.")

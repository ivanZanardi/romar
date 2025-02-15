import numpy as np

from typing import Optional, Tuple, Union

SCALINGS = {"std", "pareto", None}


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
  check_scaling(scaling)
  nb_feat = X.shape[0]
  xref = np.mean(X, axis=-1)
  std = np.std(X, axis=-1)
  if (scaling == "std"):
    xscale = std
  elif (scaling == "pareto"):
    xscale = np.sqrt(std)
  else:
    xref = np.zeros(nb_feat)
    xscale = np.ones(nb_feat)
  return {"xref": xref, "xscale": xscale}

def check_scaling(
  scaling: str
) -> None:
  if (scaling not in SCALINGS):
    raise ValueError(f"Invalid scaling method: '{scaling}'. " \
                     f"Must be one of {SCALINGS}.")

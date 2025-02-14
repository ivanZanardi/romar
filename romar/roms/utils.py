import numpy as np

from typing import Optional, Union


def build_scaling_param(
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

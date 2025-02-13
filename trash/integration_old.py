import numpy as np

from typing import Tuple, Union


def get_quad_nd(
  x: Tuple[np.ndarray],
  dist: Union[Tuple[str], str] = "uniform",
  quad: Union[Tuple[str], str] = "gl",
  deg: int = 3,
  joint: bool = True
) -> Tuple[np.ndarray]:
  """
  Compute n-dimensional quadrature points and weights over a domain
  defined by multiple axes. Supports Gauss-Legendre quadrature or
  trapezoidal integration, with optional scaling based on a probability
  distribution.

  :param x: A tuple of 1D arrays, each defining the interval for quadrature
            along a different dimension.
  :type x: Tuple[np.ndarray]
  :param dist: Distribution type for scaling weights. Defaults to "uniform".
               Options include:
               - "uniform" (default)
               - "loguniform"
  :type dist: Union[Tuple[str], str], optional
  :param quad: Quadrature type to use for each dimension.
               Defaults to "gl" (Gauss-Legendre). Options include:
               - "gl" (Gauss-Legendre)
               - "trapz" (Trapezoidal)
  :type quad: Union[Tuple[str], str], optional
  :param deg: Degree of the Gauss-Legendre quadrature.
              Only relevant if `quad` is 'gl'. Defaults to 3.
  :type deg: int, optional
  :param joint: If True, returns a joint grid of points (default: True).
                If False, returns individual 1D arrays of quadrature points
                and weights for each dimension.
  :type joint: bool, optional

  :return: A tuple containing:
            - `x`: Quadrature points (size N x D, where N is the number of
                   points and D is the number of dimensions).
            - `w`: Quadrature weights (size N, one weight per point).
  :rtype: Tuple[np.ndarray]

  :notes:
    - If `joint=True`, the function creates a full grid of points using the
      Cartesian product of all 1D quadrature points. The weights are computed
      as the product of the individual 1D weights.
    - If `joint=False`, quadrature points and weights are returned as
      separate 1D arrays.
  """
  # Number of dimensions
  nb_dim = len(x)
  # Check inputs
  if isinstance(dist, str):
    dist = tuple([dist]*nb_dim)
  if isinstance(quad, str):
    quad = tuple([quad]*nb_dim)
  # Get 1D quadrature points and weights for each dimension
  xw = [get_quad_1d(x[i], quad[i], deg, dist[i]) for i in range(nb_dim)]
  x, w = list(zip(*xw))
  if joint:
    # Create 2D grid of points using meshgrid and reshape them into (N, D)
    x = [z.reshape(-1) for z in np.meshgrid(*x)]
    x = np.vstack(x).T
    # Compute 2D quadrature weights by the product of the 1D weights
    w = [z.reshape(-1) for z in np.meshgrid(*w)]
    w = np.prod(w, axis=0)
  return x, w

def get_quad_1d(
  x: np.ndarray,
  quad: str = "gl",
  deg: int = 3,
  dist: str = "uniform"
) -> Tuple[np.ndarray]:
  """
  Compute 1D quadrature points and weights over a given interval.

  :param x: Array of points defining the interval for quadrature.
  :type x: np.ndarray
  :param quad: Quadrature type. Options:
               - "gl" (Gauss-Legendre, default)
               - "trapz" (Trapezoidal)
  :type quad: str
  :param deg: Degree of the Gauss-Legendre quadrature
              (only used if `quad="gl"`). Default is 3.
  :type deg: int
  :param dist: Probability distribution for scaling weights. Options:
              - "uniform" (default)
              - "loguniform"
  :type dist: str

  :return: Tuple of quadrature points and corresponding weights.
  :rtype: Tuple[np.ndarray, np.ndarray]
  """
  if (len(x) == 1):
    return x, np.ones(1)
  else:
    x = np.sort(x)
    a, b = np.amin(x), np.amax(x)
    if (quad == "gl"):
      x, w = _get_quad_gl_1d(x, deg)
    else:
      x, w = _get_quad_trapz_1d(x)
    f = _compute_dist(x, a, b, dist)
    return x, w*f

def _get_quad_gl_1d(
  x: np.ndarray,
  deg: int = 3
) -> Tuple[np.ndarray]:
  """
  Compute 1D Gauss-Legendre quadrature points and weights over an interval.

  :param x: Array of interval points.
  :type x: np.ndarray
  :param deg: Degree of Gauss-Legendre quadrature
              (number of points per subinterval). Default is 3.
  :type deg: int

  :return: Tuple containing quadrature points and weights.
  :rtype: Tuple[np.ndarray, np.ndarray]
  """
  # Compute Gauss-Legendre quadrature points
  # and weights for reference interval [-1, 1]
  xlg, wlg = np.polynomial.legendre.leggauss(deg)
  _x, _w = [], []
  # Loop over each interval in x
  for i in range(len(x) - 1):
    # Scaling and shifting from the reference
    # interval to the current interval
    a = 0.5 * (x[i+1] - x[i])
    b = 0.5 * (x[i+1] + x[i])
    _x.append(a * xlg + b)
    _w.append(a * wlg)
  # Concatenate all points and weights
  x = np.concatenate(_x).squeeze()
  w = np.concatenate(_w).squeeze()
  return x, w

def _get_quad_trapz_1d(
  x: np.ndarray
) -> Tuple[np.ndarray]:
  """
  Compute 1D quadrature weights using the trapezoidal rule.

  :param x: Array of interval points.
  :type x: np.ndarray

  :return: Tuple containing quadrature points and weights.
  :rtype: Tuple[np.ndarray, np.ndarray]
  """
  w = np.zeros_like(x)
  w[0] = 0.5 * (x[1] - x[0])
  w[-1] = 0.5 * (x[-1] - x[-2])
  w[1:-1] = 0.5 * (x[2:] - x[:-2])
  return x, w

def _compute_dist(
  x: np.ndarray,
  a: float,
  b: float,
  model: str = "uniform"
) -> np.ndarray:
  """
  Compute the probability distribution over a set of points based on the
  specified distribution model.

  :param x: Array of points defining the intervals along the x-axis.
  :type x: np.ndarray
  :param model: The type of distribution to compute. Options are 'uniform' or
                'loguniform'. Default is 'uniform'.
  :type model: str

  :raises ValueError: If the specified model is not recognized.

  :return: Computed distribution values.
  :rtype: np.ndarray
  """
  if (model == "uniform"):
    return np.full(x.shape, 1/(b-a))
  elif (model == "loguniform"):
    dx = np.log(b) - np.log(a)
    return 1/(x*dx)
  else:
    raise ValueError(f"Distribution model not recognized: '{model}'")

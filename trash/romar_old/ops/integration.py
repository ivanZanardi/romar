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
  Compute n-dimensional quadrature points and weights.

  This function constructs quadrature points and weights for multiple
  dimensions using either Gauss-Legendre quadrature or trapezoidal integration.
  The results can be returned as a Cartesian grid of points or as separate 1D
  arrays.

  :param x: A tuple of 1D arrays, each defining an interval along one
            dimension.
  :type x: Tuple[np.ndarray]
  :param dist: Type of distribution used for scaling weights. Options:
               - "uniform" (default)
               - "loguniform"
  :type dist: Union[Tuple[str], str], optional
  :param quad: Quadrature method for each dimension. Options:
               - "gl" (Gauss-Legendre, default)
               - "trapz" (Trapezoidal)
  :type quad: Union[Tuple[str], str], optional
  :param deg: Degree of Gauss-Legendre quadrature (only used if `quad="gl"`).
              Default is 3.
  :type deg: int, optional
  :param joint: If True, returns a full grid of points; otherwise, returns
                individual 1D arrays.
  :type joint: bool, optional

  :return: A tuple containing:
           - `x`: Quadrature points (shape: N x D if `joint=True`, otherwise
                  individual 1D arrays).
           - `w`: Quadrature weights (shape: N).
  :rtype: Tuple[np.ndarray]

  :notes:
    - If `joint=True`, the function computes a Cartesian product of quadrature
      points across dimensions.
    - If `joint=False`, it returns separate quadrature points and weights for
      each dimension.
  """
  nb_dim = len(x)
  # Convert string inputs to tuples for consistency
  dist = tuple([dist]*nb_dim) if isinstance(dist, str) else dist
  quad = tuple([quad]*nb_dim) if isinstance(quad, str) else quad
  # Compute 1D quadrature for each dimension
  xw = [get_quad_1d(x[i], quad[i], deg, dist[i]) for i in range(nb_dim)]
  x, w = list(zip(*xw))
  if joint:
    # Generate Cartesian grid and reshape
    x = np.vstack([z.reshape(-1) for z in np.meshgrid(*x)]).T
    w = np.prod([z.reshape(-1) for z in np.meshgrid(*w)], axis=0)
  return x, w

def get_quad_1d(
  x: np.ndarray,
  quad: str = "gl",
  deg: int = 3,
  dist: str = "uniform"
) -> Tuple[np.ndarray]:
  """
  Compute 1D quadrature points and weights over a given interval.

  :param x: Array of interval-defining points.
  :type x: np.ndarray
  :param quad: Quadrature method. Options:
               - "gl" (Gauss-Legendre, default)
               - "trapz" (Trapezoidal)
  :type quad: str
  :param deg: Degree of Gauss-Legendre quadrature (only relevant for
              `quad="gl"`). Default is 3.
  :type deg: int
  :param dist: Scaling distribution for weights. Options:
               - "uniform" (default)
               - "loguniform"
  :type dist: str

  :return: Quadrature points and corresponding weights.
  :rtype: Tuple[np.ndarray]
  """
  if (len(x) == 1):
    return x, np.ones(1)
  x = np.sort(x)
  a, b = np.amin(x), np.amax(x)
  if (quad == "gl"):
    x, w = _get_quad_gl_1d(x, deg)
  else:
    x, w = _get_quad_trapz_1d(x)
  # Apply probability distribution scaling
  f = _compute_dist(x, a, b, dist)
  return x, w*f

def _get_quad_gl_1d(
  x: np.ndarray,
  deg: int = 3
) -> Tuple[np.ndarray]:
  """
  Compute 1D Gauss-Legendre quadrature points and weights over an interval.

  :param x: Interval points.
  :type x: np.ndarray
  :param deg: Degree of quadrature (number of points per subinterval).
              Default is 3.
  :type deg: int

  :return: Quadrature points and corresponding weights.
  :rtype: Tuple[np.ndarray]
  """
  # Get Gauss-Legendre nodes and weights for [-1,1]
  xlg, wlg = np.polynomial.legendre.leggauss(deg)
  _x, _w = [], []
  # Loop over each interval in x
  for i in range(len(x) - 1):
    # Scaling and shifting from the reference interval to the current interval
    a = 0.5 * (x[i+1] - x[i])
    b = 0.5 * (x[i+1] + x[i])
    _x.append(a * xlg + b)
    _w.append(a * wlg)
  # Concatenate all points and weights
  x = np.concatenate(_x).reshape(-1)
  w = np.concatenate(_w).reshape(-1)
  return x, w

def _get_quad_trapz_1d(
  x: np.ndarray
) -> Tuple[np.ndarray]:
  """
  Compute 1D quadrature points and weights using the trapezoidal rule.

  :param x: Interval points.
  :type x: np.ndarray

  :return: Quadrature points and corresponding weights.
  :rtype: Tuple[np.ndarray]
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
  Compute probability distribution weights for quadrature points.

  :param x: Quadrature points.
  :type x: np.ndarray
  :param a: Lower bound of the interval.
  :type a: float
  :param b: Upper bound of the interval.
  :type b: float
  :param model: Distribution model to apply. Options:
          - "uniform" (default)
          - "loguniform"
  :type model: str

  :raises ValueError: If an unsupported distribution model is specified.

  :return: Scaling factors for quadrature weights.
  :rtype: np.ndarray
  """
  if (model == "uniform"):
    return np.full_like(x, 1.0/(b-a))
  elif (model == "loguniform"):
    dx = np.log(b) - np.log(a)
    return 1.0/(x*dx)
  else:
    raise ValueError(f"Unsupported distribution model: '{model}'")

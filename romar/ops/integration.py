import numpy as np

from typing import Tuple


def get_quad_2d(
  x: np.ndarray,
  y: np.ndarray,
  dist_x: str = "uniform",
  dist_y: str = "uniform",
  quad_x: str = "gl",
  quad_y: str = "gl",
  deg: int = 3,
  joint: bool = True
) -> Tuple[np.ndarray]:
  """
  Compute 2D Gauss-Legendre quadrature points and weights over the domain
  defined by `x` and `y`.

  :param x: Array of points defining the intervals along the x-axis.
  :type x: np.ndarray
  :param y: Array of points defining the intervals along the y-axis.
  :type y: np.ndarray
  :param deg: Degree of the Gauss-Legendre quadrature (number of points
              within each subinterval). Default is 3.
  :type deg: int
  :param dist_x: The type of distribution to scale the quadrature weights
                 along the x-axis. Default is "uniform".
  :type dist_x: str
  :param dist_y: The type of distribution to scale the quadrature weights
                 along the y-axis. Default is "uniform".
  :type dist_y: str

  :return: Tuple containing:
           - `xy`: (N, 2) array of quadrature points where N is
             the number of points.
           - `w`: Array of quadrature weights.
  :rtype: Tuple[np.ndarray]
  """
  # Get 1D quadrature points and weights for x and y axes
  x, wx = get_quad_1d(x, quad_x, deg, dist_x)
  y, wy = get_quad_1d(y, quad_y, deg, dist_y)
  if joint:
    # Create 2D grid of points using meshgrid and reshape them into (N, 2)
    xy = [z.reshape(-1) for z in np.meshgrid(x, y)]
    xy = np.vstack(xy).T
    # Compute 2D quadrature weights by the product of the 1D weights
    w = [z.reshape(-1) for z in np.meshgrid(wx, wy)]
    w = np.prod(w, axis=0)
    return xy, w
  else:
    return (x, y), (wx, wy)

def get_quad_1d(
  x: np.ndarray,
  quad: str = "gl",
  deg: int = 3,
  dist: str = "uniform"
) -> Tuple[np.ndarray]:
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
  Compute 1D Gauss-Legendre quadrature points and weights over the domain
  defined by `x`.

  :param x: Array of points defining the intervals for quadrature.
  :type x: np.ndarray
  :param deg: Degree of the Gauss-Legendre quadrature (number of points
              within each subinterval). Default is 3.
  :type deg: int
  :param dist: The type of distribution to scale the quadrature weights.
               Default is "uniform".
  :type dist: str

  :raises ValueError: If the input `x` contains fewer than two points.

  :return: Tuple of arrays containing quadrature points and weights.
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
  # Compute trapezoidal rule quadrature weights
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

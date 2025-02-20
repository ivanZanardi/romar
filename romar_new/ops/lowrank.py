import torch

from typing import Tuple
from torch.overrides import handle_torch_function, has_torch_function


def svd_lowrank_xy(
  X: torch.Tensor,
  Y: torch.Tensor,
  q: int = 6,
  niter: int = 2
) -> Tuple[torch.Tensor]:
  """
  Compute a low-rank Singular Value Decomposition (SVD) approximation.

  This function approximates the SVD of `Y^T @ X` using randomized
  algorithms based on Halko et al. (2009). It efficiently computes the
  dominant singular values and vectors, making it suitable for large-scale
  matrix computations.

  :param X: Input tensor of shape (m, n).
  :param Y: Input tensor of shape (m, p).
  :param q: Target rank for the approximation. Default is 6.
  :param niter: Number of power iterations to improve approximation.
                Default is 2.

  :return: A tuple (U, s, V), where:
           - U: Left singular vectors of shape (p, q).
           - s: Singular values of shape (q,).
           - V: Right singular vectors of shape (n, q).
  """
  if (not torch.jit.is_scripting()):
    tensor_ops = (X, Y)
    tensor_types = set(map(type, tensor_ops))
    tensor_valid = tensor_types.issubset((torch.Tensor, type(None)))
    if ((not tensor_valid) and has_torch_function(tensor_ops)):
      return handle_torch_function(
        svd_lowrank_xy, tensor_ops, X, Y, q=q, niter=niter
      )
  return _svd_lowrank(X, Y, q=q, niter=niter)

def _svd_lowrank(
  X: torch.Tensor,
  Y: torch.Tensor,
  q: int = 6,
  niter: int = 2
) -> Tuple[torch.Tensor]:
  """
  Internal function to compute the low-rank SVD approximation.

  This function approximates the SVD of `Y^T @ X` using randomized
  algorithms based on Halko et al. (2009). It efficiently computes the
  dominant singular values and vectors, making it suitable for large-scale
  matrix computations.

  :param X: Input tensor of shape (m, n).
  :param Y: Input tensor of shape (m, p).
  :param q: Target rank for the approximation. Default is 6.
  :param niter: Number of power iterations to improve approximation.
                Default is 2.

  :return: A tuple (U, s, V), where:
           - U: Left singular vectors of shape (p, q).
           - s: Singular values of shape (q,).
           - V: Right singular vectors of shape (n, q).
  """
  # Compute an approximate basis for the column space of Y^T @ X
  Q = _get_approximate_basis(X, Y, q, niter=niter)
  # Project the original matrix onto the lower-dimensional subspace
  B = (Q.T @ Y.T) @ X
  # Perform SVD on the reduced matrix
  U, s, Vh = torch.linalg.svd(B, full_matrices=False)
  # Hermitian transpose (conjugate transpose)
  V = Vh.mH
  # Project back to the original space
  U = Q.matmul(U)
  return U, s, V

def _get_approximate_basis(
  X: torch.Tensor,
  Y: torch.Tensor,
  q: int = 6,
  niter: int = 2
) -> torch.Tensor:
  """
  Compute an approximate basis for the column space of `Y^T @ X`.

  This function applies a randomized range finder algorithm to estimate
  a basis for `Y^T @ X`, which is used in the low-rank SVD computation.

  :param X: Input tensor of shape (m, n).
  :param Y: Input tensor of shape (m, p).
  :param q: Target rank for the approximation. Default is 6.
  :param niter: Number of power iterations to improve approximation.
                Default is 2.

  :return: An orthonormal basis Q of shape (n, q).
  """
  # Generate a random test matrix
  R = torch.randn(X.shape[-1], q, dtype=X.dtype, device=X.device)
  P = Y.T @ (X @ R)
  # Orthonormalize P
  Q = torch.linalg.qr(P).Q
  for _ in range(niter):
    P = X.T @ (Y @ Q)
    Q = torch.linalg.qr(P).Q
    P = Y.T @ (X @ Q)
    Q = torch.linalg.qr(P).Q
  return Q

import torch

from .. import backend as bkd
from typing import Tuple
from torch.overrides import handle_torch_function, has_torch_function


def _svd_lowrank_xy(
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
  :param Y: Input tensor of shape (p, m).
  :param q: Target rank for the approximation. Default is 6.
  :param niter: Number of power iterations to improve approximation.
                Default is 2.

  :return: A tuple (U, s, V), where:
           - U: Left singular vectors of shape (m, q).
           - s: Singular values of shape (q,).
           - V: Right singular vectors of shape (n, q).
  """
  if (not torch.jit.is_scripting()):
    tensor_ops = (X, Y)
    if not set(map(type, tensor_ops)).issubset(
      (torch.Tensor, type(None))
    ) and has_torch_function(tensor_ops):
      return handle_torch_function(
        _svd_lowrank_xy, tensor_ops, X, Y, q=q, niter=niter
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

  Uses the randomized SVD method based on Halko et al., 2009.
  The method constructs an approximate basis for the range of `Y^T @ X`
  and then computes SVD on the reduced matrix.

  :param X: Input tensor of shape (m, n).
  :param Y: Input tensor of shape (p, m).
  :param q: Target rank for the approximation.
  :param niter: Number of power iterations to refine the basis.

  :return: A tuple (U, s, V) with:
           - U: Approximate left singular vectors.
           - s: Singular values.
           - V: Approximate right singular vectors.
  """
  # Compute an approximate basis for the column space of Y^T @ X
  Q = _get_approximate_basis(X, Y, q, niter=niter)
  # Project the original matrix onto the lower-dimensional subspace
  B = (Q.T @ Y.T) @ X
  # Perform SVD on the reduced matrix
  U, s, Vh = torch.linalg.svd(B, full_matrices=False)
  V = Vh.mH  # Hermitian transpose (conjugate transpose)
  U = Q.matmul(U)  # Project back to the original space
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
  :param Y: Input tensor of shape (p, m).
  :param q: Target rank for the approximation.
  :param niter: Number of power iterations to improve accuracy.

  :return: An orthonormal basis Q of shape (n, q).
  """
  # Generate a random test matrix
  R = torch.randn(X.shape[-1], q, dtype=X.dtype, device=X.device)
  P = Y.T @ (X @ R)
  Q = torch.linalg.qr(P).Q  # Orthonormalize P
  for _ in range(niter):
    P = X.T @ (Y @ Q)
    Q = torch.linalg.qr(P).Q
    P = Y.T @ (X @ Q)
    Q = torch.linalg.qr(P).Q
  return Q


svd_lowrank_x = bkd.make_fun_np(torch.svd_lowrank)

svd_lowrank_xy = bkd.make_fun_np(_svd_lowrank_xy)

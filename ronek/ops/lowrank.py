import torch

from typing import Tuple
from torch.overrides import handle_torch_function, has_torch_function


def svd_lowrank(
  X: torch.Tensor,
  Y: torch.Tensor,
  q: int = 6,
  niter: int = 2
) -> Tuple[torch.Tensor]:
  if (not torch.jit.is_scripting()):
    tensor_ops = (X, Y)
    if not set(map(type, tensor_ops)).issubset(
      (torch.torch.Tensor, type(None))
    ) and has_torch_function(tensor_ops):
      return handle_torch_function(
        svd_lowrank, tensor_ops, X, Y, q=q, niter=niter
      )
  return _svd_lowrank(X, Y, q=q, niter=niter)

def _svd_lowrank(
  X: torch.Tensor,
  Y: torch.Tensor,
  q: int = 6,
  niter: int = 2
) -> Tuple[torch.Tensor]:
  # Algorithm 5.1 in Halko et al., 2009
  # Set A = Y.T @ X
  Q = _get_approximate_basis(X, Y, q, niter=niter)
  B = (Q.T @ Y.T) @ X
  U, s, Vh = torch.linalg.svd(B, full_matrices=False)
  V = Vh.mH
  U = Q.matmul(U)
  return U, s, V

def _get_approximate_basis(
  X: torch.Tensor,
  Y: torch.Tensor,
  q: int = 6,
  niter: int = 2
) -> torch.Tensor:
  R = torch.randn(X.shape[-1], q, dtype=X.dtype, device=X.device)
  P = Y.T @ (X @ R)
  Q = torch.linalg.qr(P).Q
  for _ in range(niter):
    P = X.T @ (Y @ Q)
    Q = torch.linalg.qr(P).Q
    P = Y.T @ (X @ Q)
    Q = torch.linalg.qr(P).Q
  return Q

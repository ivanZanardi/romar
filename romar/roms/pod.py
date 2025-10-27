import sys
import torch
import numpy as np
import joblib as jl
import multiprocessing

from tqdm import tqdm
from typing import *

from .. import env
from .. import utils
from .basic import Basic
from .. import backend as bkd
from typing import List, Optional


class POD(Basic):
  """
  Proper Orthogonal Decomposition (POD) model reduction class.

  This class implements methods for computing state covariance matrices
  and extracting reduced-order bases via randomized SVD, with optional
  component rotation.
  """

  # Compute covariance matrices
  # ===================================
  def compute_cov_mats(
    self,
    irange: List[int],
    use_quad_w: bool = False,
    nb_workers: int = 1
  ) -> Dict[str, np.ndarray]:
    """
    Compute the state covariance matrix from simulation trajectories.

    This method computes and aggregates the covariance contributions
    over a given index range, using either sequential or parallel execution.

    :param irange: List specifying the index range [start, stop) of trajectories.
    :type irange: List[int]
    :param use_quad_w: Whether to use quadrature weights for time and parameters.
                       Default is ``False``.
    :type use_quad_w: bool, optional
    :param nb_workers: Number of parallel workers to use. If ``1``, runs
                       sequentially. Default is ``1``.
    :type nb_workers: int, optional

    :return: Dictionary containing the aggregated state covariance matrix.
    :rtype: Dict[str, np.ndarray]
    """
    # Loop over computed solutions
    iterable = tqdm(
      iterable=range(*irange),
      ncols=80,
      desc="Trajectories",
      file=sys.stdout
    )
    with multiprocessing.Manager() as manager:
      # Define input arguments for covariance matrices calculation
      kwargs = dict(
        X=manager.list(),
        nb_mu=irange[1]-irange[0],
        use_quad_w=use_quad_w
      )
      if (nb_workers > 1):
        # Run jobs in parallel
        fun = env.make_fun_parallel(self._compute_cov_mats)
        jl.Parallel(nb_workers)(
          jl.delayed(fun)(index=i, **kwargs) for i in iterable
        )
      else:
        # Run jobs in series
        for i in iterable:
          self._compute_cov_mats(index=i, **kwargs)
      # Stack matrices
      X = np.vstack(list(kwargs["X"])).T
    return {"X": X}

  def _compute_cov_mats(
    self,
    index: int,
    X: List[np.ndarray],
    nb_mu: int,
    use_quad_w: bool = False
  ) -> None:
    """
    Compute and store a single sample's contribution to the state covariance
    matrix.

    This method loads simulation data for one trajectory, computes its scaled
    state matrix, and appends it to the shared list `X`.

    :param index: Index of the trajectory to process.
    :type index: int
    :param X: Shared list for collecting state matrix contributions.
    :type X: List[np.ndarray]
    :param nb_mu: Total number of parameter samples.
    :type nb_mu: int
    :param use_quad_w: Whether to use quadrature weights.
                       Default is ``False``.
    :type use_quad_w: bool, optional

    :return: None. Results are appended to the shared list `X`.
    :rtype: None
    """
    # Load data
    data = utils.load_case(path=self.path_to_data, index=index)
    if (data is not None):
      # State covariance matrix
      Xi = self._compute_state_cov(data, nb_mu, use_quad_w)
      X.append(Xi)

  # Compute basis
  # ===================================
  def compute_basis(
    self,
    X: np.ndarray,
    rotation: Optional[str] = None,
    xnot: Optional[List[int]] = None,
    rank: int = 100,
    niter: int = 50
  ) -> None:
    """
    Compute POD modes via randomized SVD and optionally apply rotation.

    :param X: State matrix of shape (nb_features, nb_samples).
    :type X: np.ndarray
    :param rotation: Optional rotation method to apply to the modes
                     (e.g., "varimax", "quartimax"). Must be supported
                     by `factor_analyzer`.
    :type rotation: str, optional
    :param xnot: List of feature indices to exclude from the decomposition.
    :type xnot: List[int], optional
    :param rank: Target rank (number of modes) for decomposition.
                 Default is ``100``.
    :type rank: int, optional
    :param niter: Number of power iterations for randomized SVD.
                  Default is ``50``.
    :type niter: int, optional

    :return: None. Saves computed POD components to disk.
    :rtype: None

    :notes:
      - Saves the following data to disk:
        - ``s``: Singular values
        - ``phi``: Trial basis (POD modes, optionally rotated).
        - ``psi``: Test basis (identical to ``phi`` since orthogonal projection).
        - ``mask``: Feature inclusion mask
        - ``xref``, ``xscale``: Scaling parameters
    """
    # Mask covariance matrices
    mask = self._make_mask(
      nb_feat=X.shape[0],
      xnot=xnot
    )
    X = X[mask]
    # Compute SVD
    rank = min(rank, X.shape[0])
    phi, s, _ = map(bkd.to_numpy, torch.svd_lowrank(
      A=bkd.to_torch(X),
      q=rank,
      niter=niter
    ))
    # Vanilla model
    # -------------
    phi = {r: phi[:,:r] for r in range(2,rank+1)}
    data = {
      "s": s,
      "phi": phi,
      "psi": phi,
      "mask": mask,
      "xref": self.xref,
      "xscale": self.xscale
    }
    self._save(data)
    # Rotated model
    # -------------
    if (rotation is not None):
      rot = self._get_rotator(rotation)
      phi = {r: rot.fit_transform(basis) for (r, basis) in phi.items()}
      data["phi"] = phi
      data["psi"] = phi
      self._save(data, identifier=rotation)

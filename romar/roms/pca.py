import os
import numpy as np
import dill as pickle

from typing import Tuple


class PCA(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    scaling: str = "pareto",
    rotation: str = "varimax",
    path_to_saving: str = "./",
    saving: bool = True
  ) -> None:
    self.scaling = scaling
    self.rotation = rotation
    # Configure saving options
    self.saving = saving
    self.path_to_saving = path_to_saving
    os.makedirs(self.path_to_saving, exist_ok=True)

  # Calling
  # ===================================
  def compute_modes(
    self,
    X: np.ndarray,
    xnot: list = [],
    rank: int = 100
  ) -> None:
    # Number of features
    nb_feat = X.shape[0]
    # Mask
    mask = self._make_mask(nb_feat, xnot)
    X = X[mask]
    # Scale
    X, xref, xscale = self._scale(nb_feat, X)
    # Reduce
    rank = min(nb_feat, rank)
    phi, s, _ = np.linalg.svd(X)
    phi = phi[:,:rank]
    # Rotate
    if (self.rotation is not None):
      phi = self._rotate(phi)
    # Save
    data = {}
    for (k, v) in (
      ("s", s),
      ("phi", phi),
      ("psi", phi),
      ("mask", mask),
      ("xref", xref),
      ("xscale", xscale)
    ):
      data[k] = v
    filename = self.path_to_saving + "/pca_bases.p"
    pickle.dump(data, open(filename, "wb"))

  def _make_mask(
    self,
    nb_feat: int,
    xnot: list
  ) -> np.ndarray:
    """
    Generate a mask to exclude specific states from ROM computations.

    :param xnot: List of state indices to exclude.
    :type xnot: list

    :return: Boolean mask indicating included states.
    :rtype: np.ndarray
    """
    mask = np.ones(nb_feat)
    xnot = np.array(xnot).astype(int).reshape(-1)
    mask[xnot] = 0
    return mask.astype(bool)

  def _scale(
    self,
    nb_feat: int,
    X: np.ndarray
  ) -> Tuple[np.ndarray]:
    if (self.scaling is not None):
      xref = np.mean(X, axis=-1)
      std = np.std(X, axis=-1)
      if (self.scaling == "std"):
        xscale = std
      elif (self.scaling == "pareto"):
        xscale = np.sqrt(std)
      else:
        raise ValueError(f"Scaling method '{self.scaling}' not valid.")
    else:
      xref = np.zeros(nb_feat)
      xscale = np.ones(nb_feat)
    X = ((X.T - xref) / xscale).T
    return X, xref, xscale

  def _rotate(
    self,
    phi: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100
  ) -> np.ndarray:
    nrow, ncol = phi.shape
    # Rotation matrix
    R = np.eye(ncol)
    # Variance
    var = 0
    for _ in range(max_iter):
      comp_rot = np.dot(phi, R)
      if (self.rotation == "varimax"):
        tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
      elif (self.rotation == "quartimax"):
        tmp = 0
      else:
        raise ValueError(f"Rotation method '{self.rotation}' not valid.")
      u, s, vh = np.linalg.svd(np.dot(phi.T, comp_rot**3 - tmp))
      R = np.dot(u, vh)
      var_new = np.sum(s)
      if ((var != 0) and (var_new < var * (1.0 + tol))):
        break
      var = var_new
    return np.dot(phi, R).T

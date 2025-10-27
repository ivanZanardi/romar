import numpy as np
import scipy as sp

from typing import Optional, Union
from .utils import init_scaling_param


class ROM(object):
  """
  Reduced-Order Model (ROM) class using basis projection.

  This class provides encoding and decoding of full state vectors into
  reduced coordinates using trial (`phi`) and test (`psi`) basis functions.
  It also supports masking of unreduced variables and optional projection
  using a projection operator.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    nb_eqs: int,
    use_proj: bool
  ) -> None:
    """
    Initialize the ROM model.

    :param nb_eqs: Number of state variables (full system dimension).
    :type nb_eqs: int
    :param use_proj: If True, use the projection operator `proj` for
                     encoding/decoding. Otherwise, use `phi` and `psi` bases.
    :type use_proj: bool
    """
    # Dimensions
    self.nb_eqs = int(nb_eqs)
    self.use_proj = bool(use_proj)
    # Controlling variable
    self.built = False

  # Model construction
  # ===================================
  def build(
    self,
    phi: np.ndarray,
    psi: np.ndarray,
    mask: np.ndarray,
    xref: Optional[Union[str, np.ndarray]] = None,
    xscale: Optional[Union[str, np.ndarray]] = None
  ) -> None:
    """
    Construct the reduced-order model by defining basis matrices and scaling.

    :param phi: Trial basis matrix of shape (nb_reduced_features, r).
    :type phi: np.ndarray
    :param psi: Test basis matrix of shape (nb_reduced_features, r).
    :type psi: np.ndarray
    :param mask: Boolean mask of shape (nb_eqs,) indicating which features
                 are reduced.
    :type mask: np.ndarray
    :param xref: Reference vector for scaling (default: zeros).
    :type xref: Optional[Union[str, np.ndarray]], optional
    :param xscale: Scaling factors for state variables (default: ones).
    :type xscale: Optional[Union[str, np.ndarray]], optional

    :raises ValueError: If the mask length does not match the system dimension.
    """
    # Set flag
    self.built = False
    # Ensure mask is boolean and properly shaped
    self.mask = mask.astype(bool).reshape(-1)
    if (len(self.mask) != self.nb_eqs):
      raise ValueError("Mask size mismatch: " \
                      f"expected {self.nb_eqs}, got {len(mask)}.")
    # Biorthogonalize the basis
    phi = phi @ sp.linalg.inv(psi.T @ phi)
    # Dimensions
    # > Number of excluded states
    self.size_xnot = np.sum(~self.mask)
    # > Number of reduced states
    self.size_zhat = phi.shape[1]
    # Construct full basis matrices
    self.phi = self._build_basis(phi)
    self.psi = self._build_basis(psi)
    # Compute projection operator
    self.proj = self.phi @ self.psi.T
    # Handle reference values
    self.xref = init_scaling_param(xref, self.nb_eqs, ref_value=0.0)
    self.xref[~self.mask]= 0.0
    # Handle scaling values
    self.xscale = init_scaling_param(xscale, self.nb_eqs, ref_value=1.0)
    self.xscale[~self.mask]= 1.0
    self.xscale_mat = np.diag(self.xscale)
    self.ov_xscale_mat = np.diag(1.0/self.xscale)
    # Determine ROM dimension
    self.rom_dim = self.nb_eqs if self.use_proj else self.phi.shape[1]
    # Set encoder/decoder
    self.encoder = self.proj if self.use_proj else self.psi.T
    self.decoder = self.proj if self.use_proj else self.phi
    self.encoder = self.encoder @ self.ov_xscale_mat
    self.decoder = self.xscale_mat @ self.decoder
    # Set flag
    self.built = True

  def _build_basis(
    self,
    pxi: np.ndarray
  ) -> np.ndarray:
    """
    Constructs the full basis matrix, incorporating projection basis
    and identity elements for excluded states (masked out).

    :param pxi: Basis matrix (`phi` or `psi`) of shape (nb_reduced_features, r).
    :type pxi: np.ndarray

    :return: Full basis matrix of shape (nb_eqs, r + nb_unreduced).
    :rtype: np.ndarray
    """
    # Allocate full basis matrix
    basis = np.zeros((self.nb_eqs, self.size_xnot + self.size_zhat))
    # Insert basis for projection
    basis[self.mask,:self.size_zhat] = pxi
    # Insert identity elements for not selected states
    ix = np.where(~self.mask)[0]
    iy = self.size_zhat + np.arange(self.size_xnot)
    basis[ix,iy] = 1.0
    return basis

  # Encoding/Decoding
  # ===================================
  def encode(
    self,
    x: np.ndarray,
    is_der: bool = False
  ) -> np.ndarray:
    """
    Encode full state vector(s) into reduced-order coordinates.

    :param x: Input full state of shape (nb_eqs,) or (N, nb_eqs).
    :type x: np.ndarray
    :param is_der: If True, encode a derivative vector without removing `xref`.
    :type is_der: bool, optional

    :return: Encoded reduced representation of shape (rom_dim,) or (N, rom_dim).
    :rtype: np.ndarray

    :raises ValueError: If ROM is not built or input dimensions mismatch.
    """
    if (not self.built):
      raise ValueError("ROM model not built.")
    # Ensure correct dimensions
    if (x.shape[-1] != self.nb_eqs):
      raise ValueError("Input state shape mismatch: expected last " \
                      f"dimension {self.nb_eqs}, got {x.shape[-1]}.")
    # Encode
    if (not is_der):
      x = x - self.xref
    return x @ self.encoder.T

  def decode(
    self,
    z: np.ndarray,
    is_der: bool = False
  ) -> np.ndarray:
    """
    Decode reduced-order vector(s) into full-state coordinates.

    :param z: Reduced representation of shape (rom_dim,) or (N, rom_dim).
    :type z: np.ndarray
    :param is_der: If True, decode a derivative vector without adding `xref`.
    :type is_der: bool, optional

    :return: Reconstructed full state vector of shape (nb_eqs,) or (N, nb_eqs).
    :rtype: np.ndarray

    :raises ValueError: If ROM is not built or input dimensions mismatch.
    """
    if (not self.built):
      raise ValueError("ROM model not built.")
    # Ensure correct dimensions
    if (z.shape[-1] != self.rom_dim):
      raise ValueError("Input reduced state shape mismatch: Expected " \
                      f"last dimension {self.rom_dim}, got {z.shape[-1]}.")
    # Decode
    x = z @ self.decoder.T
    if (not is_der):
      x = x + self.xref
    return x

  def reduce_jac(
    self,
    j: np.ndarray
  ) -> np.ndarray:
    """
    Reduce a Jacobian matrix via basis projection.

    :param j: Full Jacobian matrix of shape (nb_eqs, nb_eqs).
    :type j: np.ndarray

    :return: Reduced Jacobian matrix of shape (rom_dim, rom_dim).
    :rtype: np.ndarray
    """
    return self.encoder @ j @ self.decoder

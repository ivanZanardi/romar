import numpy as np

from .basic import Basic


class BoxIsoDoped(Basic):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    kin_dtb,
    rad_dtb=None,
    use_rad=False,
    use_proj=False,
    use_factorial=False,
    use_coll_int_fit=False
  ):
    super(BoxIsoDoped, self).__init__(
      species=species,
      kin_dtb=kin_dtb,
      rad_dtb=rad_dtb,
      use_rad=use_rad,
      use_proj=use_proj,
      use_factorial=use_factorial,
      use_coll_int_fit=use_coll_int_fit
    )
    self.nb_eqs = self.nb_comp

  # Function/Jacobian
  # ===================================
  def _fun(self, t, y):
    # ROM activated
    w = self._decode(y) if self.use_rom else y
    # Get number densities
    n = self.mix.get_n(w)
    # Compute sources
    # > Conservative variables
    f_rho = self.sources.call_iso(n)
    # > Primitive variables
    f_w = self.mix.ov_rho * f_rho
    # ROM activated
    f = self._encode(f_w) if self.use_rom else f_w
    return f

  # Output
  # ===================================
  def set_output(self, max_mom=2, linear=True):
    # Linear or log-scaled output
    self.output_lin = bool(linear)
    # Compose C matrix
    self.C = np.eye(self.nb_eqs)
    if (max_mom > 0):
      self.C[::] = 0.0
      # > Species
      si, ei = 0, 0
      for k in self.species_order:
        sk = self.mix.species[k]
        mm = max_mom if (sk.nb_comp > 1) else 1
        ei += mm
        self.C[si:ei,sk.indices] = sk.compute_mom_basis(mm)
        si = ei
      # > Remove zeros rows
      self.C = self.C[:ei+1]

  # Solving
  # ===================================
  def _set_up(self, y0, rho):
    w0, T, Te = y0[:-2], y0[-2], y0[-1]
    self.sources.init_iso(rho, T, Te)
    self.set_fun_jac()
    return w0

  def _encode(self, y):
    return y @ self.P.T if self.use_proj else y @ self.psi

  def _decode(self, z):
    return z @ self.P.T if self.use_proj else z @ self.phi.T

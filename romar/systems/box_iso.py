import torch

from .basic import Basic


class BoxIso(Basic):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    kin_dtb,
    rad_dtb=None,
    use_rad=False,
    use_proj=False,
    use_factorial=True,
    use_tables=True,
    fixed_ne=False
  ):
    super(BoxIso, self).__init__(
      species=species,
      kin_dtb=kin_dtb,
      rad_dtb=rad_dtb,
      use_rad=use_rad,
      use_proj=use_proj,
      use_factorial=use_factorial,
      use_tables=use_tables
    )
    self.fixed_ne = fixed_ne
    self.Th = 3e2
    self.Te = 3e2
    self.nb_eqs = self.nb_comp
    self.rom.nb_eqs = self.nb_comp

  # Function/Jacobian
  # ===================================
  def _fun_pt(self, t, y):
    # Get number densities
    n = self._get_prim(y)[0]
    # Compute sources
    # > Conservative variables
    f_rho = self.sources.call_iso(n)
    # > Primitive variables
    f_w = self.mix.ov_rho * f_rho
    if self.fixed_ne:
      f_w[-1] = 0.0
    return f_w

  def _get_prim(self, y, clip=False):
    n = self.mix.get_n(y)
    Th = torch.full(n[0].shape, self.Th)
    Te = torch.full(n[0].shape, self.Te)
    return n, Th, Te

  # Solving
  # ===================================
  def _set_up(
    self,
    y0: torch.Tensor,
    rho: torch.Tensor
  ) -> torch.Tensor:
    # Unpack the state vector
    w, self.Th, pe = y0[:-2], y0[-2], y0[-1]
    # > Enforce mass to sum to 1
    si = self.mix.species["Ar"].indices
    w[si[0]] = 0.0
    w[si[0]] = 1.0 - torch.sum(w)
    # Set density
    self.mix.set_rho(rho)
    # Get number densities
    n = self.mix.get_n(w)
    # Get electron temperature
    ne = n[self.mix.species["em"].indices].squeeze()
    self.Te = self.mix.get_Te(pe, ne)
    # Initialize the sources
    self.sources.init_iso(self.Th, self.Te)
    # Set the function and Jacobian
    self.set_fun_jac()
    return w

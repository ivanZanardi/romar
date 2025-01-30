import torch

from .. import const
from .basic import Basic


class BoxAd(Basic):

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
    use_coll_int_fit=True
  ):
    super(BoxAd, self).__init__(
      species=species,
      kin_dtb=kin_dtb,
      rad_dtb=rad_dtb,
      use_rad=use_rad,
      use_proj=use_proj,
      use_factorial=use_factorial,
      use_coll_int_fit=use_coll_int_fit
    )

  # Function/Jacobian
  # ===================================
  def _fun(self, t, y):
    # ROM activated
    y = self._decode(y) if self.use_rom else y
    # Extract primitive variables
    n, T, Te = self.get_prim(y)
    # Compute sources
    # > Conservative variables
    f_rho, f_eh, f_ee = self.sources.call_ad(n, T, Te)
    # > Primitive variables
    f = torch.cat([
      self.mix.ov_rho * f_rho,
      self.omega_T(f_rho, f_eh),
      self.omega_pe(f_ee)
    ])
    # ROM activated
    f = self._encode(f) if self.use_rom else f
    return f

  def get_prim(self, y):
    # Unpacking
    w, T, pe = y[:-2], y[-2], y[-1]
    # Get number densities
    n = self.mix.get_n(w)
    # Get electron temperature
    Te = self.mix.get_Te(pe=pe, ne=n[-1])
    # Clip temperatures
    T, Te = [self.clip_temp(z) for z in (T, Te)]
    return n, T, Te

  def clip_temp(self, T):
    return torch.clip(T, const.TMIN, const.TMAX)

  def omega_T(self, f_rho, f_eh):
    # Translational temperature
    f_T = f_eh - self.mix._e_h(f_rho)
    f_T = f_T / (self.mix.rho * self.mix.cv_h)
    return f_T.reshape(1)

  def omega_pe(self, f_ee):
    # Electron pressure
    gamma = self.mix.species["em"].gamma
    f_pe = (gamma - 1.0) * f_ee
    return f_pe.reshape(1)

  # Solving
  # ===================================
  def _set_up(
    self,
    y0: torch.Tensor,
    rho: torch.Tensor
  ) -> torch.Tensor:
    # Set density
    self.mix.set_rho(rho)
    # Set function and Jacobian
    self.set_fun_jac()
    return y0

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
    use_tables=True
  ):
    super(BoxAd, self).__init__(
      species=species,
      kin_dtb=kin_dtb,
      rad_dtb=rad_dtb,
      use_rad=use_rad,
      use_proj=use_proj,
      use_factorial=use_factorial,
      use_tables=use_tables
    )

  # Function/Jacobian
  # ===================================
  def _fun(self, t, y):
    print(float(t))
    # ROM activated
    y = self._decode(y) if self.use_rom else y
    # Extract primitive variables
    n, Th, Te = self._get_prim(y)
    # Compute sources
    # > Conservative variables
    f_rho, f_et, f_ee = self.sources.call_ad(n, Th, Te)
    # > Primitive variables
    f = torch.cat([
      self.mix.ov_rho * f_rho,
      self.omega_T(f_rho, f_et, f_ee),
      self.omega_pe(f_ee)
    ])
    ii = int(torch.argmax(torch.abs(f)))
    print(ii, float(f[ii]))
    # ROM activated
    f = self._encode(f) if self.use_rom else f
    return f

  def _get_prim(self, y):
    # Unpacking
    w, Th, pe = y[:-2], y[-2], y[-1]
    # Get number densities
    n = self.mix.get_n(w)
    # Get electron temperature
    Te = self.mix.get_Te(pe=pe, ne=n[-1])
    # Clip temperatures
    Th, Te = [self.clip_temp(z) for z in (Th, Te)]
    return n, Th, Te

  def clip_temp(self, T):
    return torch.clip(T, const.TMIN, const.TMAX)

  def omega_T(self, f_rho, f_e, f_ee):
    # Translational temperature
    f_T = f_e - (f_ee + self.mix._e_h(f_rho))
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

import torch

from .box_ad import BoxAd


class BoxAdNorm(BoxAd):

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
    super(BoxAdNorm, self).__init__(
      species=species,
      kin_dtb=kin_dtb,
      rad_dtb=rad_dtb,
      use_rad=use_rad,
      use_proj=use_proj,
      use_factorial=use_factorial,
      use_tables=use_tables
    )
    self.wn_no_gs = None

  # Function/Jacobian
  # ===================================
  def _fun_fom(self, t, y):
    # Extract primitive variables
    n, Th, Te = self._get_prim(y)
    # Compute sources
    # > Conservative variables
    f_rho, f_et, f_ee = self.sources.call_ad(n, Th, Te)
    # > Primitive variables
    f_w = self.mix.ov_rho * f_rho
    f_T = self.omega_T(f_rho, f_et, f_ee)
    f_pe = self.omega_pe(f_ee)
    # Normalize neutral Argon states (not ground state)
    normalize_source(self, n, f_w)
    # Return source terms
    return torch.cat([f_w, f_T, f_pe])

  def _get_prim(self, y, clip=True):
    # Unpacking
    wnorm, Th, pe = y[:-2], y[-2], y[-1]
    # De-normalize neutral Argon states (not ground state)
    w = denormalize_state(self, wnorm)
    # Get number densities
    n = self.mix.get_n(w)
    # Get electron temperature
    si = self.mix.species["em"].indices
    ne = n[si].squeeze()
    Te = self.mix.get_Te(pe, ne)
    # Clip temperatures
    if clip:
      Th, Te = [self.clip_temp(z) for z in (Th, Te)]
    return n, Th, Te

  # Solving
  # ===================================
  def _set_up(
    self,
    y0: torch.Tensor,
    rho: torch.Tensor
  ) -> torch.Tensor:
    # Set density
    self.mix.set_rho(rho)
    # Normalize neutral Argon states (not ground state)
    si = self.mix.species["Ar"].indices
    w = y0[:-2]
    # > Enforce mass to sum to 1
    w[si[:1]] = 0.0
    w[si[:1]] = 1.0 - torch.sum(w)
    # > Normalize
    w[si[1:]] /= torch.sum(w[si[1:]])
    y0[:-2] = w
    # Set function and Jacobian
    self.set_fun_jac()
    return y0

# Public functions
# =====================================
def normalize_source(model, n, f_w):
  w = model.mix.get_w(n)
  si = model.mix.species["Ar"].indices[1:]
  f_w[si] = (1.0/model.wn_no_gs)*f_w[si] \
          - (1.0/model.wn_no_gs**2)*w[si]*torch.sum(f_w[si])

def denormalize_state(model, wnorm):
  model.wn_no_gs = compute_wn_no_gs(model, wnorm)
  w = []
  for s in model.species_order:
    si = model.mix.species[s].indices
    if (s == "Ar"):
      w.append(wnorm[si[:1]])
      w.append(wnorm[si[1:]]*model.wn_no_gs)
    else:
      w.append(wnorm[si])
  return torch.cat(w)

def compute_wn_no_gs(model, wnorm):
  # Argon mass fraction (not ground state)
  wn_no_gs = torch.ones_like(wnorm[:1])
  for s in model.mix.species.values():
    si = s.indices
    if (s.name == "Ar"):
      si = si[:1]
    wn_no_gs -= torch.sum(wnorm[si], dim=0, keepdim=True)
  return wn_no_gs

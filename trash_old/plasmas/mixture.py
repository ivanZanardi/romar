import torch

from .. import const
from .species import Species


class Mixture(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    species_order=(),
    use_factorial=True
  ):
    self.species_order = tuple(species_order)
    self.use_factorial = bool(use_factorial)
    self._init_species(species)
    # Mixture density
    self.rho = None
    self.ov_rho = None

  def _init_species(self, species):
    # Initialize species
    self.species = {}
    for k in species.keys():
      self.species[k] = Species(species[k], self.use_factorial)
    # Index species
    si = 0
    for k in self.species_order:
      ei = si + self.species[k].nb_comp
      self.species[k].indices = torch.arange(si, ei).tolist()
      si = ei
    self.nb_comp = ei

  # Build
  # ===================================
  def build(self):
    for s in self.species.values():
      s.build()
    self._build_mass_matrix()
    self._build_delta_energies()

  def _build_mass_matrix(self):
    # Build vector of masses
    self.m = torch.ones(self.nb_comp)
    for k in self.species_order:
      s = self.species[k]
      self.m[s.indices] *= s.m
    self.m_inv = 1.0/self.m
    # Build mass matrix
    self.m_mat = torch.diag(self.m)
    self.m_inv_mat = torch.diag(self.m_inv)

  def _build_delta_energies(self):
    self.de = {}
    for i in self.species_order:
      si = self.species[i]
      ei = si.e_int * si.m
      for j in self.species_order:
        sj = self.species[j]
        ej = sj.e_int * sj.m
        # Compute energy difference [J]
        self.de[j+"-"+i] = ej.reshape(1,-1) - ei.reshape(-1,1)

  # Update
  # ===================================
  def update(self, n, T, Te=None):
    # Update composition
    self.update_composition(n)
    # Update species thermo
    self.update_species_thermo(T, Te)
    # Update mixture thermo
    self.update_mixture_thermo()

  # Composition
  # -----------------------------------
  def update_composition(self, n):
    x = n / torch.sum(n)
    w = self.get_w(n)
    for s in self.species.values():
      s.x = x[s.indices]
      s.n = n[s.indices]
      s.w = w[s.indices]
      s.rho = self.rho * s.w
    self._M()
    self._R()

  # Thermodynamics
  # -----------------------------------
  def update_species_thermo(self, T, Te=None):
    if (Te is None):
      Te = T
    for s in self.species.values():
      Ti = Te if (s.name == "em") else T
      s.update(Ti)

  def update_mixture_thermo(self):
    # Constant-volume specific heats
    self.cv_e = self._cv_e()
    self.cv_h = self._cv_h()
    self.cv = self.cv_e + self.cv_h
    # Energies
    self.e_e = self._e_e()
    self.e_h = self._e_h()
    self.e_int_h = self._e_int_h()
    self.e = self.e_e + self.e_h

  # Conversions
  # ===================================
  def get_n(self, w):
    return self.rho * self.m_inv_mat @ w

  def get_rho(self, n):
    return self.m_mat @ n

  def get_w(self, n):
    return self.ov_rho * self.m_mat @ n

  def get_Te(self, pe, ne):
    return pe / (ne * const.UKB)

  def get_pe(self, Te, ne):
    return ne * const.UKB * Te

  # Mixture properties
  # ===================================
  def set_rho(self, rho):
    self.rho = rho
    self.ov_rho = 1.0/rho

  def _M(self, qoi_used="w"):
    self.M = torch.zeros(1)
    if (qoi_used == "w"):
      for s in self.species.values():
        self.M += torch.sum(s.w) / s.M
      self.M = 1.0/self.M
    elif (qoi_used == "x"):
      for s in self.species.values():
        self.M += torch.sum(s.x) * s.M

  def _R(self):
    self.R = torch.zeros(1)
    for s in self.species.values():
      self.R += torch.sum(s.w) * s.R

  def _convert_mass_mole(self, qoi_used="w"):
    if (qoi_used == "w"):
      self._set_xs()
    elif (qoi_used == "x"):
      self._set_ws()

  def _set_xs(self):
    for s in self.species.values():
      s.x = self.M / s.M * s.w

  def _set_ws(self):
    for s in self.species.values():
      s.w = s.M / self.M * s.x

  def _get_ys(self, species, y):
    # Default: Mass fractions
    return species.w if (y is None) else y[species.indices]

  # Constant-volume specific heats
  # -----------------------------------
  def _cv(self, y=None):
    # Total [J/(kg K)]
    cv = torch.zeros(1)
    for s in self.species.values():
      ys = self._get_ys(s, y)
      cv += torch.sum(ys * s.cv)
    return cv

  def _cv_e(self, y=None):
    # Electron [J/(kg K)]
    s = self.species["em"]
    ys = self._get_ys(s, y)
    return ys * s.cv

  def _cv_h(self, y=None):
    # Heavy particle [J/(kg K)]
    cv_h = torch.zeros(1)
    for s in self.species.values():
      if (s.name != "em"):
        ys = self._get_ys(s, y)
        cv_h += torch.sum(ys * s.cv)
    return cv_h

  # Energies
  # -----------------------------------
  def _e(self, y=None):
    # Total energy [J/kg]
    e = torch.zeros(1)
    for s in self.species.values():
      ys = self._get_ys(s, y)
      e += torch.sum(ys * s.e)
    return e

  def _e_e(self, y=None):
    # Electron energy [J/kg]
    s = self.species["em"]
    ys = self._get_ys(s, y)
    return ys * s.e

  def _e_h(self, y=None):
    # Heavy particle energy [J/kg]
    e_h = torch.zeros(1)
    for s in self.species.values():
      if (s.name != "em"):
        ys = self._get_ys(s, y)
        e_h += torch.sum(ys * s.e)
    return e_h

  def _e_int_h(self, y=None):
    # Heavy particle internal energy [J/kg]
    e_int_h = torch.zeros(1)
    for s in self.species.values():
      if (s.name != "em"):
        ys = self._get_ys(s, y)
        e_int_h += torch.sum(ys * s.e_int)
    return e_int_h

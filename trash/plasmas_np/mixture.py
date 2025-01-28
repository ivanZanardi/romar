import numpy as np

from .species import Species
from typing import Dict, Union


class Mixture(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species: Dict[str, Union[str, dict]],
    species_order: tuple = (),
    use_factorial: bool = True
  ):
    self.species_order = tuple(species_order)
    self.use_factorial = bool(use_factorial)
    self._init_species(species)

  def _init_species(
    self,
    species: Dict[str, Union[str, dict]]
  ) -> None:
    # Initialize species
    self.species = {}
    for k in species.keys():
      self.species[k] = Species(species[k], self.use_factorial)
    # Index species
    si = 0
    for k in self.species_order:
      ei = si + self.species[k].nb_comp
      self.species[k].indices = np.arange(si, ei)
      si = ei
    self.nb_comp = ei

  # Build
  # ===================================
  def build(self) -> None:
    for s in self.species.values():
      s.build()
    self._build_mass_matrix()
    self._build_delta_energies()

  def _build_mass_matrix(self) -> None:
    # Build vector of masses
    self.m = np.ones(self.nb_comp)
    for k in self.species_order:
      s = self.species[k]
      self.m[s.indices] *= s.m
    self.m_inv = 1.0/self.m
    # Build mass matrix
    self.m_mat = np.diag(self.m)
    self.m_inv_mat = np.diag(self.m_inv)

  def _build_delta_energies(self) -> None:
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
  def update(self, rho, w, T, Te=None) -> None:
    # Update composition
    self.update_composition(rho, w)
    # Update species thermo
    self.update_species_thermo(T, Te)
    # Update mixture thermo
    self.update_mixture_thermo()

  # Composition
  # -----------------------------------
  def update_composition(self, rho, w) -> None:
    n = self.get_n(rho, w)
    x = (1.0/np.sum(n)) * n
    for s in self.species.values():
      s.x = x[s.indices]
      s.n = n[s.indices]
      s.w = w[s.indices]
      s.rho = rho * s.w
    self._M()
    self._R()

  # Thermodynamics
  # -----------------------------------
  def update_species_thermo(self, T, Te=None) -> None:
    if (Te is None):
      Te = T
    for s in self.species.values():
      Ti = Te if (s.name == "em") else T
      s.update(Ti)

  def update_mixture_thermo(self) -> None:
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
  def get_w(
    self,
    rho: float,
    n: np.ndarray
  ) -> np.ndarray:
    return (1.0/rho) * self.m_mat @ n

  def get_n(
    self,
    rho: float,
    w: np.ndarray
  ) -> np.ndarray:
    return rho * self.m_inv_mat @ w

  def get_rho(
    self,
    n: np.ndarray
  ) -> float:
    return self.m @ n

  def get_Te(
    self,
    rho: float,
    pe: float,
    w: np.ndarray
  ) -> float:
    s = self.species["em"]
    return pe / (rho * w[s.indices] * s.R)

  def get_pe(
    self,
    rho: float,
    Te: float,
    w: np.ndarray
  ) -> float:
    s = self.species["em"]
    return rho * w[s.indices] * s.R * Te

  # Mixture properties
  # ===================================
  def _M(
    self,
    qoi_used: str = "w"
  ):
    self.M = np.zeros(1)
    if (qoi_used == "w"):
      for s in self.species.values():
        self.M += np.sum(s.w) / s.M
      self.M = 1.0/self.M
    elif (qoi_used == "x"):
      for s in self.species.values():
        self.M += np.sum(s.x) * s.M

  def _R(self):
    self.R = np.zeros(1)
    for s in self.species.values():
      self.R += np.sum(s.w) * s.R

  def _convert_mass_mole(
    self,
    qoi_used: str = "w"
  ):
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
    cv = np.zeros(1)
    for s in self.species.values():
      ys = self._get_ys(s, y)
      cv += np.sum(ys * s.cv)
    return cv

  def _cv_e(self, y=None):
    # Electron [J/(kg K)]
    s = self.species["em"]
    ys = self._get_ys(s, y)
    return ys * s.cv

  def _cv_h(self, y=None):
    # Heavy particle [J/(kg K)]
    cv_h = np.zeros(1)
    for s in self.species.values():
      if (s.name != "em"):
        ys = self._get_ys(s, y)
        cv_h += np.sum(ys * s.cv)
    return cv_h

  # Energies
  # -----------------------------------
  def _e(self, y=None):
    # Total energy [J/kg]
    e = np.zeros(1)
    for s in self.species.values():
      ys = self._get_ys(s, y)
      e += np.sum(ys * s.e)
    return e

  def _e_e(self, y=None):
    # Electron energy [J/kg]
    s = self.species["em"]
    ys = self._get_ys(s, y)
    return ys * s.e

  def _e_h(self, y=None):
    # Heavy particle energy [J/kg]
    e_h = np.zeros(1)
    for s in self.species.values():
      if (s.name != "em"):
        ys = self._get_ys(s, y)
        e_h += np.sum(ys * s.e)
    return e_h

  def _e_int_h(self, y=None):
    # Heavy particle internal energy [J/kg]
    e_int_h = np.zeros(1)
    for s in self.species.values():
      if (s.name != "em"):
        ys = self._get_ys(s, y)
        e_int_h += np.sum(ys * s.e_int)
    return e_int_h

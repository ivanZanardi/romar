import numpy as np

from .. import const
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
    en = self.species["Ar"].lev["E"]    # [J]
    ei = self.species["Arp"].lev["E"]   # [J]
    self.de = {
      # Neutral-Neutral
      "nn": en.reshape(1,-1) - en.reshape(-1,1),
      # Ion-Neutral
      "in": ei.reshape(1,-1) - en.reshape(-1,1)
    }

  # Update
  # ===================================
  # Composition
  # -----------------------------------
  def update_composition(self, w, rho) -> None:
    n = self.get_n(w, rho)
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
    # Constant specific heats
    self.cv_h = self._cv_h()
    # Energies
    self.e = self._e()
    self.e_e = self._e_e()
    self.e_int_h = self._e_int_h()
    self.hf_h = self._hf_h()

  # Conversions
  # ===================================
  def get_w(
    self,
    n: np.ndarray,
    rho: float
  ) -> np.ndarray:
    return (1.0/rho) * self.m_mat @ n

  def get_n(
    self,
    w: np.ndarray,
    rho: float
  ) -> np.ndarray:
    return rho * self.m_inv_mat @ w

  def get_rho(
    self,
    n: np.ndarray
  ) -> float:
    return self.m @ n

  # Equilibrium composition
  # ===================================
  def compute_eq_comp(
    self,
    p: float,
    T: float
  ) -> np.ndarray:
    # Solve this system of equations:
    # -------------
    # 1) Charge neutrality:
    #    x_em = x_Arp
    # 2) Mole conservation:
    #    x_em + x_Arp + x_Ar = 1
    # 3) Detailed balance:
    #    (n_Arp*n_em)/n_Ar = (Q_Arp*Q_em)/Q_Ar
    # -------------
    # Update thermo
    self.update_species_thermo(T)
    # Compute number density
    n = p / (const.UKB*T)
    # Compute coefficient for quadratic system
    f = self.species["Arp"].Q * self.species["em"].Q
    f /= (self.species["Ar"].Q * n)
    # Solve quadratic system for 'x_em'
    a = 1.0
    b = 2.0 * f
    c = -f
    x = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    # Set molar fractions
    s = self.species["em"]
    s.x = x

    
    s = self.species["Arp"]
    s.x = x * s.q / s.Q
    s = self.species["Ar"]
    s.x = (1.0-2.0*x) * s.q / s.Q
    # Number densities
    n = n * np.concatenate([self.species[k].x for k in self.species_order])
    # Mass fractions and density
    rho = self.get_rho(n)
    w = self.get_w(n, rho)
    return w, rho

  # Initial composition
  # ===================================
  def get_init_composition(
    self,
    p: float,
    T: float,
    noise: bool = False,
    sigma: float = 1e-2
  ) -> np.ndarray:
    # Compute equilibrium mass fractions
    w, rho = self.compute_eq_comp(p, T)
    # Add random noise
    if noise:
      # > Electron
      s = self.species["em"]
      x_em = np.clip(s.x * self._add_norm_noise(s, sigma, use_q=False), 0, 1)
      s.x = x_em
      # > Argon Ion
      s = self.species["Arp"]
      s.x = x_em * self._add_norm_noise(s, sigma)
      # > Argon
      s = self.species["Ar"]
      s.x = (1.0-2.0*x_em) * self._add_norm_noise(s, sigma)
      # Set mass fractions
      self._M("x")
      self._set_w_s()
      # Mass densities
      w = np.concatenate([self.species[k].w for k in self.species_order])
    # Return mass densities
    return w, rho

  def _add_norm_noise(self, species, sigma, use_q=True):
    f = 1.0 + sigma * np.random.rand(species.nb_comp)
    if use_q:
      f *= species.q * f
      f /= np.sum(f)
    return f

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
      self._set_x_s()
    elif (qoi_used == "x"):
      self._set_w_s()

  def _set_x_s(self):
    for s in self.species.values():
      s.x = self.M / s.M * s.w

  def _set_w_s(self):
    for s in self.species.values():
      s.w = s.M / self.M * s.x

  # Constant specific heats
  # -----------------------------------
  def _cv_h(self):
    # Heavy particle constant-volume specific heat [J/(kg K)]
    cv_h = np.zeros(1)
    for s in self.species.values():
      if (s.name != "em"):
        cv_h += np.sum(s.w * s.cv)
    return cv_h

  # Energies
  # -----------------------------------
  def _e(self):
    # Total energy [J/kg]
    e = np.zeros(1)
    for s in self.species.values():
      e += np.sum(s.w * s.e)
    return e

  def _e_e(self):
    # Electron energy [J/kg]
    s = self.species["em"]
    return np.sum(s.w * s.e, keepdims=True)

  def _e_int_h(self):
    # Heavy particle internal energy [J/kg]
    e_int_h = np.zeros(1)
    for s in self.species.values():
      if (s.name != "em"):
        e_int_h += np.sum(s.w * s.e_int)
    return e_int_h

  def _hf_h(self):
    # Heavy particle enthalpy of formation [J/kg]
    hf_h = np.zeros(1)
    for s in self.species.values():
      if (s.name != "em"):
        hf_h += np.sum(s.w) * s.hf
    return hf_h

  # Temperatures
  # -----------------------------------
  def compute_temp(self, e, e_e):
    # Heavy particle temperature [K]
    T = (e - e_e - self._e_int_h() - self._hf_h()) / self._cv_h()
    # Free electron temperature [K]
    s = self.species["em"]
    Te = e_e / (s.w * s.cv)
    print(T, Te)
    # Clipping
    T = np.maximum(T, const.TMIN)
    Te = np.maximum(Te, const.TMIN)
    print(T, Te)
    return T, Te

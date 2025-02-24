import json
import numpy as np

from .. import const


class Species(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    properties,
    use_factorial=True
  ):
    # Load properties
    if (not isinstance(properties, dict)):
      with open(properties, "r") as file:
        properties = json.load(file)
    # Set properties
    for (k, v) in properties.items():
      setattr(self, k, v)
    self.lev = {k: np.array(v).reshape(-1) for (k, v) in self.lev.items()}
    # Control variables
    self.use_factorial = use_factorial
    # Indexing
    self.indices = None

  # Properties
  # ===================================
  @property
  def w(self):
    return self._w

  @w.setter
  def w(self, value):
    self._w = value

  @property
  def x(self):
    return self._x

  @x.setter
  def x(self, value):
    self._x = value

  @property
  def rho(self):
    return self._rho

  @rho.setter
  def rho(self, value):
    self._rho = value

  @property
  def n(self):
    return self._n

  @n.setter
  def n(self, value):
    self._n = value

  # Moments
  # ===================================
  def compute_mom(self, n, m=0):
    e = self.lev["E"] / const.eV_to_J
    if (n.shape[-1] != self.nb_comp):
      n = n.T
    return np.sum(n * e**m, axis=-1)

  def compute_mom_basis(self, max_mom):
    e = self.lev["E"] / const.eV_to_J
    m = [np.ones_like(e)]
    for i in range(max_mom-1):
      mi = m[-1]*e
      if self.use_factorial:
        mi /= (i+1)
      m.append(mi)
    return np.vstack(m) / self.M

  # Build and update
  # ===================================
  def build(self):
    # Thermo properties
    # > Charge number
    self.Z = int(self.Z)
    # > Mass [kg]
    self.m = self.M / const.UNA
    # > Specific gas constants [J/(kg K)]
    self.R = const.URG / self.M
    # Translational partition function factor
    self.q_tr_fac = 2.0 * np.pi * self.m * const.UKB / (const.UH**2)
    # Constant-volume and -pressure specific heats [J/(kg K)]
    self.cv = self.cv_tr = 1.5 * self.R
    self.cp = self.cv_tr + self.R
    # Specific heat ratio
    self.gamma = 5.0 / 3.0
    # Internal energy [J/kg]
    self.e_int = self.lev["E"] / self.m
    # Enthalpy of formation [J/kg]
    self.hf = self.Hf / self.m

  def update(self, T):
    # Partition functions
    self.ov_kT = 1.0/(const.UKB*T)
    self.q = self._q(T)
    self.Q = self._Q()
    # Energies [J/kg]
    self.e_tr = self.cv_tr * T
    self.e = self.e_tr + self.e_int + self.hf

  # Partition functions
  # ===================================
  def _Q(self):
    return np.sum(self.q, keepdims=True)

  def _q(self, T):
    return self._q_zero() * self._q_tr(T) * self._q_int()

  def _q_zero(self):
    return np.exp(-self.Hf * self.ov_kT)

  def _q_tr(self, T):
    return np.power(self.q_tr_fac * T, 1.5)

  def _q_int(self):
    return self.lev["g"] * np.exp(-self.lev["E"] * self.ov_kT)

import json
import torch

from ... import const
from ... import backend as bkd


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
    for (k, v) in self.lev.items():
      self.lev[k] = bkd.to_torch(v).reshape(-1)
    self.nb_comp = len(self.lev["E"])
    # Control variables
    self.use_factorial = bool(use_factorial)
    # Indexing
    self.indices = []
    # Methods
    self.compute_mom = bkd.make_fun_np(self._compute_mom)

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
  def _compute_mom(self, n, m=0):
    e = self.lev["E"] / const.UE
    if (n.shape[-1] != self.nb_comp):
      n = n.T
    return torch.sum(n * e**m, dim=-1)

  def compute_mom_basis(self, max_mom):
    e = self.lev["E"] / const.UE
    m = [torch.ones_like(e)]
    for i in range(max_mom-1):
      mi = m[-1]*e
      if self.use_factorial:
        mi /= (i+1)
      m.append(mi)
    return torch.vstack(m) / self.M

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
    self.q_tr_fac = 2.0 * torch.pi * self.m * const.UKB / (const.UH * const.UH)
    # Constant-volume and -pressure specific heats [J/(kg K)]
    self.cv = self.cv_tr = 1.5 * self.R
    self.cp = self.cv_tr + self.R
    # Specific heat ratio
    self.gamma = 5.0 / 3.0
    # Internal energy [J/kg]
    self.e_int = (self.lev["E"] + self.Hf) / self.m

  def update(self, T):
    # Partition functions
    self.ov_kT = 1.0/(const.UKB*T)
    self.q = self._q(T)
    self.Q = self._Q()
    # Energies [J/kg]
    self.e_tr = self.cv_tr * T
    self.e = self.e_tr + self.e_int

  # Partition functions
  # ===================================
  def _Q(self):
    return torch.sum(self.q).reshape(1)

  def _q(self, T):
    return self._q_zero() * self._q_tr(T) * self._q_int()

  def _q_zero(self):
    return torch.exp(-self.Hf * self.ov_kT)

  def _q_tr(self, T):
    return torch.pow(self.q_tr_fac * T, 1.5)

  def _q_int(self):
    return self.lev["g"] * torch.exp(-self.lev["E"] * self.ov_kT)

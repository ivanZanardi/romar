import numpy as np

from .species import Species
from typing import Optional, Tuple


class Mixture(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    use_factorial=True
  ):
    self.nb_eqs = 0
    self.species = {}
    for k in ("atom", "molecule"):
      if (k not in species):
        raise ValueError(
          "The 'species' input parameter should be a " \
          "dictionary with 'atom' and 'molecule' as keys."
        )
      self.species[k] = Species(species[k], use_factorial)
      self.nb_eqs += self.species[k].nb_comp
    self._set_mass_matrix()
    self._set_mass_ratio()

  def _set_mass_matrix(self) -> None:
    # Compose mass matrix
    m = np.full(self.nb_eqs, self.species["molecule"].m)
    m[0] = self.species["atom"].m
    self.M = np.diag(m)
    self.Minv = np.diag(1.0/m)

  def _set_mass_ratio(self) -> None:
    self.m_ratio = self.species["molecule"].m / self.species["atom"].m
    self.m_ratio = np.full(self.nb_eqs-1, self.m_ratio).reshape(1,-1)

  # Update
  # ===================================
  def update(self, T) -> None:
    for sp in self.species.values():
      sp.update(T)
    self._set_gamma()

  def _set_gamma(self) -> None:
    q_m, q_a = [self.species[k].q for k in ("molecule", "atom")]
    self.gamma = q_m / q_a**2

  def compute_eq_comp(
    self,
    rho: float
  ) -> Tuple[np.ndarray]:
    # Solve this system of equations:
    # 1) rho_a + sum(rho_m) = rho
    # 2) n_m = gamma * n_a^2
    a = np.sum(self.gamma) * self.species["molecule"].m
    b = self.species["atom"].m
    c = -rho
    n_a = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    n_m = self.gamma*n_a**2
    return np.concatenate([n_a.reshape(1), n_m])

  # Solving
  # ===================================
  def get_init_sol(
    self,
    T: float,
    wx_a: float,
    rho_p: Optional[float] = None,
    noise: bool = False,
    sigma: float = 1e-2,
    mu_type: str = "mass"
  ) -> np.ndarray:
    if (mu_type == "mass"):
      w_a, rho = wx_a, rho_p
    else:
      w_a, rho = self._convert_mu(T, x_a=wx_a, p=rho_p)
    return self._get_init_sol(T, w_a, rho, noise, sigma)

  def _convert_mu(self, T, x_a, p):
    for (s, xs) in (
      ("atom", x_a),
      ("molecule", 1-x_a)
    ):
      self.species[s].x = xs
    M = self._get_M("x")
    self._convert_mass_mole(M, "x")
    R = self._get_R()
    w_a = self.species["atom"].w
    rho = p / (R * T)
    return w_a, rho

  def _get_init_sol(
    self,
    T: float,
    w_a: float,
    rho: Optional[float] = None,
    noise: bool = False,
    sigma: float = 1e-2
  ) -> np.ndarray:
    # > Atom
    w_a = np.array(w_a).reshape(1)
    if noise:
      w_a = np.clip(w_a + sigma*np.random.rand(1), 0, 1)
    # > Molecule
    q_m = self.species["molecule"].q_int(T)
    if noise:
      q_m *= (1 + sigma*np.random.rand(*q_m.shape))
    w_m = (1-w_a)*(q_m / np.sum(q_m))
    w = np.concatenate([w_a, w_m])
    # Return mass fractions / number densities
    return w if (rho is None) else self.get_n(w, rho)

  # Mixture properties
  # ===================================
  def get_w(
    self,
    n: np.ndarray,
    rho: float
  ) -> np.ndarray:
    return (1/rho) * self.M @ n

  def get_n(
    self,
    w: np.ndarray,
    rho: float
  ) -> np.ndarray:
    return rho * self.Minv @ w

  def get_rho(
    self,
    n: np.ndarray
  ) -> float:
    return np.diag(self.M) @ n

  def _get_M(self, qoi_used="w"):
    M = 0.0
    if (qoi_used == "w"):
      for s in self.species.values():
        M += np.sum(s.w) / s.M
      M = 1.0/M
    elif (qoi_used == "x"):
      for s in self.species.values():
        M += np.sum(s.x) * s.M
    return M

  def _get_R(self):
    R = 0.0
    for s in self.species.values():
      R += np.sum(s.w) * s.R
    return R

  def _convert_mass_mole(self, M, qoi_used="w"):
    if (qoi_used == "w"):
      self._set_x_s(M)
    elif (qoi_used == "x"):
      self._set_w_s(M)

  def _set_x_s(self, M):
    for s in self.species.values():
      s.x = M / s.M * s.w

  def _set_w_s(self, M):
    for s in self.species.values():
      s.w = s.M / M * s.x

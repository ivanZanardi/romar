import torch
import numpy as np
import scipy as sp

from typing import *
from ... import const
from .mixture import Mixture
from ... import backend as bkd

MU_VARS = ("rho", "Th", "Te")


class Equilibrium(object):
  """
  Class to compute the equilibrium state (mass fractions and temperature)
  of a system involving argon and its ionized species (Ar, Ar+, and e^-).

  The equilibrium state is determined by solving the following system of
  equations:
  1) Charge neutrality
  2) Mole conservation
  3) Detailed balance, describing the ionization equilibrium between neutral
    argon, argon ions, and electrons in the system.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mixture: Mixture
  ) -> None:
    """
    Initialize the equilibrium solver with a specified gas mixture.

    :param mixture: Chemical mixture object containing species definitions
                    and thermodynamic properties.
    :type mixture: Mixture
    """
    self.mix = mixture
    self.lsq_opts = dict(
      method="trf",
      ftol=bkd.epsilon(),
      xtol=bkd.epsilon(),
      gtol=0.0,
      max_nfev=int(1e4)
    )
    self.set_fun_jac()

  def set_fun_jac(self) -> None:
    """
    Register functions and Jacobians for the least-squares optimization
    for both the primitive (`from_prim`) and conservative (`from_cons`)
    variable formulations.
    """
    for name in ("from_cons",):
      # Function
      fun = getattr(self, f"_{name}_fun")
      setattr(self, f"{name}_fun", bkd.make_fun_np(fun))
      # Jacobian
      jac = torch.func.jacrev(fun, argnums=0)
      setattr(self, f"{name}_jac", bkd.make_fun_np(jac))

  # Initial solution
  # ===================================
  def get_init_sol(
    self,
    mu: np.ndarray,
    noise: bool = False,
    sigma: float = 1e-1
  ) -> Tuple[np.ndarray, float]:
    """
    Generate an equilibrium initial condition from input primitive variables.

    Optionally adds random perturbation for stochastic initial states, and
    overrides the electron temperature with the translational one for heat
    bath simulations.

    :param mu: Vector of primitive variables (rho, Th, Te).
    :type mu: np.ndarray
    :param noise: Whether to apply random noise to composition.
    :type noise: bool, optional
    :param sigma: Standard deviation of random noise.
    :type sigma: float, optional

    :return: Tuple of (equilibrium state vector, mass density).
    :rtype: Tuple[np.ndarray, float]
    """
    # Unpack the input array into individual parameters
    rho, Th, Te = mu
    # Compute the equilibrium state based on rho and Te
    y, _ = self.from_prim(rho, Te)
    # If noise requested, update the composition and recompute the state vector
    if noise:
      self._update_composition(
        ze=self.mix.species["em"].x,
        noise=noise,
        sigma=sigma
      )
      y = self._compose_state_vector(
        T=bkd.to_torch(Te).reshape(1)
      )
    # Replace the equilibrium temperature Te with Th for heat bath simulation
    y[-2] = Th
    return bkd.to_numpy(y), float(rho)

  # Primitive variables
  # ===================================
  def from_prim(
    self,
    rho: float,
    T: float
  ) -> Tuple[np.ndarray, float]:
    """
    Compute the equilibrium state from primitive macroscopic variables
    such as density and temperature.

    :param rho: Mass density.
    :type rho: float
    :param T: Translational temperature.
    :type T: float

    :return: Tuple of (state vector, mass density).
    :rtype: Tuple[np.ndarray, float]
    """
    # Convert to 'torch.Tensor'
    rho, T = [bkd.to_torch(z).reshape(1) for z in (rho, T)]
    # Update mixture
    self.mix.set_rho(rho)
    self.mix.update_species_thermo(T)
    # Compute electron mass fraction
    we = self._we_from_prim(rho)
    # Compose state vector
    self._update_composition(we, by_mass=True)
    y = self._compose_state_vector(T)
    return bkd.to_numpy(y), float(rho)

  def _we_from_prim(
    self,
    rho: float
  ) -> float:
    """
    Compute electron mass fraction at equilibrium using closed-form solution.

    Solves a quadratic relation derived from detailed balance and mass
    conservation for Ar ↔ Ar⁺ + e⁻.

    :param rho: Mass density.
    :type rho: float

    :return: Electron mass fraction.
    :rtype: float
    """
    # Compute coefficients for quadratic system
    m, Q = [self._get_species_attr(k) for k in ("m", "Q")]
    f = [z["Arp"]*z["em"]/z["Ar"] for z in (m, Q)]
    f = (1.0/rho) * f[0] * f[1]
    r = m["Arp"]/m["em"]
    # Solve quadratic system for 'we'
    a = r
    b = f * (1.0 + r)
    c = -f
    return (-b+torch.sqrt(b**2-4*a*c))/(2*a)

  # Conservative variables
  # ===================================
  def from_cons(
    self,
    rho: float,
    e: float
  ) -> Tuple[np.ndarray, float]:
    """
    Solve for equilibrium state using conservative variables (rho, e).

    Uses nonlinear least-squares optimization to satisfy:
      - Energy conservation
      - Detailed balance

    :param rho: Mass density.
    :type rho: float
    :param e: Total energy per unit volume.
    :type e: float

    :return: Tuple of (state vector, mass density).
    :rtype: Tuple[np.ndarray, float]
    """
    # Convert to 'torch.Tensor'
    rho, e = [bkd.to_torch(z).reshape(1) for z in (rho, e)]
    # Update mixture
    self.mix.set_rho(rho)
    # Compute electron molar fraction and temperaure
    x = sp.optimize.least_squares(
      fun=self.from_cons_fun,
      x0=np.log([1e-2,1e4]),
      jac=self.from_cons_jac,
      bounds=([-np.inf, -np.inf], [np.log(0.5), np.log(1e5)]),
      args=(e,),
      **self.lsq_opts
    ).x
    # Extract variables
    xe, T = [z.reshape(1) for z in bkd.to_torch(np.exp(x))]
    # Update species thermo
    self.mix.update_species_thermo(T)
    # Compose state vector
    self._update_composition(xe)
    y = self._compose_state_vector(T)
    return bkd.to_numpy(y), float(rho)

  def _from_cons_fun(
    self,
    x: torch.Tensor,
    e: torch.Tensor
  ) -> torch.Tensor:
    """
    Residuals for detailed balance and energy conservation.

    :param x: Logarithmic variables: [log(xe), log(T)].
    :type x: torch.Tensor
    :param e: Target total energy.
    :type e: torch.Tensor

    :return: Vector of residuals: [detailed_balance, energy_error].
    :rtype: torch.Tensor
    """
    # Extract variables
    xe, T = torch.exp(x)
    # Update species thermo
    self.mix.update_species_thermo(T)
    # Update composition
    self._update_composition(xe)
    # Update mixture thermo
    self.mix.update_mixture_thermo()
    # Enforce detailed balance
    f0 = self._detailed_balance()
    # Enforce conservation of energy
    f1 = self.mix.e / e - 1.0
    return torch.cat([f0,f1])

  # Utils
  # ===================================
  def _compose_state_vector(
    self,
    T: torch.Tensor
  ) -> torch.Tensor:
    """
    Construct full equilibrium state vector.

    :param T: Translational/electron temperature.
    :type T: torch.Tensor

    :return: Vector including [mass fractions, T, pe].
    :rtype: torch.Tensor
    """
    pe = self.mix.get_pe(Te=T, ne=self.mix.species["em"].n)
    w = self.mix.get_qoi_vec("w")
    return torch.cat([w, T, pe])

  def _update_composition(
    self,
    ze: torch.Tensor,
    by_mass: bool = False,
    noise: bool = False,
    sigma: float = 1e-1
  ) -> None:
    """
    Update molar or mass composition using electron fraction.

    :param ze: Electron molar or mass fraction.
    :type ze: torch.Tensor
    :param by_mass: If True, interpret `ze` as mass fraction.
    :type by_mass: bool, optional
    :param noise: Whether to add random perturbation to composition.
    :type noise: bool, optional
    :param sigma: Standard deviation of added noise.
    :type sigma: float, optional
    """
    # Vector of molar/mass fractions
    z = torch.zeros(self.mix.nb_comp)
    # Electron
    sk = self.mix.species["em"]
    z[sk.indices] = ze
    # Compute coefficient
    if by_mass:
      m = self._get_species_attr("m")
      r = m["Arp"]/m["em"]
    else:
      r = 1.0
    # Argon neutral/ion
    for k in ("Ar", "Arp"):
      sk = self.mix.species[k]
      zk = r*ze if (k == "Arp") else 1.0-(1.0+r)*ze
      z[sk.indices] = zk * sk.q / sk.Q
    # Get molar fractions
    x = self.mix.get_x_from_w(z) if by_mass else z
    # Conservation of elements
    x = torch.clip(x, const.XMIN, 1.0)
    x = self._enforce_elem_cons(x)
    # Add noise
    if noise:
      x = self._add_noise(x, sigma)
      x = self._enforce_elem_cons(x)
    # Update composition
    self.mix.update_composition_x(x)

  def _enforce_elem_cons(
    self,
    x: torch.Tensor
  ) -> torch.Tensor:
    """
    Enforce elements conservation and renormalize electron and argon species.

    :param x: Raw molar composition vector.
    :type x: torch.Tensor

    :return: Renormalized and clipped molar composition.
    :rtype: torch.Tensor
    """
    # Electron
    sk = self.mix.species["em"]
    xe = torch.clip(x[sk.indices], const.XMIN, 0.5)
    x[sk.indices] = xe
    # Argon neutral/ion
    for k in ("Ar", "Arp"):
      sk = self.mix.species[k]
      xk = xe if (k == "Arp") else 1.0-2.0*xe
      xi = x[sk.indices]
      x[sk.indices] = xk * xi / torch.sum(xi)
    return x

  def _add_noise(
    self,
    x: torch.Tensor,
    sigma: float = 1e-2
  ) -> torch.Tensor:
    """
    Add multiplicative random noise to a vector.

    :param x: Vector to perturb (e.g., composition).
    :type x: torch.Tensor
    :param sigma: Noise strength (as standard deviation).
    :type sigma: float

    :return: Noisy vector.
    :rtype: torch.Tensor
    """
    f = 1.0 + sigma * torch.rand(x.shape)
    return f * x

  def _detailed_balance(self) -> torch.Tensor:
    r"""
    This method calculates the detailed balance error for the reaction:
    \[
    \text{Ar} \leftrightarrow \text{Ar}^+ + e^-
    \]

    :return: A tensor representing the deviation from detailed balance.
             A value close to zero indicates equilibrium.
    :rtype: torch.Tensor
    """
    n, Q = [self._get_species_attr(k) for k in ("n", "Q")]
    l = torch.sum(n["Arp"]) * n["em"] / torch.sum(n["Ar"])
    r = Q["Arp"] * Q["em"] / Q["Ar"]
    f = l/r - 1.0
    return f.reshape(1)

  def _get_species_attr(
    self,
    attr: str
  ) -> Dict[str, torch.Tensor]:
    """
    Retrieve a species-level attribute across the mixture.

    :param attr: Attribute name (e.g., 'n', 'Q', 'm').
    :type attr: str

    :return: Mapping from species names to attribute tensors.
    :rtype: Dict[str, torch.Tensor]
    """
    return {k: getattr(s, attr) for (k, s) in self.mix.species.items()}

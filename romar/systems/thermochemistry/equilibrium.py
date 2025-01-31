import torch
import numpy as np
import scipy as sp

from ... import const
from .mixture import Mixture
from .species import Species
from ... import backend as bkd
from typing import Dict, Tuple

MU_VARS = ("rho", "T", "Te")


class Equilibrium(object):

  """
  Class to compute the equilibrium state (mass fractions and temperature)
  of a system involving argon and its ionized species (Ar, Ar+, and e^-).

  The equilibrium state is determined by solving the following system of
  equations:
  1) **Charge neutrality**:
    \[
    x_{e^-} = x_{\text{Ar}^+}
    \]
    where \( x_{e^-} \) is the electron molar fraction and
    \( x_{\text{Ar}^+} \) is the argon ion molar fraction.
  2) **Mole conservation**:
    \[
    x_{e^-} + x_{\text{Ar}^+} + x_{\text{Ar}} = 1
    \]
    where \( x_{\text{Ar}} \) is the molar fraction of neutral argon.
  3) **Detailed balance**:
    \[
    \frac{n_{\text{Ar}^+} n_{e^-}}{n_{\text{Ar}}} =
    \frac{Q_{\text{Ar}^+} Q_{e^-}}{Q_{\text{Ar}}}
    \]
    This describes the ionization equilibrium between neutral argon, argon
    ions, and electrons in the system, where \( n \) represents the number
    density and \( Q \) represents the charge.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mixture: Mixture
  ) -> None:
    """
    Initializes the equilibrium solver for a given mixture.

    :param mixture: The `Mixture` object representing the chemical mixture.
    :type mixture: `Mixture`
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
    Set up functions and their Jacobians for the least-squares optimization
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
    sigma: float = 1e-1,
  ) -> Tuple[np.ndarray, float]:
    """
    Generates an initial solution for heat bath simulations based on the input
    parameters `mu`.

    This method computes the equilibrium state of the system, including partial
    densities, translational temperature, and electron pressure. It optionally
    adds random noise to the equilibrium state and replaces the equilibrium
    initial electron temperature with the translational temperature for heat
    bath simulations.

    :param mu: A NumPy array containing density, translational temperature,
               and electron temperature.
    :type mu: np.ndarray
    :param noise: Flag indicating whether to add random noise to the
                  equilibrium state.
    :type noise: bool, optional, default is False
    :param sigma: Standard deviation for the noise (if `noise` is True).
    :type sigma: float, optional, default is 1e-2

    :return: The equilibrium state vector, including partial densities,
             translational temperature, and electron pressure.
    :rtype: np.ndarray
    """
    # Unpack the input array into individual parameters
    rho, T, Te = mu
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
        T=bkd.to_torch(Te).reshape(1),
        ze=self.mix.species["em"].x
      )
    # Replace the equilibrium temperature Te with T for heat bath simulation
    y[-2] = T
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

    :param rho: Density of the system.
    :type rho: float
    :param T: Temperature of the system.
    :type T: float

    :return: The equilibrium state vector, including partial densities,
             translational temperature, and electron pressure.
    :rtype: np.ndarray
    """
    # Convert to 'torch.Tensor'
    rho, T = [bkd.to_torch(z).reshape(1) for z in (rho, T)]
    # Update mixture
    self.mix.set_rho(rho)
    self.mix.update_species_thermo(T)
    # Compute electron mass fraction
    we = self._we_from_prim(rho)
    # Compose state vector
    y = self._compose_state_vector(T, we, by_mass=True)
    return bkd.to_numpy(y), float(rho)

  def _we_from_prim(
    self,
    rho: float
  ) -> float:
    """
    Compute the residuals of the detailed balance condition based on the
    electron molar fraction.

    This method takes the electron molar fraction (as the logarithm of
    the fraction), updates the species composition based on the value,
    and enforces the detailed balance condition for the reaction:
    \[
    \text{Ar} \rightleftharpoons \text{Ar}^+ + e^-
    \]

    :param x: Logarithm of the electron molar fraction.
    :type x: torch.Tensor

    :return: The residuals of the detailed balance condition.
    :rtype: torch.Tensor
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
    Computes the equilibrium state from conservative macroscopic variables
    such as density and total energy.

    It determines the electron molar fraction (\(x_{\text{e}^-}\)) and
    temperature (\(T\)) that satisfy the conservation equations and
    detailed balance conditions.

    :param rho: Mass density of the system.
    :type rho: float
    :param e: Total energy of the system (per unit volume).
    :type e: float

    :return: The equilibrium state vector, including partial densities,
             translational temperature, and electron pressure.
    :rtype: np.ndarray
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
    y = self._compose_state_vector(T, xe)
    return bkd.to_numpy(y), float(rho)

  def _from_cons_fun(
    self,
    x: torch.Tensor,
    e: torch.Tensor
  ) -> torch.Tensor:
    """
    This method compute the residuals of two key constraints:
    1. **Detailed Balance**: Ensures equilibrium for the reaction
       \(\text{Ar} \leftrightarrow \text{Ar}^+ + e^-\).
    2. **Energy Conservation**: Ensures the total energy of the system matches
       the specified input energy `e`.

    :param x: Tensor containing logarithmic values of the electron molar
              fraction and temperature \([ \ln(x_{\text{e}^-}), \ln(T) ]\).
    :type x: torch.Tensor
    :param e: Total energy of the system (per unit volume).
    :type e: torch.Tensor

    :return: Tensor of residuals from the conservation equations. The first
             component corresponds to the detailed balance equation, and the
             second component to energy conservation.
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
    T: torch.Tensor,
    ze: torch.Tensor,
    by_mass: bool = False
  ) -> torch.Tensor:
    """
    Composes the state vector for the system's equilibrium state.

    This method combines partial densities, translational temperature,
    and electron pressure into a single vector. It updates the species
    composition based on the electron molar fraction and computes
    electron pressure and other quantities.

    :param rho: Density of the system.
    :type rho: torch.Tensor
    :param T: Translational temperature of the system.
    :type T: torch.Tensor
    :param xe: Electron molar fraction of the species.
    :type xe: torch.Tensor

    :return: Equilibrium state vector, including partial densities,
             translational temperature, and electron pressure.
    :rtype: torch.Tensor
    """
    self._update_composition(ze, by_mass=by_mass)
    pe = self.mix.get_pe(Te=T, ne=self.mix.species["em"].n)
    w = self.mix.get_qoi_vec("w")
    return torch.cat([w, T, pe])

  def _update_composition(
    self,
    ze: torch.Tensor,
    noise: bool = False,
    sigma: float = 1e-2,
    by_mass: bool = False,
    clipping: bool = True
  ) -> None:
    """
    Updates the species composition based on the electron molar fraction
    and optionally adds random noise to the composition.

    This method modifies the composition vector using the electron molar
    fraction `xe`, applying noise if specified, and updates the species
    states for various elements like "Ar" and "Arp".

    :param xe: Electron molar fraction of the species.
    :type xe: torch.Tensor
    :param noise: Whether to add random noise to the composition.
    :type noise: bool, optional, default is False
    :param sigma: Standard deviation of the noise if `noise` is True.
    :type sigma: float, optional, default is 1e-2
    """
    # Vector of molar/mass fractions
    z = torch.zeros(self.mix.nb_comp)
    # Electron
    s = self.mix.species["em"]
    if noise:
      ze *= self._add_norm_noise(s, sigma, use_pf=False)
    z[s.indices] = ze
    # Compute coefficient
    if by_mass:
      m = self._get_species_attr("m")
      r = m["Arp"]/m["em"]
    else:
      r = 1.0
    # Argon neutral/ion
    for k in ("Ar", "Arp"):
      sk = self.mix.species[k]
      xk = r*ze if (k == "Arp") else 1.0-(1.0+r)*ze
      fk = self._add_norm_noise(sk, sigma) if noise else sk.q / sk.Q
      fk = (fk+const.XMIN) / torch.sum(fk+const.XMIN)
      z[sk.indices] = xk * fk
    # Get molar fractions
    x = self.mix.get_x_from_w(z) if by_mass else z
    # Update composition
    self.mix.update_composition_x(x)
    if clipping:
      xe = self.mix.species["em"].x
      xe = torch.clip(xe, const.XMIN, 0.5)
      self._update_composition(xe, clipping=False)

  def _add_norm_noise(
    self,
    species: Species,
    sigma: float = 1e-2,
    use_pf: bool = True
  ) -> torch.Tensor:
    """
    Adds unit norm random noise to the species composition.

    The noise is generated according to the specified standard deviation
    `sigma`, and optionally scaled by the partition function `q` for the
    species.

    :param species: The species for which the noise is applied.
    :type species: `Species`
    :param sigma: Standard deviation of the noise to be added.
    :type sigma: float
    :param use_pf: Whether to scale the noise by the partition function `q`.
    :type use_pf: bool, optional, default is True

    :return: A tensor containing the noisy composition values.
    :rtype: torch.Tensor
    """
    f = 1.0 + sigma * torch.rand(species.nb_comp)
    if use_pf:
      f *= species.q
      f /= torch.sum(f)
    return f

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

  def _get_species_attr(self, attr: str) -> Dict[str, torch.Tensor]:
    """
    Retrieve the specified attribute for all species in the mixture.

    This method fetches the requested attribute (e.g., number density,
    partition function) for each species in the mixture and returns a
    dictionary mapping species names to their corresponding attribute values.

    :param attr: The name of the attribute to retrieve (e.g., "n" for
                 number density, "Q" for partition function).
    :type attr: str

    :return: A dictionary where the keys are species names and the values are
             the corresponding attribute tensors.
    :rtype: Dict[str, torch.Tensor]
    """
    return {k: getattr(s, attr) for (k, s) in self.mix.species.items()}

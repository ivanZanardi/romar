import torch
import numpy as np
import scipy as sp

from ... import const
from ... import backend as bkd
from .mixture import Mixture
from typing import Dict, Optional


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
    solve: callable,
    mixture: Mixture,
    clipping: bool = True
  ) -> None:
    """
    Initialize the equilibrium solver with a given system and
    optional clipping.

    :param solve: The callable used to solve the full system of equations.
    :type solve: callable
    :param mixture: The `Mixture` object representing the chemical mixture.
    :type mixture: `Mixture`
    :param clipping: Flag to control whether molar fractions are
                     clipped to avoid numerical issues.
    :type clipping: bool, optional, default is True
    """
    self.solve = solve
    self.mix = mixture
    self.clipping = clipping
    self.lsq_opts = dict(
      method="trf",
      ftol=1e-08,
      xtol=1e-08,
      gtol=0.0,
      max_nfev=int(1e5)
    )
    self.set_fun_jac()

  def set_fun_jac(self):
    """
    Set up functions and their Jacobians for the least-squares optimization
    for both the primitive (`from_prim`) and conservative (`from_cons`)
    variable formulations.
    """
    for name in ("from_prim", "from_cons"):
      # Function
      fun = getattr(self, f"_{name}_fun")
      setattr(self, f"{name}_fun", bkd.make_fun_np(fun))
      # Jacobian
      jac = torch.func.jacrev(fun, argnums=0)
      setattr(self, f"{name}_jac", bkd.make_fun_np(jac))

  # Primitive variables
  # ===================================
  def from_prim(
    self,
    rho: float,
    T: float,
    Te: Optional[float] = None
  ) -> np.ndarray:
    """
    Compute the equilibrium state (mass fractions and temperature) from
    primitive macroscopic variables, such as density, temperature, and
    electron temperature.

    If the electron temperature (`Te`) is not provided, the assumption
    is made that the system is in thermal equilibrium (\(T_e = T\)).

    :param rho: Density of the system.
    :type rho: float
    :param T: Temperature of the system.
    :type T: float
    :param Te: Electron temperature (if not provided, assumes
               thermal equilibrium).
    :type Te: Optional[float], optional
    :return: The equilibrium state vector, including mass fractions
             and temperatures.
    :rtype: np.ndarray
    """
    # Setting up
    if (Te is None):
      # Thermal equilibrium assumption
      solve_full_sys = False
      Te = T
    else:
      # Thermal nonequilibrium assumption
      solve_full_sys = True
    # Convert to 'torch.Tensor'
    rho, T, Te = [bkd.to_torch(z).reshape(1) for z in (rho, T, Te)]
    # Update mixture
    self.mix.set_rho(rho)
    self.mix.update_species_thermo(T, Te)
    # Compute electron molar fraction
    x = sp.optimize.least_squares(
      fun=self.from_prim_fun,
      x0=np.log([1e-2]),
      jac=self.from_prim_jac,
      bounds=(-np.inf, 0.0),
      **self.lsq_opts
    ).x
    # Extract variables
    x_em = bkd.to_torch(np.exp(x))
    x_em = self._clipping(x_em)
    # Update composition
    self._update_composition(x_em)
    # Compose state vector
    w = self.mix.get_qoi_vec("w")
    y = bkd.to_numpy(torch.cat([w, T, Te]))
    if solve_full_sys:
      # Solve the full system to determine the equilibrium state
      # under thermal nonequilibrium conditions.
      y = self.solve(t=[1e3], y0=y, rho=rho)[0].squeeze()
    return y

  def _from_prim_fun(self, x: torch.Tensor) -> torch.Tensor:
    """
    Enforce detailed balance based on the electron molar fraction.

    This method takes the electron molar fraction (as the logarithm of
    the fraction), updates the species composition based on the value,
    and enforces the detailed balance condition for the equilibrium reaction:
    \[
    \text{Ar} \rightleftharpoons \text{Ar}^+ + e^-
    \]

    :param x: Logarithm of the electron molar fraction.
    :type x: torch.Tensor

    :return: The residuals of the detailed balance condition.
    :rtype: torch.Tensor
    """
    # Extract variables
    x_em = torch.exp(x)
    # Update composition
    self._update_composition(x_em)
    # Enforce detailed balance
    return self._detailed_balance()

  # Conservative variables
  # ===================================
  def from_cons(
    self,
    rho: float,
    e: float
  ) -> np.ndarray:
    """
    Compute the equilibrium state from conservative macroscopic variables.

    This method calculates the equilibrium state based on the provided density
    (\(\rho\)) and total energy (\(e\)). It determines the electron molar
    fraction (\(x_{\text{e}^-}\)) and temperature (\(T\)) that satisfy the
    conservation equations and detailed balance conditions.

    :param rho: Mass density of the system.
    :type rho: float
    :param e: Total energy of the system (per unit volume).
    :type e: float

    :return: State vector containing the mass fractions of all species,
             the equilibrium temperature \(T\), and the electron temperature
             \(T_{\text{e}}\) (equal to \(T\) for equilibrium).
    :rtype: np.ndarray
    """
    # Convert to 'torch.Tensor'
    rho, e = [bkd.to_torch(z) for z in (rho, e)]
    # Update mixture
    self.mix.set_rho(rho)
    # Compute electron molar fraction and temperaure
    x = sp.optimize.least_squares(
      fun=self.from_cons_fun,
      x0=np.log([1e-1,1e4]),
      jac=self.from_cons_jac,
      bounds=([-np.inf, -np.inf], [0.0, np.log(1e5)]),
      args=(e,),
      **self.lsq_opts
    ).x
    # Extract variables
    x_em, T = [z.reshape(1) for z in bkd.to_torch(np.exp(x))]
    x_em = self._clipping(x_em)
    # Update species thermo
    self.mix.update_species_thermo(T)
    # Update composition
    self._update_composition(x_em)
    # Compose state vector
    w = self.mix.get_qoi_vec("w")
    y = bkd.to_numpy(torch.cat([w, T, T]))
    return y

  def _from_cons_fun(
    self,
    x: torch.Tensor,
    e: torch.Tensor
  ) -> torch.Tensor:
    """
    Compute the residuals of the conservation equations for equilibrium.

    This method evaluates two key constraints:
    1. **Detailed Balance**: Ensures equilibrium between the reaction
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
    x_em, T = torch.exp(x)
    # Update species thermo
    self.mix.update_species_thermo(T)
    # Update composition
    self._update_composition(x_em)
    # Update mixture thermo
    self.mix.update_mixture_thermo()
    # Enforce detailed balance
    f0 = self._detailed_balance()
    # Enforce conservation of energy
    f1 = self.mix.e / e - 1.0
    return torch.cat([f0,f1])

  # Utils
  # ===================================
  def _update_composition(self, x_em: torch.Tensor) -> None:
    """
    Update the species composition based on the electron molar fraction.

    This method sets the number densities of the species using conservation
    of charge (eq. 1) and mass (eq. 2).

    :param x_em: Electron molar fraction, representing the proportion of
                 electrons in the mixture.
    :type x_em: torch.Tensor
    """
    x = torch.zeros(self.mix.nb_comp)
    # Electron
    s = self.mix.species["em"]
    x[s.indices] = x_em
    # Argon ion
    s = self.mix.species["Arp"]
    x[s.indices] = x_em * s.q / s.Q
    # Argon neutral
    s = self.mix.species["Ar"]
    x[s.indices] = (1.0-2.0*x_em) * s.q / s.Q
    # Update composition
    self.mix.update_composition_x(x)

  def _detailed_balance(self) -> torch.Tensor:
    r"""
    Enforce detailed balance for the reaction.

    This method calculates the detailed balance error for the reaction
    equilibrium:
    \[
    \text{Ar} \leftrightarrow \text{Ar}^+ + e^-
    \]

    It compares the ratio of number densities and partition functions for the
    involved species and returns the deviation.

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

  def _clipping(self, x: torch.Tensor) -> torch.Tensor:
    """
    Clip molar fractions to avoid values that are too small, ensuring
    numerical stability.

    This method ensures that the molar fractions remain within a valid range.
    If clipping is enabled, the values are constrained between a predefined
    minimum (`const.XMIN`) and a maximum of 1.0. If clipping is disabled,
    the input tensor is returned unchanged.

    :param x: Tensor of molar fractions to be clipped.
    :type x: torch.Tensor

    :return: Tensor of molar fractions after applying clipping (if enabled).
    :rtype: torch.Tensor
    """
    if self.clipping:
      return torch.clip(x, const.XMIN, 1.0)
    else:
      return x

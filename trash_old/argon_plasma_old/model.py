import numpy as np
import scipy as sp

from .. import const
from .mixture import Mixture
from .kinetics import Kinetics
from typing import Dict, Optional


class Model(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species: Dict[str, str],
    kin_dtb: str,
    rad_dtb: Optional[str] = None,
    use_rad: bool = False,
    use_proj: bool = False,
    use_einsum: bool = False,
    use_factorial: bool = False,
    use_coll_int_fit: bool = False
  ) -> None:
    # Thermochemistry database
    # -------------
    # Mixture
    self.mix = Mixture(
      species,
      species_order=("Ar", "Arp", "em"),
      use_factorial=use_factorial
    )
    self.mix.build()
    self.nb_eqs = self.mix.nb_comp + 2
    # Kinetics
    self.kin = Kinetics(
      mixture=self.mix,
      reactions=kin_dtb,
      use_fit=use_coll_int_fit
    )
    # Radiation
    self.use_rad = use_rad
    self.rad = rad_dtb
    # FOM
    # -------------
    # Solving
    self.use_einsum = use_einsum
    self.fun = None
    self.jac = None
    # ROM
    # -------------
    self.use_proj = use_proj
    # Basis
    self.phi = None
    self.psi = None
    self.runtime = 0.0

  def _is_einsum_used(self, identifier: str) -> None:
    if self.use_einsum:
      raise NotImplementedError(
        "This functionality is not supported " \
        f"when using 'einsum': '{identifier}'."
      )

  # ROM
  # ===================================
  def set_basis(
    self,
    phi: np.ndarray,
    psi: np.ndarray
  ) -> None:
    self.phi, self.psi = phi, psi
    # Biorthogonalize
    self.phi = self.phi @ sp.linalg.inv(self.psi.T @ self.phi)
    # Projector
    self.P = self.phi @ self.psi.T

  # FOM
  # ===================================
  def _fun(self, t, y, rho):
    # Extract variables
    w, e, e_e = self._extract_vars(y)
    # Update composition
    self.mix.update_composition(w, rho)
    # Compute temperatures
    T, Te = self.mix.compute_temp(e, e_e)
    # Update thermo
    self.mix.update_species_thermo(T, Te)
    # Kinetics
    # -------------
    kops = self._compose_kin_ops(T, Te)
    omega_exc = self._compute_kin_omega_exc(kops)
    omega_ion = self._compute_kin_omega_ion(kops)
    # Compose RHS
    # -------------
    f = np.zeros(self.nb_eqs)
    # > Argon
    i = self.mix.species["Ar"].indices
    f[i] = omega_exc - np.sum(omega_ion, axis=1)
    # > Argon Ion
    i = self.mix.species["Arp"].indices
    fion = np.sum(omega_ion, axis=0)
    f[i] = fion
    # > Electron mass
    i = self.mix.species["em"].indices
    f[i] = np.sum(fion)
    # > Electron energy
    f[-1] = self._omega_ee(kops, T, Te)
    # Convert RHS
    # -------------
    # Number densities to mass fractions: [1/(m^3 s)] -> [1/s]
    f[:self.mix.nb_comp] = self.mix.get_w(f[:self.mix.nb_comp], rho)
    # Volumetric to massic energy: [J/(m^3 s)] -> [J/(kg s)]
    f[-2:] /= rho
    return f

  def _extract_vars(self, y):
    # Mass fractions
    w = y[:self.mix.nb_comp]
    # Total and electron energies
    e, e_e = y[self.mix.nb_comp:]
    return w, e, e_e

  # Kinetic operators
  # -----------------------------------
  def _compose_kin_ops(self, T, Te):
    """Compose kinetic operators"""
    # Update kinetics
    self.kin.update(T, Te)
    # Compose operators
    ops = {}
    # > Excitation processes
    for k in ("EXh", "EXe"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_kin_ops_exc(rates)
      if (k == "EXe"):
        ops[k+"_e"] = self._compose_kin_ops_exc(rates, apply_energy=True)
    # > Ionization processes
    for k in ("Ih", "Ie"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_kin_ops_ion(rates)
      if (k == "Ie"):
        ops[k+"_e"] = self._compose_kin_ops_ion(rates, apply_energy=True)
    return ops

  def _compose_kin_ops_exc(self, rates, apply_energy=False):
    k = rates["fwd"] + rates["bwd"]
    k = (k - np.diag(np.sum(k, axis=-1))).T
    if apply_energy:
      k *= self.mix.de["nn"]
    return k

  def _compose_kin_ops_ion(self, rates, apply_energy=False):
    k = {d: rates[d].T for d in ("fwd", "bwd")}
    if apply_energy:
      k["fwd"] *= self.mix.de["in"].T
      k["bwd"] *= self.mix.de["in"]
    return k

  # Kinetic production terms
  # -----------------------------------
  def _compute_kin_omega_exc(self, kops):
    nn, ne = [self.mix.species[k].n for k in ("Ar", "em")]
    return (kops["EXh"] * nn[0] + kops["EXe"] * ne) @ nn

  def _compute_kin_omega_ion(self, kops):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    ops_fwd = (kops["Ih"]["fwd"] * nn[0] + kops["Ie"]["fwd"] * ne) * nn
    ops_bwd = (kops["Ih"]["bwd"] * nn[0] + kops["Ie"]["bwd"] * ne) * ni * ne
    return ops_fwd.T - ops_bwd

  # Thermal nonequilibrium
  # -----------------------------------
  def _omega_ee(self, kops, T, Te):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    f = np.sum(kops["EXe_e"] @ nn) \
      - np.sum(kops["Ie_e"]["fwd"] * nn) \
      + np.sum(kops["Ie_e"]["bwd"] * ni * ne) \
      + 1.5 * const.UKB * (T-Te) * self._get_nu_eh()
    return f * ne

  def _get_nu_eh(self):
    """Electron-heavy particle relaxation frequency [1/s]"""
    s = self.mix.species
    nu = self.kin.rates["EN"] * np.sum(s["Ar"].n) / s["Ar"].m \
       + self.kin.rates["EI"] * np.sum(s["Arp"].n) / s["Arp"].m
    return const.UME * nu

  # Solving
  # ===================================
  def solve(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float
  ) -> np.ndarray:
    sol = sp.integrate.solve_ivp(
      fun=self._fun,
      t_span=[0.0,t[-1]],
      y0=y0,
      method="LSODA",
      t_eval=t,
      args=(rho,),
      first_step=1e-14,
      rtol=1e-6,
      atol=1e-15,
      jac=None
    )
    return sol.y

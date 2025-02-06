import torch

from ... import const
from .mixture import Mixture
from .kinetics import Kinetics
from .radiation import Radiation
from typing import Optional


class Sources(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    mixture: Mixture,
    kinetics: Kinetics,
    radiation: Optional[Radiation] = None
  ) -> None:
    self.mix = mixture
    self.kin = kinetics
    self.rad = radiation
    self.kin_ops = None
    self.rad_ops = None

  # Calling
  # ===================================
  # Adiabatic case
  # -----------------------------------
  def call_ad(self, n, Th, Te):
    # Mixture
    self.mix.update(n, Th, Te)
    # Kinetics/Radiation operators
    kin_ops = self.compose_kin_ops(Th, Te)
    rad_ops = self.compose_rad_ops(Th, Te) if self.rad.active else None
    # Partial densities [kg/(m^3 s)]
    f_rho = self.omega_mass(kin_ops, rad_ops)
    # Energies [J/(kg s)]
    f_et, f_ee = self.omega_energy(Th, Te, kin_ops, rad_ops)
    # Return
    return f_rho, f_et, f_ee

  # Isothermal case
  # -----------------------------------
  def init_iso(self, Th, Te):
    # Mixture
    self.mix.update_species_thermo(Th, Te)
    # Kinetics
    self.kin_ops = self.compose_kin_ops(Th, Te, isothermal=True)
    # Radiation
    if self.rad.active:
      self.rad_ops = self.compose_rad_ops(Th, Te, isothermal=True)

  def call_iso(self, n):
    # Mixture
    self.mix.update_composition(n)
    # Partial densities [kg/(m^3 s)]
    return self.omega_mass(self.kin_ops, self.rad_ops)

  # Kinetics/Radiation
  # ===================================
  def compose_kin_ops(self, Th, Te, isothermal=False):
    """Compose kinetics operators"""
    # Rates
    self.kin.update(Th, Te, isothermal)
    # Operators
    ops = {}
    # Excitation processes
    for k in ("EXh", "EXe"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_ops_exc(rates)
      if (k == "EXe"):
        ops[k+"_e"] = self._compose_ops_exc(rates, apply_energy=True)
    # Ionization processes
    for k in ("Ih", "Ie"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_ops_ion(rates)
      if (k == "Ie"):
        ops[k+"_e"] = self._compose_ops_ion(rates, apply_energy=True)
    return ops

  def compose_rad_ops(self, Th, Te, isothermal=False):
    """Compose radiation operators"""
    # Rates
    self.rad.update(Th, Te, isothermal)
    # Operators
    ops = {}
    # Excitation processes
    for k in ("BB",):
      if (k in self.rad.rates):
        rates = self.rad.rates[k]
        ops[k] = self._compose_ops_exc(rates)
        ops[k+"_e"] = self._compose_ops_exc(rates, apply_energy=True)
    # Ionization processes
    for k in ("BF", "BFp"):
      if (k in self.rad.rates):
        rates = self.rad.rates[k]
        ops[k] = self._compose_ops_ion(rates)
    return ops

  def _compose_ops_exc(self, rates, apply_energy=False):
    k = rates["fwd"] + rates["bwd"]
    k = (k - torch.diag(torch.sum(k, dim=-1))).T
    if apply_energy:
      k = k * self.mix.de["Ar-Ar"]
    return k

  def _compose_ops_ion(self, rates, apply_energy=False):
    k = {d: rates[d].T for d in ("fwd", "bwd")}
    if apply_energy:
      k["fwd"] = k["fwd"] * self.mix.de["Arp-Ar"].T
      k["bwd"] = k["bwd"] * self.mix.de["Arp-Ar"]
    return k

  # Masses
  # ===================================
  def omega_mass(self, kin_ops, rad_ops):
    # Production terms
    omega_exc = self.omega_exc(kin_ops, rad_ops)
    omega_ion = self.omega_ion(kin_ops, rad_ops)
    # Argon nd
    f_nn = omega_exc - torch.sum(omega_ion, dim=1)
    # Argon ion nd
    f_ni = torch.sum(omega_ion, dim=0)
    # Electron nd
    f_ne = torch.sum(omega_ion).reshape(1)
    # Concatenate
    f_n = torch.cat([f_nn, f_ni, f_ne])
    # Convert
    return self.mix.get_rho(f_n)

  def omega_exc(self, kin_ops, rad_ops):
    nn, ne = [self.mix.species[k].n for k in ("Ar", "em")]
    omega = kin_ops["EXh"] * nn[0] + kin_ops["EXe"] * ne
    if self.rad.active:
      omega += rad_ops["BB"]
    return omega @ nn

  def omega_ion(self, kin_ops, rad_ops):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    omega = {}
    for k in ("fwd", "bwd"):
      omega[k] = kin_ops["Ih"][k] * nn[0] + kin_ops["Ie"][k] * ne
      if self.rad.active:
        omega[k] += rad_ops["BF"][k]
      omega[k] *= nn if (k == "fwd") else (ni * ne)
    return omega["fwd"].T - omega["bwd"]

  # Energies
  # ===================================
  def omega_energy(self, Th, Te, kin_ops, rad_ops):
    omegas = {}
    omegas = self.omega_kin(omegas, Th, Te, kin_ops)
    omegas = self.omega_rad(omegas, rad_ops)
    f_et = self.omega_energy_t(omegas)
    f_ee = self.omega_energy_e(omegas)
    return f_et, f_ee

  def omega_energy_t(self, omegas):
    f = torch.zeros(1)
    for k in ("rad_bb",): # "rad_bf"):
      f += omegas[k]
    return f

  def omega_energy_e(self, omegas):
    f = torch.zeros(1)
    for k in ("kin_exc", "kin_ion", "kin_ela", "rad_ff", "rad_bf"):
      f += omegas[k]
    return f

  # Kinetics
  # -----------------------------------
  def omega_kin(self, omegas, Th, Te, kin_ops):
    omegas["kin_ela"] = self._omega_kin_ela(Th, Te)
    omegas["kin_exc"] = self._omega_kin_exc(kin_ops)
    omegas["kin_ion"] = self._omega_kin_ion(kin_ops)
    return omegas

  def _omega_kin_ela(self, Th, Te):
    """Elastic collisions"""
    sn, si, se = [self.mix.species[k] for k in ("Ar", "Arp", "em")]
    # Electron-heavy particle relaxation frequency [1/s]
    nu = const.UME * self.kin.rates["EN"] * torch.sum(sn.n) / sn.m \
       + const.UME * self.kin.rates["EI"] * torch.sum(si.n) / si.m
    return 1.5 * const.UKB * (Th-Te) * se.n * nu

  def _omega_kin_exc(self, kin_ops):
    nn, ne = [self.mix.species[k].n for k in ("Ar", "em")]
    return torch.sum(kin_ops["EXe_e"] @ nn) * ne

  def _omega_kin_ion(self, kin_ops):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    return torch.sum(kin_ops["Ie_e"]["bwd"] * ni * ne) * ne \
         - torch.sum(kin_ops["Ie_e"]["fwd"] * nn) * ne

  # Radiation
  # -----------------------------------
  def omega_rad(self, omegas, rad_ops):
    if self.rad.active:
      omegas["rad_bb"] = self._omega_rad_bb(rad_ops)
      omegas["rad_bf"] = self._omega_rad_bf(rad_ops)
      omegas["rad_ff"] = self._omega_rad_ff()
    else:
      for k in ("rad_bb", "rad_bf", "rad_ff"):
        omegas[k] = torch.zeros(1)
    return omegas

  def _omega_rad_bb(self, rad_ops):
    nn = self.mix.species["Ar"].n
    return - torch.sum(rad_ops["BB_e"] @ nn)

  def _omega_rad_bf(self, rad_ops):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    return torch.sum(rad_ops["BFp"]["fwd"] * nn) \
         - torch.sum(rad_ops["BFp"]["bwd"] * ni * ne)

  def _omega_rad_ff(self):
    ni, ne = [self.mix.species[k].n for k in ("Arp", "em")]
    return - self.rad.rates["FF"] * torch.sum(ni) * ne

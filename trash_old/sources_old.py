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
  def call_ad(self, n, T, Te):
    # Mixture
    # -------------
    self.mix.update(n, T, Te)
    # Operators
    # -------------
    # > Kinetics
    kin_ops = self.compose_kin_ops(T, Te)
    # > Radiation
    rad_ops = self.compose_rad_ops(T, Te) if self.rad.active else None
    # Partial densities [kg/(m^3 s)]
    # -------------
    # > Production terms
    omega_exc = self.omega_exc(kin_ops, rad_ops)
    omega_ion = self.omega_ion(kin_ops, rad_ops)
    # > Argon nd
    f_nn = omega_exc - torch.sum(omega_ion, dim=1)
    # > Argon ion nd
    f_ni = torch.sum(omega_ion, dim=0)
    # > Electron nd
    f_ne = torch.sum(omega_ion).reshape(1)
    # > Concatenate
    f_n = torch.cat([f_nn, f_ni, f_ne])
    # > Convert
    f_rho = self.mix.get_rho(f_n)
    # Energies [J/(kg s)]
    # -------------
    # > Production terms
    omega_eh = self.omega_eh(T, Te)
    # # > Total energy
    # f_et = self.omega_energy(rad_ops)
    # > Heavy-particle energy
    f_eh = self.omega_energy_h(omega_eh, kin_ops)
    # > Electron energy
    f_ee = self.omega_energy_e(omega_eh, kin_ops, rad_ops)
    # Return
    # -------------
    return f_rho, f_eh, f_ee

  # Isothermal case
  # -----------------------------------
  def init_iso(self, T, Te):
    # Mixture
    self.mix.update_species_thermo(T, Te)
    # Kinetics
    self.kin_ops = self.compose_kin_ops(T, Te, isothermal=True)
    # Radiation
    if self.rad.active:
      self.rad_ops = self.compose_rad_ops(T, Te, isothermal=True)

  def call_iso(self, n):
    # Mixture
    # -------------
    self.mix.update_composition(n)
    # Partial densities [kg/(m^3 s)]
    # -------------
    # > Production terms
    omega_exc = self.omega_exc(self.kin_ops, self.rad_ops)
    omega_ion = self.omega_ion(self.kin_ops, self.rad_ops)
    # > Argon nd
    f_nn = omega_exc - torch.sum(omega_ion, dim=1)
    # > Argon ion nd
    f_ni = torch.sum(omega_ion, dim=0)
    # > Electron nd
    f_ne = torch.sum(omega_ion).reshape(1)
    # > Concatenate
    f_n = torch.cat([f_nn, f_ni, f_ne])
    # > Convert
    f_rho = self.mix.get_rho(f_n)
    # Return
    # -------------
    return f_rho

  # Kinetics/Radiation
  # ===================================
  def compose_kin_ops(self, T, Te, isothermal=False):
    """Compose kinetics operators"""
    # Rates
    self.kin.update(T, Te, isothermal)
    # Operators
    ops = {}
    # > Excitation processes
    for k in ("EXh", "EXe"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_ops_exc(rates)
      ops[k+"_e"] = self._compose_ops_exc(rates, apply_energy=True)
    # > Ionization processes
    for k in ("Ih", "Ie"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_ops_ion(rates)
      ops[k+"_e"] = self._compose_ops_ion(rates, apply_energy=True)
    return ops

  def compose_rad_ops(self, T, Te, isothermal=False):
    """Compose radiation operators"""
    # Rates
    self.rad.update(T, Te, isothermal)
    # Operators
    ops = {}
    # > Excitation processes
    for k in ("BB",):
      rates = self.rad.rates[k]
      ops[k] = self._compose_ops_exc(rates)
      ops[k+"_e"] = self._compose_ops_exc(rates, apply_energy=True)
    # > Ionization processes
    for k in ("BF",):
      rates = self.rad.rates[k]
      ops[k] = self._compose_ops_ion(rates)
      ops[k+"_e"] = self._compose_ops_ion(rates, apply_energy=True)
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
  def omega_exc(self, kin_ops, rad_ops=None):
    nn, ne = [self.mix.species[k].n for k in ("Ar", "em")]
    omega = kin_ops["EXh"] * nn[0] + kin_ops["EXe"] * ne
    if (rad_ops is not None):
      omega += rad_ops["BB"]
    return omega @ nn

  def omega_ion(self, kin_ops, rad_ops=None):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    omega = {}
    for k in ("fwd", "bwd"):
      omega[k] = kin_ops["Ih"][k] * nn[0] + kin_ops["Ie"][k] * ne
      if (rad_ops is not None):
        omega[k] += rad_ops["BF"][k]
      omega[k] *= nn if (k == "fwd") else (ni * ne)
    return omega["fwd"].T - omega["bwd"]

  # Energies
  # ===================================
  # def omega_energy_t(self, rad_ops=None):
  #   if (rad_ops is not None):
  #     nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
  #     return -torch.sum(rad_ops["BB_e"] @ nn) \
  #        + torch.sum(rad_ops["BF_e"]["fwd"] * nn) \
  #        - torch.sum(rad_ops["BF_e"]["bwd"] * ni * ne) \
  #        - self.rad.rates["FF"] * torch.sum(ni) * ne
  #   else:
  #     return torch.zeros(1)

  def omega_energy_h(self, omega_eh, kin_ops):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    # Kinetics
    f = torch.sum(kin_ops["EXh_e"] @ nn) \
      - torch.sum(kin_ops["Ih_e"]["fwd"] * nn) \
      + torch.sum(kin_ops["Ih_e"]["bwd"] * ni * ne)
    f = f * ne - omega_eh
    return f.reshape(1)

  def omega_energy_e(self, omega_eh, kin_ops, rad_ops=None):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    # Kinetics
    f = torch.sum(kin_ops["EXe_e"] @ nn) \
      - torch.sum(kin_ops["Ie_e"]["fwd"] * nn) \
      + torch.sum(kin_ops["Ie_e"]["bwd"] * ni * ne)
    f = f * ne + omega_eh
    # Radiation
    if (rad_ops is not None):
      f += torch.sum(rad_ops["BF_e"]["fwd"] * nn) \
         - torch.sum(rad_ops["BF_e"]["bwd"] * ni * ne) \
         - self.rad.rates["FF"] * torch.sum(ni) * ne
    return f.reshape(1)

  def omega_eh(self, T, Te):
    """Elastic collisions"""
    ne = self.mix.species["em"].n
    return 1.5 * const.UKB * (T-Te) * ne * self._get_nu_eh()

  def _get_nu_eh(self):
    """Electron-heavy particle relaxation frequency [1/s]"""
    sn, si = [self.mix.species[k] for k in ("Ar", "Arp")]
    nu = self.kin.rates["EN"] * torch.sum(sn.n) / sn.m \
       + self.kin.rates["EI"] * torch.sum(si.n) / si.m
    return const.UME * nu

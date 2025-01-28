import torch

from ... import const
from .mixture import Mixture
from .kinetics import Kinetics


class Sources(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    mixture: Mixture,
    kinetics: Kinetics,
    radiation=None
  ):
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
    self.mix.update(n, T, Te)
    # Kinetics
    # > Operators
    kin_ops = self.compose_kin_ops(T, Te)
    # > Production terms
    omega_exc = self.omega_kin_exc(kin_ops)
    omega_ion = self.omega_kin_ion(kin_ops)
    # Partial densities [kg/(m^3 s)]
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
    # > Total energy
    f_et = self.omega_energy()
    # > Electron energy
    f_ee = self.omega_energy_el(T, Te, kin_ops)
    # Return
    return f_rho, f_et, f_ee

  # Isothermal case
  # -----------------------------------
  def init_iso(self, T, Te):
    # Mixture
    self.mix.update_species_thermo(T, Te)
    # Kinetics
    self.kin_ops = self.compose_kin_ops(T, Te, isothermal=True)

  def call_iso(self, n):
    # Mixture
    self.mix.update_composition(n)
    # Kinetics
    omega_exc = self.omega_kin_exc(self.kin_ops)
    omega_ion = self.omega_kin_ion(self.kin_ops)
    # Partial densities [kg/(m^3 s)]
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
    return f_rho

  # Kinetics
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
    k = (k - torch.diag(torch.sum(k, dim=-1))).T
    if apply_energy:
      k = k * self.mix.de["Ar-Ar"]
    return k

  def _compose_kin_ops_ion(self, rates, apply_energy=False):
    k = {d: rates[d].T for d in ("fwd", "bwd")}
    if apply_energy:
      k["fwd"] = k["fwd"] * self.mix.de["Arp-Ar"].T
      k["bwd"] = k["bwd"] * self.mix.de["Arp-Ar"]
    return k

  # Masses
  # ===================================
  def omega_kin_exc(self, kin_ops):
    nn, ne = [self.mix.species[k].n for k in ("Ar", "em")]
    return (kin_ops["EXh"] * nn[0] + kin_ops["EXe"] * ne) @ nn

  def omega_kin_ion(self, kin_ops):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    omega = {}
    for k in ("fwd", "bwd"):
      omega[k] = kin_ops["Ih"][k] * nn[0] + kin_ops["Ie"][k] * ne
      omega[k] *= nn if (k == "fwd") else (ni * ne)
    return omega["fwd"].T - omega["bwd"]

  # Energies
  # ===================================
  def omega_energy(self):
    return torch.zeros(1)

  def omega_energy_el(self, T, Te, kin_ops):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    f = torch.sum(kin_ops["EXe_e"] @ nn) \
      - torch.sum(kin_ops["Ie_e"]["fwd"] * nn) \
      + torch.sum(kin_ops["Ie_e"]["bwd"] * ni * ne) \
      + 1.5 * const.UKB * (T-Te) * self._get_nu_eh()
    f = f * ne
    return f.reshape(1)

  def _get_nu_eh(self):
    """Electron-heavy particle relaxation frequency [1/s]"""
    sn, si = [self.mix.species[k] for k in ("Ar", "Arp")]
    nu = self.kin.rates["EN"] * torch.sum(sn.n) / sn.m \
       + self.kin.rates["EI"] * torch.sum(si.n) / si.m
    return const.UME * nu

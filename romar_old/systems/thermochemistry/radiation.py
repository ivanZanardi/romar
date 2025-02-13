import torch
import numpy as np
import dill as pickle

from ... import const
from ... import utils
from ... import backend as bkd


class Radiation(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    processes,
    use_tables=False,
    active=True
  ):
    # Collision integrals look-up tables
    self.use_tables = use_tables
    # Initialize processes rates
    self._init_processes(processes)
    # Radiation enabled
    self.active = active
    # Constant
    self.fac_emis_ff = float((32.0*np.pi/3.0) \
      * np.sqrt(2.0*np.pi*const.UKB/(3.0*const.UME)) * const.UE**6 \
      / ((4.0*np.pi*const.UEPS0*const.UC0)**3 * const.UH*const.UME))

  def _init_processes(self, processes):
    # Load processes
    self.processes = processes
    if (not isinstance(self.processes, dict)):
      self.processes = pickle.load(open(self.processes, "rb"))
    # Convert processes
    self.processes = utils.map_nested_dict(bkd.to_torch, self.processes)
    self.processes = utils.map_nested_dict(
      lambda x: x.squeeze() if torch.is_tensor(x) else x, self.processes
    )
    # Initialize rates container
    self.rates = {}
    if ("BB" in self.processes):
      self.rates["BB"] = self._compute_BB_rates()

  # Rates
  # ===================================
  def update(self, Th, Te, isothermal=False):
    # Compute radiation rates
    # > Zeroth order moment
    if ("BF" in self.processes):
      self.rates["BF"] = self._compute_BF_rates(Te, identifier="BF")
    # > First order moment
    if (not isothermal):
      if ("BFp" in self.processes):
        self.rates["BFp"] = self._compute_BF_rates(Te, identifier="BFp")
      if ("FF" in self.processes):
        self.rates["FF"] = self._compute_FF_rate(Te)
    # Squeeze tensors
    self.rates = utils.map_nested_dict(torch.squeeze, self.rates)

  # Forward and backward rates
  # -----------------------------------
  def _compute_fwd_rates(self, T, A, beta, Ta):
    # Arrhenius law
    return A * torch.exp(beta*torch.log(T) - Ta/T)

  # Zeroth order moment
  # -----------------------------------
  def _compute_BB_rates(self, identifier="BB"):
    """
    Spontaneous emission and absorption (BB)
    - Equation:       Ar(*)=Ar(*)
    - Forward rate:   kf = k*(1-lambda)
    - Backward rate:  kb = k*lambda
    """
    process = self.processes[identifier]
    k = process["values"]
    l = process["lambda"]
    return {"fwd": (1.0-l)*k, "bwd": (l*k).T}

  def _compute_BF_rates(self, Te, identifier="BF"):
    """
    Photo-ionization and radiative recombination (BF)
    - Equation:       Ar(*)=Arp(*)+em
    - Forward rate:   kf = k(Te)*(1-lambda)
    - Backward rate:  kb = k(Te)*lambda
    """
    process = self.processes[identifier]
    k = self._compute_fwd_rates(Te, **process["values"])
    l = process["lambda"]
    return {"fwd": (1.0-l)*k, "bwd": (l*k).T}

  # First order moment
  # -----------------------------------
  def _compute_FF_rate(self, Te):
    """Bremsstrahlung emission (FF)"""
    return self.fac_emis_ff * torch.sqrt(Te) \
      * self.processes["FF"]["Z_sq_eff"] \
      * self.processes["FF"]["g_bar"]

import torch
import dill as pickle

from ... import utils
from ... import backend as bkd


class Radiation(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    reactions,
    active=True
  ):
    # Initialize reactions rates
    self._init_reactions(reactions)
    # Radiation enabled
    self.active = active

  def _init_reactions(self, reactions):
    # Load reactions
    self.reactions = reactions
    if (not isinstance(self.reactions, dict)):
      self.reactions = pickle.load(open(self.reactions, "rb"))
    # Convert reactions
    self.reactions = utils.map_nested_dict(self.reactions, bkd.to_torch)
    # Initialize rates container
    self.rates = {}
    if ("BB" in self.reactions):
      self.rates["BB"] = self._compute_BB_rates()

  # Rates
  # ===================================
  def update(self, T, Te, isothermal=False):
    # Compute radiation rates
    # > Zeroth order moment
    if ("BF" in self.reactions):
      self.rates["BF"] = self._compute_BF_rates(Te)
    # > First order moment
    if ((not isothermal) and ("FF" in self.reactions)):
      self.rates["FF"] = self._compute_FF_rate(Te)
    # Squeeze tensors
    self.rates = utils.map_nested_dict(self.rates, torch.squeeze)

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
    reaction = self.reactions[identifier]
    k = reaction["values"]
    l = reaction["lambda"]
    return {"fwd": (1.0-l)*k, "bwd": (l*k).T}

  def _compute_BF_rates(self, Te, identifier="BF"):
    """
    Photo-ionization and radiative recombination (BF)
    - Equation:       Ar(*)=Arp(*)+em
    - Forward rate:   kf = k(Te)*(1-lambda)
    - Backward rate:  kb = k(Te)*lambda
    """
    reaction = self.reactions[identifier]
    k = self._compute_fwd_rates(Te, **reaction["values"])
    l = reaction["lambda"]
    return {"fwd": (1.0-l)*k, "bwd": (l*k).T}

  # First order moment
  # -----------------------------------
  def _compute_FF_rate(self, Te):
    """Bremsstrahlung emission (FF)"""
    return 1.42e-40 * torch.sqrt(Te) \
      * self.reactions["FF"]["Z_sq_eff"] \
      * self.reactions["FF"]["g_bar"]

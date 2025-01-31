import torch
import numpy as np
import dill as pickle

from ... import utils
from ... import backend as bkd
from pyharm import PolyHarmInterpolator


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

  def _init_processes(self, processes):
    # Load processes
    self.processes = processes
    if (not isinstance(self.processes, dict)):
      self.processes = pickle.load(open(self.processes, "rb"))
    # Convert processes
    self.processes = utils.map_nested_dict(self.processes, bkd.to_torch)
    # Photo-ionization and radiative recombination (BF-prime)
    self._init_BFp_rates()
    # Initialize rates container
    self.rates = {}
    if ("BB" in self.processes):
      self.rates["BB"] = self._compute_BB_rates()

  def _init_BFp_rates(self):
    if (("BFp" in self.processes) and self.use_tables):
      # Input/Output tensors
      x = self.processes["T"].squeeze()
      y = self.processes["BFp"]["values"]
      # Tensor shapes
      xshape = np.array(x.shape)
      yshape = np.array(y.shape)
      # Permute output tensor
      i = np.where(yshape == xshape[0])[0][0]
      dims = [i] + [j for j in range(len(yshape)) if (j != i)]
      y = torch.permute(y, dims=dims)
      # Interpolate
      self.processes["BFp"]["shape"] = np.array(y.shape).tolist()[1:]
      self.processes["BFp"]["interp"] = PolyHarmInterpolator(
        c=x.reshape(1,xshape[0],-1),
        f=y.reshape(1,xshape[0],-1),
        order=1,
        smoothing=0.0,
        dtype=bkd.floatx(bkd="torch")
      )

  # Rates
  # ===================================
  def update(self, Th, Te, isothermal=False):
    # Compute radiation rates
    # > Zeroth order moment
    if ("BF" in self.processes):
      self.rates["BF"] = self._compute_BF_rates(Te)
    # > First order moment
    if (not isothermal):
      if (("BFp" in self.processes) and self.use_tables):
        self.rates["BFp"] = self._compute_BFp_rates(Te)
      if ("FF" in self.processes):
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
  def _compute_BFp_rates(self, Te, identifier="BFp"):
    """
    Photo-ionization and radiative recombination (BF-prime)
    """
    process = self.processes[identifier]
    k = process["interp"](Te.reshape(1,1,1)).reshape(process["shape"])
    l = process["lambda"]
    return {"fwd": (1.0-l)*k, "bwd": (l*k).T}

  def _compute_FF_rate(self, Te):
    """Bremsstrahlung emission (FF)"""
    return 1.42e-40 * torch.sqrt(Te) \
      * self.processes["FF"]["Z_sq_eff"] \
      * self.processes["FF"]["g_bar"]

import numpy as np
import scipy as sp
import pandas as pd

from .coarse_graining_m0 import CoarseGrainingM0


class CoarseGrainingM1(CoarseGrainingM0):
  """
  See:
    http://dx.doi.org/10.1063/1.4915926
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    molecule
  ):
    super(CoarseGrainingM1, self).__init__(molecule)

  # Calling
  # ===================================
  def __call__(
    self,
    T,
    filename,
    teval=None,
    mapping=None,
    nb_bins=1
  ):
    super(CoarseGrainingM1, self).__call__(T, mapping, nb_bins)
    x = self.read_sol(filename, teval)
    return self.decode(x)

  def build(
    self,
    mapping=None,
    nb_bins=1
  ):
    self.set_probmat(mapping, nb_bins)
    # Test bases
    e = self.molecule.lev["e"].reshape(-1,1)
    self.psi = np.hstack([self.P, self.P*e])

  def read_sol(self, filename, teval=None):
    data = pd.read_csv(filename)
    bins = [i+1 for i in range(self.nb_bins)]
    Xg = data[[f"X_{self.molecule.name}_{i}" for i in bins]].values
    ng = data[["n"]].values * Xg
    Tg = data[[f"Tg_{i}" for i in bins]].values
    x = np.hstack([ng, Tg])
    if (teval is not None):
      t = data["t"].values
      x = sp.interpolate.interp1d(t, x, kind="cubic", axis=0)(teval)
    return x

  def encode(self, x):
    return x @ self.psi

  def decode(self, x):
    # Setting up
    is_2d = (len(x.shape) > 1)
    if (is_2d and (x.shape[-1] == 2*self.nb_bins)):
      x = x.T
    pfun = self.molecule.q_int_2d if is_2d else self.molecule.q_int
    # Extract group number densities and temperatures
    ng, Tg = np.split(x, 2, axis=0)
    # Loop over groups/bins
    n = 0.0
    for g in range(self.nb_bins):
      q = (self.P[:,g] * pfun(Tg[g])).T
      Q = np.sum(q, axis=0)
      n = n + ng[g]*q/Q
    return n

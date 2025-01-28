"""Module for reading chemical equations"""

import numpy as np

from pyvalem import reaction as pvl_reac


# Chemical equation string corrections (Do not change the order!)
_CHEMEQ_REPLACES = (
  (' ', ''),
  ("\n", ''),
  ('(', ' '),
  (')', ''),
  ('+', " + "),
  ('=', " -> "),
  ('p', '+'),
  ("em", "e-")
)

def get_param(eq, min_i=0):
  # Equation name
  for (old, new) in _CHEMEQ_REPLACES:
    eq = eq.replace(old, new)
  eq = pvl_reac.Reaction(eq)
  # Parameters
  param = {}
  species = []
  for side in ("reactants", "products"):
    side_param = _get_side(eq, side, min_i)
    param[side] = side_param[0]
    species.append(side_param[1])
    param[f"nb_{side}"] = len(side_param[0])
  param["species"] = list(np.unique(sum(species, [])))
  return param

def _get_side(eq, side, min_i=0):
  species = []
  names = []
  side = getattr(eq, side)
  for (coeff, sp) in side:
    # Name
    name = str(sp.formula)
    for (old, new) in (('+','p'),('-',"m")):
      name = name.replace(old, new)
    # Stoichiometric coefficient
    coeff = int(coeff)
    # Quantum level
    i = sp.states
    if (len(i) > 0):
      i = str(i[-1])
      if (i != '*'):
        i = int(i.split('=')[-1])
        i -= min_i
    else:
      i = 0
    # Storing
    names.append(name)
    species.append((coeff, name, i))
  # Eliminate species duplicates
  species_unique = []
  for (s, sp) in enumerate(species):
    sp = list(sp)
    if (s == 0):
      species_unique.append(sp)
    else:
      for sp_u in species_unique:
        if (sp_u[1:] == sp[1:]):
          sp_u[0] += sp[0]
        else:
          species_unique.append(sp)
          break
  species = tuple([tuple(sp_u) for sp_u in species_unique])
  return species, names

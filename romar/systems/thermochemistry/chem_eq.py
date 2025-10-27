"""
Module for parsing and extracting parameters from chemical reaction strings.

This module provides utilities to parse chemical equations and extract
quantitative and qualitative information such as:

- Reactants and products
- Stoichiometric coefficients
- Species names
- Quantum state indices

Example usage:
  Given the chemical equation "O2(*)+O=3O", the function `get_param`
  will return the following dictionary of parameters:

  param = {
    "species": ("O", "O2"),
    "reactants": [(1, "O2", "*"), (1, "O", 0)],
    "products": [(3, "O", 0)],
    "nb_reactants": 2,
    "nb_products": 1
  }

The dictionary includes:
  - "species": A tuple of unique species names involved in the reaction.
  - "reactants": A list of tuples for reactants, each containing:
    (stoichiometric coefficient, species name, quantum level).
  - "products": A list of tuples for products, in the same format.
  - "nb_reactants": Number of reactant species.
  - "nb_products": Number of product species.
"""

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

def get_param(
  eq: str,
  min_i: int = 0
) -> dict:
  """
  Parse a chemical equation string into structured reaction parameters.

  :param eq: A chemical equation string (e.g., "O2(*) + O = 3O").
  :type eq: str
  :param min_i: Minimum quantum index used to offset state labels. Default is 0.
  :type min_i: int

  :return: Dictionary with extracted fields:
           - "species": List of unique species names.
           - "reactants": List of (coefficient, species, quantum level).
           - "products": List of (coefficient, species, quantum level).
           - "nb_reactants": Number of reactant terms.
           - "nb_products": Number of product terms.
  :rtype: dict
  """
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
  param["species"] = np.unique(sum(species, [])).tolist()
  return param

def _get_side(
  eq: pvl_reac.Reaction,
  side: str,
  min_i: int = 0
) -> tuple:
  """
  Parse one side (reactants or products) of the equation.

  :param eq: Parsed reaction object from `pyvalem.reaction.Reaction`.
  :type eq: pvl_reac.Reaction
  :param side: "reactants" or "products".
  :type side: str
  :param min_i: Minimum allowed quantum index (used to offset levels).
  :type min_i: int

  :return: Tuple of:
           - Parsed species list: List of (coefficient, species, quantum level)
           - Raw species names (before deduplication)
  :rtype: tuple
  """
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

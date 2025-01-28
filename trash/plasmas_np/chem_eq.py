"""
Module for parsing and extracting parameters from chemical equations.

This module defines functions to parse a chemical equation string and extract
various parameters such as reactants, products, stoichiometric coefficients,
species, and quantum levels. The primary function, `get_param`, returns a
dictionary containing the parsed details from the equation, while helper
functions facilitate processing individual sides (reactants or products) of
the equation.

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
  Parse a chemical equation string and return a dictionary of parameters.

  :param eq: A string representing the chemical equation.
  :type eq: str
  :param min_i: Minimum index for quantum levels (default is 0).
  :type min_i: int
  :return: A dictionary containing the parsed parameters, including the
           species, reactants, products, and process information.
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
  param["species"] = list(np.unique(sum(species, [])))
  return param

def _get_side(
  eq: pvl_reac.Reaction,
  side: str,
  min_i: int = 0
) -> tuple:
  """
  Extract species and their details from a side of the equation
  (reactants/products).

  :param eq: The parsed chemical reaction object.
  :type eq: pvl_reac.Reaction
  :param side: The side of the equation, either 'reactants' or 'products'.
  :type side: str
  :param min_i: Minimum index for quantum levels (default is 0).
  :type min_i: int
  :return: A tuple containing:
           - A list of species as tuples (stoichiometric coeff, name, level).
           - A list of species names.
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

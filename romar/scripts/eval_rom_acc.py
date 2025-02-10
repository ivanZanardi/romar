"""
Evaluate accuracy of ROM model.
"""

import os
import sys
import copy
import json
import argparse
import importlib

import tensorflow as tf

# Inputs
# =====================================
parser = argparse.ArgumentParser()
parser.add_argument("--inpfile", type=str, help="path to JSON input file")
args = parser.parse_args()

with open(args.inpfile) as file:
  inputs = json.load(file)

# Import 'romar' package
# =====================================
if (importlib.util.find_spec("romar") is None):
  sys.path.append(inputs["path_to_lib"])

# Environment
# =====================================
from romar import env
env.set(**inputs["env"])

# Libraries
# =====================================
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
plt.style.use(inputs.get("mpl_style", "default"))

from romar import utils
from romar import postproc as pp
from romar import systems as sys_mod

_VALID_MODELS = ("cobras", "pod")

# Main
# =====================================
if (__name__ == "__main__"):

  print("Initialization ...")

  # System
  # -----------------------------------
  system = utils.get_class(
    modules=[sys_mod],
    name=inputs["system"]["name"]
  )(**inputs["system"]["init"])

  # Testing
  # -----------------------------------
  # Initialization
  # ---------------
  # Path to saving
  path_to_saving = inputs["paths"]["saving"] + "/error/"
  os.makedirs(path_to_saving, exist_ok=True)
  # ROM models
  models = {}
  for (name, model) in inputs["models"].items():
    if model.get("active", False):
      model = copy.deepcopy(model)
      if (name in _VALID_MODELS):
        model["bases"] = pickle.load(open(model["bases"], "rb"))
        if (model.get("error", None) is not None):
          model["error"] = pickle.load(open(model["error"], "rb"))
        else:
          model["error"] = None
      else:
        raise ValueError(
          f"Name '{name}' not valid! Valid ROM models are {_VALID_MODELS}."
        )
      models[name] = model

  # Loop over ROM models
  # ---------------
  for (name, model) in models.items():
    print("Evaluating accuracy of ROM '%s' ..." % model["name"])
    rrange = np.sort(inputs["rom_range"])
    if (model["error"] is None):
      t = None
      error, runtime, not_conv = {}, {}, {}
      # Loop over ROM dimensions
      for r in range(*rrange):
        print("> Solving with %i dimensions ..." % r)
        system.set_rom(
          phi=model["bases"]["phi"][:,:r],
          psi=model["bases"]["phi"][:,:r],
          mask=model["bases"]["mask"].squeeze()
        )
        idata, iruntime, inot_conv = system.compute_err(**inputs["data"])
        if (idata is not None):
          if (t is None):
            t = idata["t"]
          r = str(r)
          error[r], runtime[r], not_conv[r] = idata["err"], iruntime, inot_conv
      # Save error statistics
      print("> Saving statistics ...")
      # > Error
      filename = path_to_saving + f"/{name}_err.p"
      pickle.dump({"t": t, "data": error}, open(filename, "wb"))
      # > Runtime
      filename = path_to_saving + f"/{name}_runtime.json"
      with open(filename, "w") as file:
        json.dump(runtime, file, indent=2)
      # > Not converged
      filename = path_to_saving + f"/{name}_not_conv.json"
      with open(filename, "w") as file:
        json.dump(not_conv, file, indent=2)
    else:
      t = model["error"]["t"]
      error = {}
      for r in range(*rrange):
        k = str(r)
        if (k in model["error"]["data"]):
          error[k] = model["error"]["data"][k]
    # Plot error statistics
    print("> Plotting error evolution ...")
    pp.plot_err_evolution(
      path=path_to_saving+f"/{name}/",
      t=t,
      error=error,
      species=system.mix.species,
      **inputs["plot"]
    )

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  print("Done!")

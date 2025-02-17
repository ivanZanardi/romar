"""
Evaluate accuracy of ROM model.
"""

import sys
import json
import argparse
import importlib

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
import os
import copy
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
plt.style.use(inputs.get("mpl_style", "default"))

from romar import utils
from romar import postproc as pp
from romar import systems as sys_mod

_VALID_MODELS = {"cobras", "pca"}

# Main
# =====================================
if (__name__ == "__main__"):

  print("Initialization ...")

  # Path to saving
  path_to_saving = inputs["paths"]["saving"] + "/error/"
  os.makedirs(path_to_saving, exist_ok=True)

  # Copy input file
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

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
  models = {}
  for (name, model) in inputs["models"].items():
    if model.get("active", False):
      _model = copy.deepcopy(model)
      if (name in _VALID_MODELS):
        # Load basis
        with open(model["basis"], "rb") as file:
          _model["basis"] = pickle.load(file)
        # Load error
        if (model.get("error", None) is not None):
          with open(model["error"], "rb") as file:
            _model["error"] = pickle.load(file)
        else:
          _model["error"] = None
      else:
        raise ValueError(
          f"Name '{name}' not valid! Valid ROM models are {_VALID_MODELS}."
        )
      models[name] = _model
      del _model

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
        system.rom.build(
          phi=model["basis"]["phi"][r],
          psi=model["basis"]["psi"][r],
          **{k: model["basis"][k] for k in ("mask", "xref", "xscale")}
        )
        idata, iruntime, not_conv[r] = system.compute_err(**inputs["data"])
        if (idata is not None):
          if (t is None):
            t = idata["t"]
          error[r], runtime[r] = idata["err"], iruntime
      # Save error statistics
      print("> Saving statistics ...")
      # > Error
      filename = path_to_saving + f"/{name}_err.p"
      with open(filename, "wb") as file:
        pickle.dump({"t": t, "data": error}, file)
      # > Runtime and not converged cases
      for (k, v) in (
        ("runtime", runtime),
        ("not_conv", not_conv)
      ):
        filename = path_to_saving + f"/{name}_{k}.json"
        with open(filename, "w") as file:
          json.dump(v, file, indent=2)
    else:
      t = model["error"]["t"]
      error = {}
      for r in range(*rrange):
        if (r in model["error"]["data"]):
          error[r] = model["error"]["data"][r]
    # Plot error statistics
    print("> Plotting error evolution ...")
    pp.plot_err_evolution(
      path=path_to_saving+f"/{name}/",
      t=t,
      error=error,
      species=system.mix.species,
      rrange=rrange,
      **inputs["plot"]
    )

  print("Done!")

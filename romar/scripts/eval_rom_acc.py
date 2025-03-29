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
import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt
plt.style.use(inputs.get("mpl_style", "default"))

from romar import utils
from romar import systems
from romar import postproc as pp

# Main
# =====================================
if (__name__ == "__main__"):

  print("Initialization ...")

  # Path to saving
  path_to_saving = inputs["paths"]["saving"]
  os.makedirs(path_to_saving, exist_ok=True)

  # Copy input file
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  # System
  # -----------------------------------
  system = utils.get_class(
    modules=[systems],
    name=inputs["system"]["name"]
  )(**inputs["system"]["init"])

  # Testing
  # -----------------------------------
  # Initialization
  # ---------------
  models = {}
  for (name, model) in inputs["models"].items():
    if model.get("active", False):
      models[name] = copy.deepcopy(model)
      # Load basis
      with open(model["basis"], "rb") as file:
        models[name]["basis"] = pickle.load(file)
      # Load error
      if (model.get("error", None) is not None):
        with open(model["error"], "rb") as file:
          models[name]["error"] = pickle.load(file)
      else:
        models[name]["error"] = None

  # Loop over ROM models
  # ---------------
  for (name, model) in models.items():
    print("Evaluating accuracy of ROM '%s' ..." % model["name"])
    ipath_to_saving = path_to_saving + f"/{name}/"
    os.makedirs(ipath_to_saving, exist_ok=True)
    if (model["error"] is None):
      t = None
      err_time, err_mean, runtime, not_conv = {}, {}, {}, {}
      # Loop over ROM dimensions
      for i in range(*inputs["rom_range"]):
        print("> Solving with %i dimensions ..." % i)
        system.rom.build(
          **{k: model["basis"][k][i] for k in ("phi", "psi")},
          **{k: model["basis"][k] for k in ("mask", "xref", "xscale")}
        )
        ierr, iruntime, not_conv[i] = system.compute_err(**inputs["data"])
        if (ierr is not None):
          if (t is None):
            t = ierr["t"]
          runtime[i] = iruntime
          err_time[i] = ierr["err_time"]
          err_mean[i] = pd.json_normalize(
            ierr["err_mean"], sep='_'
          ).to_dict(orient='records')[0]
      # Save statistics
      print("> Saving statistics ...")
      # > Error over time
      filename = ipath_to_saving + "/err_time.p"
      with open(filename, "wb") as file:
        pickle.dump({"t": t, "data": err_time}, file)
      # > Mean error
      pd.DataFrame.from_dict(data=err_mean, orient="columns").to_csv(
        path_or_buf=ipath_to_saving + "/err_mean.csv",
        float_format='%.3f'
      )
      # > Runtime and not converged cases
      for (k, v) in (
        ("runtime", runtime),
        ("not_conv", not_conv)
      ):
        filename = ipath_to_saving + f"/{k}.json"
        with open(filename, "w") as file:
          json.dump(v, file, indent=2)
    else:
      t = model["error"]["t"]
      err_time = {}
      for i in range(*inputs["rom_range"]):
        if (i in model["error"]["data"]):
          err_time[i] = model["error"]["data"][i]
    # Plot error statistics
    if inputs["plot"]["active"]:
      print("> Plotting error evolution ...")
      pp.plot_err_evolution(
        path=ipath_to_saving+"/figs/",
        x=t,
        error=err_time,
        species=system.mix.species,
        rrange=inputs["rom_range"],
        **inputs["plot"]["kwargs"]
      )

  print("Done!")

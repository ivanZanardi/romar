"""
Visualize ROM vs FOM trajectories.
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
  path_to_saving = inputs["paths"]["saving"] + "/visual/"
  os.makedirs(path_to_saving, exist_ok=True)
  # ROM models
  models = {}
  for (name, model) in inputs["models"].items():
    if model.get("active", False):
      model = copy.deepcopy(model)
      if (name in _VALID_MODELS):
        model["bases"] = pickle.load(open(model["bases"], "rb"))
      else:
        raise ValueError(
          f"Name '{name}' not valid! Valid ROM models are {_VALID_MODELS}."
        )
      models[name] = model

  # Loop over test cases
  # ---------------
  for icase in inputs["data"]["cases"]:
    print(f"Evaluating case '{icase}' ...")
    # > Loop over ROM dimensions
    rrange = np.sort(inputs["rom_range"])
    for r in range(*rrange):
      # > Saving folder
      path_to_saving_i = path_to_saving + f"/case_{icase}/r{r}/"
      os.makedirs(path_to_saving_i, exist_ok=True)
      # > Loop over ROM models
      t = None
      sols, errs = {}, {}
      for (name, model) in models.items():
        print("> Solving ROM '%s' with %i dimensions ..." % (model["name"], r))
        system.set_rom(
          phi=model["bases"]["phi"][:,:r],
          psi=model["bases"]["psi"][:,:r],
          mask="/home/zanardi/Codes/ML/ROMAr/run/rad_on_test3/max_mom_2/rom_mask.txt" #model["bases"]["mask"]
        )
        isol, _ = system.compute_sol_rom(
          filename=inputs["data"]["path"]+f"/case_{icase}.p"
        )
        if (isol is not None):
          if (t is None):
            t = isol.pop("t")
          if ("FOM" not in sols):
            sols["FOM"] = isol.pop("FOM")
          sols[model["name"]] = isol.pop("ROM")
          errs[model["name"]] = isol.pop("err")

      # > Postprocessing
      print(f"> Postprocessing with {r} dimensions ...")
      plot_kwargs = dict(
        path=path_to_saving_i,
        t=t,
        y=sols,
        err=errs,
        err_scale=inputs["plot"].get("err_scale", "log"),
        tlim=inputs["plot"]["tlim"][icase],
        hline=inputs["plot"].get("hline", None),
        ylim_err=inputs["plot"].get("ylim_err", None)
      )
      pp.plot_temp_evolution(
        **plot_kwargs
      )
      pp.plot_mom_evolution(
        **plot_kwargs,
        species=system.mix.species_order,
        labels=inputs["plot"]["labels"],
        max_mom=inputs["plot"].get("max_mom", 2)
      )
      # pp.plot_multi_dist_2d(
      #   teval=inputs["data"]["teval"][icase],
      #   markersize=inputs["plot"].get("markersize", 1),
      #   subscript=inputs["plot"].get("subscript", "i"),
      #   **common_kwargs
      # )
      # if inputs["plot"]["animate"]:
      #   pp.animate_dist(
      #     markersize=inputs["plot"]["markersize"],
      #     **common_kwargs
      #   )

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  print("Done!")

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

from romar import roms
from romar import utils
from romar import postproc as pp
from romar import systems as sys_mod
from silx.io.dictdump import h5todict

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
        if ("mask" not in model):
          raise ValueError(
            f"Please, provide the path to ROM mask for '{name}' model."
          )
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
      path_to_saving_i = path_to_saving + f"/{icase}/r{r}/"
      os.makedirs(path_to_saving_i, exist_ok=True)
      # > Loop over ROM models
      sols, errs = {}, {}
      for (name, model) in models.items():
        print("> Solving ROM '%s' with %i dimensions ..." % (model["name"], r))
        system.set_rom(
          phi=model["bases"]["phi"][:,:r],
          psi=model["bases"]["phi"][:,:r],
          mask=model["mask"]
        )
        isol, _ = system.compute_sol_rom(
          filename=inputs["data"]["path"]+f"/{icase}.p",
          err_only=False
        )
        if (isol is not None):
          isol[model["name"]] = isol.pop("ROM")
          errs[model["name"]] = isol.pop("err")
          sols.update(isol)
        
      filename = path_to_saving + "/sols.json"
      with open(filename, "w") as file:
        json.dump(tf.nest.map_structure(np.shape, sols), file, indent=2)
      filename = path_to_saving + "/errs.json"
      with open(filename, "w") as file:
        json.dump(tf.nest.map_structure(np.shape, errs), file, indent=2)
      # # > Postprocessing
      # print(f"> Postprocessing with {r} dimensions ...")
      # common_kwargs = dict(
      #   path=path_to_saving_i,
      #   t=t,
      #   n_m=sols,
      #   molecule=system.mix.species["molecule"]
      # )
      # pp.plot_mom_evolution(
      #   max_mom=inputs["plot"].get("max_mom", 2),
      #   molecule_label=inputs["plot"]["molecule_label"],
      #   ylim_err=inputs["plot"].get("ylim_err", None),
      #   err_scale=inputs["plot"].get("err_scale", "linear"),
      #   hline=inputs["plot"].get("hline", None),
      #   tlim=inputs["data"]["tlim"][icase],
      #   **common_kwargs
      # )
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

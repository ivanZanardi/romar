"""
Visualize ROM vs FOM trajectories.
"""

import os
import sys
import copy
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
import matplotlib.pyplot as plt
plt.style.use(inputs["plot"].get("style", None))

from romar import roms
from romar import utils
from romar import postproc as pp
from romar import systems as sys_mod
from silx.io.dictdump import h5todict

_VALID_MODELS = ("cobras", "pod", "cg", "mt")

# Main
# =====================================
if (__name__ == '__main__'):

  print("Initialization ...")

  # Isothermal master equation model
  # -----------------------------------
  path_to_dtb = inputs["paths"]["dtb"]
  system = utils.get_class(
    modules=[sys_mod],
    name=inputs["system"]["name"]
  )(
    species={
      k: path_to_dtb + f"/species/{k}.json" for k in ("atom", "molecule")
    },
    rates_coeff=path_to_dtb + "/kinetics.hdf5",
    **inputs["system"]["kwargs"]
  )

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
      if (name in ("cobras", "pod")):
        model["bases"] = h5todict(model["bases"])
      elif (name == "cg"):
        model["class"] = roms.CoarseGrainingM1(
          molecule=path_to_dtb+"/species/molecule.json"
        )
      elif (name == "mt"):
        model["class"] = roms.MultiTemperature(
          molecule=path_to_dtb+"/species/molecule.json"
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
    # > Load test case
    filename = inputs["data"]["path"]+f"/case_{icase}.p"
    data = utils.load_case(filename=filename)
    T, t, n0, n_fom = [data[k] for k in ("T", "t", "n0", "n")]
    # > Update FOM operators
    system.update_fom_ops(T)
    # > Loop over ROM dimensions
    for r in range(*inputs["rom_range"]):
      # > Solutions container
      sols = {"FOM": n_fom[1]}
      # > Saving folder
      path_to_saving_i = path_to_saving + f"/case_{icase}/r{r}/"
      os.makedirs(path_to_saving_i, exist_ok=True)
      # > Loop over ROM models
      for (name, model) in models.items():
        if (name in ("cobras", "pod")):
          pdata = (model["name"], r)
          print("> Solving ROM '%s' with %i dimensions ..." % pdata)
          system.update_rom_ops(
            phi=model["bases"]["phi"][:,:r],
            psi=model["bases"]["psi"][:,:r]
          )
          sols[model["name"]] = system.solve_rom(t, n0)[1]
        # elif ((name == "cg") and (2*int(model["nb_bins"]) == r)):
        elif (name == "cg"):
          if (model["cases"].get(icase, None) is not None):
            pdata = (model["name"], int(model["nb_bins"]))
            print("> Reading ROM '%s' solution with %i bins ..." % pdata)
            sols[model["name"]] = model["class"](
              T=T,
              filename=model["cases"][icase],
              teval=t,
              mapping=model["mapping"],
              nb_bins=int(model["nb_bins"])
            )
        elif (name == "mt"):
          if (model["cases"].get(icase, None) is not None):
            pdata = (model["name"], 2)
            print("> Reading ROM '%s' solution with %i dimensions ..." % pdata)
            sols[model["name"]] = model["class"](
              filename=model["cases"][icase],
              teval=t
            )
      # > Postprocessing
      print(f"> Postprocessing with {r} dimensions ...")
      common_kwargs = dict(
        path=path_to_saving_i,
        t=t,
        n_m=sols,
        molecule=system.mix.species["molecule"]
      )
      pp.plot_mom_evolution(
        max_mom=inputs["plot"].get("max_mom", 2),
        molecule_label=inputs["plot"]["molecule_label"],
        ylim_err=inputs["plot"].get("ylim_err", None),
        err_scale=inputs["plot"].get("err_scale", "linear"),
        hline=inputs["plot"].get("hline", None),
        tlim=inputs["data"]["tlim"][icase],
        **common_kwargs
      )
      pp.plot_multi_dist_2d(
        teval=inputs["data"]["teval"][icase],
        markersize=inputs["plot"].get("markersize", 1),
        subscript=inputs["plot"].get("subscript", "i"),
        **common_kwargs
      )
      if inputs["plot"]["animate"]:
        pp.animate_dist(
          markersize=inputs["plot"]["markersize"],
          **common_kwargs
        )

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  print("Done!")

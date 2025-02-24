"""
Evaluate accuracy of ROM model.
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
import numpy as np
import joblib as jl
import dill as pickle
import matplotlib.pyplot as plt
plt.style.use(inputs["plot"].get("style", None))

from tqdm import tqdm
from romar import utils
from romar import postproc as pp
from romar import systems as sys_mod
from silx.io.dictdump import dicttoh5, h5todict

_VALID_MODELS = ("cobras", "pod")

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
  path_to_saving = inputs["paths"]["saving"]+"/error/"
  os.makedirs(path_to_saving, exist_ok=True)
  # Time grid
  t = utils.load_case(path=inputs["data"]["path"], index=0, key="t")

  # ROM models
  models = {}
  for (name, model) in inputs["models"].items():
    if model.get("active", False):
      model = copy.deepcopy(model)
      if (name in _VALID_MODELS):
        model["basis"] = pickle.load(open(model["basis"], "rb"))
        if (model.get("error", None) is not None):
          model["error"] = pickle.load(open(model["error"], "rb"))
        else:
          model["error"] = None
        if ("mask" not in model):
          raise ValueError(
            f"Please, provide the path to ROM mask for '{name}' model."
          )
      else:
        raise ValueError(
          f"Name '{name}' not valid! Valid ROM models are {_VALID_MODELS}."
        )
      models[name] = model

  # Util functions
  # ---------------
  def compute_err_parallel():
    irange = np.sort(inputs["data"]["range"])
    nb_samples = irange[1]-irange[0]
    iterable = tqdm(
      iterable=range(*irange),
      ncols=80,
      desc="  Cases",
      file=sys.stdout
    )
    kwargs = dict(
      path=inputs["data"]["path"],
      eval_err=True,
      eps=1e-8
    )
    nb_workers = inputs["data"]["nb_workers"]
    if (nb_workers > 1):
      sols = jl.Parallel(nb_workers)(
        jl.delayed(system.compute_sol_rom)(index=i, **kwargs) for i in iterable
      )
    else:
      sols = [system.compute_sol_rom(index=i, **kwargs) for i in iterable]
    # Split error values and running times
    err, runtime = list(zip(*sols))
    err = [x for x in err if (x is not None)]
    runtime = [x for x in runtime if (x is not None)]
    converged = len(runtime)/nb_samples
    print(f"  Total converged cases: {len(runtime)}/{nb_samples}")
    if (converged >= 0.8):
      # Stack error values
      if (inputs["eval_err"] == "mom"):
        err = np.stack(err, axis=0)
      else:
        err = np.vstack(err)
      # Compute statistics
      err = compute_err_stats(err)
      runtime = compute_runtime_stats(runtime)
      return err, runtime
    else:
      return None, None

  def compute_err_stats(err):
    return {
      "t": t,
      "mean": np.mean(err, axis=0),
      "std": np.std(err, axis=0)
    }

  def compute_runtime_stats(runtime):
    return {
      "mean": np.mean(runtime),
      "std": np.std(runtime)
    }

  def save_err_stats(model, stats):
    dicttoh5(
      treedict=stats,
      h5file=path_to_saving + f"/{model}_err.hdf5",
      overwrite_data=True
    )

  def save_runtime_stats(model, stats):
    filename = path_to_saving + f"/{model}_runtime.json"
    with open(filename, "w") as file:
      json.dump(stats, file, indent=2)

  # Loop over models
  # ---------------
  for (name, model) in models.items():
    print("Evaluating accuracy of ROM '%s' ..." % model["name"])
    if (model["error"] is None):
      err, runtime = {}, {}
      # Loop over dimensions
      for r in range(*inputs["rom_range"]):
        print("> Solving with %i dimensions ..." % r)
        system.set_rom(
          phi=model["basis"]["phi"][:,:r],
          psi=model["basis"]["psi"][:,:r],
          mask=model["mask"]
        )
        result = compute_err_parallel()
        if (None not in result):
          r = str(r)
          err[r], runtime[r] = result
      # Save error statistics
      print("> Saving statistics ...")
      save_err_stats(name, err)
      save_runtime_stats(name, runtime)
    else:
      err = {}
      for r in range(*inputs["rom_range"]):
        k = str(r)
        if (k in model["error"]):
          err[k] = model["error"][k]
    # Plot error statistics
    print("> Plotting error evolution ...")
    common_kwargs = dict(
      eval_err=inputs["eval_err"],
      hline=inputs["plot"].get("hline", None),
      err_scale=inputs["plot"].get("err_scale", "linear"),
      molecule_label=inputs["plot"]["molecule_label"],
      ylim_err=inputs["plot"].get("ylim_err", None),
      subscript=inputs["plot"].get("subscript", "i"),
      max_mom=inputs["plot"].get("max_mom", 2)
    )
    pp.plot_err_evolution(
      path=path_to_saving+f"/{name}/",
      err=err,
      **common_kwargs
    )

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  print("Done!")

"""
Build reduced-order model.
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
import dill as pickle

from romar import roms
from romar import utils
from romar import systems

# Main
# =====================================
if (__name__ == "__main__"):

  print("Initialization ...")

  # Path to saving
  path_to_saving = inputs["paths"]["saving"] + "/models/"
  os.makedirs(path_to_saving, exist_ok=True)

  # Copy input file
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  # System
  # ---------------
  system = utils.get_class(
    modules=[systems],
    name=inputs["system"]["name"]
  )(**inputs["system"]["init"])
  system.compute_c_mat(**inputs["system"]["c_mat"])

  # Scaling
  # ---------------
  scale = False
  scaling = {}
  scaling_opts = inputs.get("scaling", {})
  if ("filename" in scaling_opts):
    scale = scaling_opts.get("active", False)
    method = scaling_opts.get("method", None)
    if (method is not None):
      with open(scaling_opts["filename"], "rb") as file:
        scalings = pickle.load(file)
      scaling = scalings[method]

  # ROMs
  # ---------------
  for (name, opts) in inputs["models"].items():
    # Model checking
    if (name not in roms.VALID_ROMS):
      raise ValueError(
        f"Unsupported ROM model: '{name}'. Valid options: {roms.VALID_ROMS}"
      )
    # Model initialization
    print("-"*20)
    print(f"'{name}' model")
    print("-"*20)
    model = utils.get_class(
      modules=[roms],
      name=name
    )(
      system=system,
      path_to_data=inputs["paths"]["data"],
      scale=scale,
      path_to_saving=path_to_saving + f"/{name.lower()}/",
      **scaling
    )
    # Covariance matrices
    if (not opts["cov_mats"].get("read", False)):
      print("> Computing covariance matrices ...")
      cov_mats = model.compute_cov_mats(**opts["cov_mats"]["compute"])
      if opts["cov_mats"].get("save", False):
        filename = model.path_to_saving + "/cov_mats.p"
        with open(filename, "wb") as file:
          pickle.dump(cov_mats, file)
    else:
      print("> Reading covariance matrices ...")
      filename = opts["cov_mats"]["filename"]
      with open(filename, "rb") as file:
        cov_mats = pickle.load(file)
    # Modes
    print("> Computing modes ...")
    model.compute_modes(**cov_mats, **opts["modes"])

  print("Done!")

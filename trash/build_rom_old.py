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

# Utils
# =====================================
def _get_cov_mats(model, opts):
  if (not opts.get("read", False)):
    print("> Computing covariance matrices ...")
    cov_mats = model.compute_cov_mats(**opts["compute"])
    if opts.get("save", False):
      filename = os.path.join(path_to_saving, f"{model.name}_cov_mats.p")
      with open(filename, "wb") as file:
        pickle.dump(cov_mats, file)
  else:
    print("> Reading covariance matrices ...")
    with open(opts["filename"], "rb") as file:
      cov_mats = pickle.load(file)
  return cov_mats

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
  # ---------------
  system = utils.get_class(
    modules=[systems],
    name=inputs["system"]["name"]
  )(**inputs["system"]["init"])
  system.compute_c_mat(**inputs["system"]["c_mat"])

  # Scaling
  # ---------------
  scaling = None
  scaling_opts = inputs.get("scaling", {})
  if ("filename" in scaling_opts):
    scaling = scaling_opts.get("method", None)
    if (scaling is not None):
      with open(scaling_opts["filename"], "rb") as file:
        scalings = pickle.load(file)
      scaling = scalings[scaling]

  # CoBRAS
  # ---------------
  # Model
  print("CoBRAS model")
  cobras_opts = inputs["cobras"].get("init", {})
  cobras_opts.update(dict(
    system=system,
    path_to_data=inputs["paths"]["data"],
    path_to_saving=path_to_saving
  ))
  if (scaling is not None):
    cobras_opts.update(scaling)
  cobras = roms.CoBRAS(**cobras_opts)
  # Covariance matrices
  X, Y = _get_cov_mats(model=cobras, opts=inputs["cobras"]["cov_mats"])
  # Modes
  print("> Computing modes ...")
  cobras.compute_modes(X=X, Y=Y, **inputs["cobras"]["modes"])

  # PCA
  # ---------------
  # Model
  print("PCA model")
  pca_opts = inputs["pca"].get("init", {})
  pca_opts.update(dict(
    system=system,
    path_to_data=inputs["paths"]["data"],
    path_to_saving=path_to_saving
  ))
  if (scaling is not None):
    pca_opts.update(scaling)
  pca = roms.PCA(**pca_opts)
  # Covariance matrices
  X = _get_cov_mats(model=pca, opts=inputs["pca"]["cov_mats"])
  # Modes
  print("> Computing modes ...")
  pca.compute_modes(X=X, **inputs["pca"]["modes"])

  print("Done!")

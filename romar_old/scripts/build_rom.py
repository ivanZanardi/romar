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
import numpy as np
import dill as pickle

from romar import ops
from romar import roms
from romar import utils
from romar import systems as sys_mod
from romar.systems.thermochemistry.equilibrium import MU_VARS

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
    modules=[sys_mod],
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

  # Quadrature points
  # ---------------
  # > Initial conditions space (mu)
  x, dist = [], []
  for k in MU_VARS:
    kfun = np.linspace if (k == "T") else np.geomspace
    x.append(kfun(**inputs["grids"]["mu"][k]))
    kdist = "loguniform" if (k == "Te") else "uniform"
    dist.append(kdist)
  mu, w_mu = ops.get_quad_nd(
    x=x,
    deg=2,
    dist=dist
  )
  quad_mu = {"x": mu, "w": np.sqrt(w_mu)}
  # > Save quadrature points
  filename = path_to_saving + "/quad_mu.p"
  with open(filename, "wb") as file:
    pickle.dump(quad_mu, file)

  # CoBRAS
  # ---------------
  # Model
  print("CoBRAS model")
  cobras_opts = inputs["cobras"].get("init", {})
  cobras_opts.update(dict(
    system=system,
    tgrid=inputs["grids"]["t"],
    quad_mu=quad_mu,
    path_to_saving=path_to_saving
  ))
  if (scaling is not None):
    cobras_opts.update(scaling)
  cobras = roms.CoBRAS(**cobras_opts)
  # Covariance matrices
  cov_mats_opts = inputs["cobras"]["cov_mats"]
  if (not cov_mats_opts.get("read", False)):
    print("> Computing covariance matrices ...")
    cov_mats = cobras.compute_cov_mats(**cov_mats_opts["compute"])
    if cov_mats_opts.get("save", False):
      filename = os.path.join(path_to_saving, "cov_mats.p")
      with open(filename, "wb") as file:
        pickle.dump(cov_mats, file)
  else:
    print("> Reading covariance matrices ...")
    with open(cov_mats_opts["filename"], "rb") as file:
      cov_mats = pickle.load(file)
  X, Xw, Yw = cov_mats
  # Modes
  print("> Computing modes ...")
  modes_opts = inputs["cobras"]["modes"]
  _ = cobras.compute_modes(Xw=Xw, Yw=Yw, **modes_opts)

  # PCA
  # ---------------
  # Model
  print("PCA model")
  pca_opts = inputs["pca"].get("init", {})
  pca_opts.update(dict(
    path_to_saving=path_to_saving
  ))
  pca = roms.PCA(**pca_opts)
  # Modes
  print("> Computing modes ...")
  modes_opts = inputs["pca"]["modes"]
  if (scaling is not None):
    modes_opts.update(scaling)
  _ = pca.compute_modes(X=X, **modes_opts)

  print("Done!")

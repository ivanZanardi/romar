"""
Build reduced-order model.
"""

import os
import sys
import json
import time
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

  runtime = time.time()

  # System
  # -----------------------------------
  system = utils.get_class(
    modules=[sys_mod],
    name=inputs["system"]["name"]
  )(**inputs["system"]["init"])
  system.compute_c_mat(**inputs["system"]["c_mat"])

  # Model reduction
  # -----------------------------------
  # Path to saving
  path_to_saving = inputs["paths"]["saving"]
  os.makedirs(path_to_saving, exist_ok=True)

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
  pickle.dump(quad_mu, open(filename, "wb"))

  # CoBRAS
  # ---------------
  cobras = roms.CoBRAS(
    system=system,
    tgrid=inputs["grids"]["t"],
    quad_mu=quad_mu,
    path_to_saving=path_to_saving,
    saving=True
  )
  # > Compute covariance matrices
  cov_mats_opts = inputs["cobras"]["cov_mats"]
  if (not cov_mats_opts.get("read", False)):
    print("Computing covariance matrices ...")
    X, Y = cobras.compute_cov_mats(**cov_mats_opts["compute"])
    if cov_mats_opts.get("save", False):
      np.save(path_to_saving + "/X.npy", X)
      np.save(path_to_saving + "/Y.npy", Y)
  else:
    print("Reading covariance matrices ...")
    X = np.load(cov_mats_opts["path_x"])
    Y = np.load(cov_mats_opts["path_y"])
  # > Compute modes
  print("Computing modes ...")
  modes_opts = inputs["cobras"]["modes"]
  cobras.compute_modes(X=X, Y=Y, **modes_opts["compute"])

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  print("Done!")

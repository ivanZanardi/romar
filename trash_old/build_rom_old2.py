"""
Build balanced truncation-based ROM.
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

from tqdm import tqdm
from romar import ops
from romar import utils
from romar.roms import LinCoBRAS
from romar import systems as sys_mod

# Main
# =====================================
if (__name__ == "__main__"):

  print("Initialization ...")

  runtime = time.time()

  # Isothermal master equation system
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

  # Balanced truncation
  # -----------------------------------
  # Path to saving
  path_to_saving = inputs["paths"]["saving"]
  os.makedirs(path_to_saving, exist_ok=True)

  # Quadrature points
  quad = {}
  # > Time
  t, w_t = ops.get_quad_1d(
    x=system.get_tgrid(**inputs["grids"]["t"]),
    quad="gl",
    deg=2,
    dist="uniform"
  )
  quad["t"] = {"x": t, "w": np.sqrt(w_t)}
  # > Initial conditions space (mu)
  mu, w_mu = ops.get_quad_2d(
    x=np.geomspace(**inputs["grids"]["mu"]["T0"]),
    y=np.linspace(**inputs["grids"]["mu"]["w0_a"]),
    deg=2,
    dist_x="loguniform",
    dist_y="uniform"
  )
  quad["mu"] = {"x": mu, "w": np.sqrt(w_mu)}
  # > Equilibrium parameters space (theta)
  T_grid = inputs["grids"]["theta"]["T"]
  if isinstance(T_grid, dict):
    T = np.linspace(**T_grid)
  else:
    T = np.sort(np.array(T_grid).reshape(-1))
  theta, w_theta = ops.get_quad_2d(
    x=T,
    y=np.geomspace(**inputs["grids"]["theta"]["rho"]),
    deg=2,
    dist_x="uniform",
    dist_y="uniform",
    quad_x="trapz",
    quad_y="gl",
    joint=False
  )
  quad["theta"] = {}
  for (i, k) in enumerate(("T", "rho")):
    quad["theta"][k] = {"x": theta[i], "w": np.sqrt(w_theta[i])}
  # > Save quadrature points
  filename = path_to_saving + "/quad_theta.json"
  quad_theta = utils.map_nested_dict(quad["theta"], lambda x: x.tolist())
  with open(filename, "w") as file:
    json.dump(quad_theta, file, indent=2)

  # Model reduction
  # ---------------
  cov_mats = inputs.get("cov_mats", {"read": False})
  if (not cov_mats["read"]):
    X, Y = [], []
    print("Looping over temperatures:")
    for (i, Ti) in enumerate(quad["theta"]["T"]["x"]):
      print("> T = %.4e K" % Ti)
      # > FOM operators
      system.update_fom_ops(Ti)
      for (j, rhoj) in enumerate(
        tqdm(quad["theta"]["rho"]["x"], ncols=80, desc="  Densities")
      ):
        # > Linear operators
        lin_ops = system.compute_lin_fom_ops(
          mu=quad["mu"]["x"],
          rho=rhoj,
          max_mom=int(inputs["max_mom"])
        )
        cobras = LinCoBRAS(
          operators=lin_ops,
          quadrature=quad,
          path_to_saving=path_to_saving,
          saving=False,
          verbose=False
        )
        # > Covariance matrices
        Xij, Yij = cobras(
          xnot=[0],
          modes=False
        )
        wij = quad["theta"]["T"]["w"][i] * quad["theta"]["rho"]["w"][j]
        X.append(wij*Xij)
        Y.append(wij*Yij)
    X = np.hstack(X)
    Y = np.hstack(Y)
    if cov_mats.get("save", False):
      np.save(path_to_saving + "/X.npy", X)
      np.save(path_to_saving + "/Y.npy", Y)
  else:
    print("Reading X and Y matrices ...")
    X = np.load(cov_mats["path_x"])
    Y = np.load(cov_mats["path_y"])
    cobras = LinCoBRAS(
      operators=None,
      quadrature=None,
      path_to_saving=path_to_saving,
      saving=False,
      verbose=False
    )

  # > Compute balancing modes
  cobras.verbose = True
  cobras(
    X=X,
    Y=Y,
    modes=True,
    pod=True,
    runtime=cobras.runtime
  )

  cobras.runtime["tot"] = time.time()-runtime
  filename = path_to_saving + "/runtime.json"
  with open(filename, "w") as file:
    json.dump(cobras.runtime, file, indent=2)

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  print("Done!")

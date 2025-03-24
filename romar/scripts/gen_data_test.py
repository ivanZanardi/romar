"""
Generate FOM testing data.
"""

import os
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
import numpy as np

from romar import utils
from romar import systems
from romar.data import Data

# Main
# =====================================
if (__name__ == "__main__"):

  print("Initialization ...")

  # Path to saving
  path_to_saving = inputs["paths"]["saving"] + "/test/"
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

  # Data
  # ---------------
  data = Data(
    system=system,
    grids=inputs["data"]["grids"],
    path_to_saving=path_to_saving
  )
  data.generate_data_test(**inputs["data"]["sampled"])

  # Defined cases
  # ---------------
  if ("defined" in inputs["data"]):
    t = np.geomspace(**data.grids["t"])
    for (k, muk) in inputs["data"]["defined"]["cases"].items():
      print(f"Running case '{k}' ...")
      runtime = data.compute_sol(
        mu=muk,
        t=t,
        filename=path_to_saving + f"/case_{k}.p"
      )
      if (runtime is None):
        print(f"Case '{k}' not converged!")

    print("Done!")

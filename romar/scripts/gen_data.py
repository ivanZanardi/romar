"""
Generate FOM data.
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
from romar import systems as sys_mod

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

  # Data generation
  # -----------------------------------
  # Initialization
  # ---------------
  # Path to saving
  path_to_saving = inputs["paths"]["saving"]
  os.makedirs(path_to_saving, exist_ok=True)
  # Time grid
  t = np.geomspace(**inputs["grids"]["t"])

  # Sampled cases
  # ---------------
  # Construct design matrix
  mu_opts = inputs["param_space"]["sampled"]["mu"]
  if (mu_opts["nb_samples"] > 0):
    # Sampled initial conditions parameters
    mu = system.construct_design_mat_mu(**mu_opts)
    # Generate data
    print("Running sampled cases ...")
    runtime = utils.generate_case_parallel(
      sol_fun=system.compute_sol_fom,
      irange=[0,mu_opts["nb_samples"]],
      sol_kwargs=dict(
        t=t,
        mu=mu.values,
        noise=False,
        path=path_to_saving
      ),
      nb_workers=inputs["param_space"]["nb_workers"],
      desc=None,
      delimiter="> "
    )
    # Save parameters
    mu.to_csv(
      path_to_saving + "/samples_mu.csv",
      float_format="%.8e",
      index=True
    )
    # Save runtime
    with open(path_to_saving + "/runtime.txt", "w") as file:
      file.write("Mean running time: %.8e s" % runtime)

  # Defined cases
  # ---------------
  for (k, muk) in inputs["param_space"]["defined"]["cases"].items():
    print(f"Running case '{k}' ...")
    runtime = system.compute_sol_fom(
      t=t,
      mu=muk,
      noise=False,
      filename=path_to_saving + f"/case_{k}.p"
    )
    if (runtime is None):
      print(f"Case '{k}' not converged!")

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  print("Done!")

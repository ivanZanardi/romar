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
import pandas as pd

from romar import utils
from romar import systems as sys_mod

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
  T, mu = [inputs["param_space"]["sampled"][k] for k in ("T", "mu")]
  if ((T["nb_samples"] > 0) and (mu["nb_samples"] > 0)):
    # > Sampled temperatures
    if isinstance(T, dict):
      T = system.construct_design_mat_temp(**T)
    else:
      T = np.sort(np.array(T.reshape(-1)))
      T = pd.DataFrame(data=T, columns=["T"])
    nb_samples_temp = len(T)
    # > Sampled initial conditions parameters
    mu = system.construct_design_mat_mu(**mu)
    nb_samples_mu = len(mu)
    # Generate data
    print("Looping over sampled temperatures:")
    runtime = 0.0
    for (i, Ti) in enumerate(T.values.reshape(-1)):
      print("> T = %.4e K" % Ti)
      system.update_fom_ops(Ti)
      runtime += utils.generate_case_parallel(
        sol_fun=system.compute_fom_sol,
        irange=[0,nb_samples_mu],
        sol_kwargs=dict(
          T=Ti,
          t=t,
          mu=mu.values,
          update=False,
          path=path_to_saving,
          shift=nb_samples_mu*i,
          filename=None
        ),
        nb_workers=inputs["param_space"]["nb_workers"]
      )
    # Save parameters
    for (name, df) in (
      ("mu", mu),
      ("T", T)
    ):
      df.to_csv(
        path_to_saving + f"/samples_{name}.csv",
        float_format="%.8e",
        index=True
      )
    # Save runtime
    runtime /= nb_samples_temp
    with open(path_to_saving + "/runtime.txt", "w") as file:
      file.write("Mean running time: %.8e s" % runtime)

  # Defined cases
  # ---------------
  for (k, param) in inputs["param_space"]["defined"]["cases"].items():
    print(f"Running case '{k}' ...")
    runtime = system.compute_fom_sol(
      T=float(param["T"]),
      t=t,
      mu=param["mu"],
      mu_type=inputs["param_space"]["defined"].get("mu_type", "mass"),
      update=True,
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

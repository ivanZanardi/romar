#!/bin/bash -i
set -euo pipefail

# -----------------------------------------------------------------------------
# Script purpose:
#   Automates data generation for ROM training and evaluation datasets.
#   Includes training set, general test sets, and temperature-conditioned tests.
#
# Requirements:
#   - jq:           For parsing JSON configuration
#   - conda:        Must be initialized via `conda init`
#   - Python:       Environment must contain all ROMAr dependencies
# -----------------------------------------------------------------------------

# --------------------------
# USER CONFIGURATION
# --------------------------
# Modify only the two lines below:
#
#   1. paths_file → Path to your custom paths.json
#   2. conda_env  → Name of the conda environment with required packages
#
# Your paths.json must define the following keys:
#   - paths.saving     → Base directory for saving results
#   - paths.library    → Absolute path to the ROMAr codebase
#   - paths.anaconda   → Path to Anaconda installation
# --------------------------

conda_env="sciml"
paths_file="./../paths.json"

# --------------------------
# Validate Environment
# --------------------------
if ! command -v jq &> /dev/null; then
  echo " Error: 'jq' is not installed. Install it with:"
  echo " > sudo apt install jq     # or"
  echo " > brew install jq         # on macOS"
  exit 1
fi

if [[ ! -f "$paths_file" ]]; then
  echo "Error: paths.json not found at $paths_file"
  exit 1
fi

# --------------------------
# Load Paths from JSON
# --------------------------
path_to_inputs="$(jq -r '.paths.saving' "$paths_file")/inputs"
path_to_scripts="$(jq -r '.paths.library' "$paths_file")/scripts"

# --------------------------
# Activate Conda
# --------------------------
source "$(jq -r '.paths.anaconda' "$paths_file")/bin/activate"
conda activate $conda_env

# --------------------------
# Run Workflow
# --------------------------
echo -e "\n[1/6] Generating input files ..."
python -u "$path_to_scripts/gen_input_files.py" --inpfile "$paths_file"

echo -e "\n[2/6] Generating training dataset ..."
python -u "$path_to_scripts/gen_data_train.py" --inpfile "$path_to_inputs/gen_data_train.json"

echo -e "\n[3/6] Generating test dataset (nominal) ..."
python -u "$path_to_scripts/gen_data_test.py" --inpfile "$path_to_inputs/gen_data_test.json"

echo -e "\n[4/6] Generating test dataset (fixed density) ..."
python -u "$path_to_scripts/gen_data_test.py" --inpfile "$path_to_inputs/gen_data_test_rho_fixed.json"

echo -e "\n[5/6] Generating test dataset (high temperatures) ..."
python -u "$path_to_scripts/gen_data_test.py" --inpfile "$path_to_inputs/gen_data_test_temp_high.json"

echo -e "\n[6/6] Generating test dataset (low temperatures) ..."
python -u "$path_to_scripts/gen_data_test.py" --inpfile "$path_to_inputs/gen_data_test_temp_low.json"

# --------------------------
# Cleanup
# --------------------------
echo -e "\nCleaning up generated input files ..."
rm -rf "$path_to_inputs"

# --------------------------
# Deactivate Conda
# --------------------------
conda deactivate

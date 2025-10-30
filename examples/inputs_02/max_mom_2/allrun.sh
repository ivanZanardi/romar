#!/bin/bash -i
set -euo pipefail

# -----------------------------------------------------------------------------
# Script purpose:
#   Automates the end-to-end Reduced-Order Model (ROM) pipeline:
#     - Generates input configuration files
#     - Builds the reduced-order model (ROM)
#     - Evaluates ROM accuracy under multiple test scenarios
#     - Visualizes results
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

echo -e "\n[2/6] Building Reduced-Order Model (ROM) ..."
python -u "$path_to_scripts/build_rom.py" --inpfile "$path_to_inputs/build_rom.json"

echo -e "\n[3/6] Evaluating ROM Accuracy ..."
python -u "$path_to_scripts/eval_rom_acc.py" --inpfile "$path_to_inputs/eval_rom_acc.json"

echo -e "\n[4/6] Evaluating ROM Accuracy (high temperatures) ..."
python -u "$path_to_scripts/eval_rom_acc.py" --inpfile "$path_to_inputs/eval_rom_acc_temp_high.json"

echo -e "\n[5/6] Evaluating ROM Accuracy (low temperatures) ..."
python -u "$path_to_scripts/eval_rom_acc.py" --inpfile "$path_to_inputs/eval_rom_acc_temp_low.json"

echo -e "\n[6/6] Visualizing ROM Results ..."
python -u "$path_to_scripts/visual_rom.py" --inpfile "$path_to_inputs/visual_rom.json"

# --------------------------
# Cleanup
# --------------------------
echo -e "\nCleaning up generated input files ..."
rm -rf "$path_to_inputs"

# --------------------------
# Deactivate Conda
# --------------------------
conda deactivate

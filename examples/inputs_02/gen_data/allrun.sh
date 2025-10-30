#!/bin/bash -i

# Set paths
path_to_scripts=/u/zanardi3/Codes/ML/ROMAr/romar/scripts/
path_to_saving=/u/zanardi3/Workspace/cr_argon/roms/run02/

# Load Conda environment
source /sw/apps/anaconda3/2024.10/bin/activate
conda activate sciml

# Run scripts
echo -e "\nRunning 'gen_input_files' script ..."
python -u $path_to_scripts/gen_input_files.py --inpfile ./../paths.json

echo -e "\nRunning 'gen_data_train' script ..."
python -u $path_to_scripts/gen_data_train.py --inpfile $path_to_saving/inputs/gen_data_train.json

echo -e "\nRunning 'gen_data_test' script ..."
python -u $path_to_scripts/gen_data_test.py --inpfile $path_to_saving/inputs/gen_data_test.json

echo -e "\nRunning 'gen_data_test' script (fixed rho) ..."
python -u $path_to_scripts/gen_data_test.py --inpfile $path_to_saving/inputs/gen_data_test_rho_fixed.json

echo -e "\nRunning 'gen_data_test' script (high temperatures) ..."
python -u $path_to_scripts/gen_data_test.py --inpfile $path_to_saving/inputs/gen_data_test_temp_high.json

echo -e "\nRunning 'gen_data_test' script (low temperatures) ..."
python -u $path_to_scripts/gen_data_test.py --inpfile $path_to_saving/inputs/gen_data_test_temp_low.json

# Remove generated inputs
rm -rf $path_to_saving/inputs/

# Purge Conda environment
conda deactivate

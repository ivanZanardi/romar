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

echo -e "\nRunning 'build_rom' script ..."
python -u $path_to_scripts/build_rom.py --inpfile $path_to_saving/inputs/build_rom.json

# echo -e "\nRunning 'eval_rom_acc' script ..."
# python -u $path_to_scripts/eval_rom_acc.py --inpfile $path_to_saving/inputs/eval_rom_acc.json

# echo -e "\nRunning 'eval_rom_acc' script (high temperatures) ..."
# python -u $path_to_scripts/eval_rom_acc.py --inpfile $path_to_saving/inputs/eval_rom_acc_temp_high.json

# echo -e "\nRunning 'eval_rom_acc' script (low temperatures) ..."
# python -u $path_to_scripts/eval_rom_acc.py --inpfile $path_to_saving/inputs/eval_rom_acc_temp_low.json

# echo -e "\nRunning 'visual_rom' script ..."
# python -u $path_to_scripts/visual_rom.py --inpfile $path_to_saving/inputs/visual_rom.json

# # Remove generated inputs
# rm -rf $path_to_saving/inputs/

# Purge Conda environment
conda deactivate

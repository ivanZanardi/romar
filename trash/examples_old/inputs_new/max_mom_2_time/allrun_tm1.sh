#!/bin/bash -i

path_to_scripts=/home/zanardi/Codes/ML/ROMAr/romar/romar/scripts/

echo -e "\nRunning 'eval_rom_acc' script ..."
python -u $path_to_scripts/eval_rom_acc.py --inpfile eval_rom_acc_tm1.json

echo -e "\nRunning 'visual_rom' script ..."
python -u $path_to_scripts/visual_rom.py --inpfile visual_rom_tm1.json

#!/bin/bash -i

path_to_scripts=/home/zanardi/Codes/ML/ROMAr/romar/romar/scripts/

echo -e "\nRunning 'build_rom' script ..."
python -u $path_to_scripts/build_rom.py --inpfile build_rom_Thigh.json

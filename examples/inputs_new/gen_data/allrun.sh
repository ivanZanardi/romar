#!/bin/bash -i

path_to_scripts=/home/zanardi/Codes/ML/ROMAr/romar/romar/scripts/

echo -e "\nRunning 'gen_data_train' script ..."
python -u $path_to_scripts/gen_data_train.py --inpfile gen_data_train.json

echo -e "\nRunning 'gen_data_test' script ..."
python -u $path_to_scripts/gen_data_test.py --inpfile gen_data_test.json

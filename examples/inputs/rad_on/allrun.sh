#!/bin/bash -i

cd gen_data
bash allrun.sh
cd ../

cd max_mom_2
bash allrun.sh
cd ../

cd max_mom_10
bash allrun.sh
cd ../

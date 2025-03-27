#!/bin/bash -i

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
##SBATCH --exclude=ccc[0370,0371,0372,0373,0374,0375,0376,0377,0378,0379,0380,0381,0382,0383,0384,0385]
##SBATCH --exclude=ccc[0400,0401,0402,0403,0404,0405,0406,0407,0408,0409,0410,0411,0412,0413,0414,0415]
#SBATCH --job-name=roms_r04_a
#SBATCH --account=chess
#SBATCH --partition=chess
#SBATCH --output=log.output_%j
#SBATCH --error=log.error_%j
#SBATCH --mail-user=zanardi3@illinois.edu
#SBATCH --mail-type=BEGIN,END

# Activate the environment
source /sw/apps/anaconda3/2024.10/bin/activate
conda activate sciml

# Run the scripts
# cd gen_data
# bash allrun.sh
# cd ../

cd max_mom_2
bash allrun.sh
cd ../

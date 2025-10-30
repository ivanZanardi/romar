#!/bin/bash -i

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
##SBATCH --exclude=ccc[0370-0385]
##SBATCH --exclude=ccc[0400-0415]
##SBATCH --nodelist=ccc[0370-0385]
##SBATCH --nodelist=ccc[0400-0415]
#SBATCH --job-name=roms_run01
#SBATCH --account=chess
#SBATCH --partition=chess
#SBATCH --output=log.output_%j
#SBATCH --error=log.error_%j
##SBATCH --mail-user=zanardi3@illinois.edu
##SBATCH --mail-type=BEGIN,END

# cd gen_data
# bash allrun.sh
# cd ../

cd max_mom_2
bash allrun.sh
cd ../

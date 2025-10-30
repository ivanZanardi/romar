#!/bin/bash -i

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --account=chess
#SBATCH --partition=chess
#SBATCH --job-name=run01
#SBATCH --output=log.output_%j
#SBATCH --error=log.error_%j

##SBATCH --exclude=ccc[0370-0385]
##SBATCH --exclude=ccc[0400-0415]
##SBATCH --nodelist=ccc[0370-0385]
##SBATCH --nodelist=ccc[0400-0415]

# Phase 1: Generate training and test data
cd gen_data
bash allrun.sh
cd ../

# Phase 2: Build and evaluate ROM
cd max_mom_2
bash allrun.sh
cd ../

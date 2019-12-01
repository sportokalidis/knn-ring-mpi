#!/bin/bash
#SBATCH -p pdlabs
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=8
#SBATCH --time=0:05:00

module load gcc openmpi openblas

make all

srun ./src/knnring_mpi







#!/bin/sh
#SBATCH -o %J.o
#SBATCH -e %J.e
#SBATCH -t 24:00:00
#SBATCH -n 16
#SBATCH --job-name= {seedname}
#SBATCH --exclusive
module load castep/19
mpirun castep.mpi {seedname}

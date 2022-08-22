#!/bin/sh
#SBATCH -o %J.o
#SBATCH -e %J.e
#SBATCH -t 24:00:00
#SBATCH -n 16
#SBATCH --job-name= Si2_geom_opt
#SBATCH --exclusive
module load castep/19
mpirun castep.mpi Si2_geom_opt

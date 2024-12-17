#! /bin/bash
#SBATCH -N 2
#SBATCH -n 6
#SBATCH -c 4
#SBATCH -J conv_hybrid
#SBATCH --output=output.%j
#SBATCH --error=error.%j
#
module load openmpi 
srun $HOME/final/conv_hybrid testcases/1024_1024_96.in output
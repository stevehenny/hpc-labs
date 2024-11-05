#!/bin/bash

#SBATCH --time=00:01:00  # walltime
#SBATCH --ntasks-per-node=1  # number of tasks per node
#SBATCH --mem-per-cpu=1024M  # memory per CPU core
#SBATCH --partition=m8 # partition to run the job

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#module load gcc openmpi

export PMIX_MCA_psec=^munge

for n in 2 3 6 13 32; do
  if [ $n -gt $SLURM_JOB_NUM_NODES ]; then break; fi
  #mpirun -N 1 -np $n ./solution 1 $((256 * 1024))
  srun --mpi=pmix -N $n -n $n ./solution 1 $((256 * 1024))
done

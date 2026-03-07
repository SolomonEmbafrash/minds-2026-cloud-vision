#!/bin/bash

#SBATCH --job-name=helloworld
#SBATCH --account=project_2001220
#SBATCH --time=00:01:00
#SBATCH --partition=small

echo "Hello, world from SLURM job $SLURM_JOB_ID on node $(hostname)"
#!/bin/bash

#SBATCH --job-name=pipeline
#SBATCH --account=project_2018175
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

module load pytorch/2.7
pip install -U --user pandas ollama numpy

OLLAMA_SCRATCH=/scratch/project_2018175/embafras/ollama
export OLLAMA_MODELS=${OLLAMA_SCRATCH}/models
export PATH=${OLLAMA_SCRATCH}/bin:$PATH

mkdir -p ${OLLAMA_SCRATCH}/logs
ollama serve > ${OLLAMA_SCRATCH}/logs/${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

sleep 15

ollama pull ministral-3:3b
ollama pull embeddinggemma
ollama list

python pipeline.py images.txt model.csv

kill $OLLAMA_PID
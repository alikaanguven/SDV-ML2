#!/bin/bash
#SBATCH --job-name=SDV
#SBATCH --output=/scratch-cbe/users/alikaan.gueven/job_outs/job_%j.out 
#SBATCH --partition=g
#SBATCH --constraint="g2|g3|g4"
#SBATCH --qos=g_medium
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=3500M
echo -----------------------------------------------
IMAGE=/software/system/jupyter/jupyter-conda_v10012025.sif
apptainer exec --nv "$IMAGE" bash -lc "
mamba activate SDV
echo \"Using env: $CONDA_DEFAULT_ENV \"
echo \"COMMAND: $1\" 
$1
"
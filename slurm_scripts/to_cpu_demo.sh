#!/bin/bash 

# Usage: sbatch to_gpu.sh "python ParT_customised.py"


#SBATCH --job-name=ML_thingy
#SBATCH --output=/scratch-cbe/users/alikaan.gueven/job_outs/job_%j.out 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=250M 
#SBATCH --nodes=1-1 
#SBATCH --partition=c
#SBATCH --qos=c_rapid 
#SBATCH --time=01:00:00 
echo -----------------------------------------------
IMAGE=/software/system/jupyter/jupyter-conda_v10012025.sif
apptainer exec --nv "$IMAGE" bash -lc "
mamba activate SDV
echo \"Using env: $CONDA_DEFAULT_ENV \"
echo \"COMMAND: $1\" 
$1
"
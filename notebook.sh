#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err


port=8887
node=$(hostname -s)

module load cuda
module load nvhpc

# run jupyter notebook
jupyter-notebook --no-browser --port=${port} --ip=${node}



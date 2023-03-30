#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err


port=8888
node=$(hostname -s)

module load cuda
module load nvhpc

# run jupyter notebook
jupyter-notebook --no-browser --port=${port} --ip=${node}



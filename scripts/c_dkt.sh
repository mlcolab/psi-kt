#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-12:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=50G  
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --array=0



# print info about current job
scontrol show job $SLURM_JOB_ID 


A=({5,10,20})
python main.py --dataset assistment12 \
--model_name DKT --load 0 \
--max_step 50 --lr 5e-3 --l2 1e-5 \
--epoch 200 --emb_history 1 \
--overfit 0 
# --expername test --epoch 10
# --graph_params [['correct_transition_graph.json', True]]
# --mail-user=hanqi.zhou@uni-tuebingen.de   # Email to which notifications will be sent
# 
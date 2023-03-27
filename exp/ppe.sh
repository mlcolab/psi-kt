#!/bin/bash

#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=2         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=1-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G  
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --array=0

A=(0.2)

python single_learner_single_skill_predict.py \
--dataset junyi/single_user_multi_skill \
--model_name PPE --max_step 100 --gpu 0 \
--epoch 200 --overfit 16 \
--batch_size 512 --validate --train_time_ratio ${A[$SLURM_ARRAY_TASK_ID]} --test_time_ratio 0.4 \
--train_mode ls_split_time --multi_node 0 --expername test

wait
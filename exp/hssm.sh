#!/bin/bash

#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-12:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=64G  
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --array=0

python exp_learner_predict.py --dataset assistment17/multi_skill \
--model_name TestHSSM \
--em_train 0 \
--overfit 16 --num_sample 50 --batch_size 64 --eval_batch_size 16 \
--test 1 --test_every 5 --validate 0 \
--lr_decay 50 --lr 5e-2 --vcl 0 \
--max_step 50 --gpu 0 \
--multi_node 1 \
--train_mode ls_split_time \
--epoch 200 --train_time_ratio 0.4 --test_time_ratio 0.5 --early_stop 0 \
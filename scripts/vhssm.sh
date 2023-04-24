#!/bin/bash

#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-12:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=64G  
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --array=0


python exp_learner_predict.py --dataset junyi15/multi_skill \
--model_name VanillaHSSM \
--max_step 100 --gpu 0 \
--epoch 1000 --overfit 512 \
--train_time_ratio 0.3 --test_time_ratio 0.7 --early_stop 0 \
--batch_size 16 --eval_batch_size 16 \
--lr_decay 50 --lr 0.04 \
--train_mode ls_split_time --multi_node 1 \
--validate 1 --validate_every 10 --experiname testdata \
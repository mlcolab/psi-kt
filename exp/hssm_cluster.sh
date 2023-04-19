#!/bin/bash

#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G  
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --array=0

python exp_learner_predict.py --dataset junyi15/single_skill --multi_node 0 \
--model_name TestHSSM \
--max_step 200 --gpu 0 \
--epoch 100 --overfit 0 \
--train_time_ratio 0.1 --test_time_ratio 0.5 --early_stop 0 \
--batch_size 512 --eval_batch_size 16 \
--lr_decay 50 --lr 0.04 \
--train_mode ls_split_time \
--validate 0 --experiname whole \
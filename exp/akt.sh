#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-12:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G  
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --array=0

python exp_baseline.py --dataset junyi15/multi_skill --max_step 50 \
--model_name AKT \
--em_train 0 \
--epoch 200 --vcl 0 --multi_node 1 \
--train_mode ls_split_time --overfit 16 \
--batch_size 256 \
--test 1 --test_every 5 --save_every 5 --validate 1 \
--train_time_ratio 0.4 --test_time_ratio 0.5 \

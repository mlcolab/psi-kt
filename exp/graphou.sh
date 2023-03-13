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

python single_learner_single_skill_predict.py \
--dataset junyi/single_user_multi_skill \
--model_name GraphOU --max_step 200 --gpu 0 \
--epoch 100 --overfit 0 \
--batch_size 512 --validate --train_time_ratio 0.5 --test_time_ratio 0.4 \
--train_mode ls_split_time --multi_node 1 \
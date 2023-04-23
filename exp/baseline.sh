#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G  
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --array=0

# A=({DKTForgetting,DKT,HKT})

python exp_baseline.py --dataset junyi15/multi_skill --max_step 50 \
--model_name HLR \
--em_train 0 --epoch 200 --vcl 0 --multi_node 1 \
--train_mode ls_split_time --overfit 0 \
--batch_size 256 --eval_batch_size 256 \
--test 1 --test_every 5 --save_every 5 --validate 1 \
--train_time_ratio 0.4 --test_time_ratio 0.5 \
--early_stop 0

# # ------ vcl -----
# python exp_baseline.py --dataset assistment12/multi_skill --max_step 50 \
# --model_name ${A[$SLURM_ARRAY_TASK_ID]} --load 0 \
# --gpu 0 \
# --epoch 100 --vcl 1 \
# --train_mode simple_split_time --overfit 0 \
# --batch_size 256 \
# --test 1 --test_every 1 --save_every 5 \
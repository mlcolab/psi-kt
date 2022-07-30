#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-12:00            # Runtime in D-HH:MM
#SBATCH --mem=50G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=zzzzzhq@outlook.com   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

python main.py --model_name DKT --emb_size 64 --hidden_size 64 --max_step 50 --lr 1e-3 \
--path /mnt/qb/work/mlcolab/hzhou52/kt \
--l2 0 --gpu 0 --dataset test

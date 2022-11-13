#!/bin/bash
#SBATCH --job-name=cluster_kt
#SBATCH --partition=gpu-2080ti
#SBATCH --time=0-12:00 # Runtime in D-HH:MM

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --array=0

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# ### init virtual environment if needed
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate mykt

### the command to run

# print info about current job
scontrol show job $SLURM_JOB_ID 
A=({30,40,45})
srun python main.py --dataset assistment12 \
--model_name CausalKT --load 0 \
--max_step 50 --lr 5e-3 --l2 1e-5 \
--batch_size 4 --epoch 200 \
--expername time_lag_${A[$SLURM_ARRAY_TASK_ID]} \
--overfit 50 --emb_history 0 \
--time_lag ${A[$SLURM_ARRAY_TASK_ID]} --num_graph 5 --dense_init 0



# --expername test --epoch 10
# --graph_params [['correct_transition_graph.json', True]]
# --mail-user=hanqi.zhou@uni-tuebingen.de   # Email to which notifications will be sent
# 
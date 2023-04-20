#!/bin/bash

python exp_baseline.py --dataset assistment12/multi_skill --max_step 50 \
--model_name DKT --load 0 \
--lr 5e-3 --l2 1e-5 --gpu 0 \
--epoch 2 --vcl 1 \
--train_mode simple_split_time --overfit 16 \
--batch_size 16 \
--validate 1 --validate_every 1
# --graph_params [["correct_transition_graph.json", True]]
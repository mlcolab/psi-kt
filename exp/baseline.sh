#!/bin/bash

python exp_baseline.py --dataset assistment12 \
--model_name DKT --load 0 \
--max_step 50 --lr 5e-3 --l2 1e-5 --gpu 0 \
--epoch 200 --emb_history 1 --emb_size 16 \
# --graph_params [["correct_transition_graph.json", True]]
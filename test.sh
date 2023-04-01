#!/bin/bash

python main.py  \
--model_name DKT --max_step 50 --dataset assistment12 \
--lr 5e-3 --l2 1e-5 --gpu 0 \
--epoch 200 --overfit 16 \
# --num_graph 10 --decoder_type oudebug \
# --latent_rep basic --batch_size 32 
# --graph_params [["correct_transition_graph.json", True]]
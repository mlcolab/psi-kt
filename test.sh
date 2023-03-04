#!/bin/bash

python main.py --dataset assistment12 \
--model_name CausalKT --load 0 \
--max_step 50 --lr 5e-3 --l2 1e-5 --time_log 5 --gpu 0 \
--epoch 200 --emb_history 1 --overfit 0 \
--num_graph 10 --decoder_type oudebug \
--latent_rep basic --batch_size 32 # --regenerate_corpus
# --graph_params [["correct_transition_graph.json", True]]
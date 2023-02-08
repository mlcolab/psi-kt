#!/bin/bash

python main.py --dataset synthetic/2023-01-18T16:52:27.319345_node_3mean_0.7speed_0.02var_0.01 \
--model_name CausalKT --load 0 \
--max_step 50 --lr 5e-3 --l2 1e-5 --time_log 5 --gpu 0 \
--quick_test --epoch 200 --emb_history 1 --overfit 0 \
--num_graph 10 --decoder_type oudebug \
--latent_rep basic # --regenerate_corpus
# --graph_params [["correct_transition_graph.json", True]]
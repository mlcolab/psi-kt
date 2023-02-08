#!/bin/bash

python learner_predict_junyi.py --dataset junyi \
--model_name CausalKT --load 0 \
--max_step 200 --lr 5e-3 --l2 1e-5 --time_log 5 --gpu 0 \
--decoder_type graphou_fixedgraph --latent_rep gt_fixedgraph \
--gt_adj_path /mnt/qb/work/mlcolab/hzhou52/kt/junyi/adj.npy \
--num_graph 1 --decoder_type graphou_fixedgraph \
--quick_test --epoch 200 --overfit 16 --emb_size 16 --time_lag 50 --emb_size 8 \

 # --regenerate_corpus
# --graph_params [["correct_transition_graph.json", True]]
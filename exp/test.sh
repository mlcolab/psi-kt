#!/bin/bash

python single_learner_single_skill_predict.py --dataset junyi/single_user_single_skill \
--model_name SwitchingNLDS --load 0 \
--max_step 200 --lr 5e-3 --l2 1e-5 --time_log 5 --gpu 0 \
--epoch 200 --overfit 16 --emb_size 16 --time_lag 50 --emb_size 8 \
--batch_size 512 --validate --train_time_ratio 0.6 --test_time_ratio 0.4

 # --regenerate_corpus
# --graph_params [["correct_transition_graph.json", True]]
# --gt_adj_path /mnt/qb/work/mlcolab/hzhou52/kt/junyi/adj.npy \
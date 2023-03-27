#!/bin/bash

python single_learner_single_skill_predict.py \
--dataset Duolingo \
--model_name VanillaOU --max_step 200 --gpu 0 \
--epoch 100 --overfit 0 \
--batch_size 512 --validate --train_time_ratio 0.5 --test_time_ratio 0.4 \
--train_mode ls_split_time --multi_node 0

#!/bin/bash

python exp_baseline.py \
--dataset assistment12/multi_skill --model_name AKT --random_seed 2023 \
--epoch 500 --vcl 0 --multi_node 1 \
--train_mode ls_split_time --overfit 100 \
--batch_size 256 --eval_batch_size 256 \
--test 1 --test_every 1 --save_every 10 --validate 1 \
--train_time_ratio 0.2 --test_time_ratio 0.2 \
--early_stop 1 \
--lr 5e-3 --lr_decay 150 --expername early_stop \
--save_folder /mnt/qb/work/mlcolab/hzhou52/0729_new_exp2_logs

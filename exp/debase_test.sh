#!/bin/bash
python exp_baseline.py --dataset assistment17/multi_skill \
--model_name HKT --em_train 0 \
--epoch 200 --vcl 0 --multi_node 1 --train_mode ls_split_time --overfit 16 \
--batch_size 256 --eval_batch_size 256 \
--test 1 --test_every 5 --save_every 5 --validate 1 \
--train_time_ratio 0.4 --test_time_ratio 0.5 \
--early_stop 0 --max_step 50 \
--lr_decay 5000 --gamma 1 \
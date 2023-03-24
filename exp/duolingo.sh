#!/bin/bash

python reload_results.py --model_name visualize --dataset junyi/single_user_multi_skill --max_step 200 \
--train_mode ls_split_time --eval_batch_size 128
# python duolingo.py /mnt/qb/work/mlcolab/hzhou52/kt/Duolingo/learning_traces.csv
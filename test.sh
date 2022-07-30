# python main.py --model_name hkt --emb_size 64 --max_step 50 --lr 5e-4 --l2 1e-5 --time_log 5 --gpu 0 \
# --path /mnt/qb/work/mlcolab/hzhou52/kt \
# --dataset test


python main.py --model_name SKTF --max_step 50 --lr 5e-3 --l2 1e-5 --time_log 5 --gpu 0 \
--path /mnt/qb/work/mlcolab/hzhou52/skt_ass13_test \
--dataset ass13 \
# --graph_params [["correct_transition_graph.json", True]]
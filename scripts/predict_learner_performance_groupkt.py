# -*- coding: UTF-8 -*-
import sys
sys.path.append("..")

import os
import argparse
import numpy as np
import datetime
from pathlib import Path

import torch

from knowledge_tracing.data import data_loader
from knowledge_tracing.runner import runner_groupkt, runner_vcl
from knowledge_tracing.utils import utils, arg_parser, logger
from knowledge_tracing.groupkt.groupkt import GroupKT, ContinualGroupKT

def global_parse_args():
    """
    Model-specific arguments are defined in corresponding files.
    """
    parser = argparse.ArgumentParser(description="Global")
    parser.add_argument("--model_name", type=str, help="Choose a model to run.")

    parser.add_argument(
        "--learned_graph",
        type=str,
        default="w_gt",
        help="none: no graph is learner; b_gt: graph with binary edge; w_gt: graph with weighted graph",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="/mnt/qb/work/mlcolab/hzhou52/kt/junyi15/adj.npy",
        help="if the data has ground-truth graph we can compare our inferred graph with ground truth. Note the GT graph is not used for training.",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=10,
        help="number of samples when we use MC for non-analytical solution",
    )
    parser.add_argument('--em_train', type=int, default=0)

    return parser

if __name__ == "__main__":
    # ----- add aditional arguments for this exp. -----
    parser = global_parse_args()
    parser = arg_parser.parse_args(parser)
    global_args, extras = parser.parse_known_args()
    
    global_args.time = datetime.datetime.now().isoformat()
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logs = logger.Logger(global_args)

    # ----- data part -----
    corpus_path = Path(
        global_args.data_dir,
        global_args.dataset,
        "Corpus_{}.pkl".format(global_args.max_step),
    )
    data = data_loader.DataReader(global_args, logs)
    if not corpus_path.exists() or global_args.regenerate_corpus:
        data.create_corpus()
        data.show_columns()
    corpus = data.load_corpus()

    # ----- logger information -----
    log_args = [
        global_args.model_name,
        global_args.dataset,
        str(global_args.random_seed),
    ]
    logs.write_to_log_file("-" * 45 + " BEGIN: " + utils.get_time() + " " + "-" * 45)
    logs.write_to_log_file(utils.format_arg_str(global_args, exclude_lst=[]))

    # ----- random seed -----
    torch.manual_seed(global_args.random_seed)
    torch.cuda.manual_seed(global_args.random_seed)
    np.random.seed(global_args.random_seed)

    # ----- GPU & CUDA -----
    if global_args.device.type != "cpu":
        if global_args.GPU_to_use is not None:
            torch.cuda.set_device(global_args.GPU_to_use)
        global_args.num_GPU = torch.cuda.device_count()
        global_args.batch_size_multiGPU = global_args.batch_size * global_args.num_GPU
    else:
        global_args.num_GPU = None
        global_args.batch_size_multiGPU = global_args.batch_size
    logs.write_to_log_file("# cuda devices: {}".format(torch.cuda.device_count()))

    # ----- Model initialization -----
    if "ls_" or "ln_" in global_args.train_mode:
        num_seq = corpus.n_users
    else:
        num_seq = 1

    adj = np.load(global_args.graph_path)

    if global_args.vcl == 0:
        model = GroupKT(
            mode=global_args.train_mode,
            num_seq=num_seq,
            num_node=1 if not global_args.multi_node else corpus.n_skills,
            nx_graph=None if not global_args.multi_node else adj,
            device=global_args.device,
            args=global_args,
            logs=logs,
        )
    else:
        model = ContinualGroupKT(
            mode=global_args.train_mode,
            num_seq=num_seq,
            num_node=1 if not global_args.multi_node else corpus.n_skills,
            nx_graph=None if not global_args.multi_node else adj,
            device=global_args.device,
            args=global_args,
            logs=logs,
        )
    # shutil.copy(
    #     "/home/mlcolab/hzhou52/knowledge_tracing/models/HSSM.py",
    #     global_args.log_path,
    # )
    # shutil.copy(
    #     "/home/mlcolab/hzhou52/knowledge_tracing/models/learner_hssm_vcl_model.py",
    #     global_args.log_path,
    # )
    # shutil.copy(
    #     "/home/mlcolab/hzhou52/knowledge_tracing/VCLRunner.py", global_args.log_path
    # )
    # shutil.copy(
    #     "/home/mlcolab/hzhou52/knowledge_tracing/KTRunner.py", global_args.log_path
    # )

    if global_args.load > 0:
        model.load_model(model_path=global_args.load_folder)
    logs.write_to_log_file(model)
    model.actions_before_train()

    # Move to current device
    if torch.cuda.is_available():
        if global_args.distributed:
            model, _ = utils.distribute_over_GPUs(
                global_args, model, num_GPU=global_args.num_GPU
            )
        else:
            model = model.to(global_args.device)

    # Running
    if global_args.vcl:
        runner = runner_vcl.VCLRunner(global_args, logs)
    else:
        runner = runner_groupkt.GroupKTRunner(global_args, logs)

runner.train(model, corpus)
logs.write_to_log_file(
    os.linesep + "-" * 45 + " END: " + utils.get_time() + " " + "-" * 45
)

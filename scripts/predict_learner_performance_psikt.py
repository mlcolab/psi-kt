import sys

sys.path.append("..")

import os
import argparse
import datetime
from pathlib import Path

import numpy as np

import torch

from knowledge_tracing.data import data_loader
from knowledge_tracing.runner import runner_psikt, runner_vcl
from knowledge_tracing.utils import utils, arg_parser, logger
from knowledge_tracing.psikt.psikt import AmortizedPSIKT, ContinualPSIKT


def global_parse_args():
    """
    Model-specific arguments are defined in corresponding files.
    """
    parser = argparse.ArgumentParser(description="Global")
    parser.add_argument("--model_name", type=str, help="Choose a model to run.")

    # Data loading parameters
    parser.add_argument(
        "--graph_path",
        type=str,
        default="../kt/junyi15/adj.npy",
        help="if the data has ground-truth graph we can compare our inferred graph with ground truth. Note the GT graph is not used for training.",
    )
    
    # Training parameters
    parser.add_argument(
        "--learned_graph",
        type=str,
        default="w_gt",
        help="none: no graph is learner; b_gt: graph with binary edge; w_gt: graph with weighted graph",
    )
    parser.add_argument(
        "--em_train",
        type=int,
        default=0,
        help="whether to use the EM version of training, i.e., separate the generative model and inference model",
    )
    parser.add_argument(
        "--node_dim",
        type=int,
        default=16,
        help="the dimension of the node embedding of learned concept graph",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=1e3,
        help="number of samples when we use MC for non-analytical solution",
    )
    parser.add_argument(
        "--var_log_max",
        type=int,
        default=10,
        help="the maximum value of the variance of the logit of the Bernoulli distribution",
    )
    parser.add_argument(
        "--num_category",
        type=int,
        default=1e1,
        help="the number of categories in the categorical distribution in GMVAE",
    )
    
    # Loss weight parameters
    parser.add_argument(
        "--s_entropy_weight",
        type=float,
        default=1e-1,
        help="the weight of the entropy of the s",
    )
    parser.add_argument(
        "--z_entropy_weight",
        type=float,
        default=1e-1,
        help="the weight of the entropy of the z",
    )
    parser.add_argument(
        "--s_log_weight",
        type=float,
        default=1,
        help="the weight of the log likelihood of the s",
    )
    parser.add_argument(
        "--z_log_weight",
        type=float,
        default=1,
        help="the weight of the log likelihood of the z",
    )
    parser.add_argument(
        "--y_log_weight",
        type=float,
        default=1,
        help="the weight of the log likelihood of the y",
    )
    parser.add_argument(
        "--sparsity_loss_weight",
        type=float,
        default=1e-12,
        help="the weight of the sparsity loss",
    )
    parser.add_argument(
        "--cat_weight",
        type=float,
        default=10,
        help="the weight of the categorical loss",
    )

    return parser


if __name__ == "__main__":
    # Parse global arguments and additional experiment-specific arguments
    parser = global_parse_args()  # Global argument parser setup
    parser = arg_parser.parse_args(parser)  # Add experiment-specific arguments
    global_args, extras = parser.parse_known_args()  # Parse known and extra arguments

    # Set current time and device (GPU or CPU) in global_args
    global_args.time = datetime.datetime.now().isoformat()
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize logger with global arguments
    logs = logger.Logger(global_args)

    # Setup path for corpus data and initialize data reader
    corpus_path = Path(
        global_args.data_dir,
        global_args.dataset,
        "Corpus_{}.pkl".format(global_args.max_step),
    )
    data = data_loader.DataReader(global_args, logs)
    
    # If corpus does not exist or regeneration is requested, create and save corpus
    if not corpus_path.exists() or global_args.regenerate_corpus:
        data.create_corpus()
        data.show_columns()
    corpus = data.load_corpus(global_args)

    # Log experiment setup information
    log_args = [
        global_args.model_name,
        global_args.dataset,
        str(global_args.random_seed),
    ]
    logs.write_to_log_file("-" * 45 + " BEGIN: " + utils.get_time() + " " + "-" * 45)
    logs.write_to_log_file(utils.format_arg_str(global_args, exclude_lst=[]))

    # Set random seed for reproducibility
    torch.manual_seed(global_args.random_seed)
    torch.cuda.manual_seed(global_args.random_seed)
    np.random.seed(global_args.random_seed)

    # Setup GPU and CUDA if available
    if global_args.device.type != "cpu":
        if global_args.GPU_to_use is not None:
            torch.cuda.set_device(global_args.GPU_to_use)
        global_args.num_GPU = torch.cuda.device_count()  # Number of GPUs available
        global_args.batch_size_multiGPU = global_args.batch_size * global_args.num_GPU
    else:
        global_args.num_GPU = None
        global_args.batch_size_multiGPU = global_args.batch_size
    logs.write_to_log_file("# cuda devices: {}".format(torch.cuda.device_count()))

    # Setup GPU and CUDA if available
    if global_args.device.type != "cpu":
        if global_args.GPU_to_use is not None:
            torch.cuda.set_device(global_args.GPU_to_use)
        global_args.num_GPU = torch.cuda.device_count()
        global_args.batch_size_multiGPU = global_args.batch_size * global_args.num_GPU
    else:
        global_args.num_GPU = None
        global_args.batch_size_multiGPU = global_args.batch_size
    logs.write_to_log_file("# cuda devices: {}".format(torch.cuda.device_count()))

    # Model initialization based on training mode and whether using VCL or not
    num_seq = corpus.n_users if not global_args.num_learner else global_args.num_learner
    adj = np.load(global_args.graph_path)  # Load adjacency matrix for graph

    if global_args.vcl == 0:
        model = AmortizedPSIKT(
            mode=global_args.train_mode,
            num_node=1 if not global_args.multi_node else corpus.n_skills,
            nx_graph=None if not global_args.multi_node else adj,
            device=global_args.device,
            args=global_args,
            logs=logs,
        )
    else:
        model = ContinualPSIKT(
            mode=global_args.train_mode,
            num_node=1 if not global_args.multi_node else corpus.n_skills,
            nx_graph=None if not global_args.multi_node else adj,
            device=global_args.device,
            args=global_args,
            logs=logs,
            num_seq=num_seq,
        )

    # Load model state from file if specified
    if global_args.load > 0:
        model.load_state_dict(torch.load(global_args.load_folder), strict=False)
        # Disable gradient updates for generator parameters in VCL mode
        if global_args.vcl:
            for name, param in model.named_parameters():
                if "gen" in name:
                    param.requires_grad = False

    # Log model details and prepare for training
    logs.write_to_log_file(model)
    model.actions_before_train()  # Perform any pre-training actions

    # Move model to the current device (GPU or CPU)
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
        runner = runner_psikt.PSIKTRunner(global_args, logs)

runner.train(model, corpus)
# Log the end of the experiment
logs.write_to_log_file(
    os.linesep + "-" * 45 + " END: " + utils.get_time() + " " + "-" * 45
)

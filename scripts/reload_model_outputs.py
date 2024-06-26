import sys

sys.path.append("..")

import pickle
import argparse
import datetime
import copy

import numpy as np
from pathlib import Path

import torch

from knowledge_tracing.data import data_loader

from knowledge_tracing.utils import utils, arg_parser, logger
from knowledge_tracing.psikt.psikt import *
from knowledge_tracing.baseline.pykt import qikt, gkt
from knowledge_tracing.baseline import ppe
from knowledge_tracing.baseline.HawkesKT import dktforgetting, hkt
from knowledge_tracing.baseline.EduKTM import dkt, akt
from knowledge_tracing.baseline.halflife_regression import hlr


if __name__ == "__main__":
    # Initial argument parsing to determine the model to use
    parser = argparse.ArgumentParser(description="Global")

    init_parser = argparse.ArgumentParser(description="Model")
    init_parser.add_argument(
        "--model_name", type=str, default="KT", help="Choose a model to run."
    )
    init_args, init_extras = init_parser.parse_known_args()

    # Determine if a PSI model is being used based on the model name
    model_name = init_args.model_name
    if "PSI" not in model_name:
        # Dynamically load the model class based on the model name
        model = eval("{0}.{1}".format(model_name.lower(), model_name.upper()))
        # Parse additional arguments specific to the chosen model
        parser = arg_parser.parse_args(parser)
        parser = model.parse_model_args(parser)
    else:
        # Parse global arguments for PSI models
        parser = arg_parser.parse_args(parser)

    # Finalize argument parsing
    global_args, extras = parser.parse_known_args()
    global_args.model_name = model_name
    # Set execution timestamp and device (GPU or CPU)
    global_args.time = datetime.datetime.now().isoformat()
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize random seed for reproducibility
    torch.manual_seed(global_args.random_seed)
    torch.cuda.manual_seed(global_args.random_seed)
    np.random.seed(global_args.random_seed)

    if "sublearner" not in global_args.dataset:
        # Setup logging
        logs = logger.Logger(global_args)

        # Prepare data and corpus
        corpus_path = Path(
            global_args.data_dir,
            global_args.dataset,
            "Corpus_{}.pkl".format(global_args.max_step),
        )

        # Initialize data reader and possibly create a new corpus if required
        data = data_loader.DataReader(global_args, logs)
        if not corpus_path.exists() or global_args.regenerate_corpus:
            data.create_corpus()
            data.show_columns()
        corpus = data.load_corpus(global_args)

        # ----- logger information -----
        log_args = [
            global_args.model_name,
            global_args.dataset,
            str(global_args.random_seed),
        ]
        # Log experiment setup
        logs.write_to_log_file(
            "-" * 45 + " BEGIN: " + utils.get_time() + " " + "-" * 45
        )
        logs.write_to_log_file(utils.format_arg_str(global_args, exclude_lst=[]))

        # Setup GPU & CUDA if available
        if global_args.device.type != "cpu":
            if global_args.GPU_to_use is not None:
                torch.cuda.set_device(global_args.GPU_to_use)
            global_args.num_GPU = torch.cuda.device_count()
            global_args.batch_size_multiGPU = (
                global_args.batch_size * global_args.num_GPU
            )
        else:
            global_args.num_GPU = None
            global_args.batch_size_multiGPU = global_args.batch_size
        logs.write_to_log_file("# cuda devices: {}".format(torch.cuda.device_count()))

        # Prepare for model loading
        obj_folder = Path(Path(global_args.load_folder).parents[1], "Obj")
        obj_folder.mkdir(exist_ok=True)

        # Initialize the model based on whether it's a PSI model
        if "PSI" not in global_args.model_name:
            cur_model = model(global_args, corpus, logs)
        else:
            # Setup additional arguments for PSI models and initialize
            global_args.node_dim = 16
            global_args.var_log_max = 1
            global_args.num_category = 10
            global_args.learned_graph = "w_gt"
            global_args.num_sample = 50
            global_args.s_entropy_weight = 0.01
            global_args.z_entropy_weight = 0.01
            global_args.s_log_weight = 0.01
            global_args.z_log_weight = 0.01
            global_args.y_log_weight = 1
            cur_model = AmortizedPSIKT(
                mode=global_args.train_mode,
                num_node=corpus.n_skills,
                nx_graph=np.load("../kt/junyi15/adj.npy"),
                device=global_args.device,
                args=global_args,
                logs=logs,
            )
        cur_model.eval()  # Set the model to evaluation mode

        # Move the model to the specified device (GPU/CPU)
        if torch.cuda.is_available():
            if global_args.distributed:
                cur_model, _ = utils.distribute_over_GPUs(
                    global_args, cur_model, num_GPU=global_args.num_GPU
                )
            else:
                cur_model = cur_model.to(global_args.device)

        # Load the model parameters
        model_path = global_args.load_folder
        cur_model.module.load_state_dict(
            torch.load(model_path), strict=False
        )  # Assuming 'model_state_dict' is the key used during saving

        # Perform inference on your input data
        with torch.no_grad():
            train_outputs = []
            test_outputs = []
            epoch_whole_data = copy.deepcopy(corpus.data_df["whole"])
            whole_batches = cur_model.module.prepare_batches(
                corpus, epoch_whole_data, global_args.batch_size, phase="train"
            )
            train_whole_data = copy.deepcopy(corpus.data_df["train"])
            train_batches = cur_model.module.prepare_batches(
                corpus, train_whole_data, global_args.batch_size, phase="train"
            )

            # Execute model forward pass on a subset of the data
            for batch in train_batches[10:30]:
                batch = cur_model.module.batch_to_gpu(batch, global_args.device)
                out_dict = cur_model.module.forward(batch)
                train_outputs.append(out_dict)

            train_path = f"train_{model_path.split('/')[-1][:-3]}.obj"
            test_path = f"test_{model_path.split('/')[-1][:-3]}.obj"

            filehandler = open(Path(obj_folder, train_path), "wb")
            pickle.dump(train_outputs, filehandler)
            filehandler.close()

            # filehandler = open(Path(obj_folder, test_path), "wb")
            # pickle.dump(test_outputs, filehandler)
            # filehandler.close()

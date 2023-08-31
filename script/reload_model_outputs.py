# @Date: 2023/07/25

import sys

sys.path.append("..")

import os
import pickle
import argparse
import numpy as np
import datetime
import copy

import torch

from knowledge_tracing.data import data_loader
from knowledge_tracing.runner import runner_baseline

from knowledge_tracing.utils import utils, arg_parser, logger
from knowledge_tracing.baseline import DKT, DKTForgetting, HKT, AKT, HLR, PPE

if __name__ == "__main__":
    # ----- add aditional arguments for this exp. -----
    parser = argparse.ArgumentParser(description="Global")

    init_parser = argparse.ArgumentParser(description="Model")
    init_parser.add_argument(
        "--model_name", type=str, default="CausalKT", help="Choose a model to run."
    )
    init_args, init_extras = init_parser.parse_known_args()

    model_name = init_args.model_name
    model = eval("{0}.{0}".format(model_name))

    parser = arg_parser.parse_args(parser)
    parser = model.parse_model_args(parser)
    global_args, extras = parser.parse_known_args()
    global_args.model_name = model_name
    global_args.time = datetime.datetime.now().isoformat()
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(5):
        global_args.dataset = "junyi15/sublearner/sublearner_{}".format(i)

        # ----- random seed -----
        torch.manual_seed(global_args.random_seed)
        torch.cuda.manual_seed(global_args.random_seed)
        np.random.seed(global_args.random_seed)

        # ----- log -----
        logs = logger.Logger(global_args)

        # ----- data part -----
        corpus_path = os.path.join(
            global_args.data_dir,
            global_args.dataset,
            "Corpus_{}.pkl".format(global_args.max_step),
        )
        data = data_loader.DataReader(global_args, logs)
        if not os.path.exists(corpus_path) or global_args.regenerate_corpus:
            data.create_corpus()
            data.show_columns()
        corpus = data.load_corpus(global_args)

        # ----- logger information -----
        log_args = [
            global_args.model_name,
            global_args.dataset,
            str(global_args.random_seed),
        ]
        logs.write_to_log_file(
            "-" * 45 + " BEGIN: " + utils.get_time() + " " + "-" * 45
        )
        logs.write_to_log_file(utils.format_arg_str(global_args, exclude_lst=[]))

        # ----- GPU & CUDA -----
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

        runner = runner_baseline.BaselineKTRunner(global_args, logs)

        # Folder containing the model parameter files
        folder = ""
        model_folder = os.path.join(folder, "Model")
        from pathlib import Path

        Path(os.path.join(folder, "Obj")).mkdir(parents=True, exist_ok=True)

        for epoch_file in os.listdir(model_folder):
            if epoch_file.endswith("90.pt"):
                # Load model parameters

                cur_model = model(global_args, corpus, logs)
                cur_model.eval()
                # Move to current device
                if torch.cuda.is_available():
                    if global_args.distributed:
                        cur_model, _ = utils.distribute_over_GPUs(
                            global_args, cur_model, num_GPU=global_args.num_GPU
                        )
                    else:
                        cur_model = cur_model.to(global_args.device)

                model_path = os.path.join(model_folder, epoch_file)
                checkpoint = torch.load(model_path)
                cur_model.load_state_dict(
                    checkpoint, strict=False
                )  # Assuming 'model_state_dict' is the key used during saving

                # Perform inference on your input data
                with torch.no_grad():
                    outputs = []
                    epoch_train_data = copy.deepcopy(
                        corpus.data_df["train"][: global_args.overfit]
                    )
                    # Return a random sample of items from an axis of object.
                    epoch_train_data = epoch_train_data.sample(frac=1).reset_index(
                        drop=True
                    )
                    train_batches = cur_model.module.prepare_batches(
                        corpus, epoch_train_data, global_args.batch_size, phase="train"
                    )

                    for (
                        batch
                    ) in (
                        train_batches
                    ):  # tqdm(train_batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
                        batch = cur_model.module.batch_to_gpu(batch, global_args.device)
                        out_dict = cur_model.module.forward(batch)
                        outputs.append(out_dict)

                    save_path = f"predictions_{epoch_file[:-3]}.obj"

                    save_path = os.path.join(folder, "Obj", save_path)
                    filehandler = open(save_path, "wb")
                    pickle.dump(outputs, filehandler)
                    filehandler.close()

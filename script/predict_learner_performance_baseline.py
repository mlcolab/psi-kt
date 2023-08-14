# @Date: 2023/07/25

import sys
sys.path.append('..')

import os 
import pickle
import argparse
import numpy as np
import datetime

import torch

from knowledge_tracing.data import data_loader
from knowledge_tracing.runner import runner_baseline
# from VCLRunner_baseline import BaselineVCLRunner
# from FTRunner_baseline import FTRunner
from knowledge_tracing.utils import utils, arg_parser, logger
from knowledge_tracing.baseline import DKT, DKTForgetting, HKT, AKT, HLR, PPE

# TODO: this is duplicate with the one in exp

if __name__ == '__main__':

    # ----- add aditional arguments for this exp. -----
    parser = argparse.ArgumentParser(description='Global')

    parser.add_argument('--finetune', type=int, default=0)
    parser.add_argument('--vcl_predict_step', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--multi_node', type=int, default=1)
    
    parser.add_argument('--id', type=int, default=0)
    
    # Define an argument parser for the model name.
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='CausalKT', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    
    # Get the model name from the command-line arguments.
    # Evaluate the model name string as a Python expression to get the corresponding model class.
    model_name = init_args.model_name
    model = eval('{0}.{0}'.format(model_name))
    
    # ----- args -----
    # reference:
    # # https://docs.python.org/3/library/argparse.html?highlight=parse_known_args#argparse.ArgumentParser.parse_known_args
    parser = arg_parser.parse_args(parser)
    parser = model.parse_model_args(parser)
    global_args, extras = parser.parse_known_args() 
    global_args.model_name = model_name
    global_args.time = datetime.datetime.now().isoformat()
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ----- random seed -----
    torch.manual_seed(global_args.random_seed)
    torch.cuda.manual_seed(global_args.random_seed)
    np.random.seed(global_args.random_seed)
    
    # ----- log -----
    logs = logger.Logger(global_args)
    
    # ----- data part -----
    corpus_path = os.path.join(global_args.data_dir, global_args.dataset, 'Corpus_{}.pkl'.format(global_args.max_step))
    data = data_loader.DataReader(global_args, logs)
    if not os.path.exists(corpus_path) or global_args.regenerate_corpus:
        data.create_corpus()
        data.show_columns() 
    corpus = data.load_corpus(global_args) 

    # ----- logger information -----
    log_args = [global_args.model_name, global_args.dataset, str(global_args.random_seed)]
    logs.write_to_log_file('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    logs.write_to_log_file(utils.format_arg_str(global_args, exclude_lst=[]))
    
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
    if 'ls_' or 'ln_' in global_args.train_mode:
        num_seq = corpus.n_users
    else: num_seq = 1
    
    model = model(global_args, corpus, logs)
        
    if global_args.load > 0:
        model.load_state_dict(torch.load(global_args.load_folder), strict=False)
    logs.write_to_log_file(model)
    model.actions_before_train()
    
    # Move to current device
    if torch.cuda.is_available():
        if global_args.distributed:
            model, _ = utils.distribute_over_GPUs(global_args, model, num_GPU=global_args.num_GPU)
        else: 
            model = model.to(global_args.device)
            
    # Running
    # TODO: modify the runner
    if global_args.vcl: 
        runner = VCLRunner(global_args, logs)
    elif global_args.finetune:
        runner = FTRunner(global_args, logs)
    else:
        runner = runner_baseline.BaselineKTRunner(global_args, logs)

runner.train(model, corpus)
model.module.actions_after_train()
logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
    

# -*- coding: UTF-8 -*-
import sys
sys.path.append('..')

import os 
import time
import pickle
import argparse
import numpy as np
import datetime
import shutil

import torch
from torch.autograd import profiler

from data import data_loader
from KTRunner import KTRunner
from VCLRunner_baseline import VCLRunner
from utils import utils, arg_parser, logger
from models import DKT, DKTForgetting, HKT

import ipdb

if __name__ == '__main__':

    # ----- add aditional arguments for this exp. -----
    parser = argparse.ArgumentParser(description='Global')
    # debug
    parser.add_argument('--alpha_minimum', type=float, default=100)
    parser.add_argument('--learned_graph', type=str, default='w_gt')
    parser.add_argument('--vcl', type=int, default=1)

    parser.add_argument('--train_time_ratio', type=float, default=0.5, help='')
    parser.add_argument('--test_time_ratio', type=float, default=0.5, help='')

    # Training options 
    parser.add_argument('--multi_node', type=int, default=0)
    parser.add_argument('--num_sample', type=int, default=100)
    
    # Define an argument parser for the model name.
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='CausalKT', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    
    # Get the model name from the command-line arguments.
    # Evaluate the model name string as a Python expression to get the corresponding model class.
    model_name = init_args.model_name
    model = eval('{0}.{0}'.format(model_name))
    
    # ----- args -----
    parser = arg_parser.parse_args(parser)
    parser = model.parse_model_args(parser)
    global_args, extras = parser.parse_known_args() # https://docs.python.org/3/library/argparse.html?highlight=parse_known_args#argparse.ArgumentParser.parse_known_args
    global_args.model_name = model_name
    global_args.time = datetime.datetime.now().isoformat()
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ----- log -----
    logs = logger.Logger(global_args)
    
    # ----- data part -----
    corpus_path = os.path.join(global_args.data_dir, global_args.dataset, 'Corpus_{}.pkl'.format(global_args.max_step))
    if not os.path.exists(corpus_path) or global_args.regenerate_corpus:
        data = data_loader.DataReader(global_args, logs)
        data.show_columns() 
        logs.write_to_log_file('Save corpus to {}'.format(corpus_path))
        pickle.dump(data, open(corpus_path, 'wb'))
    corpus = utils.load_corpus(logs, global_args) 
    
    # ----- logger information -----
    log_args = [global_args.model_name, global_args.dataset, str(global_args.random_seed)]
    logs.write_to_log_file('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
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
    # ipdb.set_trace()
    
    # ----- Model initialization -----
    if 'ls_' or 'ln_' in global_args.train_mode:
        num_seq = corpus.n_users
    else: num_seq = 1
    
    model = model(global_args, corpus, logs)
    # TODO for debugging
    shutil.copy('/home/mlcolab/hzhou52/knowledge_tracing/models/baseline/{}.py'.format(model_name),
                    global_args.log_path)
    shutil.copy('/home/mlcolab/hzhou52/knowledge_tracing/exp/exp_baseline.py',
                    global_args.log_path)
    shutil.copy('/home/mlcolab/hzhou52/knowledge_tracing/VCLRunner.py',
                    global_args.log_path)
    shutil.copy('/home/mlcolab/hzhou52/knowledge_tracing/KTRunner.py',
                    global_args.log_path)
        
    if global_args.load > 0:
        model.load_model(model_path=global_args.load_folder)
    
    logs.write_to_log_file(model)
    model.actions_before_train()
    
    # Move to current device
    if torch.cuda.is_available():
        if global_args.distributed:
            model, _ = utils.distribute_over_GPUs(global_args, model, num_GPU=global_args.num_GPU)
        else: 
            model = model.to(global_args.device)
            
    # Running
    if global_args.vcl:
        runner = VCLRunner(global_args, logs)
    else:
        runner = KTRunner(global_args, logs)

# runner = VCLRunner(global_args, logs)
runner.train(model, corpus)
# logs.write_to_log_file('\nTest After Training: ' + runner._print_res(model, corpus))

# model.module.actions_after_train()
logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
    

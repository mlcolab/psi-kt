# -*- coding: UTF-8 -*-
import sys
sys.path.append('..')

import os 
import time
import pickle
import argparse
import numpy as np
import torch
import datetime
import builtins

from data import data_loader
from models import *
from KTRunner import *
from VCLRunner import *
from utils import utils, arg_parser, logger

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import ipdb
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


def main(args, model, logs, fun=None):
    logs.write_to_log_file('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    logs.write_to_log_file(utils.format_arg_str(args, exclude_lst=exclude))
    
    # Random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Load data
    corpus = load_corpus(logs, args) 
    
    # GPU & CUDA
    if args.device.type != "cpu":
        if args.GPU_to_use is not None:
            torch.cuda.set_device(args.GPU_to_use)
        args.num_GPU = torch.cuda.device_count()
        args.batch_size_multiGPU = args.batch_size * args.num_GPU
    else:
        args.num_GPU = None
        args.batch_size_multiGPU = args.batch_size
    logs.write_to_log_file("# cuda devices: {}".format(torch.cuda.device_count()))

    # Running
    # runner = KTRunner(args, logs)
    runner = VCLRunner(args, logs)
    
    if args.load > 0:
        model.load_model()
        
    model = model(args, corpus, logs)
    logs.write_to_log_file(model)
    model = model.double()
    model.apply(model.init_weights)
    model.actions_before_train()

    if torch.cuda.is_available():
        if args.distributed:
            model, _ = utils.distribute_over_GPUs(args, model, num_GPU=args.num_GPU)
        else: 
            model = model.to(args.device)

    runner.train(model, corpus)
    logs.write_to_log_file('\nTest After Training: ' + runner.print_res(model, corpus))

    model.actions_after_train()
    logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


def load_corpus(logs, args):
    '''
    agrs: the global arguments
    Load corupus from the corpus path, and split the data into k folds. 
    '''

    corpus_path = os.path.join(args.data_dir, args.dataset, 'Corpus_{}.pkl'.format(args.max_step))
    logs.write_to_log_file('Load corpus from {}'.format(corpus_path))
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    corpus.gen_fold_data(args.fold)
    logs.write_to_log_file('# Train: {}, # Dev: {}, # Test: {}'.format(
            len(corpus.data_df['train']), len(corpus.data_df['dev']), len(corpus.data_df['test'])
        ))
    return corpus


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='CausalKT', help='Choose a model to run.')

    init_args, init_extras = init_parser.parse_known_args()
    model_name = init_args.model_name
    model = eval('{0}.{0}'.format(model_name))

    # ----- args -----
    parser = argparse.ArgumentParser(description='Global')
    parser = arg_parser.parse_args(parser)
    parser = model.parse_model_args(parser)
    global_args, extras = parser.parse_known_args() 
    global_args.model_name = model_name
    global_args.time = datetime.datetime.now().isoformat()
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logs = logger.Logger(global_args)
    
    # ----- data part -----
    corpus_path = os.path.join(global_args.data_dir, global_args.dataset, 'Corpus_{}.pkl'.format(global_args.max_step))
    if not os.path.exists(corpus_path) or global_args.regenerate_corpus:
        data = data_loader.DataReader(global_args, logs)
        data.gen_fold_data(k=0)
        data.show_columns() 
        logs.write_to_log_file('Save corpus to {}'.format(corpus_path))
        pickle.dump(data, open(corpus_path, 'wb'))

    # ----- logger information -----
    log_args = [init_args.model_name, global_args.dataset, str(global_args.random_seed)]
    for arg in ['lr', 'l2', 'fold'] + model.extra_log_args:
        log_args.append(arg + '=' + str(eval('global_args.' + arg))) 

    main(global_args, model, logs)

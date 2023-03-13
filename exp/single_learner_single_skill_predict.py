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
from models.learner_model import HLR, PPE, VanillaOU, GraphOU
from models.new_learner_model import *

# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

import ipdb
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


def load_corpus(logs, args):
    '''
    Load corupus from the corpus path, and split the data into k folds. 
    Args:
        logs:
        args:
    '''
    corpus_path = os.path.join(args.data_dir, args.dataset, 'Corpus_{}.pkl'.format(args.max_step))
    logs.write_to_log_file('Load corpus from {}'.format(corpus_path))
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    
    if 'split_learner' in args.train_mode:
        corpus.gen_fold_data(args.fold)
        logs.write_to_log_file('# Training mode splits LEARNER')
        logs.write_to_log_file('# Train: {}, # Dev: {}, # Test: {}'.format(
                len(corpus.data_df['train']), len(corpus.data_df['dev']), len(corpus.data_df['test'])
            ))
    if 'split_time' in args.train_mode:
        corpus.gen_time_split_data(args.train_time_ratio, args.test_time_ratio)
        logs.write_to_log_file('# Training mode splits TIME')
        logs.write_to_log_file('# Train: {}, # Dev: {}, # Test: {}'.format(
                len(corpus.data_df['train']), len(corpus.data_df['dev']), len(corpus.data_df['test'])
            ))

    return corpus


if __name__ == '__main__':
    # ----- add aditional arguments for this exp. -----
    parser = argparse.ArgumentParser(description='Global')
    # init_parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--model_name', type=str, help='Choose a model to run.')
    
    # Training options
    parser.add_argument(
        '--train_mode', type=str, default='train_split_time', 
        help= 'simple_split_time' + 'simple_split_learner' 
        + 'ls_split_time' 
        + 'ns_split_time' + 'ns_split_learner'
        + 'ln_split_time', # ln can be split to time+learner
    )
    parser.add_argument('--multi_node', type=int, default=0)
    parser.add_argument('--train_time_ratio', type=float, default=0.5, help='')
    parser.add_argument('--test_time_ratio', type=float, default=0.4, help='')
    
    # general
    parser.add_argument('--num_node', type=int, default=1, help='')
    parser.add_argument('--num_seq', type=int, default=1, help='')
    
    # HLR 
    parser.add_argument('--base', type=float, default=2., help='')
    # OU 
    # PPE
    parser.add_argument('--ppe_lr', type=float, default=0.2, help='')
    # SSSM
    parser.add_argument('--hidden_dim', type=int, default=8, help='')
    
    parser.add_argument('--graph_path', type=str, default='/mnt/qb/work/mlcolab/hzhou52/kt/junyi/adj.npy')
    
    parser = arg_parser.parse_args(parser)
    
    global_args, extras = parser.parse_known_args() 
    global_args.time = datetime.datetime.now().isoformat()
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # global_args.device = torch.device('cpu')
    
    logs = logger.Logger(global_args)
    
    # ----- data part -----
    corpus_path = os.path.join(global_args.data_dir, global_args.dataset, 'Corpus_{}.pkl'.format(global_args.max_step))
    if not os.path.exists(corpus_path) or global_args.regenerate_corpus:
        data = data_loader.DataReader(global_args, logs)
        # data.gen_fold_data(k=0)
        data.show_columns() 
        logs.write_to_log_file('Save corpus to {}'.format(corpus_path))
        pickle.dump(data, open(corpus_path, 'wb'))
    # Load data
    corpus = load_corpus(logs, global_args) 
    
    # ----- logger information -----
    log_args = [global_args.model_name, global_args.dataset, str(global_args.random_seed)]

    logs.write_to_log_file('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    logs.write_to_log_file(utils.format_arg_str(global_args, exclude_lst=exclude))
    
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
    if 'ls_' or 'ln_' in global_args.train_mode:
        num_seq = corpus.n_users
    else: num_seq = 1
    
    adj = np.load(global_args.graph_path)
    
    if global_args.model_name == 'HLR':
        model = HLR(
            mode=global_args.train_mode, 
            num_seq=num_seq,
            num_node=1 if not global_args.multi_node else corpus.n_skills,
            nx_graph=None if not global_args.multi_node else adj,
            device=global_args.device, 
            logs=logs,
        )
        
    elif global_args.model_name == 'VanillaOU':
        model = VanillaOU(
            mode=global_args.train_mode, 
            num_seq=num_seq,
            num_node=1 if not global_args.multi_node else corpus.n_skills,
            nx_graph=None if not global_args.multi_node else adj,
            device=global_args.device, 
            logs=logs
        )
        
    elif global_args.model_name == 'GraphOU':
        model = GraphOU(
            mode=global_args.train_mode, 
            num_seq=num_seq,
            num_node=1 if not global_args.multi_node else corpus.n_skills,
            nx_graph=None if not global_args.multi_node else adj,
            device=global_args.device, 
            logs=logs
        )
        
    elif global_args.model_name == 'PPE':
        model = PPE(
            mode=global_args.train_mode, 
            lr=global_args.ppe_lr,
            num_seq=num_seq,
            device=global_args.device, 
            logs=logs
        )
    elif global_args.model_name == 'SwitchingNLDS':
        model = create_model(
            dim_s=3,
            dim_z=1,
            dim_y=1,
            device=global_args.device, 
            args=global_args,
            logs=logs,   
        )

        
    if global_args.load > 0:
        model.load_model(model_path=global_args.load_folder)
    logs.write_to_log_file(model)
    # model = model.double() # ??? when to use double
    # model.apply(model.init_weights)
    model.actions_before_train()
    
    
    # Move to current device
    if torch.cuda.is_available():
        if global_args.distributed:
            model, _ = utils.distribute_over_GPUs(global_args, model, num_GPU=global_args.num_GPU)
        else: 
            model = model.to(global_args.device)
            
    # Running
    runner = KTRunner(global_args, logs)
    # runner = VCLRunner(global_args, logs)
    runner.train(model, corpus)
    # logs.write_to_log_file('\nTest After Training: ' + runner._print_res(model, corpus))

    # model.module.actions_after_train()
    logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
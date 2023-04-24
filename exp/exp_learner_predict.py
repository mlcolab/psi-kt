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
# from models import *
# from KTRunner_test import *
import KTRunner
import VCLRunner
from utils import utils, arg_parser, logger
from models.learner_model import PPE, VanillaOU, GraphOU
from models.HSSM import HSSM, GraphHSSM
from models.learner_hssm_vcl_model import GraphContinualHSSM

# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

import ipdb
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html



if __name__ == '__main__':

    # ----- add aditional arguments for this exp. -----
    parser = argparse.ArgumentParser(description='Global')
    parser.add_argument('--model_name', type=str, help='Choose a model to run.')
    
    # debug
    parser.add_argument('--alpha_minimum', type=float, default=100)
    parser.add_argument('--learned_graph', type=str, default='w_gt')

    # Training options
    parser.add_argument('--multi_node', type=int, default=0)
    parser.add_argument('--graph_path', type=str, default='/mnt/qb/work/mlcolab/hzhou52/kt/junyi15/adj.npy')
    parser.add_argument('--num_sample', type=int, default=10)
    
    parser = arg_parser.parse_args(parser)
    
    global_args, extras = parser.parse_known_args() 
    global_args.time = datetime.datetime.now().isoformat()
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
            num_seq=num_seq,
            num_node=1 if not global_args.multi_node else corpus.n_skills,
            nx_graph=None if not global_args.multi_node else adj,
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
    
    # elif global_args.model_name == 'VanillaHSSM':
    #     model = VanillaHSSM( # TestHierachicalSSM(
    #         mode=global_args.train_mode, 
    #         num_seq=num_seq,
    #         num_node=1 if not global_args.multi_node else corpus.n_skills,
    #         nx_graph=None if not global_args.multi_node else adj,
    #         device=global_args.device, 
    #         args=global_args,
    #         logs=logs,
    #     )
    #     shutil.copy('/home/mlcolab/hzhou52/knowledge_tracing/models/learner_hssm_model.py',
    #                     global_args.log_path)
    elif global_args.model_name == 'TestHierachicalSSM' or global_args.model_name == 'GraphHSSM':
        
        if global_args.vcl == 0:
            model = GraphHSSM( # TestHierachicalSSM(
                mode=global_args.train_mode, 
                num_seq=num_seq,
                num_node=1 if not global_args.multi_node else corpus.n_skills,
                nx_graph=None if not global_args.multi_node else adj,
                device=global_args.device, 
                args=global_args,
                logs=logs,
            )
        else:
            model = GraphContinualHSSM(
                mode=global_args.train_mode, 
                num_seq=num_seq,
                num_node=1 if not global_args.multi_node else corpus.n_skills,
                nx_graph=None if not global_args.multi_node else adj,
                device=global_args.device, 
                args=global_args,
                logs=logs,
            )
        shutil.copy('/home/mlcolab/hzhou52/knowledge_tracing/models/HSSM.py',
                        global_args.log_path)
        shutil.copy('/home/mlcolab/hzhou52/knowledge_tracing/models/learner_hssm_vcl_model.py',
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
        runner = VCLRunner.VCLRunner(global_args, logs)
    else:
        runner = KTRunner.KTRunner(global_args, logs)

runner.train(model, corpus)
# logs.write_to_log_file('\nTest After Training: ' + runner._print_res(model, corpus))

# model.module.actions_after_train()
logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
    

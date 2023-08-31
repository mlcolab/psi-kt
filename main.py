# -*- coding: UTF-8 -*-

import os 
import time
import pickle
import argparse
import numpy as np
import torch
import datetime
import shutil
import inspect

from data import data_loader
from models import *
from KTRunner import *
# from VCLRunner import *
from utils import utils, arg_parser, logger

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import ipdb
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


def main(args, 
         model, 
         logs, 
         fun=None):
    '''
    Args:
        args:
        model:
        logs:
        fun
    '''
    # # DDP setting
    # if "WORLD_SIZE" in os.environ:
    #     args.world_size = int(os.environ["WORLD_SIZE"])
    # args.distributed = args.world_size > 1
    # ngpus_per_node = torch.cuda.device_count()
    # if args.distributed:
    #     if args.local_rank != -1: # for torch.distributed.launch
    #         args.rank = args.local_rank
    #         args.gpu = args.local_rank
    #     elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
    #         args.rank = int(os.environ['SLURM_PROCID']) # The values of SLURM_PROCID range from 0 to the number of running processes minus 1.
    #         args.gpu = args.rank % torch.cuda.device_count()
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)
    # # suppress printing if not on master gpu
    # if args.rank != 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass

    # Running
    runner = KTRunner(args, logs)
    # runner = VCLRunner(args, logs)

    # if args.distributed:
    #     mp.spawn(fun, nprocs=args.num_GPU, args=(args, corpus, runner, model, logs)) 
    # else:
    #     test_train(args.device[0], args, corpus, runner, model, logs)
    
    # -- load model from existing path
    if args.load > 0:
        model.load_model()
        
    # logs.write_to_log_file('Test Before Training: ' + runner._print_res(model, corpus))
    fun(args.device, args, corpus, runner, model, logs)
    logs.write_to_log_file('\nTest After Training: ' + runner._print_res(model, corpus))

    model.actions_after_train()
    logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


def distributed_train(gpu, args, corpus, runner, model, logs):
    '''
    Args:
        args:    global arguments
        corpus:  loaded training data
        runner:  KTRunner instance for training, testing and validation
        model:   defined model instance for training
        logs:    Logger instance for logging information
    '''
    # # Define multiple gpus
    # dev0 = (gpu*2) %  args.world_size
    # dev1 = (gpu*2+1) % args.world_size
    # print("Start the initialize of the process group")
    # dist.init_process_group(backend="nccl", world_size=args.world_size, rank=gpu, init_method="file:///distributed_test",)   
    # # NCCL_DEBUG=INFO
    # print("Finish the initialization of the process group")
    
    # Define model
    model = model(args, corpus, logs)#, dev0, dev1)
    logs.write_to_log_file(model)
    model = model.double()
    model.apply(model.init_weights)
    model.actions_before_train()

    if torch.cuda.is_available():
        if args.distributed:
            model, _ = utils.distribute_over_GPUs(args, model, num_GPU=args.num_GPU)
        else: 
            model = model.to(args.device)

    # # DPP training
    # torch.cuda.set_device(gpu)
    # model.cuda(gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model)# , device_ids=[gpu])
    # print(f"Running basic DDP example on rank {args.rank}.")
    # # TODO DEBUG
    # for name, param in model.module.named_parameters():
    #     if param.grad != None:
    #         print(name, torch.isfinite(param.grad).all())
    #     else: print(name)
    # for name, param in model.named_parameters():
    #     if param.grad != None:
    #         print(name, torch.isfinite(param.grad).all())
    #     else: print(name)
    #     if param.requires_grad:
    #         print('Grad:', name)

    runner.train(model, corpus)


if __name__ == '__main__':
    # Define an argument parser for the model name.
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='CausalKT', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    
    # Get the model name from the command-line arguments.
    # Evaluate the model name string as a Python expression to get the corresponding model class.
    model_name = init_args.model_name
    model = eval('{0}.{0}'.format(model_name))
    
    # ----- args -----
    parser = argparse.ArgumentParser(description='Global')
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
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '44144'

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
        
    # Define model
    model = model(global_args, corpus, logs)#, dev0, dev1)
    logs.write_to_log_file(model)
    model = model.double()
    model.apply(model.init_weights)
    model.actions_before_train()
    
    
    
    shutil.copy(inspect.getmodule(model).__file__, global_args.log_path)










    main(global_args, model, logs, distributed_train)

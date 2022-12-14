# -*- coding: UTF-8 -*-

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
from utils import utils, arg_parser, logger

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import ipdb
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


def main(args, model, logs, fun=None):
    # logging.info(msg, *args, **kwargs) Logs a message with level INFO on the root logger. 
    logs.write_to_log_file('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    logs.write_to_log_file(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Load data
    corpus = load_corpus(logs, args) #junyi train 178276 dev 19808 test 49522

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

    # ipdb.set_trace()
    # model_path = '/home/mlcolab/hzhou52/mykt/logs/CausalKT/2022-10-09T22:04:22.946163_whole_problem_bias_data_assistment12_overfit_0/Model_0.pt'

    # if args.distributed:
    #     mp.spawn(fun, nprocs=args.num_GPU, args=(args, corpus, runner, model, logs)) 
    # else:
    #     test_train(args.device[0], args, corpus, runner, model, logs)
    if args.load > 0:
        model.load_model()
    # logs.write_to_log_file('Test Before Training: ' + runner.print_res(model, corpus))
    test_train(args.device, args, corpus, runner, model, logs)
    logs.write_to_log_file('\nTest After Training: ' + runner.print_res(model, corpus))

    model.actions_after_train()
    logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


def test_train(gpu, args, corpus, runner, model, logs):
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
    model, _ = utils.distribute_over_GPUs(args, model, num_GPU=args.num_GPU)
    # model = model.to(args.device)

    # torch.cuda.set_device(gpu)
    # model.cuda(gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model)# , device_ids=[gpu])

    # DPP training
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

def load_corpus(logs, args):
    corpus_path = os.path.join(args.data_dir, args.dataset, 'Corpus_{}.pkl'.format(args.max_step))
    if os.path.exists(corpus_path) and not global_args.regenerate_corpus:
        logs.write_to_log_file('Load corpus from {}'.format(corpus_path))
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
    else:
        t1 = time.time()
        corpus = reader_name(args)
        logs.write_to_log_file('Done! [{:<.2f} s]'.format(time.time() - t1))
        logs.write_to_log_file('Save corpus to {}'.format(corpus_path))
        with open(corpus_path, 'wb') as f:
            pickle.dump(corpus, f)
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
    global_args, extras = parser.parse_known_args() # https://docs.python.org/3/library/argparse.html?highlight=parse_known_args#argparse.ArgumentParser.parse_known_args
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
        ipdb.set_trace()
        logs.write_to_log_file('Save corpus to {}'.format(corpus_path))
        pickle.dump(data, open(corpus_path, 'wb'))

    # Logging configuration
    log_args = [init_args.model_name, global_args.dataset, str(global_args.random_seed)]
    for arg in ['lr', 'l2', 'fold'] + model.extra_log_args:
        log_args.append(arg + '=' + str(eval('global_args.' + arg))) 
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '44144'

    main(global_args, model, logs, test_train)

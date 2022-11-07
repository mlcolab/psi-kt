# -*- coding: UTF-8 -*-

import os
import sys
import time
import pickle
import logging
import argparse
import numpy as np
import torch
import datetime

from data import data_loader
from models import *
from KTRunner import *
from utils import utils, arg_parser, logger

import ipdb



def main(args, model, logs):
    # logging.info(msg, *args, **kwargs) Logs a message with level INFO on the root logger. 
    logs.write_to_log_file('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    logs.write_to_log_file(utils.format_arg_str(args, exclude_lst=exclude))
    
    # Random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # GPU & CUDA
    if args.device.type != "cpu":
        if args.GPU_to_use is not None:
            torch.cuda.set_device(args.GPU_to_use)
        torch.cuda.manual_seed(args.random_seed)
        args.num_GPU = torch.cuda.device_count()
        args.batch_size_multiGPU = args.batch_size * args.num_GPU
    else:
        args.num_GPU = None
        args.batch_size_multiGPU = args.batch_size
    logs.write_to_log_file("# cuda devices: {}".format(torch.cuda.device_count()))

    # Load data
    corpus = load_corpus(logs, args) #junyi train 178276 dev 19808 test 49522
    
    # Define model
    model = model(args, corpus, logs)
    logs.write_to_log_file(model)
    model = model.double()
    model.apply(model.init_weights)
    model.actions_before_train()
    model, _ = utils.distribute_over_GPUs(args, model, num_GPU=args.num_GPU)
    # model = model.to(args.device)
        
    # Define
    runner = KTRunner(args, logs)

    # logs.write_to_log_file('Test Before Training: ' + runner.print_res(model, corpus))
    # if args.load > 0:
    #     model.load_model()
    # ipdb.set_trace()
    # model_path = '/home/mlcolab/hzhou52/mykt/logs/CausalKT/2022-10-09T22:04:22.946163_whole_problem_bias_data_assistment12_overfit_0/Model_0.pt'
    if args.train > 0:
        runner.train(model, corpus)
    logs.write_to_log_file('\nTest After Training: ' + runner.print_res(model, corpus))

    model.actions_after_train()
    logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


def load_corpus(logs, args):
    corpus_path = os.path.join(args.data_dir, args.dataset, 'Corpus_{}.pkl'.format(args.max_step))
    if not args.regenerate and os.path.exists(corpus_path):
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
    # print('device count!')
    # print(torch.cuda.device_count())
    global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logs = logger.Logger(global_args)
    # ipdb.set_trace()
    # # ----- data part -----
    # data = data_loader.DataReader(global_args, logs)
    # data.gen_fold_data(k=0)
    # data.show_columns()

    # corpus_path = os.path.join(global_args.data_dir, global_args.dataset, 'Corpus_{}.pkl'.format(global_args.max_step))
    # logs.write_to_log_file('Save corpus to {}'.format(corpus_path))
    # pickle.dump(data, open(corpus_path, 'wb'))

    # Logging configuration
    log_args = [init_args.model_name, global_args.dataset, str(global_args.random_seed)]
    for arg in ['lr', 'l2', 'fold'] + model.extra_log_args:
        log_args.append(arg + '=' + str(eval('global_args.' + arg))) 
        
    main(global_args, model, logs)

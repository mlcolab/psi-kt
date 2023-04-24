# -*- coding: UTF-8 -*-
import sys
sys.path.append('..')

import os 
import pickle
import argparse
import numpy as np
import copy
import torch
import datetime
import ipdb

pt_path = '/mnt/qb/work/mlcolab/hzhou52/kt/logs/GraphHSSM/assistment17/multi_skill/2023-04-24T17:42:47.120492__overfit_64/Model/Model_50_0.pt'
model = torch.load(pt_path)
ipdb.set_trace()


outdict_path = ''
with open(outdict_path, 'rb') as f:
        outdict = pickle.load(f)
ipdb.set_trace()











# from data import data_loader
# # from models import *
# from KTRunner import *
# from utils import utils, arg_parser, logger
# from models.new_learner_model_test import *
# from models.learner_model import *


# # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


# def load_corpus(logs, args):
#     '''
#     Load corupus from the corpus path, and split the data into k folds. 
#     Args:
#         logs:
#         args:
#     '''
#     corpus_path = os.path.join(args.data_dir, args.dataset, 'Corpus_{}.pkl'.format(args.max_step))
#     logs.write_to_log_file('Load corpus from {}'.format(corpus_path))
#     with open(corpus_path, 'rb') as f:
#         corpus = pickle.load(f)
    
#     if 'split_learner' in args.train_mode:
#         corpus.gen_fold_data(args.fold)
#         logs.write_to_log_file('# Training mode splits LEARNER')
#         logs.write_to_log_file('# Train: {}, # Dev: {}, # Test: {}'.format(
#                 len(corpus.data_df['train']), len(corpus.data_df['dev']), len(corpus.data_df['test'])
#             ))
#     if 'split_time' in args.train_mode:
#         corpus.gen_time_split_data(args.train_time_ratio, args.test_time_ratio)
#         logs.write_to_log_file('# Training mode splits TIME')
#         logs.write_to_log_file('# Train: {}, # Dev: {}, # Test: {}'.format(
#                 len(corpus.data_df['train']), len(corpus.data_df['dev']), len(corpus.data_df['test'])
#             ))

#     return corpus


# if __name__ == '__main__':
#     # ----- add aditional arguments for this exp. -----
#     parser = argparse.ArgumentParser(description='Global')
#     parser.add_argument('--model_name', type=str, help='Choose a model to run.')
    
#     # Training options
#     parser.add_argument(
#         '--train_mode', type=str, default='train_split_time', 
#         help= 'simple_split_time' + 'simple_split_learner' 
#         + 'ls_split_time' 
#         + 'ns_split_time' + 'ns_split_learner'
#         + 'ln_split_time', # ln can be split to time+learner
#     )
#     parser.add_argument('--multi_node', type=int, default=0)
#     parser.add_argument('--train_time_ratio', type=float, default=0.5, help='')
#     parser.add_argument('--test_time_ratio', type=float, default=0.4, help='')
    
#     # general
#     parser.add_argument('--num_node', type=int, default=1, help='')
#     parser.add_argument('--num_seq', type=int, default=1, help='')
    
#     parser.add_argument('--graph_path', type=str, default='/mnt/qb/work/mlcolab/hzhou52/kt/junyi/adj.npy')
    
    
#     parser = arg_parser.parse_args(parser)
    
#     global_args, extras = parser.parse_known_args() 
#     global_args.time = datetime.datetime.now().isoformat()
#     global_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     logs = logger.Logger(global_args)
    
#     # ----- data part -----
#     corpus_path = os.path.join(global_args.data_dir, global_args.dataset, 'Corpus_{}.pkl'.format(global_args.max_step))
#     corpus = load_corpus(logs, global_args) 
    
#     # ----- logger information -----
#     log_args = [global_args.model_name, global_args.dataset, str(global_args.random_seed)]

#     logs.write_to_log_file('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
#     exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
#                'regenerate', 'sep', 'train', 'verbose']
#     logs.write_to_log_file(utils.format_arg_str(global_args, exclude_lst=exclude))
    
#     # ----- random seed -----
#     torch.manual_seed(global_args.random_seed)
#     torch.cuda.manual_seed(global_args.random_seed)
#     np.random.seed(global_args.random_seed)
    
#     # ----- GPU & CUDA -----
#     if global_args.device.type != "cpu":
#         if global_args.GPU_to_use is not None:
#             torch.cuda.set_device(global_args.GPU_to_use)
#         global_args.num_GPU = torch.cuda.device_count()
#         global_args.batch_size_multiGPU = global_args.batch_size * global_args.num_GPU
#     else:
#         global_args.num_GPU = None
#         global_args.batch_size_multiGPU = global_args.batch_size
#     logs.write_to_log_file("# cuda devices: {}".format(torch.cuda.device_count()))
    
    
#     # ----- Model initialization -----
#     if 'ls_' or 'ln_' in global_args.train_mode:
#         num_seq = corpus.n_users
#     else: num_seq = 1
    
#     adj = np.load(global_args.graph_path)
    

#     # model = TestHierachicalSSM(
#     #     device=global_args.device, 
#     #     args=global_args,
#     #     logs=logs,   
#     # )
#     # model_path = '/mnt/qb/work/mlcolab/hzhou52/kt/logs/1exp_ngms/AmortizeHSSM/split_time/2023-03-14T12:27:13.728399__overfit_0/Model/Model_76.pt'
#     # model.load_state_dict(torch.load(model_path))
#     # model.eval()
    
#     # model = VanillaOU(
#     #     mode=global_args.train_mode,
#     #     device=global_args.device, 
#     #     logs=logs,   
#     #     num_seq=num_seq,
#     # )
#     # model_path = '/mnt/qb/work/mlcolab/hzhou52/kt/logs/0exp_ngss/OU/junyi/single_user_single_skill/2023-03-09T10:41:38.685090__overfit_0_epoch_100/Model/Model_76.pt'
#     # model.load_state_dict(torch.load(model_path))
#     # model.eval()
    
#     model = PPE(
#         mode=global_args.train_mode,
#         device=global_args.device, 
#         logs=logs,   
#         num_seq=num_seq,
#         num_node=corpus.n_skills,
#         nx_graph=adj,
#     )
#     # OU: 
#     # model_path = '/mnt/qb/work/mlcolab/hzhou52/kt/logs/VanillaOU/junyi/single_user_multi_skill/2023-03-26T09:59:02.878897__overfit_0/Model/Model_60.pt'
#     # Graphou
#     # model_path = '/mnt/qb/work/mlcolab/hzhou52/kt/logs/GraphOU/junyi/single_user_multi_skill/2023-03-26T14:44:05.947186__overfit_0/Model/Model_57.pt'
#     # HLR 
#     # model_path = '/mnt/qb/work/mlcolab/hzhou52/kt/logs/HLR/junyi/single_user_multi_skill/2023-03-26T22:14:30.148679_0.5_overfit_0/Model/Model_19.pt'
#     # PPE
#     model_path = '/mnt/qb/work/mlcolab/hzhou52/kt/logs/PPE/junyi/single_user_multi_skill/2023-03-27T01:00:39.656180_0.5_overfit_0/Model/Model_99.pt'
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
    
    
#     # Move to current device
#     if torch.cuda.is_available():
#         if global_args.distributed:
#             model, _ = utils.distribute_over_GPUs(global_args, model, num_GPU=global_args.num_GPU)
#         else: 
#             model = model.to(global_args.device)
            
#     # Running
#     runner = KTRunner(global_args, logs)
    
#     epoch_whole_data = copy.deepcopy(corpus.data_df['whole'])
#     whole_batches = model.module.prepare_batches(corpus, epoch_whole_data, batch_size=8, phase='whole')
    
#     # train_batches = model.module.prepare_batches(corpus, epoch_train_data, self.batch_size, phase='train')
#     # val_batches = model.module.prepare_batches(corpus, epoch_dev_data, self.eval_batch_size, phase='dev')
#     # test_batches = model.module.prepare_batches(corpus, epoch_test_data, self.eval_batch_size, phase='test')
    
    
#     for i in range(len(whole_batches)):
#         batch = [whole_batches[i]]
#         _, outdicts = runner.evaluate(model, corpus, set_name='whole', visualize=True, 
#                                     data_batches=batch, whole_batches=batch)
    
#         ipdb.set_trace()
#         flat_outdicts = outdicts
#         # flat_outdicts = {}
#         # for key in outdicts[0].keys():
#         #     # if key != 'prediction' and key != 'label':
#         #     flat_outdicts[key] = torch.cat([out[key] for out in outdicts], 1)
        
#         with open('model_dict_ppe_batch_id_{}.pkl'.format(i), 'wb') as f:
#             pickle.dump(flat_outdicts, f)
            
            
#     # _, outdicts = runner.evaluate(model, corpus, set_name='whole', visualize=True, 
#     #                             data_batches=whole_batches, whole_batches=whole_batches)

#     # flat_outdicts = {}
#     # for key in outdicts[0].keys():
#     #     if key != 'prediction' and key != 'label':
#     #         flat_outdicts[key] = torch.cat([out[key] for out in outdicts], 1)
    
#     # with open('/mnt/qb/work/mlcolab/hzhou52/kt/junyi/single_user_multi_skill/model_dict_graphou_batch_id.pkl', 'wb') as f:
#     #     pickle.dump(flat_outdicts, f)
                 
#     ipdb.set_trace()
#     # model.module.actions_after_train()
#     logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
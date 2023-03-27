# -*- coding: UTF-8 -*-

import os
import sys
import math
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import ipdb

class DataReader(object):
    '''
    A general data reader for different KT datasets.
    This contains the basic data features and aggregates the features according to different learners.
    For specific data features, it will be defined at get_feed_dict function in each KT model class.
    '''
    def __init__(self, args, logs):
        '''
        Args:
            prefix:    data folder path
            dataset:   the name of KT dataset
            sep:       the delimiter when loading a csv file
            k_fold:    number of k folder to do cross-validation
            max_step:  the maximum step considered during training; NOTE: sometimes it has also been defined during the pre-processing process
            logs:      the log instance where defining the saving/loading information
        '''
        self.prefix = args.data_dir
        self.dataset = args.dataset
        logs.write_to_log_file('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        
        self.sep = args.sep
        self.k_fold = args.kfold
        self.max_step = int(args.max_step)
        self.logs = logs
        self.inter_df = pd.read_csv(os.path.join(self.prefix, self.dataset, 'interactions_{}.csv'.format(args.max_step)), sep=self.sep)
        
        ##### aggregate by user
        user_wise_dict = dict()
        cnt, n_inters = 0, 0
        self.data_df = {
            'train': pd.DataFrame(), 'dev': pd.DataFrame(), 'test': pd.DataFrame()
        }
        
        for user, user_df in self.inter_df.groupby('user_id'):
            df = user_df[:self.max_step]  
            skill_unique = df.skill_id.unique()
            sk_dfs = []
            for sk in skill_unique:
                sk_df = df[df.skill_id==sk].copy()
                sk_df['num_history'] = list(np.arange(len(sk_df)))
                num_correct = 0
                for i in range(len(sk_df)):
                    sk_df.loc[sk_df.index[i], 'num_success'] = num_correct
                    if sk_df.loc[sk_df.index[i], 'correct'] > 0:
                        num_correct += 1
                sk_df['num_failure'] = sk_df.num_history - sk_df.num_success
                sk_dfs.append(sk_df)
            new_df = pd.concat(sk_dfs, axis=0).sort_values('timestamp', ascending=True)
            
            user_wise_dict[cnt] = {
                'user_id': user,                                                    # the ID of the learner
                'skill_seq': new_df['skill_id'].values.tolist(),                    # the sequence of ID of the skills
                'correct_seq': [round(x) for x in df['correct']],                   # the sequence of the performance corresponding to the skill (binary)
                'time_seq': new_df['timestamp'].values.tolist(),                    # the sequence of the time stamps; it should be in an ascending order
                'problem_seq': new_df['problem_id'].values.tolist(),                # the sequence of ID of the problems; NOTE: one skill can have multiple problems
                'num_history': new_df['num_history'].values.astype(int).tolist(),   # until the time step, the amount of interactions of this specific skill for this learner
                'num_success': new_df['num_success'].values.astype(int).tolist(),   # the amount of interactions where the learner succeeded
                'num_failure': new_df['num_failure'].values.astype(int).tolist(),   # the amount of interactions where the learner failed
            }
            
            cnt += 1
            n_inters += len(df)
        ipdb.set_trace()
        self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')
        self.n_users = max(self.inter_df['user_id'].values) + 1
        self.n_skills = max(self.inter_df['skill_id']) + 1
        self.n_problems = max(self.inter_df['problem_id']) + 1

        ##### load the ground-truth graph if available TODO 
        if self.dataset == 'synthetic': 
            self.adj = np.load(os.path.join(self.prefix, self.dataset, 'adj.npy'))
        else: self.adj = np.zeros((self.n_skills, self.n_skills))

        self.logs.write_to_log_file('"n_users": {}, "n_skills": {}, "n_problems": {}, "n_interactions": {}'.format(
            self.n_users, self.n_skills, self.n_problems, n_inters
        ))

    def gen_fold_data(self, k):
        '''
        Args:
            k: select the k-th fold to run
        '''
        assert k < self.k_fold
        n_examples = len(self.user_seq_df)
        fold_size = math.ceil(n_examples / self.k_fold)
        
        fold_begin = k * fold_size
        fold_end = min((k + 1) * fold_size, n_examples)
        self.data_df['test'] = self.user_seq_df.iloc[fold_begin:fold_end]
        
        residual_df = pd.concat([self.user_seq_df.iloc[0:fold_begin], self.user_seq_df.iloc[fold_end:n_examples]])
        dev_size = int(0.1 * len(residual_df)) # 
        dev_indices = np.random.choice(residual_df.index, dev_size, replace=False)  # random
        self.data_df['dev'] = self.user_seq_df.iloc[dev_indices]
        
        self.data_df['train'] = residual_df.drop(dev_indices)
        self.data_df['whole'] = self.user_seq_df

    def gen_time_split_data(self, train_ratio, test_ratio):
        '''
        Split the train/test/dev based on time steps. 
        E.g., for each learner, the first 70% interactions are training data, the next 10% validation data, and final 20% test data.
        self.user_seq_df.keys(): ['user_id', 'skill_seq', 'correct_seq', 'time_seq', 'problem_seq',
                                  'num_history', 'num_success', 'num_failure']
        Args:
            
        '''
        assert train_ratio + test_ratio <= 1
        n_time_steps = len(self.user_seq_df['time_seq'][0])
        self.data_df = {
            'train': dict(),
            'dev': dict(),
            'test': dict(),
            'whole': dict(),
        }
        
        train_size = math.ceil(n_time_steps * train_ratio)
        test_size = math.ceil(n_time_steps * test_ratio)
        dev_size = n_time_steps-train_size-test_size
        
        for key in self.user_seq_df.keys():
            if key != 'user_id':
                value = np.stack(self.user_seq_df[key].values)
                self.data_df['train'][key] = value[:, :train_size].tolist()
                self.data_df['test'][key] = value[:, -test_size:].tolist()
                self.data_df['dev'][key] = value[:, train_size:dev_size+train_size].tolist()
                self.data_df['whole'][key] = value[:, :].tolist()

        for key in self.data_df.keys():     
            self.data_df[key] = pd.DataFrame.from_dict(self.data_df[key], orient='columns')   
            self.data_df[key]['user_id'] = self.user_seq_df['user_id']
        
    def show_columns(self):
        self.logs.write_to_log_file('Data columns:')
        self.logs.write_to_log_file(self.user_seq_df.iloc[np.random.randint(0, len(self.user_seq_df))])





# class KTDataset(Dataset):
#     '''
#     Dataset for different KT 
#     '''
#     def __init__(self, args, logs):
#         '''
#         Args:
#             prefix:    data folder path
#             dataset:   the name of KT dataset
#             sep:       the delimiter when loading a csv file
#             k_fold: 
#             max_step:  the maximum step considered during training; NOTE: sometimes it has also been defined during the pre-processing process
#             logs:      the log instance where defining the saving/loading information
#         '''
#         self.prefix = args.data_dir
#         self.dataset = args.dataset
#         logs.write_to_log_file('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        
#         self.sep = args.sep
#         self.k_fold = args.kfold
#         self.max_step = int(args.max_step)
#         self.logs = logs
#         self.inter_df = pd.read_csv(os.path.join(self.prefix, self.dataset, 'interactions_{}.csv'.format(args.max_step)), sep=self.sep)[:100]
        
#         ##### aggregate by user
#         user_wise_dict = dict()
#         cnt, n_inters = 0, 0
#         self.data_df = {
#             'train': pd.DataFrame(), 'dev': pd.DataFrame(), 'test': pd.DataFrame()
#         }
        
#         for user, user_df in self.inter_df.groupby('user_id'):
#             df = user_df[:self.max_step]  
#             skill_unique = df.skill_id.unique()
#             sk_dfs = []
#             for sk in skill_unique:
#                 sk_df = df[df.skill_id==sk].copy()
#                 sk_df['num_history'] = list(np.arange(len(sk_df)))
#                 num_correct = 0
#                 for i in range(len(sk_df)):
#                     sk_df.loc[sk_df.index[i], 'num_success'] = num_correct
#                     if sk_df.loc[sk_df.index[i], 'correct'] > 0:
#                         num_correct += 1
#                 sk_df['num_failure'] = sk_df.num_history - sk_df.num_success
#                 sk_dfs.append(sk_df)
#             new_df = pd.concat(sk_dfs, axis=0).sort_values('timestamp', ascending=True)

#             user_wise_dict[cnt] = {
#                 'user_id': user,                                                    # the ID of the learner
#                 'skill_seq': new_df['skill_id'].values.tolist(),                    # the sequence of ID of the skills
#                 'correct_seq': [round(x) for x in df['correct']],                   # the sequence of the performance corresponding to the skill (binary)
#                 'time_seq': new_df['timestamp'].values.tolist(),                    # the sequence of the time stamps; it should be in an ascending order
#                 'problem_seq': new_df['problem_id'].values.tolist(),                # the sequence of ID of the problems; NOTE: one skill can have multiple problems
#                 'num_history': new_df['num_history'].values.astype(int).tolist(),   # until the time step, the amount of interactions of this specific skill for this learner
#                 'num_success': new_df['num_success'].values.astype(int).tolist(),   # the amount of interactions where the learner succeeded
#                 'num_failure': new_df['num_failure'].values.astype(int).tolist(),   # the amount of interactions where the learner failed
#             }
            
#             cnt += 1
#             n_inters += len(df)
#         self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')

#         self.n_users = max(self.inter_df['user_id'].values) + 1
#         self.n_skills = max(self.inter_df['skill_id']) + 1
#         self.n_problems = max(self.inter_df['problem_id']) + 1

#         ##### load the ground-truth graph if available TODO 
#         if self.dataset == 'synthetic': 
#             self.adj = np.load(os.path.join(self.prefix, self.dataset, 'adj.npy'))
#         else: self.adj = np.zeros((self.n_skills, self.n_skills))

#         self.logs.write_to_log_file('"n_users": {}, "n_skills": {}, "n_problems": {}, "n_interactions": {}'.format(
#             self.n_users, self.n_skills, self.n_problems, n_inters
#         ))

#     def show_columns(self):
#         self.logs.write_to_log_file('Data columns:')
#         self.logs.write_to_log_file(self.user_seq_df.iloc[np.random.randint(0, len(self.user_seq_df))])

#     def __len__(self):
#         return len(self.user_seq_df)

#     def __getitem__(self, idx):
#         single_data = {}
#         for k, v in self.user_seq_df.items():
#             single_data[k] = self.user_seq_df[k][idx]
#         return single_data
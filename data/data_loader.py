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
    
    def __init__(self, args, logs):
        self.prefix = args.data_dir
        self.sep = args.sep
        self.k_fold = args.kfold
        self.max_step = int(args.max_step)

        self.dataset = args.dataset
        self.data_df = {
            'train': pd.DataFrame(), 'dev': pd.DataFrame(), 'test': pd.DataFrame()
        }
        
        self.logs = logs
        
        logs.write_to_log_file('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))

        self.inter_df = pd.read_csv(os.path.join(self.prefix, self.dataset, 'interactions_{}.csv'.format(args.max_step)), sep=self.sep)
        # aggregate by user
        user_wise_dict = dict()
        cnt, n_inters = 0, 0

        for user, user_df in self.inter_df.groupby('user_id'):

            df = user_df[:self.max_step]  # consider the first 50 interactions
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
            new_df = pd.concat(sk_dfs, 0).sort_values('timestamp', ascending=True)

            user_wise_dict[cnt] = {
                'user_id': user,
                'skill_seq': new_df['skill_id'].values.tolist(), 
                'correct_seq': [round(x) for x in df['correct']],
                'time_seq': new_df['timestamp'].values.tolist(),
                'problem_seq': new_df['problem_id'].values.tolist(),
                'num_history': new_df['num_history'].values.astype(int).tolist(),
                'num_success': new_df['num_success'].values.astype(int).tolist(),
                'num_failure': new_df['num_failure'].values.astype(int).tolist(),
            }
            
            cnt += 1
            n_inters += len(df)

        self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')
        self.n_users = max(self.inter_df['user_id'].values) + 1
        self.n_skills = max(self.inter_df['skill_id']) + 1
        # TODO: now, every user list is not in the same length. Add 0 to make them all in max_step length
        # TODO the number of unique exercise in log and exercise is not the same
        # self.inter_df['exercise'].unique().shape[0] # max(self.inter_df['skill_id']) + 1
        self.n_problems = max(self.inter_df['problem_id']) + 1

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
        

    def show_columns(self):
        self.logs.write_to_log_file('Data columns:')
        self.logs.write_to_log_file(self.user_seq_df.iloc[np.random.randint(0, len(self.user_seq_df))])


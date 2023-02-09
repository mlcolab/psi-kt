import sys
sys.path.append('..')

import numpy as np
from numpy.random import default_rng

import pandas as pd
import networkx as nx 
import scipy
from scipy import stats

import os
import argparse
import datetime
import torch

from utils.parametric_models import *
from utils.visualize import *
# rom data.data_loader import KTData

import ipdb

def parse_args(parser):
    # ----- global -----
    parser.add_argument('--random_seed', type=int, default=1,)
    parser.add_argument('--num_sequence', type=int, default=4,)
    parser.add_argument('--learner_model', type=str, default=['graph_ou'], help=['hlr', 'ou', 'graph_ou', 'egraph_ou', 'ppe'])
    
    # ----- time points -----
    parser.add_argument('--time_random_type', type=str, default='random', help=['random', 'uniform'])
    parser.add_argument('--time_step', type=int, default=20,)
    parser.add_argument('--max_time_step', type=int, default=250,)

    # ----- random graph -----
    parser.add_argument('--num_node', type=int, default=10,)
    parser.add_argument('--edge_prob', type=float, default=0.4,)

    # ----- ou process -----
    parser.add_argument('--mean_rev_speed', type=float, default=0.02,)
    parser.add_argument('--mean_rev_level', type=float, default=0.7,)
    parser.add_argument('--vola', type=float, default=0.01,)
    parser.add_argument('--gamma', type=float, default=0.75,)
    parser.add_argument('--rho', type=float, default=2,)
    parser.add_argument('--omega', type=float, default=0.75,)
    
    # ----- extend graph ou process -----

    # ----- hlr process -----
    parser.add_argument('--theta', type=list, default=[1/4, 1/2, -1/3],)

    # ----- ppe process -----
    parser.add_argument('--learning_rate', type=float, default=0.1)
    # parser.add_argument('--decay_rate', type=float, default=0.2)

    

    # ----- save path -----
    parser.add_argument('--save_path', type=str, default='/mnt/qb/work/mlcolab/hzhou52/kt/synthetic',)

    return parser 


class KTData(object):
    def __init__(self, data_type='syn', df=None, times=None, items=None, graph_adj=None):
        if data_type == 'syn':
            self.init_synthetic_data(times, items, graph_adj)
        else:
            self.init_real_data(df)


    def init_synthetic_data(self, times, items, graph_adj):
        ipdb.set_trace()
        self.num_seq = times.shape[0]

        self.user_id = np.arange(self.num_seq)
        self.time_seq = times
        self.skill_id = items

        self.adj = graph_adj
        

    def init_real_data(self, df):
        pass

    def count_history(self):
        for i in range(self.num_seq):
            skill_unique = self.skill_id.unique()





def save_as_unified_format(args, path, times, items, adj):

    df = []
    timestamp = times.flatten()
    dwell_time = np.zeros_like(timestamp)

    correct = (p_items >=0.5) * 1 # (sigmoid(p_items) >= 0.5)*1 # TODO???

    problem_id = items.flatten()
    skill_id = items.flatten()

    user_id = np.tile(np.arange(args.num_sequence).reshape(-1, 1), (1, args.time_step)).flatten()

    df = np.stack([timestamp, dwell_time, correct, problem_id, skill_id, user_id], -1)
    df = pd.DataFrame(df, columns=['timestamp', 'dwell_time', 'correct', 'problem_id', 'skill_id', 'user_id'])

    df = df.astype({
    'timestamp': np.float64,
    'dwell_time': np.float64,
    'correct': np.float64,
    'problem_id': np.int64,
    'skill_id': np.int64,
    'user_id': np.int64
    })

    # Save
    adj_path = os.path.join(args.log_path, 'adj.npy')
    np.save(adj_path, adj)
    df_path = os.path.join(args.log_path, 'interactions_{}.csv'.format(args.time_step))
    df.to_csv(df_path, sep='\t', index=False)


def time_point_generate():
    '''
    generate random or uniform time points for the interaction
    Main Args:
        args.time_random_type: uniform or random
        args.max_time_step
        args.time_step: the interval between two time points. It is useful ONLY IF 'uniform' is chosen
    '''
    if args.time_random_type == 'uniform':
        times = np.arange(0, args.max_time_step, args.max_time_step//args.time_step)
        times = np.tile(np.expand_dims(times, 0), (args.num_sequence, 1)) # [num_deq, time_step]
        
    elif args.time_random_type == 'random':
        rng = default_rng(args.random_seed)
        times = []
        for i in range(args.num_sequence):
            time = rng.choice(np.arange(args.max_time_step), args.time_step, False)
            time.sort()
            times.append(time)
        times = np.stack(times)

    return times


def review_item_generate():
    rng = default_rng(args.random_seed)
    items = []
    # p_items = []
    for i in range(args.num_sequence):
        item = rng.choice(np.arange(args.num_node), args.time_step, True)
        items.append(item)
        # p_items.append(path[i, np.arange(args.time_step), item]) 
    items = np.stack(items)
    # p_items = np.stack(p_items).flatten()

    return items


def sigmoid(x):
    return 1/(1 + np.exp(-x))





if __name__ == '__main__':
    # ----- args -----
    parser = argparse.ArgumentParser(description='Global')
    parser = parse_args(parser)

    global args
    args, extras = parser.parse_known_args()

    args.time = datetime.datetime.now().isoformat()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.log_path = os.path.join(args.save_path, args.time + '_' + 'node_' + str(args.num_node))
    

    # -- generate random graphs
    graph = nx.erdos_renyi_graph(args.num_node, args.edge_prob, seed=args.random_seed, directed=True)
    adj = nx.adjacency_matrix(graph).toarray()
    # draw_graph(graph, args)

    # -- generate time points & reviewing items
    times = time_point_generate()
    items = review_item_generate()
    
    # -- initialize a KT Data Object
    # ktdata = KTData(data_type='syn', times=times, items=items, graph_adj=adj) TODO

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
        
        
    # # -- PPE process 
    # for learning_rate in [0.01, 0.05, 0.1, 0.2, 0.5, 1]:
    #     ppe_generator = PPE(learning_rate, num_seq=args.num_sequence)
    #     ppe_path, params = ppe_generator.simulate_path(np.zeros((args.num_node,1)), times, items)
    #     prefix = 'ppe_lr_{}'.format(learning_rate)
        
    #     draw_path(ppe_path[0], args, times[0], items=items[0], prefix=prefix)#, scatter=True)
    #     draw_params(params, args, times, items, prefix=prefix)
    
    
    # # -- HLR process 
    # weight_total = [1/12, 1/6, 1/4, 1/3, 1/2, 2/3]
    # weight_success = [1/2]
    # weight_failure = [-1/3]
    # for w in weight_total:
    #     theta = [w] + weight_success + weight_failure
    #     theta = np.array(theta)
    #     hlr_generator = HLR(theta=theta, num_seq=args.num_sequence)
    #     hlr_path, params = hlr_generator.simulate_path(np.zeros((args.num_node,1)), times, items)
    #     prefix = 'hlr_theta_{:.2f}_{:.2f}_{:.2f}'.format(theta[0], theta[1], theta[2])
    
    #     draw_path(hlr_path[0], args, times[0], items=items[0], prefix=prefix)
    #     draw_params(params, args, times, items, prefix=prefix)
    # ipdb.set_trace()
    
    # -- Extend Graph OU process 
    # -- Graph OU process 
    # for speed in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
    for gamma in [0.1, 0.2, 0.5, 0.75, 1]: 
        ou_generator = GraphOU(args.mean_rev_speed, args.mean_rev_level, args.vola, args.num_sequence, 
                                nx_graph=graph, gamma=gamma, rho=args.rho, omega=args.omega) # test
        ou_path, params = ou_generator.simulate_path(np.zeros((args.num_node,1)), times, items=items)
        prefix = 'GraphOU_speed_{}_vola_{}_gamma_{}_rho_{}_omega_{}'.format(
            args.mean_rev_speed, args.vola, gamma, args.rho, args.omega
        )
        # args.mean_rev_speed
        
        draw_path(ou_path[0], args, times[0], items=items[0], prefix=prefix)
        draw_path(sigmoid(ou_path[0]), args, times[0], items=items[0], prefix=prefix+'sigmoid')
        draw_params(params, args, times, items, prefix=prefix)
    ipdb.set_trace()

    # -- Vanilla OU process
    ou_vanilla_generator = VanillaOU(args.mean_rev_speed, args.mean_rev_level, args.vola, args.num_sequence)
    vanilla_path = ou_vanilla_generator.simulate_path(np.zeros((args.num_node,1)), times)
    draw_path(vanilla_path[0], args, times[0], items=items[0], prefix='ou')
    
    ipdb.set_trace()








    


    ipdb.set_trace()
    save_as_unified_format(args, path, times, items, adj)

    # TODO debug: visualize
    draw_path(path[0], args, times[0], prefix='graph')
    draw_path(vanilla_path[0], args, times[0], prefix='vanilla')
    draw_path(sigmoid(path[0]), args, times[0], prefix='sigmoid')
    visualize_ground_truth(graph, args, adj)

    ipdb.set_trace()
    
    
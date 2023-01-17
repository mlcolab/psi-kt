import sys
sys.path.append('..')

import numpy as np
from numpy.random import default_rng

import pandas as pd
import networkx as nx 
import scipy
from scipy import stats

import argparse
import torch
import datetime

from data.ou_process import VanillaOU, VanillaGraphOU, ExtendGraphOU
from utils.visualize import *

import ipdb

def parse_args(parser):
    # ----- global -----
    parser.add_argument('--random_seed', type=int, default=1,)
    parser.add_argument('--num_sequence', type=int, default=100,)

    # ----- random graph -----
    parser.add_argument('--num_node', type=int, default=5,)
    parser.add_argument('--edge_prob', type=float, default=0.2,)

    # ----- ou process -----
    parser.add_argument('--mean_rev_speed', type=float, default=0.02,)
    parser.add_argument('--mean_rev_level', type=float, default=0.5,)
    parser.add_argument('--mean_base_level', type=float, default=1.0,)
    parser.add_argument('--vola', type=float, default=0.08,)


    parser.add_argument('--time_step', type=int, default=50,)
    parser.add_argument('--max_time_step', type=int, default=350,)
    parser.add_argument('--random_type', type=str, default='random',)

    return parser 

def save_as_unified_format(args, path, times, save_path):
    df = []
    timestamp = times.flatten()
    dwell_time = np.zeros_like(timestamp)

    rng = default_rng(args.random_seed)
    items = []
    p_items = []
    for i in range(args.num_sequence):
        item = rng.choice(np.arange(args.num_node), args.time_step, True)
        items.append(item)
        p_items.append(path[i, np.arange(args.time_step), item]) 
    items = np.stack(items)
    p_items = np.stack(p_items).flatten()
    correct = (sigmoid(p_items) >= 0.5)*1

    problem_id = items.flatten()
    skill_id = items.flatten()

    user_id = np.tile(np.arange(args.num_sequence).reshape(-1, 1), (1, args.time_step)).flatten()

    df = np.stack([timestamp, dwell_time, correct, problem_id, skill_id, user_id], -1)
    df = pd.DataFrame(df, columns=['timestamp', 'dwell_time', 'correct', 'problem_id', 'skill_id', 'user_id'])

    ipdb.set_trace()
    df = df.astype({
    'timestamp': np.float64,
    'dwell_time': np.float64,
    'correct': np.float64,
    'problem_id': np.int64,
    'skill_id': np.int64,
    'user_id': np.int64
    })

    # Save
    df.to_csv('/mnt/qb/work/mlcolab/hzhou52/kt/synthetic/'+'interactions_{}.csv'.format(args.time_step), sep='\t', index=False)

    # return df

def time_point_generate(args):
    if args.random_type == 'uniform':
        times = np.arange(0, args.max_time_step, args.max_time_step//args.time_step)
        times = np.tile(np.expand_dims(times, 0), (args.num_sequence, 1)) # [num_deq, time_step]
        
    elif args.random_type == 'random':
        rng = default_rng(args.random_seed)
        times = []
        for i in range(args.num_sequence):
            time = rng.choice(np.arange(args.max_time_step), args.time_step, False)
            time.sort()
            times.append(time)
        times = np.stack(times)
    return times

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def review_item_generate():
    pass



def main(args, graph):
    pass


if __name__ == '__main__':
    # ----- args -----
    parser = argparse.ArgumentParser(description='Global')
    parser = parse_args(parser)
    args, extras = parser.parse_known_args()

    args.time = datetime.datetime.now().isoformat()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph = nx.erdos_renyi_graph(args.num_node, args.edge_prob, seed=args.random_seed, directed=True)
    adj = nx.adjacency_matrix(graph).toarray()
    np.save('/mnt/qb/work/mlcolab/hzhou52/kt/synthetic/adj.npy', adj)
    ipdb.set_trace()
    times = time_point_generate(args)

    ou_generator = VanillaGraphOU(args.mean_rev_speed, args.mean_rev_level, args.vola, args.num_sequence, graph)
    path = ou_generator.simulate_path(np.zeros((args.num_node,1)), times)

    save_as_unified_format(args, path, times, save_path=None)
    ipdb.set_trace()
    # # TODO debug: compare with vanilla
    # ou_generator = VanillaGraphOU(args.mean_rev_speed, args.mean_rev_level, args.vola, graph)
    # path = ou_generator.simulate_path(np.zeros((args.num_node,1)), times)

    ou_vanilla_generator = VanillaOU(args.mean_rev_speed, args.mean_rev_level, args.vola, args.num_sequence)
    vanilla_path = ou_vanilla_generator.simulate_path(np.zeros((args.num_node,1)), times)

    # TODO debug: visualize
    draw_path(path[0], args, times[0], prefix='graph')
    draw_path(vanilla_path[0], args, times[0], prefix='vanilla')
    draw_path(sigmoid(path[0]), args, times[0], prefix='sigmoid')
    draw_graph(graph)
    

    # ipdb.set_trace()
    
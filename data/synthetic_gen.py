import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import networkx as nx 
import scipy
from scipy import stats

import argparse
import torch
import datetime

from data.ou_process import VanillaOU, VanillaGraphOU, RewriteGraphOU, ExtendGraphOU
from utils.visualize import *

import ipdb

def parse_args(parser):
    # ----- global -----
    parser.add_argument('--random_seed', type=int, default=100,)
    parser.add_argument('--num_sequence', type=int, default=1,)

    # ----- random graph -----
    parser.add_argument('--num_node', type=int, default=2,)
    parser.add_argument('--edge_prob', type=float, default=0.2,)

    # ----- ou process -----
    parser.add_argument('--mean_rev_speed', type=float, default=0.05,)
    parser.add_argument('--mean_rev_level', type=float, default=1.0,)
    parser.add_argument('--mean_base_level', type=float, default=1.0,)
    parser.add_argument('--vola', type=float, default=0.05,)
    parser.add_argument('--time_step', type=int, default=200,)

    return parser 

def save_as_unified_format():

    df = df.astype({
    'timestamp': np.float64,
    'dwell_time': np.float64,
    'correct': np.float64,
    'problem_id': np.int64,
    'skill_id': np.int64,
    'user_id': np.int64
    })

def time_point_generate(args):
    return np.arange(args.time_step)

def review_item_generate():
    pass

def sigmoid(x):
    return 1/(1 + np.exp(-x))

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

    # TODO synthetic time
    times = time_point_generate(args)

    # ou_generator = VanillaGraphOU(args.mean_rev_speed, args.mean_rev_level, args.vola, graph)
    # path = ou_generator.simulate_path(np.zeros((args.num_node,1)), times)

    ou_generator = RewriteGraphOU(args.mean_rev_speed, args.mean_rev_level, args.vola, graph)
    path = ou_generator.simulate_path(np.zeros((args.num_node,1)), times)

    # TODO debug: compare with vanilla
    ou_vanilla_generator = VanillaOU(args.mean_rev_speed, args.mean_rev_level, args.vola)
    vanilla_path = ou_vanilla_generator.simulate_path(np.zeros((args.num_node,1)), times)

    # TODO debug: visualize
    draw_path(path, args, times, prefix='graph')
    draw_path(vanilla_path, args, times, prefix='vanilla')
    # draw_path(sigmoid(path), args, times, prefix='sigmoid')
    # draw_graph(graph)
    
    ipdb.set_trace()
    
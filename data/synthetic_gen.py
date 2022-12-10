import numpy as np

import argparse
import torch
import datetime
import networkx as nx 

import scipy
from scipy import stats

from ou_process import VanillaOU, GraphOU
# from utils.visualize import *
import ipdb

def parse_args(parser):
    # ----- global -----
    parser.add_argument('--random_seed', type=int, default=100,)

    # ----- random graph -----
    parser.add_argument('--num_node', type=int, default=2,)
    parser.add_argument('--edge_prob', type=float, default=0.2,)

    # ----- ou process -----
    parser.add_argument('--mean_rev_speed', type=float, default=0.1,)
    parser.add_argument('--mean_rev_level', type=float, default=-0.05,)
    parser.add_argument('--vola', type=float, default=0.1,)
    parser.add_argument('--time_step', type=int, default=200,)


    return parser # args

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
def draw_path(path, args, times, prefix=None):
    plt.clf()
    color = cm.rainbow(np.linspace(0, 1, args.num_node))
    for i, c in zip(range(args.num_node), color):
        plt.plot(times, path[:,i], c=c, label='{}'.format(i))
    plt.legend()
    plt.savefig(prefix+'_path_speed{}_level{}_noise{}_time{}.png'.format(args.mean_rev_speed, args.mean_rev_level, args.vola, args.time_step))
def draw_graph(graph):
    plt.clf()
    nx.draw(graph, with_labels=True)
    plt.savefig('graph.png', dpi=300, bbox_inches='tight')


def main():
    pass


if __name__ == '__main__':
    # ----- args -----
    parser = argparse.ArgumentParser(description='Global')
    parser = parse_args(parser)
    args, extras = parser.parse_known_args()

    args.time = datetime.datetime.now().isoformat()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph = nx.erdos_renyi_graph(args.num_node, args.edge_prob, seed=args.random_seed, directed=True)
    # ipdb.set_trace()

    # TODO synthetic time
    times = np.arange(args.time_step)
    ou_generator = GraphOU(args.mean_rev_speed, args.mean_rev_level, args.vola, graph)
    path = ou_generator.simulate_path(np.zeros((args.num_node,1)), times)

    # TODO debug: compare with vanilla
    ou_vanilla_generator = VanillaOU(args.mean_rev_speed, args.mean_rev_level, args.vola)
    vanilla_path = ou_vanilla_generator.simulate_path(np.zeros((args.num_node,1)), times)

    # TODO debug: visualize
    draw_path(path, args, times, prefix='graph')
    draw_path(vanilla_path, args, times, prefix='vanilla')
    draw_graph(graph)
    ipdb.set_trace()


    main(global_args, model, logs, test_train)
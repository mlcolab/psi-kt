import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import networkx as nx
import os
import imageio
import ipdb


def draw_path(path, args, times, items=None, prefix=None, scatter=False):
    plt.clf()
    color = cm.rainbow(np.linspace(0, 1, args.num_node))
    plt.figure(figsize=(20,8))
    # ipdb.set_trace()
    for i, c in zip(range(args.num_node), color):
        if scatter:
            ind = np.where(items==i)[0]
            plt.plot(times[ind], path[ind,i], c=c, label='{}'.format(i))
        else:
            plt.plot(times, path[:,i], c=c, label='{}'.format(i))
            
        # put labels on interacted nodes
        ind = np.where(items==i)[0]
        plt.scatter(times[ind], path[ind, i], marker='*')

    plt.legend()
    plt.savefig(os.path.join(args.log_path, prefix+'_process.png'))


def visualize_ground_truth(graph, args, adj, size=4.0):
    plt.clf()
    nx.draw(graph, with_labels=True)
    plt.savefig(os.path.join(args.log_path, 'graph_raw.png'), dpi=300, bbox_inches='tight')

    plt.clf()
    plt.rcParams['figure.figsize'] = [size, size]
    fig, ax = plt.subplots(1, 1)
    ax.matshow(adj, vmin=0, vmax=1)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title(r'Ground truth $G^*$', pad=10)
    plt.savefig(os.path.join(args.log_path, 'graph_adj.png'))
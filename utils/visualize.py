import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import networkx as nx


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
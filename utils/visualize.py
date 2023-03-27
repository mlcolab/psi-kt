import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import networkx as nx
import os
import imageio
import ipdb














def draw_path(path, args, times, items=None, prefix=None, scatter=False):
    '''
    Args:
        path: [num_node, t]
        times: [t]
    '''
    path = path.cpu().numpy() if args.device.type == 'cuda' else path.numpy()
    times = times.cpu().numpy() if args.device.type == 'cuda' else times.numpy()
    items = items.cpu().numpy() if args.device.type == 'cuda' else items.numpy()
    
    process = prefix.split('_')[0]
    color = cm.rainbow(np.linspace(0, 1, args.num_node))

    plt.clf()
    plt.figure(figsize=(20,8))
    plt.xlabel('time t')
    plt.ylabel('recall_probability')
    plt.title(
        label=process.upper()+' Model',
        fontsize=20,
        color="black"
    )
        
    for i, c in zip(range(args.num_node), color):
        if scatter:
            ind = np.where(items==i)[0]
            plt.plot(times[ind], path[ind,i], color=c, label='{}'.format(i), linewidth=4)
        else:
            plt.plot(times, path[i], color=c, label='{}'.format(i), linewidth=4)
            
        # put labels on interacted nodes
        ind = np.where(items==i)[0]
        plt.scatter(times[ind], path[i, ind], s=150, marker='*')
        plt.vlines(x=times[ind], ymin = np.min(path), ymax = np.max(path),
                colors = 'grey', linestyles='dashdot')
                
    plt.legend()
    if not os.path.exists(os.path.join(args.log_path, process)):
        os.makedirs(os.path.join(args.log_path, process))
    plt.savefig(os.path.join(args.log_path, process, prefix+'.png'))



def draw_params(params, args, times, items=None, prefix=None, scatter=False):
    seq = 0
    process = prefix.split('_')[0]
    for keys, values in params.items():
        plt.clf()
        color = cm.rainbow(np.linspace(0, 1, args.num_node))
        plt.figure(figsize=(15,8))
        plt.xlabel('time t')
        plt.ylabel('number counts')
        plt.title(label=keys,
                fontsize=20,
                color="black")
        
        for i, c in zip(range(args.num_node), color):
            if np.issubdtype(values.dtype, np.integer):
                plt.scatter(times[seq, 1:], values[seq, :, i], color=c, label='{}'.format(i))
            else:
                plt.plot(times[seq, 1:], values[seq, :, i], color=c, label='{}'.format(i))
                
            # put labels on interacted nodes
            ind = np.where(items[seq]==i)[0]
            # ipdb.set_trace()
            plt.scatter(times[seq, ind], values[seq, ind-1, i], marker='*')

        plt.legend()
        plt.savefig(os.path.join(args.log_path, process, prefix+'_params_'+keys+'.png'))




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
    
    
    
# --- taking samples from posterier gaussian distribution --- # 

# sample_num = 0 # number of samples
# fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(10,3))
# pos = ax1.imshow(post_cov)
# fig.colorbar(pos,ax=ax1)
# ax1.title.set_text('Covariance Matrix')
# ax2.plot(x_real,y_real,c='k',label='real function')
# ax2.plot(x_test,post_mean,label='mean',c='r',linestyle='--')
# plt.fill_between(x_test, post_mean+np.diagonal(post_cov)*2, post_mean-np.diagonal(post_cov)*2,label='95% variance',alpha=0.2)
# plt.title('Gaussian process conditioned on test data')
# ax2.scatter(x_data,y_data,c='b')
# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax2.grid()
# ax2.set_xlabel('$x$'); ax2.set_ylabel('$f(x)$')
# plt.show()

# # plotting with some samples from the conditional distribution 

# sample_num = 10 # number of samples
# fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(10,3))
# pos = ax1.imshow(post_cov)
# fig.colorbar(pos,ax=ax1)
# ax1.title.set_text('Covariance Matrix')
# ax2.plot(x_real,y_real,c='k',label='real function')
# ax2.plot(x_test,post_mean,label='mean',c='r',linestyle='--')
# plt.fill_between(x_test, post_mean+np.diagonal(post_cov)*2, post_mean-np.diagonal(post_cov)*2,label='95% variance',alpha=0.2)
# for j in range(sample_num):
#     sample = np.random.multivariate_normal(post_mean,post_cov)
#     ax2.plot(x_test,sample)
# plt.title('Gaussian process conditioned on test data')
# ax2.scatter(x_data,y_data,c='b')
# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax2.grid()
# ax2.set_xlabel('$x$'); ax2.set_ylabel('$f(x)$')
# plt.show()
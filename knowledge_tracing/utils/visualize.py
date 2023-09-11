import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns

from numpy.random import default_rng

import scipy
from scipy import stats


##########################################################################################
# Compare different models, different tasks, single metrics (ideally f1 score)
# Put all of the results in a single figure
##########################################################################################

def compare_model_task(
    figsize=(12, 3), 
    title='Compare Model Performance on Different Tasks',
    save_path=None
    ):
    """
    Compare the performance of different models.

    Returns:
    None
    """
    # Define the data
    train_sizes = ['Train 40%', 'Train 30%', 'Train 20%']
    metrics = ['HLR_acc', 'OU_acc', 'HLR_f1', 'OU_f1']
    results = np.array([
        [0.7716, 0.8581, 0.8648, 0.9206],
        [0.7412, 0.8540, 0.8348, 0.9179],
        [0.7066, 0.8476, 0.8145, 0.9137],
    ])

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.minorticks_on()
    ax.set_xticks(np.arange(len(train_sizes) * len(metrics)))
    ax.set_xticklabels(np.repeat(metrics, len(train_sizes)), fontsize=8)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel('F1 Score')

    colors = ['lightsteelblue', 'lightblue', 'cornflowerblue', 'steelblue']
    for i, train_size in enumerate(train_sizes):
        offset = i * len(metrics)
        bars = ax.bar(
            x=np.arange(offset, offset + len(metrics)),
            height=results[i],
            color=colors[i],
            width=0.8
        )
        ax.bar_label(bars, size=8)
    
    ax.legend(train_sizes, loc='lower left')
    plt.title(title)
    plt.savefig(save_path)


##########################################################################################









##########################################################################################
# Compare different models, different tasks, different metrics
##########################################################################################

def _create_heatmap(data, tasks, model_names, metrics, save_path, prefix):
    """
    Create a heatmap to visualize the performance of different models on different tasks.

    Parameters:
    - data: a dictionary containing the performance metrics for each model and task
    - tasks: a list of task names
    - model_names: a list of model names
    - metrics: a list of metric names

    Returns:
    None
    """

    # Prepare data for heatmap
    num_tasks = len(tasks)
    num_models = len(model_names)
    num_metrics = len(metrics)
    heatmap_data = np.zeros((num_models * num_tasks, num_metrics))

    for task_idx, task in enumerate(tasks):
        for model_idx, model in enumerate(model_names):
            metric_values = data[model][task]
            heatmap_data[task_idx * num_models + model_idx] = metric_values

    # Create a DataFrame for heatmap labels
    heatmap_labels = pd.DataFrame({
        'Task': np.repeat(tasks, num_models),
        'Model': np.tile(model_names, num_tasks)
    })

    # Create heatmap
    plt.clf()
    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=metrics, yticklabels=False, cbar_kws={'label': 'Metric Value'})

    # Add labels
    for i in range(num_models * num_tasks):
        plt.text(-1, i + 0.5, heatmap_labels.iloc[i]['Model'] + ', ' + heatmap_labels.iloc[i]['Task'], ha='left', va='center', fontsize=10)

    # Display the plot
    # plt.show()
    plt.savefig(Path(save_path, 'heatmap' + prefix +'.png'))
    
    
def _create_bar(data, tasks, model_names, metrics, save_path, prefix):
    num_tasks = len(tasks)
    num_models = len(model_names)
    num_metrics = len(metrics)
    
    bar_width = 1 / (num_models + 1)
    opacity = 0.8
    colors = ['lightblue', 'lightsteelblue', 'cornflowerblue', 'darkblue']

    # Create subplots for each task
    fig, axes = plt.subplots(nrows=1, ncols=num_tasks, figsize=(20, 5), sharey=True)
    fig.subplots_adjust(wspace=0.1)

    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx]
        
        # Plot bars for each model
        for model_idx, model in enumerate(model_names):
            metric_values = data[model][task]
            bar_positions = np.arange(num_metrics) + model_idx * bar_width
            bars = ax.bar(bar_positions, metric_values, bar_width, alpha=opacity, label=model, color=colors[model_idx])

            # Label the values in the bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate('{:.2f}'.format(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        # Set axis labels and ticks
        ax.set_title(task)
        ax.set_xticks(np.arange(num_metrics) + bar_width * (num_models - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1])
        
        # Add grid lines
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        # Add legend
        ax.legend(loc='center left')
    fig.savefig(Path(save_path, 'bar' + prefix +'.png'))


def _create_line(data, tasks, model_names, metrics, save_path, prefix):
    num_tasks = len(tasks)
    num_models = len(model_names)
    num_metrics = len(metrics)
    
    colors = cm.rainbow(np.linspace(0, 1, num_metrics))

    fig, axes = plt.subplots(nrows=1, ncols=num_tasks, figsize=(20, 5), sharey=True)
    fig.subplots_adjust(wspace=0.1)

    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx]
        
        # Plot lines for each model
        for metric_idx, metric in enumerate(metrics):
            # if metric_idx == 1: continue
            metric_values = [data[model][task][metric_idx] for model in model_names]
            ax.plot(model_names, metric_values, marker='o', label=metric, color=colors[metric_idx])

        # Set axis labels and ticks
        ax.set_title(task)
        ax.set_ylim([0, 1])
        
        # Add grid lines
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        # Add legend outside the plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # Adjust the layout to accommodate the legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(Path(save_path, 'line' + prefix +'.png'))
    
    
def compare_model_task_metric(data, tasks, model_names, metrics, save_path=None, fig_format='heatmap', prefix=''):
    '''
    Example data:
    model_names = ['PPE', 'HLR', 'HOU']
    tasks = ['Train 50%', 'Train 40%']
    metrics = ['Accuracy', 'AUC', 'F1 Score', 'Precision', 'Recall']

    data = {
        'PPE': {
            'Train 50%': [0.1731, 0.5178, 0.1481, 0.9159, 0.0806],
            'Train 40%': [0.1713, 0.5178, 0.1443, 0.9164, 0.0783],
        },
        'HLR': {
            'Train 50%': [0.8477, 0.5094, 0.9169, 0.8928, 0.9424],
            'Train 40%': [0.7716, 0.5019, 0.8648, 0.8619, 0.8678],
        },
        'HOU': {
            'Train 50%': [0.8718, 0.5391,  0.9310, 0.8957, 0.9692],
            'Train 40%': [0.8581, 0.8021, 0.9206, 0.8927, 0.9502],
        }
    }
    '''
    
    num_models = len(model_names)
    num_tasks = len(tasks)
    num_metrics = len(metrics)
    
    if fig_format == 'heatmap':
        _create_heatmap(data, tasks, model_names, metrics, save_path)
    elif fig_format == 'bar':
        _create_bar(data, tasks, model_names, metrics, save_path, prefix)
    elif fig_format == 'line':
        _create_line(data, tasks, model_names, metrics, save_path, prefix)











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
    Path(args.log_path, process).touch()
    plt.savefig(Path(args.log_path, process, prefix+'.png'))



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
        plt.savefig(Path(args.log_path, process, prefix+'_params_'+keys+'.png'))




def visualize_ground_truth(graph, args, adj, size=4.0):
    plt.clf()
    nx.draw(graph, with_labels=True)
    plt.savefig(Path(args.log_path, 'graph_raw.png'), dpi=300, bbox_inches='tight')

    plt.clf()
    plt.rcParams['figure.figsize'] = [size, size]
    fig, ax = plt.subplots(1, 1)
    ax.matshow(adj, vmin=0, vmax=1)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title(r'Ground truth $G^*$', pad=10)
    plt.savefig(Path(args.log_path, 'graph_adj.png'))
    
    

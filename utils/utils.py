# -*- coding: UTF-8 -*-

import os, pickle, datetime
import torch
import numpy as np
from collections import defaultdict

import ipdb

import os
import pickle

# a dict to store the activations
activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
 

def load_corpus(logs, args):
    '''
    Load corpus from the corpus path, and split the data into k folds. 

    Args:
        logs: An object to write logs to.
        args: An object that contains command-line arguments.

    Returns:
        The corpus object that contains the loaded data.
    '''

    # Construct the path to the corpus file based on the dataset and max_step arguments.
    corpus_path = os.path.join(args.data_dir, args.dataset, 'Corpus_{}.pkl'.format(args.max_step))
    logs.write_to_log_file(f"Load corpus from {corpus_path}")
    
    # Load the corpus object from the pickle file at the specified path.
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    
    # Check the value of the train_mode argument to determine the type of data split.
    if 'split_learner' in args.train_mode:
        # Generate folds of data for a learner-based split using the gen_fold_data method of the corpus object.
        corpus.gen_fold_data(args.fold)
        logs.write_to_log_file('# Training mode splits LEARNER')

    elif 'split_time' in args.train_mode:
        # Generate splits of data based on time ratios using the gen_time_split_data method of the corpus object.
        corpus.gen_time_split_data(args.train_time_ratio, args.test_time_ratio)
        logs.write_to_log_file('# Training mode splits TIME')
        
    # Write to logs the number of examples in the train, val, and test sets.
    logs.write_to_log_file('# Train: {}, # val: {}, # Test: {}'.format(
            len(corpus.data_df['train']), len(corpus.data_df['val']), len(corpus.data_df['test'])
        ))

    return corpus


def _get_feed_dict(keys, data, start, batch_size, pad_list=False, device='cpu'):
    '''
    Creates a PyTorch feed_dict for a batch of data.

    Args:
        keys: A dictionary mapping keys to their corresponding column names in the input data.
        data: A pandas DataFrame containing the input data.
        start: The starting index of the batch.
        batch_size: The size of the batch.
        pad_list: A boolean indicating whether to pad the sequences in the input data.

    Returns:
        A dictionary containing the input data as PyTorch tensors.
    '''

    # Create an empty dictionary to hold the feed_dict values
    feed_dict = {}
    
    # Iterate over the keys in the provided list
    for key, value in keys.items():
        # Extract the sequence of values for the current key from the input data
        seq = data[value][start: start + batch_size].values
        
        # If the key ends in '_seq' and the pad_list flag is True, pad the sequence
        if '_seq' in key or 'num_' in key:
            seq = pad_lst(seq)
        
        # Convert the sequence to a PyTorch tensor and add it to the feed_dict dictionary
        feed_dict[key] = torch.as_tensor(seq).to(device)
        
    return feed_dict


def format_arg_str(args, exclude_lst, max_len=20):
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len-3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str


def format_metric(metric):
    assert(type(metric) == dict)
    format_str = []
    for name in np.sort(list(metric.keys())):
        m = metric[name]
        if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
            format_str.append('{}:{:<.4f}'.format(name, m))
        elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
            format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)


def check_dir(file_name):
    '''
    Checks if the directory containing the specified file exists, and creates it if necessary.

    Args:
        file_name: The name of the file to check.

    Returns:
        None
    '''

    # Get the path to the directory containing the specified file.
    dir_path = os.path.dirname(file_name)

    # If the directory doesn't exist, create it.
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)



def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def strictly_increasing(l):
    return all(x < y for x, y in zip(l, l[1:]))


def strictly_decreasing(l):
    return all(x > y for x, y in zip(l, l[1:]))


def non_increasing(l):
    return all(x >= y for x, y in zip(l, l[1:]))


def non_decreasing(l):
    return all(x <= y for x, y in zip(l, l[1:]))


def monotonic(l):
    return non_increasing(l) or non_decreasing(l)


# https://github.com/tswsxk/longling/blob/2cc45688e183b3395ada129ec54db7bd00959cbb/longling/lib/candylib.py#L17
def as_list(obj) -> list:
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]


def create_rel_rec_send(num_atoms, device):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    
    # Generate off-diagonal interaction graph
    off_diag = np.ones([num_atoms, num_atoms])
    # ipdb.set_trace()
    rel_rec = np.array(np.where(off_diag)[0])
    rel_send = np.array(np.where(off_diag)[1])
    rel_rec = torch.tensor(rel_rec).to(device)
    rel_send = torch.tensor(rel_send).to(device)

    return rel_rec, rel_send


def distribute_over_GPUs(args, model, num_GPU=None):
    ## distribute over GPUs
    # if torch.device("cpu") not in args.device:
    if args.device.type != "cpu":
        if num_GPU is None:
            model = torch.nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            args.batch_size_multiGPU = args.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = torch.nn.DataParallel(model, device_ids=list(range(num_GPU)))
            args.batch_size_multiGPU = args.batch_size * num_GPU
    else:
        model = torch.nn.DataParallel(model)
        args.batch_size_multiGPU = args.batch_size

    model = model.to(args.device)

    return model, num_GPU


class ConfigDict(dict):
    """Configuration dictionary that allows the `.` access.
    Example:
    ```python
    config = ConfigDict()
    config.test_number = 1
    ```
    The content could be access by
    ```python
    print(config.test_number)  # 1 will be returned.
    ```
    """
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v
    def __getattr__(self, attr):
        return self.get(attr)
    def __setattr__(self, key, value):
        self.__setitem__(key, value)
    def __setitem__(self, key, value):
        super(ConfigDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})
    def __delattr__(self, item):
        self.__delitem__(item)
    def __delitem__(self, key):
        super(ConfigDict, self).__delitem__(key)
        del self.__dict__[key]
        
        
        
def pad_lst(lst, value=-1, dtype=np.int64):
    # Find the maximum length of any row in the input list
    inner_max_len = max(map(len, lst))
    
    # Create a new array with the same number of rows as the input list
    # and the maximum row length as the number of columns
    result = np.ones([len(lst), inner_max_len], dtype) * value
    
    # Iterate over each row of the input list
    for i, row in enumerate(lst):
        # Iterate over each element in the row
        for j, val in enumerate(row):
            # Copy the element value to the corresponding position in the new array
            result[i][j] = val
            
    return result



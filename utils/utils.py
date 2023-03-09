# -*- coding: UTF-8 -*-

import os, torch
import logging
import datetime
import numpy as np
from collections import defaultdict

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


def pad_lst(lst, value=0, dtype=np.int64):
    inner_max_len = max(map(len, lst))
    result = np.ones([len(lst), inner_max_len], dtype) * value
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result


def check_dir(file_name):
    dir_path = os.path.dirname(file_name)
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
    r"""A utility function that converts the argument to a list
    if it is not already.
    Parameters
    ----------
    obj : object
        argument to be converted to a list
    Returns
    -------
    list_obj: list
        If `obj` is a list or tuple, return it. Otherwise,
        return `[obj]` as a single-element list.
    Examples
    --------
    >>> as_list(1)
    [1]
    >>> as_list([1])
    [1]
    >>> as_list((1, 2))
    [1, 2]
    """
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]


def append_losses(losses_list, losses):
    for loss, value in losses.items():
        if type(value) == float:
            losses_list[loss].append(value)
        elif type(value) == defaultdict:
            if losses_list[loss] == []:
                losses_list[loss] = defaultdict(list)
            for idx, elem in value.items():
                losses_list[loss][idx].append(elem)
        else:
            losses_list[loss].append(value.item())
    return losses_list


# def create_rel_rec_send(args, num_atoms):
#     """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
#     if args.unobserved > 0 and args.model_unobserved == 1:
#         num_atoms -= args.unobserved

#     # Generate off-diagonal interaction graph
#     off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
#     # ipdb.set_trace()
#     rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
#     rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
#     rel_rec = torch.FloatTensor(rel_rec)
#     rel_send = torch.FloatTensor(rel_send)

#     if args.cuda:
#         rel_rec = rel_rec.cuda()
#         rel_send = rel_send.cuda()

#     return rel_rec, rel_send


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
import torch
import torch.optim as optim

OPTIMIZER_MAP = {
    'gd': optim.SGD,
    'adagrad': optim.Adagrad,
    'adadelta': optim.Adadelta,
    'adam': optim.Adam
}
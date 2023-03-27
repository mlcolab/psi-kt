from typing import List, Optional, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Dropout, LayerNorm, Linear, Module, Sequential


def generate_fully_connected(
    input_dim: int,
    output_dim: int,
    hidden_dims,
    non_linearity,
    activation,
    device,
    p_dropout = 0.0,
    normalization = None,
    res_connection = False,
):
    layers: List[Module] = []

    prev_dim = input_dim
    for idx, hidden_dim in enumerate(hidden_dims):

        block: List[Module] = []

        if normalization is not None and idx > 0:
            block.append(normalization(prev_dim).to(device))
        block.append(Linear(prev_dim, hidden_dim).to(device))

        if non_linearity is not None:
            block.append(non_linearity())
        if p_dropout != 0:
            block.append(Dropout(p_dropout))

        if res_connection and (prev_dim == hidden_dim):
            layers.append(resBlock(Sequential(*block)))
        else:
            layers.append(Sequential(*block))
        prev_dim = hidden_dim

    if normalization is not None:
        layers.append(normalization(prev_dim).to(device))
    layers.append(Linear(prev_dim, output_dim).to(device))

    if activation is not None:
        layers.append(activation())

    fcnn = Sequential(*layers)

    return fcnn


class MLP(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.0, use_batch_norm=True, final_linear=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.use_batch_norm = use_batch_norm
        self.final_linear = final_linear
        if self.final_linear:
            self.fc_final = nn.Linear(n_out, n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        if self.final_linear:
            x = self.fc_final(x)
        if self.use_batch_norm:
            return self.batch_norm(x)
        else:
            return x


class resBlock(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)
        


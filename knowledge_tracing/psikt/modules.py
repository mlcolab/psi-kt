from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, Module, Sequential


class VAEEncoder(nn.Module):
    """
    A simple implementation of Gaussian MLP Encoder and Decoder
    Modified from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        tanh: bool = False,
    ) -> None:
        super(VAEEncoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Tanh = nn.Tanh()
        self.tanh = tanh

        self.training = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        if self.tanh:
            return self.Tanh(mean), self.Tanh(log_var)
        else:
            return mean, log_var


def build_rnn_cell(rnn_type: str, hidden_dim_rnn: int, rnn_input_dim: int) -> nn.Module:
    """
    Build a PyTorch RNN cell with the specified type, hidden dimension, and input dimension.

    Args:
        rnn_type: a string representing the type of RNN cell to use (e.g., "GRU", "LSTM", "SimpleRNN")
        hidden_dim_rnn: an integer representing the size of the hidden state of the RNN cell
        rnn_input_dim: an integer representing the size of the input to the RNN cell

    Returns:
        rnn_cell: a PyTorch RNN cell of the specified type
    """
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        rnn_cell = nn.GRUCell(input_size=rnn_input_dim, hidden_size=hidden_dim_rnn)
    elif rnn_type == "lstm":
        rnn_cell = nn.LSTMCell(input_size=rnn_input_dim, hidden_size=hidden_dim_rnn)
    elif rnn_type == "simplernn":
        rnn_cell = nn.RNNCell(input_size=rnn_input_dim, hidden_size=hidden_dim_rnn)
    else:
        raise ValueError(f"Invalid RNN type: {rnn_type}")
    return rnn_cell


def build_dense_network(
    input_size: int, layer_sizes: int, layer_activations: List[nn.Module]
) -> nn.Module:
    """
    Build a multi-layer neural network with the specified input size, layer sizes, and layer activations.

    Args:
    - input_size: an integer representing the size of the input layer
    - layer_sizes: a list of integers representing the sizes of the hidden layers
    - layer_activations: a list of activation functions, or None for linear activations

    Returns:
    - nets: a PyTorch sequential model representing the multi-layer neural network
    """
    modules = []
    for lsize, activation in zip(layer_sizes, layer_activations):
        modules.append(nn.Linear(input_size, lsize))
        if activation is not None:
            modules.append(activation)
        input_size = lsize
    nets = nn.Sequential(*modules)
    return nets


def generate_fully_connected(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    non_linearity: nn.Module,
    activation: nn.Module,
    device: torch.device,
    p_dropout: float = 0.0,
    normalization: nn.Module = None,
    res_connection: bool = False,
):
    """
    Generate a fully connected neural network module.

    This function constructs a fully connected neural network (FCNN) module with
    customizable architecture, including the number of hidden layers, activation
    functions, dropout layers, normalization layers, and optional residual connections.

    Args:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_dims (List[int]): A list of integers specifying the dimensions of the
            hidden layers in the FCNN.
        non_linearity (nn.Module): The non-linearity function (e.g., ReLU) applied
            between layers.
        activation (nn.Module): The activation function applied to the final output
            layer (e.g., softmax for classification).
        device (torch.device): The device (e.g., 'cpu' or 'cuda') on which the
            FCNN will be placed.
        p_dropout (float, optional): The dropout probability. Default is 0.0 (no dropout).
        normalization (nn.Module, optional): The normalization layer (e.g., BatchNorm)
            applied between hidden layers. Default is None (no normalization).
        res_connection (bool, optional): Whether to use residual connections between
            hidden layers with matching dimensions. Default is False (no residual
            connections).

    Returns:
        nn.Module: A fully connected neural network module with the specified architecture.

    """
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

    def __init__(
        self, n_in, n_hid, n_out, do_prob=0.0, use_batch_norm=True, final_linear=False
    ):
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

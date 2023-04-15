from typing import List, Optional, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Dropout, LayerNorm, Linear, Module, Sequential
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import ipdb


"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""
# https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
class VAEEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEEncoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     #
        
        return mean, log_var


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
# https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

# https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class CausalTransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead=4, nhid=32, nlayers=2, dropout=0.0):
        super(CausalTransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.hidden_dim = nhid
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        # self.encoder = nn.Embedding(ntoken, ninp)
        # self.decoder = nn.Linear(ninp, ntoken)

        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).contiguous()
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _to_device(self, device):
        for _, value in self.transformer_encoder.state_dict().items():
            value = value.to(device)
    # def init_weights(self):
    #     initrange = 0.1
    #     # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    #     # nn.init.zeros_(self.decoder.bias)
    #     # nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(src.shape[1])# .to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        self._to_device(src.device)
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        # device = self.transformer_encoder.state_dict()['layers.0.self_attn.in_proj_weight'].device
        output = self.transformer_encoder(src, self.src_mask.to(src.device)) # (batch_size, seq_len, hidden_dim)
        # output = self.decoder(output)
        return output# F.log_softmax(output, dim=-1)
    
    
    
# https://github.com/BasselMahrousseh/GPT/blob/main/GPT_Model.ipynb
# def causal_attention_mask(batch_size, n_dest, n_src, dtype):
#     """
#     Mask the upper half of the dot product matrix in self attention.
#     This prevents flow of information from future tokens to current token.
#     1's in the lower triangle, counting from the lower right corner.
#     """
#     i = tf.range(n_dest)[:, None]
#     j = tf.range(n_src)
#     m = i >= j - n_src + n_dest
#     mask = tf.cast(m, dtype)
#     mask = tf.reshape(mask, [1, n_dest, n_src])
#     mult = tf.concat(
#         [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
#     )
#     return tf.tile(mask, mult)


# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super().__init__()
#         self.att = layers.MultiHeadAttention(num_heads, embed_dim)
#         self.ffn = keras.Sequential(
#             [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
#         )
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)

#     def forward(self, inputs):
#         input_shape = tf.shape(inputs)
#         batch_size = input_shape[0]
#         seq_len = input_shape[1]
#         causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
#         attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
#         attention_output = self.dropout1(attention_output)
#         out1 = self.layernorm1(inputs + attention_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output)
#         return self.layernorm2(out1 + ffn_output)















def build_rnn_cell(rnn_type: str, hidden_dim_rnn: int, rnn_input_dim: int):
    """
    Build a PyTorch RNN cell with the specified type, hidden dimension, and input dimension.

    Parameters:
    - rnn_type: a string representing the type of RNN cell to use (e.g., "GRU", "LSTM", "SimpleRNN")
    - hidden_dim_rnn: an integer representing the size of the hidden state of the RNN cell
    - rnn_input_dim: an integer representing the size of the input to the RNN cell

    Returns:
    - rnn_cell: a PyTorch RNN cell of the specified type
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


def build_dense_network(input_size, layer_sizes, layer_activations):
    """
    Build a multi-layer neural network with the specified input size, layer sizes, and layer activations.

    Parameters:
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
        


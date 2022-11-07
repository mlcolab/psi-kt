from typing import List, Optional, Type

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from models.modules import generate_fully_connected

from torch.nn import Dropout, LayerNorm, Linear, Module, Sequential
from utils.utils import create_rel_rec_send

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

class Graph(nn.Module):
    def __init__(self, device, num_nodes, args):
        super(Graph, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.args = args

        self.norm_layer = LayerNorm
        self.dropout = args.dropout
        self.res_connection = True
        self.encoder_layer_sizes = None
        self.decoder_layer_sizes = None # TODO

        self.time_log = args.time_log
        self.time_lag = args.time_lag

        self.emb_history = args.emb_history
        self.skill_base = args.skill_base
        self.problem_base = args.problem_base

        self.rec_ind, self.send_ind = create_rel_rec_send(self.num_nodes, self.device)

    def _init_weights(self):
        pass
    def forward(self, feed_dict):
        pass


class GraphTargetMessage(Graph):
    def __init__(self, device, num_nodes, args, emb_size=None):
        super().__init__(device, num_nodes, args)
        self.emb_size = emb_size or args.emb_size
        self._init_weights()

    def _init_weights(self):
        a = max(self.emb_size, 64)
        layers_f = self.decoder_layer_sizes or [a, a]
        in_dim_f = self.emb_size
        layers_g = self.encoder_layer_sizes or [a, a]
        in_dim_g = self.emb_size*2

        if self.emb_history == 1:
            self.emb_history_layer = generate_fully_connected(
                input_dim=in_dim_g,
                output_dim=self.emb_size,
                hidden_dims=layers_g,
                p_dropout=self.dropout,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=self.device,
                normalization=self.norm_layer,
                res_connection=self.res_connection,
            )

        self.interv_edges = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=self.emb_size,
            hidden_dims=layers_g,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )
        
        self.final_output_layer = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=2,
            hidden_dims=layers_f,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )


    def forward(self, feed_dict, adj, node_embeddings, node_base=None, node_bias_base=None):

        skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        problems = feed_dict['problem_seq']  # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]
        
        bs, time_steps = labels.shape
        time_lag = self.args.time_lag

        bs_skill_emb = torch.tile(node_embeddings, (bs, 1, 1)).to(self.device)
        # bs_skill_base = torch.tile(node_base, (bs, 1)).to(self.device)
        # bs_problem_base = torch.tile(node_bias_base, (bs, 1)).to(self.device)

        if self.emb_history == 0:
            labels_pro = torch.where(labels==0, -2, labels)
            labels_pro = torch.where(labels_pro==-1, 0, labels_pro)
            labels_pro = torch.where(labels_pro==-2, -1, labels_pro)
        else: labels_pro = labels
        
        delta_t = (times[:, :, None] - times[:, None, :]).abs().double() # [bs, seq_len, seq_len] # symmetry ranging from 13.47e8-13.74e8
        delta_t = torch.log(delta_t + 1e-10) / torch.log(torch.tensor(self.time_log)) # ??? time_log # ranging from 0 to 17
        
        preds = []

        # TODO
        for t in range(time_lag, time_steps):
            
            label_history = labels_pro[:, t - time_lag:t] # [bs, time_lag]
            skill_history = skills[:, t - time_lag:t] # [bs, time_lag]
        
            skill_target = skills[:, t:t+1] # [bs, 1]
            problem_target = problems[:, t:t+1]
            # skill_base_target = bs_skill_base[torch.arange(0, bs), skill_target[:, 0]].unsqueeze(-1) # [bs, 1]
            # problem_base_target = bs_problem_base[torch.arange(0, bs), problem_target[:, 0]].unsqueeze(-1) # [bs, 1]

            delta_t_weight = delta_t.transpose(-1,-2)[:, t, t - time_lag:t] # [bs, time_lag]  range from 0 to around 10
            emb_history = [bs_skill_emb[torch.arange(0, bs), skill_history[:, i]] for i in range(time_lag)]
            emb_history = torch.stack(emb_history, dim=1) # [bs, time_lag, emd_size]

            if self.emb_history == 0:
                emb_history *= label_history.unsqueeze(-1)
            elif self.emb_history == 1:
                label_history = torch.tile(label_history.unsqueeze(-1), (1, 1, self.embedding_size))
                emb_history = torch.cat([emb_history, label_history], dim=-1).double()
                emb_history = self.emb_history_layer(emb_history)

            # w_ij means the effect from i to j 
            # column j has all of the effects coming to j
            # so index in j-th row of w_transpose  
            graph_weight = adj.transpose(-1,-2)[:, torch.arange(0, bs), skill_target[:, 0]] # [num_graph, bs, num_nodes]
            cross_weight = [graph_weight[:, torch.arange(0, bs), skill_history[:, i]] for i in range(time_lag)]
            cross_weight = torch.stack(cross_weight, -1) # [bs, time_lag]
            
            emb_target = torch.tile(bs_skill_emb[torch.arange(0, bs), skill_target[:,0]].unsqueeze(1), (1, time_lag, 1))
            edge_feat = torch.cat([emb_target, emb_history], dim=-1).double() # [bs, time_lag, emb_size*2]

            msg = self.interv_edges(edge_feat) # [bs, time_lag, emb_size]

            # use a softmax here? TODO
            cross_effect = (cross_weight * torch.exp(-delta_t_weight)).unsqueeze(-1) * msg
            cross_effect = cross_effect.sum(-2) # [num_graph, bs, emb_size]

            # pred = (self.final_output_layer(cross_effect) + problem_base_target + skill_base_target).sigmoid()
            # pred = (self.final_output_layer(cross_effect)).sigmoid()
            pred = F.softmax(self.final_output_layer(cross_effect), dim=-1)[..., 1:] # problem_base_target + skill_base_target
            preds.append(pred)



class GraphRecurrent(Graph): 
    def __init__(self, device, num_nodes, args, emb_size=None):
        super().__init__(device, num_nodes, args)
        self.emb_size = emb_size or args.emb_size
        self._init_weights()

    def _init_weights(self):
        a = max(self.emb_size, 64)
        n_hid = self.emb_size # ???

        layers_g = self.encoder_layer_sizes or [a, a]
        in_dim_g = self.emb_size*2

        if self.emb_history == 1:
            self.emb_history_layer = generate_fully_connected(
                input_dim=in_dim_g,
                output_dim=self.emb_size,
                hidden_dims=layers_g,
                p_dropout=self.dropout,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=self.device,
                normalization=self.norm_layer,
                res_connection=self.res_connection,
            )

        self.msg_fc1 = nn.Linear(2 * n_hid, n_hid)
        self.msg_fc2 = nn.Linear(n_hid, n_hid) 
        
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = True

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        # n_in_node = self.num_nodes
        n_in_node = n_hid
        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

    def forward(self, feed_dict, adj, node_embeddings):
        '''
        node_embeddings: [num_graph, num_nodes, emb_size]
        '''
        skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        problems = feed_dict['problem_seq']  # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]
        
        bs, time_steps = labels.shape
        time_lag = self.args.time_lag
        num_graph = adj.shape[0]
        
        node_embeddings = node_embeddings.double()
        hidden = torch.zeros(bs, self.num_nodes, self.msg_out_shape).to(self.device)

        # inputs = data.transpose(1, 2).contiguous()
        # time_steps = inputs.size(1)
        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]
        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        pred_all = []

        for t in range(0, time_steps - 1):
            ins = skills[:, t:t+1]

            # node2edge
            receivers = torch.index_select(hidden, dim=1, index=self.rec_ind) # [bs, num_edges, dim]
            senders = torch.index_select(hidden, dim=1, index=self.send_ind)
            pre_msg = torch.cat([senders, receivers], dim=-1).double()

            # Run separate MLP for every edge type
            rel_type = adj.reshape(num_graph, bs, -1) # [num_graph, bs, num_edge]
            # NOTE: To exclude one edge type, simply offset range by 1
            msg = torch.tanh(self.msg_fc1(pre_msg))
            msg = F.dropout(msg, p=self.dropout)
            msg = torch.tanh(self.msg_fc2(msg)) # [bs, num_edge, dim]
            
            msg = msg * rel_type.unsqueeze(-1) # [num_graph, bs, num_edge, dim]
            agg_msgs = msg.reshape(num_graph, bs, self.num_nodes, self.num_nodes, -1)
            agg_msgs = agg_msgs.sum(-2)
            agg_msgs = agg_msgs.contiguous() / self.num_nodes  # Average

            # GRU-style gated aggregation
            bs_emb = node_embeddings.repeat(bs, 1, 1)
            inputs = bs_emb[torch.arange(bs), ins[:,0]]
            inputs = inputs.reshape(1, bs, 1, self.emb_size)
            r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
            i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
            n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
            hidden = (1 - i) * n + i * hidden # [num_graph, bs, num_nodes, emb_size]

            # Output MLP
            pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout)
            pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout)
            pred = self.out_fc3(pred)

            # if burn_in:
            #     if step <= burn_in_steps:
            #         ins = inputs[:, step, :, :]
            #     else:
            #         ins = pred_all[step - 1]
            # else:
            #     assert pred_steps <= time_steps
            #     # Use ground truth trajectory input vs. last prediction
            #     if not step % pred_steps:
            #         ins = inputs[:, step, :, :]
            #     else:
            #         ins = pred_all[step - 1]

            # # node2edge
            # receivers = torch.matmul(rel_rec, hidden) # hidden states of receiver nodes [bs, num_nodes, hid_dim]
            # senders = torch.matmul(rel_send, hidden)
            # pre_msg = torch.cat([senders, receivers], dim=-1)

            # agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
            # agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

            # # GRU-style gated aggregation
            # r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
            # i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
            # n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
            # hidden = (1 - i) * n + i * hidden

            # # Output MLP
            # pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout)
            # pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout)
            # pred = self.out_fc3(pred)

            # # Predict position/velocity difference
            # pred = inputs + pred
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()








class GraphWholeMessage(Graph): # TODO
    def __init__(self, device, num_nodes, args, emb_size=None):
        super().__init__(device, num_nodes, args)
        self.emb_size = emb_size or args.emb_size
        self._init_weights()

    def _init_weights(self):
        a = max(self.emb_size, 64)
        layers_f = self.decoder_layer_sizes or [a, a]
        in_dim_f = self.emb_size
        layers_g = self.encoder_layer_sizes or [a, a]
        in_dim_g = self.emb_size*2

        if self.emb_history == 1:
            self.emb_history_layer = generate_fully_connected(
                input_dim=in_dim_g,
                output_dim=self.emb_size,
                hidden_dims=layers_g,
                p_dropout=self.dropout,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=self.device,
                normalization=self.norm_layer,
                res_connection=self.res_connection,
            )

        self.interv_edges = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=self.emb_size,
            hidden_dims=layers_g,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )
        
        self.final_output_layer = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=2,
            hidden_dims=layers_f,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )


    def forward(self, feed_dict, adj, node_embeddings, node_base=None, node_bias_base=None):

        skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        problems = feed_dict['problem_seq']  # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]
        
        bs, time_steps = labels.shape
        time_lag = self.args.time_lag

        bs_skill_emb = torch.tile(node_embeddings, (bs, 1, 1)).to(self.device)
        # bs_skill_base = torch.tile(node_base, (bs, 1)).to(self.device)
        # bs_problem_base = torch.tile(node_bias_base, (bs, 1)).to(self.device)

        if self.emb_history == 0:
            labels_pro = torch.where(labels==0, -2, labels)
            labels_pro = torch.where(labels_pro==-1, 0, labels_pro)
            labels_pro = torch.where(labels_pro==-2, -1, labels_pro)
        else: labels_pro = labels
        
        delta_t = (times[:, :, None] - times[:, None, :]).abs().double() # [bs, seq_len, seq_len] # symmetry ranging from 13.47e8-13.74e8
        delta_t = torch.log(delta_t + 1e-10) / torch.log(torch.tensor(self.time_log)) # ??? time_log # ranging from 0 to 17
        
        preds = []

        # TODO
        for t in range(time_lag, time_steps):
            
            label_history = labels_pro[:, t - time_lag:t] # [bs, time_lag]
            skill_history = skills[:, t - time_lag:t] # [bs, time_lag]
        
            skill_target = skills[:, t:t+1] # [bs, 1]
            problem_target = problems[:, t:t+1]
            
            # skill_base_target = bs_skill_base[torch.arange(0, bs), skill_target[:, 0]].unsqueeze(-1) # [bs, 1]
            # problem_base_target = bs_problem_base[torch.arange(0, bs), problem_target[:, 0]].unsqueeze(-1) # [bs, 1]

            delta_t_weight = delta_t.transpose(-1,-2)[:, t, t - time_lag:t] # [bs, time_lag]  range from 0 to around 10
            emb_history = [bs_skill_emb[torch.arange(0, bs), skill_history[:, i]] for i in range(time_lag)]
            emb_history = torch.stack(emb_history, dim=1) # [bs, time_lag, emd_size]

            if self.emb_history == 0:
                emb_history *= label_history.unsqueeze(-1)
            elif self.emb_history == 1:
                label_history = torch.tile(label_history.unsqueeze(-1), (1, 1, self.embedding_size))
                emb_history = torch.cat([emb_history, label_history], dim=-1).double()
                emb_history = self.emb_history_layer(emb_history)

            # w_ij means the effect from i to j 
            cross_weight = [adj[:, torch.arange(0, bs), skill_history[:,i]] for i in range(time_lag)]
            cross_weight = torch.stack(cross_weight, 2) # [num_graph, bs, time_lag, num_nodes]
            # before edge feat, the emb should be broadcasted
            emb_history1 = emb_history.unsqueeze(2).repeat(1, 1, self.num_nodes, 1)
            bs_skill_emb1 = bs_skill_emb.unsqueeze(1).repeat(1, time_lag, 1, 1)
            edge_feat = torch.cat([emb_history1, bs_skill_emb1], dim=-1).double() # [bs, time_lag, num_nodes, dim_edge_feat]

            msg = self.interv_edges(edge_feat) # [bs, time_lag, num_nodes, emb_size]
            # use a softmax here? TODO
            cross_effect = (cross_weight * torch.exp(-delta_t_weight.unsqueeze(-1))).unsqueeze(-1) * msg # [num_graph, bs, time_lag, num_nodes, dim]
            cross_effect = cross_effect.sum(-2) # [num_graph, bs, emb_size]

            # column j has all of the effects coming to j
            # so index in j-th row of w_transpose  

            graph_weight = adj.transpose(-1,-2)[:, torch.arange(0, bs), skill_target[:, 0]] # [num_graph, bs, num_nodes]
            cross_weight = [graph_weight[:, torch.arange(0, bs), skill_history[:, i]] for i in range(time_lag)]
            cross_weight = torch.stack(cross_weight, -1) # [bs, time_lag]
            
            emb_target = torch.tile(bs_skill_emb[torch.arange(0, bs), skill_target[:,0]].unsqueeze(1), (1, time_lag, 1))
            edge_feat = torch.cat([emb_target, emb_history], dim=-1).double() # [bs, time_lag, emb_size*2]

            



            # pred = (self.final_output_layer(cross_effect) + problem_base_target + skill_base_target).sigmoid()
            # pred = (self.final_output_layer(cross_effect)).sigmoid()
            pred = F.softmax(self.final_output_layer(cross_effect), dim=-1)[..., 1:] # problem_base_target + skill_base_target
            preds.append(pred)

from typing import List, Optional, Type

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from models.modules import generate_fully_connected

from time import time 

import ipdb

class InterventionalGraph(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        dropout_prob = 0.,
        norm_layer = None,
        res_connection = True,
        encoder_layer_sizes = None,
        decoder_layer_sizes = None,
        embedding_size = None, 
        time_lag = 1, 
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.dropout_prob = dropout_prob
        self.time_lag = time_lag
        
        self.embedding_size = embedding_size
        self._init_embeddings()
        self.W = self._initialize_W()

        a = max(self.embedding_size, 64)
        layers_g = encoder_layer_sizes or [a, a]
        layers_f = decoder_layer_sizes or [a, a]
        in_dim_g = self.embedding_size*2
        in_dim_f = self.embedding_size
        
        self.g = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=self.embedding_size,
            hidden_dims=layers_g,
            p_dropout=dropout_prob,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )

        self.f = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=1,
            hidden_dims=layers_f,
            p_dropout=dropout_prob,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )

        self.interv_edges = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=self.embedding_size,
            hidden_dims=layers_g,
            p_dropout=dropout_prob,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )

    def _initialize_W(self):
        W = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        return nn.Parameter(W, requires_grad=True)

    def _init_embeddings(self):
        skill_base = torch.randn(1, self.num_nodes, device=self.device)
        self.skill_base = nn.Parameter(skill_base, requires_grad=True)

        alpha_skill_embeddings = torch.randn(1, self.num_nodes, self.embedding_size, device=self.device)
        self.alpha_skill_embeddings = nn.Parameter(alpha_skill_embeddings, requires_grad=True)
        # alpha_inter_embeddings = torch.randn(1, self.skill_num*2, self.embedding_size, device=self.device)
        # self.alpha_inter_embeddings = nn.Parameter(alpha_inter_embeddings, requires_grad=True)

        # self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        # self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        # self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        # self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

    def get_weighted_adjacency(self):
        W_adj = self.W * (1.0 - torch.eye(self.num_nodes, device=self.device))  # Shape (num_nodes, num_nodes)
        return W_adj

    def predict(self, X, W_adj): # X [bs, 50, time, 2]
        skills, times, labels = X # [bs, time_seq]
        bs, time_steps = labels.shape
        time_lag = self.time_lag

        if len(W_adj.shape) == 2:
            W_adj = W_adj.unsqueeze(0)
            bs_W = torch.tile(W_adj, (bs, 1, 1))

        bs_skill_emd = torch.tile(self.alpha_skill_embeddings, (bs, 1, 1)).to(self.device)
        bs_skill_base = torch.tile(self.skill_base, (bs, 1)).to(self.device)

        labels_pro = torch.where(labels==0, -2, labels)
        labels_pro = torch.where(labels_pro==-1, 0, labels_pro)
        labels_pro = torch.where(labels_pro==-2, -1, labels_pro)
        
        time_log = torch.tensor(5,)
        delta_t = (times[:, :, None] - times[:, None, :]).abs().double() # [bs, seq_len, seq_len] # symmetry
        delta_t = torch.log(delta_t + 1e-10) / torch.log(time_log) # ??? time_log

        preds = []
        for t in range(time_lag, time_steps):
            ipdb.set_trace()
            label_history = labels_pro[:, t - time_lag:t] # [bs, time_lag]
            skill_history = skills[:, t - time_lag:t] # [bs, time_lag]
            
            skill_target = skills[:, t:t+1] # [bs, 1]
            skill_base_target = bs_skill_base[torch.arange(0, bs), skill_target[:, 0]].unsqueeze(-1) # [bs, 1]

            # w_ij means the effect from i to j 
            # column j has all of the effects coming to j
            # so index in j-th row of w_transpose  
            graph_weight = bs_W.transpose(-1,-2)[torch.arange(0, bs), skill_target[:, 0]] # [bs, num_nodes]
            cross_weight = [graph_weight[torch.arange(0, bs), skill_history[:, i]] for i in range(time_lag)]
            cross_weight = torch.stack(cross_weight, -1) # [bs, time_lag]

            # TODO: why it is not monotonic
            delta_t_weight = delta_t.transpose(-1,-2)[:, t, t - time_lag:t] # [bs, time_lag]

            emb_history = [bs_skill_emd[torch.arange(0, bs), skill_history[:, i]] for i in range(time_lag)]
            emb_history = torch.stack(emb_history, dim=1) # [bs, time_lag, emd_size]
            emb_history *= label_history.unsqueeze(-1)

            emb_target = torch.tile(bs_skill_emd[torch.arange(0, bs), skill_target[:,0]].unsqueeze(1), (1, time_lag, 1))
            edge_feat = torch.cat([emb_target, emb_history], dim=-1) # [bs, time_lag, emb_size*2]
            msg = self.interv_edges(edge_feat) # [bs, time_lag, emb_size]

            cross_effect = (cross_weight * torch.exp(-delta_t_weight)).unsqueeze(-1) * msg
            cross_effect = cross_effect.sum(1) + skill_base_target
            pred = self.f(cross_effect).sigmoid()
            
            preds.append(pred)
        preds = torch.cat(preds, dim=-1)
        if skills.is_cuda:
            preds = preds.cuda()
        return preds


    def simulate_SEM(
        self,
        Z: torch.Tensor,
        W_adj: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None,
        gumbel_max_regions: Optional[List[List[int]]] = None,
        gt_zero_region: Optional[List[int]] = None,
    ):
        X = torch.zeros_like(Z)

        for _ in range(self.num_nodes):
            if intervention_mask is not None and intervention_values is not None:
                X[:, intervention_mask] = intervention_values.unsqueeze(0)
            X = self.f.feed_forward(X, W_adj) + Z
            if gumbel_max_regions is not None:
                for region in gumbel_max_regions:
                    maxes = X[:, region].max(-1, keepdim=True)[0]
                    X[:, region] = (X[:, region] >= maxes).float()
            if gt_zero_region is not None:
                X[:, gt_zero_region] = (X[:, gt_zero_region] > 0).float()

        if intervention_mask is not None and intervention_values is not None:
            if intervention_values.shape == X.shape:
                X[:, intervention_mask] = intervention_values
            else:
                X[:, intervention_mask] = intervention_values.unsqueeze(0)
        return X



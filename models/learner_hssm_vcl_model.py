import sys
sys.path.append('..')

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import torch
from torch import nn, distributions
from torch.nn.parameter import Parameter

from collections import defaultdict

import ipdb

from models.modules import build_rnn_cell, build_dense_network
from models.BaseModel import BaseModel
from models.modules import CausalTransformerModel, VAEEncoder
from models.HSSM import GraphHSSM

from enum import Enum

torch.autograd.set_detect_anomaly(True)

EPS = 1e-6
T_SCALE = 60*60*24


class GraphContinualHSSM(GraphHSSM):
    def __init__(
        self,
        mode='train', 
        num_node=1,
        num_seq=1,
        args=None,
        device='cpu',
        logs=None,
        nx_graph=None,
    ):
        super().__init__(mode, num_node, num_seq, args, device, logs, nx_graph)
        
        time_step_save = args.max_step
        num_seq_save = num_seq

        s_shape = (num_seq_save, 1, time_step_save, self.dim_s)
        z_shape = (num_seq_save, 1, time_step_save, self.dim_z)
        z_shape_pred = (num_seq_save, 1, time_step_save, 1)
        
        self.pred_s_means = Parameter(torch.zeros(s_shape, device=self.device), requires_grad=False)
        self.pred_s_vars = Parameter(torch.zeros(s_shape, device=self.device), requires_grad=False)
        self.infer_s_means = Parameter(torch.zeros(s_shape, device=self.device), requires_grad=False)
        self.infer_s_vars = Parameter(torch.zeros(s_shape, device=self.device), requires_grad=False)
        
        self.pred_s_means_update = Parameter(torch.zeros(s_shape, device=self.device), requires_grad=False)
        self.pred_s_vars_update = Parameter(torch.zeros(s_shape, device=self.device), requires_grad=False)
        self.infer_s_means_update = Parameter(torch.zeros(s_shape, device=self.device), requires_grad=False)
        self.infer_s_vars_update = Parameter(torch.zeros(s_shape, device=self.device), requires_grad=False)
        
        self.pred_z_means = Parameter(torch.zeros(z_shape_pred, device=self.device), requires_grad=False)
        self.pred_z_vars = Parameter(torch.zeros(z_shape_pred, device=self.device), requires_grad=False)
        self.infer_z_means = Parameter(torch.zeros(z_shape, device=self.device), requires_grad=False)
        self.infer_z_vars = Parameter(torch.zeros(z_shape, device=self.device), requires_grad=False)
        
        self.pred_z_means_update = Parameter(torch.zeros(z_shape_pred, device=self.device), requires_grad=False)
        self.pred_z_vars_update = Parameter(torch.zeros(z_shape_pred, device=self.device), requires_grad=False)
        self.infer_z_means_update = Parameter(torch.zeros(z_shape, device=self.device), requires_grad=False)
        self.infer_z_vars_update = Parameter(torch.zeros(z_shape, device=self.device), requires_grad=False)
        
        # TODO
        self.var_minimum = torch.log(torch.tensor(0.05).to(self.device))
        
        self.gen_network_transition_s = build_dense_network(                
                self.node_dim,
                [self.node_dim, self.node_dim],
                [nn.ReLU(), None]
            )
        
        self.transformer, self.rnn = 0,1
        self.infer_network_posterior_s = nn.LSTM(
                        input_size=self.node_dim, 
                        hidden_size=self.node_dim,  
                        bidirectional = False, 
                        batch_first = True,
                    )
        self.infer_network_posterior_z = nn.LSTM(
                        input_size=self.node_dim, 
                        hidden_size=self.node_dim,  
                        bidirectional = False, 
                        batch_first = True,
                    )
            
            
    def s_transition_infer(
        self, 
        feed_dict: Dict[str, torch.Tensor] = None,
        emb_inputs: torch.Tensor = None, 
        idx: int = None,
    ):
        
        emb_rnn_inputs = emb_inputs
        
        if self.transformer:
            output = self.infer_network_posterior_s(emb_rnn_inputs)
                
        elif self.rnn:
            output, _ = self.infer_network_posterior_s(emb_rnn_inputs)
            
        output = output[:, -1:]
        mean, log_var = self.infer_network_posterior_mean_var_s(output)  # [batch_size, time_step, dim_s]
        log_var = torch.minimum(log_var, self.var_minimum)
        cov_mat = torch.diag_embed(torch.exp(log_var) + EPS)
        dist_s = distributions.multivariate_normal.MultivariateNormal(
            loc=mean, 
            scale_tril=torch.tril(cov_mat)
        )
            
        return dist_s
 

    def z_transition_infer(
        self, 
        feed_dict: Dict[str, torch.Tensor] = None,
        emb_inputs: torch.Tensor = None,
        idx: int = None,
    ):
        '''
        Compute the posterior distribution of the latent variable `z_t` given the input and output sequences.

        Args:
            inputs (tuple): A tuple containing the feed dictionary and the sampled skills `s_t`.
            num_sample (int): Number of samples for Monte Carlo estimation of the posterior distribution.
            emb_inputs (torch.Tensor): Optional embedded input sequence of shape [batch_size, time_step, dim_item+dim_time].

        Returns:
            z_sampled (torch.Tensor): Sampled latent variable `z_t` of shape [batch_size*num_sample, 1, time_step, out_dim].
            z_entropy (torch.Tensor): Entropy of the posterior distribution of `z_t`.
            z_log_prob_q (torch.Tensor): Log probability of the posterior distribution of `z_t`.
            rnn_states (torch.Tensor): Output states of the posterior network.
            z_mean (torch.Tensor): Mean of the posterior distribution of `z_t`.
            z_log_var (torch.Tensor): Log variance of the posterior distribution of `z_t`.
        '''
        
        # NOTE: Embed the input sequence, if needed
        
        emb_rnn_inputs = emb_inputs
        
        if self.transformer:
            output = self.infer_network_posterior_z(emb_rnn_inputs)
            
        elif self.rnn:
            output, _ = self.infer_network_posterior_z(emb_rnn_inputs)
            
        output = output[:, -1:]
        mean, log_var = self.infer_network_posterior_mean_var_z(output)  # [batch_size, time_step, dim_s]
        log_var = torch.minimum(log_var, self.var_minimum)
        cov_mat = torch.diag_embed(torch.exp(log_var) + EPS)
        dist_z = distributions.multivariate_normal.MultivariateNormal(
            loc=mean, 
            scale_tril=torch.tril(cov_mat)
        )
            
        return dist_z


    def predictive_model(
        self, 
        feed_dict: Dict[str, torch.Tensor] = None,
        idx: int = None,
        eval: bool = False,
        update: bool = False,
        s_prior: torch.distributions.MultivariateNormal = None,
        z_prior: torch.distributions.MultivariateNormal = None,
    ):
        user = feed_dict['user_id']
        y_idx = feed_dict['label_seq'][:, idx:idx+1] # [bs, 1]
        t_idx = feed_dict['time_seq'][:, idx-1: idx+1] # [bs, 2]
        dt = torch.diff(t_idx, dim=-1)/T_SCALE + EPS 
        device = y_idx.device
        bs = y_idx.shape[0]
        
        if idx == 0:
            # p_theta(s_0)
            s_tilde_dist = distributions.MultivariateNormal(
                self.gen_s0_mean.to(device),  
                scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_s0_log_var.to(device)) + EPS))
            )
            s_tilde_dist_mean = s_tilde_dist.mean[None,...] # [1, 1, dim_s]
            s_tilde_dist_var = torch.diagonal(s_tilde_dist.scale_tril, dim1=1, dim2=2)[None,...]
            
            # p_theta(z_0)
            z_tilde_dist = distributions.MultivariateNormal(
                self.gen_z0_mean.to(device), 
                scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_z0_log_var.to(device)) + EPS))
            )
            z_tilde_dist_mean = z_tilde_dist.mean[None,...] # [1, 1, dim_z]
            z_tilde_dist_var = torch.diagonal(z_tilde_dist.scale_tril, dim1=1, dim2=2)[None,...]

            if self.dim_z > 1:
                # choose and save the items' mean and var 
                item_idx = feed_dict['skill_seq'][:, idx]# [bs, 1]
                node_emb = self.node_dist._get_node_embedding() # [bs, num_node]   [item_idx] # [bs, 1, dim_item]
                
                z_tilde_dist_mean_all = (z_tilde_dist_mean * node_emb).sum(-1, keepdim=True) # [1, num_node, 1]
                z_tilde_dist_mean = z_tilde_dist_mean_all[0][item_idx].unsqueeze(-1) # [bs, 1, 1]
                
                z_tilde_dist_var_all = (z_tilde_dist_var * node_emb).sum(-1, keepdim=True) # [1, num_node, 1]
                z_tilde_dist_var_all = torch.minimum(torch.maximum(z_tilde_dist_var_all, torch.tensor(EPS).to(self.device)), torch.exp(self.var_minimum)) # [1, num_node, 1]
                z_tilde_dist_var = z_tilde_dist_var_all[0][item_idx].unsqueeze(-1) # [bs, 1, 1]

                z_tilde_dist = distributions.MultivariateNormal(
                    z_tilde_dist_mean_all,
                    scale_tril=torch.tril(torch.diag_embed(z_tilde_dist_var_all))
                )

        else:
            # q_phi(s_t-1) the posterior of last time step is the prior of this time step
            if s_prior != None:
                s_prior_mean = s_prior.mean
                s_prior_cov = torch.diagonal(s_prior.scale_tril, dim1=1, dim2=2)
                
            else:
                s_prior_mean = self.infer_s_means_update[user, :, idx-1]
                s_prior_cov =  self.infer_s_vars_update[user, :, idx-1]
    
            # ----- p_theta(s_t | s_t-1) -----
            dt_emb = self.positionalencoding1d(self.node_dim, dt.shape[1], dt) # [bs, 1, dim_node]
            label_emb = y_idx[..., None] # [bs, 1, 1]
            if self.dim_z > 1:
                items = feed_dict['skill_seq'][:, idx:idx+1] # [bs, 1]
                node_emb = self.node_dist._get_node_embedding().to(dt_emb.device) # [num_nodes, dim_node]
                skill_value_emb = node_emb[items] # [bs, 1, dim_node]
            else:
                skill_value_emb = torch.zeros_like(dt_emb)
            rnn_input_emb = skill_value_emb + label_emb + dt_emb 
            # self.gen_network_transition_s.flatten_parameters()
            # output, _ = self.gen_network_transition_s(rnn_input_emb) # TODO change!!! # [bs, 1, dim_node]
            # output = self.gen_network_transition_s(rnn_input_emb)
            output = rnn_input_emb
            mean_div, log_var = self.gen_network_prior_mean_var_s(output) 
            
            # # 1. use MCMC to sample s_t
            # s_last_sample = s_prior_dist.sample((self.num_sample,)).transpose(0,1).squeeze(2) # [bs, num_sample, dim_s]
            # s_tilde_dist_mean = mean_div + s_last_sample # [bs, num_sample, dim_s]
            # 2. analytical solution
            # x ~ N (m, P), y|x ~ N (Hx + u, R) -> y ~ N (Hm + u, HPH^T + R)
            s_tilde_dist_mean = mean_div + s_prior_mean # [bs, dim_s]
            log_var = torch.minimum(log_var, self.var_minimum)
            s_tilde_dist_var = torch.exp(log_var) + s_prior_cov + EPS # [bs, dim_s]
            # s_tilde_dist_var = torch.minimum(s_tilde_dist_var, torch.exp(self.var_minimum))
            s_tilde_dist = distributions.multivariate_normal.MultivariateNormal(
                loc=s_tilde_dist_mean, 
                scale_tril=torch.tril(torch.diag_embed(s_tilde_dist_var))
            )


            # q_phi(z_t-1) the posterior of last time step is the prior of this time step
            if z_prior != None:
                z_prior_mean = z_prior.mean
                z_prior_cov = torch.diagonal(z_prior.scale_tril, dim1=1, dim2=2) # TODO
            else:
                z_prior_mean = self.infer_z_means_update[user,:,idx-1] # [bs, 1, dim_z]
                z_prior_cov = self.infer_z_vars_update[user,:,idx-1]
                
            # ----- p_theta(z_t | s_t, z_t-1) -----
            # 1. MCMC
            # z_last_sample = z_prior_dist.sample((self.num_sample,)).transpose(0,1)[..., 0] # [bs, num_sample, 1, dim_z] -> [bs, num_sample, 1]
            # s_next_sample = s_tilde_dist.rsample() # [bs, num_sample, dim_s]
            # 2. analytical solution
            s_next_sample = s_tilde_dist_mean # [bs, 1, dim_s]
            z_last_sample = z_prior_mean # [bs, 1, dim_z]
             
            sampled_alpha = torch.relu(s_next_sample[...,0:1]) + EPS*self.args.alpha_minimum # TODO change # [bs, 1, 1]
            decay = torch.exp(-sampled_alpha * dt.reshape(bs, 1, 1)) # [bs, 1, 1]
            sampled_mean = s_next_sample[..., 1:2]
            if self.dim_z > 1:
                item_idx = feed_dict['skill_seq'][:, idx]# [bs, 1]
                # ipdb.set_trace()
                omega = 0.5
                sampled_gamma = torch.sigmoid(s_next_sample[..., 3:])
                adj_t = torch.transpose(torch.exp(self.node_dist.edge_log_probs())[0].to(sampled_alpha.device), -1, -2)  # [num_nodes, num_nodes]
                in_degree = adj_t.sum(dim=-1)
                ind = torch.where(in_degree == 0)[0] 
                w_adj_t = adj_t # TODO
                z_last_sample = (node_emb * z_last_sample).sum(-1, keepdim=True) # [bs, num_node, 1]
                empower = (z_last_sample * w_adj_t).sum(-2).unsqueeze(-1) # [bs, num_node, 1]
                # torch.einsum('ij, ajm->aim', w_adj_t, z_last_sample) # [bs*n, num_node, times-1]
                empower = (1 / (in_degree[None, :, None] + EPS)) * sampled_gamma * empower
                empower[:,ind] = 0
                stable = sampled_mean
                tmp_mean_level = omega * stable + (1-omega) * empower
                
                z_tilde_dist_mean_all = z_last_sample * decay + (1.0 - decay) * tmp_mean_level
                z_tilde_dist_mean = z_tilde_dist_mean_all[0][item_idx].unsqueeze(-1)
                
                sampled_log_var = s_next_sample[..., 2:3] # [bs, 1, 1]
                sampled_log_var = torch.minimum(sampled_log_var, self.var_minimum)
                z_tilde_dist_var_all = torch.exp(sampled_log_var) * decay + EPS # [bs, 1, 1]
                z_tilde_dist_var = z_tilde_dist_var_all # [bs, 1, 1]
                
                z_tilde_dist = distributions.MultivariateNormal(
                    z_tilde_dist_mean_all,
                    scale_tril=torch.tril(torch.diag_embed(z_tilde_dist_var_all))
                ) # MultivariateNormal(loc: torch.Size([16, 837, 1]), scale_tril: torch.Size([16, 837, 1, 1]))
                
            else:
                tmp_mean_level = sampled_mean
                z_tilde_dist_mean = z_last_sample * decay + (1.0 - decay) * tmp_mean_level
        
                sampled_log_var = s_next_sample[..., 2:3]
                sampled_log_var = torch.minimum(sampled_log_var, self.var_minimum)
                z_tilde_dist_var = torch.exp(sampled_log_var) * decay + EPS
                
                z_tilde_dist = distributions.multivariate_normal.MultivariateNormal(
                    loc=z_tilde_dist_mean, 
                    scale_tril=torch.tril(torch.diag_embed(z_tilde_dist_var))
                )
        
        if not eval:
            if not update:
                self.pred_s_means[user, :, idx] = s_tilde_dist_mean.detach().clone()
                self.pred_s_vars[user, :, idx] = s_tilde_dist_var.detach().clone()
                self.pred_z_means[user, :, idx] = z_tilde_dist_mean.detach().clone()
                self.pred_z_vars[user, :, idx] = z_tilde_dist_var.detach().clone()
            else:
                self.pred_s_means_update[user, :, idx] = s_tilde_dist_mean.detach().clone()
                self.pred_s_vars_update[user, :, idx] = s_tilde_dist_var.detach().clone()
                self.pred_z_means_update[user, :, idx] = z_tilde_dist_mean.detach().clone()
                self.pred_z_vars_update[user, :, idx] = z_tilde_dist_var.detach().clone()
                
        return s_tilde_dist, z_tilde_dist
    
    
    def inference_model(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
        eval: bool = False,
        update: bool = False,
    ):
        '''
        Args:
            eval: if True, it will not update the parameters. 
                    Usually this happens during evaluation or comparison when there is no gradient.
            update: if True, it means the network is optimized by the update loss. 
                    We should save the interested parameters in the updated list.
        '''
        
        y_idx = feed_dict['label_seq'][:, :idx+1] # [bs, times]
        t_idx = feed_dict['time_seq'][:, :idx+1] 
        
        # ----- embedding -----
        t_pe = self.get_time_embedding(t_idx, 'absolute')  # [bs, times, node_dim]
        # NOTE: if t_idx contains only one time step, it will be zero; because it normalize the time by subtracting the minimum time
        y_pe = torch.tile(y_idx.unsqueeze(-1), (1,1, self.node_dim))   # [bs, times, node_dim]
        
        if self.dim_z > 1:
            item_idx = feed_dict['skill_seq'][:, :idx+1] 
            node_pe = self.node_dist._get_node_embedding()[item_idx]  # [bs, times, node_dim]
            emb_input = torch.cat([node_pe, y_pe], dim=-1) # [bs, times, node_dim*2]
            self.infer_network_emb.flatten_parameters() # NOTE: useful when distributed training
            emb_rnn_inputs, _ = self.infer_network_emb(emb_input) # [bs, times, node_dim]
        else: 
            emb_rnn_inputs = y_pe # TODO do we need an embedding layer?
            # self.infer_network_emb.flatten_parameters()
            # emb_rnn_inputs, _ = self.infer_network_emb(emb_input) # [bs, times, dim*2(32)]
        emb_rnn_inputs = emb_rnn_inputs + t_pe

        # -----  `q(s[t] | y[1:t])' -----
        s_dist = self.s_transition_infer(
            feed_dict, emb_inputs=emb_rnn_inputs, idx=idx
        )

        # ----- `q(z[t] | y[1:t])' -----
        z_dist  = self.z_transition_infer(
            feed_dict, emb_inputs=emb_rnn_inputs, idx=idx,
        )
        
        if not eval:
            users = feed_dict['user_id']
            if not update:
                self.infer_s_means[users, :, idx] = s_dist.mean.detach().clone()
                self.infer_s_vars[users, :, idx] = torch.diagonal(s_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()
                self.infer_z_means[users, :, idx] = z_dist.mean.detach().clone()
                self.infer_z_vars[users, :, idx] = torch.diagonal(z_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()
            else:
                self.infer_s_means_update[users, :, idx] = s_dist.mean.detach().clone()
                self.infer_s_vars_update[users, :, idx] = torch.diagonal(s_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()
                self.infer_z_means_update[users, :, idx] = z_dist.mean.detach().clone()
                self.infer_z_vars_update[users, :, idx] = torch.diagonal(z_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()

        return s_dist, z_dist


    def loss(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        outdict: Dict[str, torch.Tensor],
        metrics: List[str] = None,
    ):
        """
        Calculates the loss of the model based on the ground truth label and predicted label.

        Args:
        - feed_dict (dict): The input to the model
        - outdict (dict): The output of the model
        - metrics (dict): A dictionary of metrics to evaluate the performance of the model

        Returns:
        - losses (defaultdict): A defaultdict of losses
        """
        losses = defaultdict(lambda: torch.zeros(()))#, device=self.device))

        # Calculate binary cross-entropy loss -> not used for optimization only for visualization
        gt = outdict["label"] 
        pred = outdict["prediction"]
        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred, gt.float())
        losses['loss_bce'] = bceloss
        
        for key in ['elbo', 'sequence_likelihood', 
                    'st_entropy', 'zt_entropy',
                    'log_prob_yt', 'log_prob_zt', 'log_prob_st']:
            losses[key] = outdict[key].mean()
        losses['loss_total'] = -outdict['elbo'].mean()
        
        # # Still NOT for optimization
        # pred_att = torch.exp(self.edge_log_probs)# self.node_dist._get_atten_weights()[-1].clone().detach()
        # gt_adj = self.adj.to(pred.device).transpose(-1,-2)
        # losses['spasity'] = (pred_att >= 0.5).sum()
        # losses['adj_0_att_1'] = (1 * (pred_att >= 0.5) * (1-gt_adj)).sum()
        # losses['adj_1_att_0'] = (1 * (pred_att < 0.5) * gt_adj).sum()
        
        # # Register output predictions
        # self.register_buffer(name="output_predictions", tensor=pred)
        # self.register_buffer(name="output_gt", tensor=gt)
        # self.register_buffer(name="output_attention_weights", tensor=pred_att)
        # self.register_buffer(name="output_gt_graph_weights", tensor=gt_adj)
        
        # Evaluate metrics
        if metrics != None:
            self.metrics = metrics
            pred = pred.detach().cpu().data.numpy()
            gt = gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
        
        # Calculate mean and variance of the Ornstein-Uhlenbeck process
        losses['ou_speed'] = outdict["sampled_s"][...,0].mean()
        losses['ou_mean'] = outdict["sampled_s"][...,1].mean()
        losses['ou_vola'] = outdict["sampled_s"][...,2].mean()
        
        if self.dim_s > 3:
            losses['ou_gamma'] = outdict["sampled_s"][...,3].mean()
        
        return losses
    
    
    def objective_function(
        self,
        feed_dict_idx: Dict[str, torch.Tensor],
        idx: int,
        pred_dist=None,
        post_dist=None,
    ):
        
        y_idx = feed_dict_idx['label_seq'][:, idx:idx+1]
        item_idx = feed_dict_idx['skill_seq'][:, idx:idx+1]
        
        # p_tilde_theta(s_t) = \int p_theta(s_t | s_t-1) q_phi(s_t-1 | y_1:t-1) ds_t
        # p_tilde_theta(z_t) = \int p_theta(z_t | s_t, z_t-1) q_phi(z_t-1 | y_1:t-1) dz_t
        s_tilde_dist, z_tilde_dist = pred_dist
        # q_phi(s_t | y_1:t), q_phi(z_t | y_1:t) 
        s_infer_dist, z_infer_dist = post_dist
        
        # log tilde_p_theta(s_t)
        s_vp_sample = s_infer_dist.rsample((self.num_sample,)) # [num_sample, bs, 1, dim_s]
        log_prob_st = s_tilde_dist.log_prob(s_vp_sample) # [num_sample, bs, 1, dim_s]
        log_prob_st = log_prob_st.mean() / self.dim_s
        
        
        # log p_theta(y_t | z_t)
        if self.dim_z > 1:
            # log tilde_p_theta(z_t | s_t)
            node_emb = self.node_dist._get_node_embedding() # [num_node, node_dim]
            z_vp_sample = z_infer_dist.rsample((self.num_sample,)) # [num_sample, bs, 1, dim_z]
            z_vp_scalar_sample = (z_vp_sample * node_emb).sum(-1, keepdim=True) # [n, bs, num_node, 1]
            log_prob_zt = z_tilde_dist.log_prob(z_vp_scalar_sample) # [n, bs, num_node]
            log_prob_zt = log_prob_zt.mean()
            
            item_idx = item_idx.unsqueeze(0).repeat(self.num_sample, 1, 1)
            sampled_scalar_z = torch.gather(z_vp_scalar_sample[..., 0], -1, item_idx) # [n, bs, 1]
        else:
            # log tilde_p_theta(z_t | s_t)
            z_vp_sample = z_infer_dist.rsample((self.num_sample,)) # [num_sample, bs, 1, dim_z]
            log_prob_zt = z_tilde_dist.log_prob(z_vp_sample) # [num_sample, bs, 1, dim_z]
            log_prob_zt = log_prob_zt.mean() / self.dim_z
            
            sampled_scalar_z = z_vp_sample[:, :, 0]  # [n, bs, 1]
            
        emission_prob = self.y_emit(sampled_scalar_z) # [n, bs, 1]
        emission_dist = torch.distributions.bernoulli.Bernoulli(probs=emission_prob)
        prediction = emission_dist.sample() # NOTE: only for evaluation; no gradient
        label = y_idx.unsqueeze(0).repeat(self.num_sample, 1, 1).float()
        log_prob_yt = emission_dist.log_prob(label).mean() / self.dim_y
        
        st_entropy = s_infer_dist.entropy().mean()
        zt_entropy = z_infer_dist.entropy().mean()
        
        elbo = log_prob_st + log_prob_zt + log_prob_yt - st_entropy - zt_entropy
        iwae = None 
        
        return dict(
            elbo=elbo,
            iwae=iwae,
            sequence_likelihood=log_prob_yt+log_prob_zt+log_prob_st,
            log_prob_yt=log_prob_yt,
            log_prob_zt=log_prob_zt,
            log_prob_st=log_prob_st,
            st_entropy=st_entropy,
            zt_entropy=zt_entropy,
            prediction=prediction,
            label=label,
            sampled_s=s_vp_sample,
        )

    
    def comparison_function(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
    ):
        loss_fn = torch.nn.BCELoss()
        
        ipdb.set_trace()
        comparison = defaultdict(lambda: torch.zeros(()))#, device=self.device))
        users = feed_dict['user_id']
        labels = feed_dict['label_seq'][:, idx] # [bs, times]
        items = feed_dict['skill_seq'][:, idx] # [bs, times]
        
        # ------ comparison 1: check if optimization works ------
        old_z_tilde_dist = torch.distributions.MultivariateNormal( 
                loc=self.pred_z_means[users, :, idx],
                scale_tril=torch.diag_embed(self.pred_z_vars[users, :, idx]),
            )
        new_z_tilde_dist = torch.distributions.MultivariateNormal( 
                loc=self.pred_z_means_update[users, :, idx],
                scale_tril=torch.diag_embed(self.pred_z_vars_update[users, :, idx]),
            )
        old_y = self.y_emit(old_z_tilde_dist.sample((self.num_sample,)))
        new_y = self.y_emit(new_z_tilde_dist.sample((self.num_sample,)))
        gt = labels.unsqueeze(0).tile((self.num_sample, 1)).float().flatten()
        
        comparison['comp_1_tilde'] = loss_fn(new_y.flatten(), gt.float()) - loss_fn(old_y.flatten(), gt.float())
        
        
        old_z_infer_dist = torch.distributions.MultivariateNormal( 
                loc=self.infer_z_means[users, :, idx],
                scale_tril=torch.diag_embed(self.infer_z_vars[users, :, idx]),
            )
        new_z_infer_dist = torch.distributions.MultivariateNormal( 
                loc=self.infer_z_means_update[users, :, idx],
                scale_tril=torch.diag_embed(self.infer_z_vars_update[users, :, idx]),
            )
        if self.dim_z == 1:
            old_y = self.y_emit(old_z_infer_dist.sample((self.num_sample,)))
            new_y = self.y_emit(new_z_infer_dist.sample((self.num_sample,)))
            
            comparison['comp_1_infer'] = loss_fn(new_y.flatten(), gt.float()) - loss_fn(old_y.flatten(), gt.float())
        else:
            node_emb_item = self.node_dist._get_node_embedding()[items] # [bs, node_dim]
        
        # comparison 2: forward TODO not exactly right
        # for comparison 
        old_s_infer_dist = torch.distributions.MultivariateNormal( 
                loc=self.infer_s_means[users, :, idx],
                scale_tril=torch.diag_embed(self.infer_s_vars[users, :, idx]),
            )
        _, old_pred_z_tilde_dist = self.predictive_model(feed_dict, idx=idx+1, eval=True, s_prior=old_s_infer_dist, z_prior=old_z_infer_dist)
        old_y = self.y_emit(old_pred_z_tilde_dist.sample((self.num_sample,)))
        _, new_pred_z_tilde_dist = self.predictive_model(feed_dict, idx=idx+1, eval=True)
        new_y = self.y_emit(new_pred_z_tilde_dist.sample((self.num_sample,)))
        comparison['comp_2_pred'] = loss_fn(new_y.flatten(), gt.float()) - loss_fn(old_y.flatten(), gt.float())
        
        new_y = new_y.flatten()
        gt = gt.flatten()
        comparison['pred_gt_next_bce'] = loss_fn(new_y, gt)
        pred = new_y.detach().cpu().data.numpy()
        gt = gt.detach().cpu().data.numpy()
        evaluations = BaseModel.pred_evaluate_method(pred, gt, self.metrics)
        for key in evaluations.keys():
            comparison['pred_gt_new_'+key] = evaluations[key]
            
        return comparison


    def eval_future(
        self, 
        feed_dict: Dict[str, torch.Tensor] = None,
        idx: int = None,
        bafore_opt: bool = False,
        s_prior: torch.distributions.MultivariateNormal = None,
        z_prior: torch.distributions.MultivariateNormal = None,
    ):
        s_tilde_dist, z_tilde_dist = self.predictive_model(feed_dict, idx=idx+1, eval=True, s_prior=s_prior, z_prior=z_prior)
        y_pred = self.y_emit(z_tilde_dist.sample((self.num_sample,)))
        
        # user = feed_dict['user_id']
        # y_idx = feed_dict['label_seq'][:, idx:] # [bs, times]
        # t_idx = feed_dict['time_seq'][:, idx:] # [bs, 2]
        # dt = (t_idx[:,1:] - t_idx[:,:1])/T_SCALE + EPS 
        # device = y_idx.device
        # bs = y_idx.shape[0]
        
        # if bafore_opt:
        #     s_tilde_dist = torch.distributions.MultivariateNormal(
        #             loc=self.pred_s_means[user, :, idx],
        #             scale_tril=torch.diag_embed(self.pred_s_vars[user, :, idx]),
        #         )
        #     z_tilde_dist = torch.distributions.MultivariateNormal(
        #             loc=self.pred_z_means[user, :, idx],
        #             scale_tril=torch.diag_embed(self.pred_z_vars[user, :, idx]),
        #         )
        #     y_pred_idx_sample = self.y_emit(z_tilde_dist.sample((self.num_sample,)))
        #     y_pred_idx_sample = (y_pred_idx_sample>=0.5) * 1.0
        #     s_tilde_idx_sample = s_tilde_dist.sample((self.num_sample,)).transpose(0,1).squeeze(2) # [bs, num_sample, dim_s]
        #     s_idx_sample = s_tilde_idx_sample
        #     z_tilde_idx_sample = z_tilde_dist.sample((self.num_sample,)).transpose(0,1).squeeze(2) # [bs, num_sample, dim_s]
        #     z_idx_sample = z_tilde_idx_sample
            
        # else:
        #     s_infer_dist, z_infer_dist = self.inference_model()
        #     s_tilde_idx_sample = s_infer_dist.sample((self.num_sample,)).transpose(0,1).squeeze(2) # [bs, num_sample, dim_s]
        #     label_emb = y_idx[..., None] # [bs, 1, 1]
        #     s_idx_sample = s_tilde_idx_sample
        #     z_infer_idx_sample = z_infer_dist.sample((self.num_sample,)).transpose(0,1).squeeze(2) # [bs, num_sample, dim_s]
        #     z_idx_sample = z_infer_idx_sample
            
            
        # dt_emb = self.positionalencoding1d(self.node_dim, dt.shape[1], dt) # [bs, 1, dim_node]
        # label_emb = y_pred_idx_sample[..., None] # [bs, 1, 1]
        # if self.dim_z > 1:
        #     raise NotImplementedError
        #     # value_u = self.node_dist._get_node_embedding().to(sampled_s.device)# .clone().detach()
        #     # skill_value_emb = value_u[items][:,:-1] # [bs, times-1, dim_node]
        # else:
        #     skill_value_emb = torch.zeros_like(dt_emb)
        # rnn_input_emb = skill_value_emb + label_emb + dt_emb 
        # # self.gen_network_transition_s.flatten_parameters()
        # # output, _ = self.gen_network_transition_s(rnn_input_emb) # TODO change!!! # [bs, 1, dim_node]
        # # output = self.gen_network_transition_s(rnn_input_emb)
        # output = rnn_input_emb
        # mean_div, log_var = self.gen_network_prior_mean_var_s(output) 
        # s_tilde_future_dist_mean = mean_div + s_idx_sample # [bs, num_sample, dim_s]
        # log_var = torch.minimum(log_var, self.var_minimum)
        # s_tilde_future_dist_var = torch.exp(log_var) + EPS # [bs, 1, dim_s]

        # s_tilde_future_dist = distributions.multivariate_normal.MultivariateNormal(
        #     loc=s_tilde_future_dist_mean, 
        #     scale_tril=torch.tril(torch.diag_embed(s_tilde_future_dist_var))
        # ) 
        # s_tilde_future_sample = s_tilde_future_dist.sample() # [bs, num_sample, dim_s]
        
        # sampled_alpha = torch.relu(s_tilde_future_sample[...,0:1]) + EPS*100 # TODO change
        # decay = torch.exp(-sampled_alpha * dt.reshape(bs, 1, 1))
        # sampled_mean = s_tilde_future_sample[..., 1:2]
        # z_tilde_dist_mean = z_idx_sample * decay + (1.0 - decay) * sampled_mean
        
        # sampled_log_var = s_tilde_future_sample[..., 2:3]
        # sampled_log_var = torch.minimum(sampled_log_var, self.var_minimum)
        # z_tilde_dist_var = torch.exp(sampled_log_var) * decay + EPS
            
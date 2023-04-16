import sys
sys.path.append('..')

import math 
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import torch
from torch import nn, distributions
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from collections import defaultdict

import ipdb

from models.modules import build_rnn_cell
from models.BaseModel import BaseModel
from models.learner_model import BaseLearnerModel
from models.new_learner_model import build_dense_network
from models.modules import CausalTransformerModel, VAEEncoder
from models.variational_distributions import VarDIBS, VarTransformation, VarAttention
from models.learner_hssm_model import HSSM, GraphHSSM

from enum import Enum

torch.autograd.set_detect_anomaly(True)

RANDOM_SEED = 131
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
        self.fit_vi_global_s, self.fit_vi_transition_s, self.infer_global_s, self.infer_transition_s = 0,0,0,1
        
        self.pred_s_means = torch.zeros((num_seq, self.num_sample, args.max_step, self.dim_s), device=self.device)
        self.pred_s_vars = torch.zeros((num_seq, self.num_sample, args.max_step, self.dim_s), device=self.device)
        self.infer_s_means = torch.zeros((num_seq, 1, args.max_step, self.dim_s), device=self.device)
        self.infer_s_vars = torch.zeros((num_seq, 1, args.max_step, self.dim_s), device=self.device)
        
        self.pred_z_means = torch.zeros((num_seq, self.num_sample, args.max_step, self.dim_z), device=self.device)
        self.pred_z_vars = torch.zeros((num_seq, self.num_sample, args.max_step, self.dim_z), device=self.device)
        self.infer_z_means = torch.zeros((num_seq, 1, args.max_step, self.dim_z), device=self.device)
        self.infer_z_vars = torch.zeros((num_seq, 1, args.max_step, self.dim_z), device=self.device)
 
        
    def s_transition_infer(
        self, 
        feed_dict=None,
        emb_inputs=None,
        idx=None,
    ):
        # if idx is not None:
        #     t_input = feed_dict['time_seq'][:, :idx+1]
        # else: 
        #     t_input = feed_dict['time_seq'] # [bs, times]

        emb_rnn_inputs = emb_inputs
        
        if self.transformer:
            # Compute the output of the posterior network
            output = self.infer_network_posterior_s(emb_rnn_inputs)
            
            # Compute the mean and covariance matrix of the posterior distribution of `s_t`
            output = output[:, -1:]
            mean, log_var = self.infer_network_posterior_mean_var_s(output)  # [batch_size, time_step, dim_s]
            cov_mat = torch.diag_embed(torch.exp(log_var) + EPS)
            dist_s = distributions.multivariate_normal.MultivariateNormal(
                loc=mean, 
                scale_tril=torch.tril(cov_mat)
            )
                
        elif self.rnn:
            raise NotImplementedError

        return dist_s
 

    def z_transition_infer(
        self, 
        feed_dict, 
        emb_inputs=None,
        idx=None,
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
        
        # if idx is not None:
        #     t_input = feed_dict['time_seq'][:, :idx]
        # else:
        #     t_input = feed_dict['time_seq'] # [bs, times]
        
        # Embed the input sequence, if needed
        emb_rnn_inputs = emb_inputs
        output = self.infer_network_posterior_z(emb_rnn_inputs)
        
        # Compute the mean and covariance matrix of the posterior distribution of `z_t`
        output = output[:, -1:] # [bs, 1, out_dim]
        mean, log_var = self.infer_network_posterior_mean_var_z(output) # [bs, times, out_dim]
        cov_mat = torch.diag_embed(torch.exp(log_var) + EPS) 
        dist_z = distributions.multivariate_normal.MultivariateNormal(
            loc=mean, 
            scale_tril=torch.tril(cov_mat)
        )
        
        
        return dist_z


    def predictive_model(
        self, 
        feed_dict, 
        idx=None
    ):
        user = feed_dict['user_id']
        y_idx = feed_dict['label_seq'][:, idx].unsqueeze(-1) # [bs, 1]
        t_idx = feed_dict['time_seq'][:, idx-1: idx+1] # [bs, 2]
        dt = torch.diff(t_idx, dim=-1)/T_SCALE + EPS 
        device = y_idx.device
        bs = y_idx.shape[0]
        
        if idx == 0:
            s_tilde_dist = distributions.MultivariateNormal(
                self.gen_s0_mean.to(device), 
                scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_s0_log_var.to(device)) + EPS))
            )
            z_tilde_dist = distributions.MultivariateNormal(
                self.gen_z0_mean.to(device), 
                scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_z0_log_var.to(device)) + EPS))
            )
            z_last_sample = z_tilde_dist.sample((self.num_sample, )).unsqueeze(0).repeat(bs, 1, 1, 1) # [bs, num_sample, 1, dim_z]
            
            s_tilde_dist_mean = s_tilde_dist.mean[None,...]
            s_tilde_dist_var = torch.diagonal(s_tilde_dist.scale_tril, dim1=1, dim2=2)[None,...]
            z_tilde_dist_mean = z_tilde_dist.mean[None,...]
            z_tilde_dist_var = torch.diagonal(z_tilde_dist.scale_tril, dim1=1, dim2=2)[None,...]
            
        else:
            ipdb.set_trace()
            s_prior_mean = self.infer_s_means[user,:,idx-1]
            s_prior_cov =  self.infer_s_vars[user,:,idx-1]
            z_prior_mean = self.infer_z_means[user,:,idx-1]
            z_prior_cov = self.infer_z_vars[user,:,idx-1]
            
            # q_phi(s_t-1) the posterior of last time step is the prior of this time step
            s_prior_dist = torch.distributions.MultivariateNormal( 
                loc=s_prior_mean,
                scale_tril=torch.diag_embed(s_prior_cov),
            )
            # p_theta(s_t | s_t-1)
            s_last_sample = s_prior_dist.rsample((self.num_sample,)).transpose(0,1).squeeze(2) # [bs, num_sample, dim_s]
            dt_emb = self.positionalencoding1d(self.node_dim, dt.shape[1], dt) # [bs, 1, dim_node]
            label_emb = y_idx[..., None] # [bs, 1, 1]
            if self.dim_z > 1:
                pass
                # value_u = self.node_dist._get_node_embedding().to(sampled_s.device)# .clone().detach()
                # skill_value_emb = value_u[items][:,:-1] # [bs, times-1, dim_node]
            else:
                skill_value_emb = torch.zeros_like(dt_emb)
            rnn_input_emb = skill_value_emb + label_emb + dt_emb # TODO is there any other way to concat?
            self.gen_network_transition_s.flatten_parameters()
            output, _ = self.gen_network_transition_s(rnn_input_emb) # TODO change!!! # [bs, 1, dim_node]
            mean_div, log_var = self.gen_network_prior_mean_var_s(output) 
            s_tilde_dist_mean = mean_div + s_last_sample # [bs, num_sample, dim_s]
            s_tilde_dist_var = torch.exp(log_var) + EPS # [bs, 1, dim_s]

            s_tilde_dist = distributions.multivariate_normal.MultivariateNormal(
                loc=s_tilde_dist_mean, 
                scale_tril=torch.tril(torch.diag_embed(s_tilde_dist_var))
            )

        
            # q_phi(z_t-1) the posterior of last time step is the prior of this time step
            z_prior_dist = torch.distributions.MultivariateNormal(
                loc=z_prior_mean,
                scale_tril=torch.diag_embed(z_prior_cov),
            )
            # p_theta(z_t | s_t, z_t-1)
            z_last_sample = z_prior_dist.rsample((self.num_sample,)).transpose(0,1)[..., 0] # [bs, num_sample, 1, dim_z] -> [bs, num_sample, 1]
            s_next_sample = s_tilde_dist.rsample() # [bs, num_sample, dim_s]
            
            sampled_alpha = torch.relu(s_next_sample[...,0:1]) + EPS*100 # TODO change
            decay = torch.exp(-sampled_alpha * dt.reshape(bs, 1, 1))
            sampled_mean = s_next_sample[..., 1:2]
            sampled_log_var = s_next_sample[..., 2:3]
            sampled_var = torch.sigmoid(sampled_log_var) # torch.exp(sampled_log_var) * decay + EPS # TODO not constrained
            z_tilde_dist_mean = z_last_sample * decay + (1.0 - decay) * sampled_mean
            z_tilde_dist_var = torch.where(torch.isinf(sampled_var), torch.tensor(1e30, device=sampled_var.device, dtype=sampled_var.dtype), sampled_var)
            z_tilde_dist_var += EPS
            # sampled_gamma = torch.sigmoid(s_next_sample[..., 3])
            # omega = 0.5
            # adj = torch.exp(self.node_dist.edge_log_probs()).to(sampled_z.device) 
            # adj_t = adj # torch.transpose(adj, -1, -2).contiguous() # TODO test with multiple power of adj
            # in_degree = adj_t.sum(dim=-1)
            # ind = torch.where(in_degree == 0)[0] 
            # w_adj_t = adj_t
            # # adj = self.adj.float().to(sampled_s.device)
            # # adj_t = torch.transpose(adj, -1, -2).contiguous() # TODO test with multiple power of adj
            # # in_degree = adj_t.sum(dim=-1)
            # # ind = torch.where(in_degree == 0)[0] # [284,]
            # # attn_output_weights = self.node_dist._get_atten_weights()
            # # w_adj_t = adj_t * attn_output_weights[-1].to(sampled_z.device) # .clone().detach()
            # empower = torch.einsum('ij, ajm->aim', w_adj_t, z_last) # [bs*n, num_node, times-1]
            # empower = (1 / (in_degree[None, :, None] + EPS)) * sampled_gamma * empower # [bs*n, num_node, 1]
            # empower[:,ind] = 0
            # stable = sampled_mean
            # tmp_mean_level = omega * stable + (1-omega) * empower
            # z_pred = z_last * decay + (1.0 - decay) * tmp_mean_level
            
            # z_tilde_dist = distributions.multivariate_normal.MultivariateNormal(
            #     loc=z_tilde_dist_mean, 
            #     scale_tril=torch.tril(torch.diag_embed(z_tilde_dist_var))
            # )
    
        # pred_y = self.y_emit(z_last_sample) # [bs, num_sample, 1]
        # pred_dict = {
        #     'prediction': pred_y, 
        #     'label': y_idx, 
        #     's_tilde_dist': s_tilde_dist,
        #     'z_tilde_dist': z_tilde_dist,
        
        ipdb.set_trace()
        self.pred_s_means[user, :, idx] += s_tilde_dist_mean.detach().clone()
        self.pred_s_vars[user, :, idx] += s_tilde_dist_var.detach().clone()
        self.pred_z_means[user, :, idx] += z_tilde_dist_mean.detach().clone()
        self.pred_z_vars[user, :, idx] += z_tilde_dist_var.detach().clone()
        
        return True
    
    
    def inference_model(self, feed_dict, idx=None):
        ipdb.set_trace()
        
        y_idx = feed_dict['label_seq'][:, :idx+1] # [bs, times]
        t_idx = feed_dict['time_seq'][:, :idx+1] 
        item_idx = feed_dict['skill_seq'][:, :idx+1] 
        users = feed_dict['user_id']
        
        # if idx == 0:
        #     h_emb_seq_last = [torch.zeros((bs, 1, self.node_dim)).to(device)] * 2
        # else:
        #     pass
        
        # ----- embedding -----
        t_pe = self.positionalencoding1d(self.node_dim, t_idx.shape[1], t_idx)
        y_pe = torch.tile(y_idx.unsqueeze(-1), (1,1, self.node_dim)) # TODO # [bs, times, dim]]   
        
        if self.num_node > 1:
            node_pe = self.node_dist._get_node_embedding()[item_idx] 
            emb_input = torch.cat([node_pe, y_pe], dim=-1) # [bs, times, dim*4]
            self.infer_network_emb.flatten_parameters()
            emb_rnn_inputs, _ = self.infer_network_emb(emb_input) # [bs, times, dim*2(32)]
        else: 
            emb_rnn_inputs = y_pe
        emb_rnn_inputs = emb_rnn_inputs + t_pe

        # ----- Sample continuous hidden variable from `q(s[t] | y[1:t])' -----
        s_dist = self.s_transition_infer(
            feed_dict, emb_inputs=emb_rnn_inputs, idx=idx
        )

        # ----- Sample continuous hidden variable from `q(z[1:T] | y[1:T])' -----
        z_dist  = self.z_transition_infer(
            feed_dict, emb_inputs=emb_rnn_inputs, idx=idx,
        )
        
        ipdb.set_trace()
        self.infer_s_means[users, :, idx] += s_dist.mean.detach().clone()
        self.infer_s_vars[users, :, idx] += torch.diagonal(s_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()
        self.infer_z_means[users, :, idx] += z_dist.mean.detach().clone()
        self.infer_z_vars[users, :, idx] += torch.diagonal(z_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()
        # if self.dim_z > 1:
        #     z_sampled_scalar = z_sampled @ self.node_dist._get_node_embedding().transpose(-1,-2).contiguous().to(z_sampled.device)# .clone().detach()
        # else:
        #     z_sampled_scalar = z_sampled

        # return_dict["label"] = feed_dict['label_seq'][:,None,:,None]  
        # return_dict["prediction"] = recon_inputs_items   
        # return_dict["sampled_s"] = s_sampled  
        
        # self.register_buffer(name="output_mean_s", tensor=s_mean)
        # self.register_buffer(name="output_var_s", tensor=s_log_var)
        # self.register_buffer(name="output_mean_z", tensor=z_mean)
        # self.register_buffer(name="output_var_z", tensor=z_log_var)
        # self.register_buffer(name="output_emb_input", tensor=emb_input)
        # self.register_buffer(name="output_sampled_z", tensor=z_sampled)
        # self.register_buffer(name="output_sampled_y", tensor=recon_inputs)
        # self.register_buffer(name="output_sampled_s", tensor=s_sampled)
        # self.register_buffer(name="output_items", tensor=items)
        
        # return_dict['log_prob_st'] = log_prob_st.mean()
        # return_dict['log_prob_zt'] = log_prob_zt.mean()
        # return_dict['log_prob_yt'] = log_prob_yt.mean()

        return s_dist, z_dist
    
    
    def loss(self, feed_dict, outdict, metrics=None):
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
            pred = pred.detach().cpu().data.numpy()
            gt = gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
        
        # Calculate mean and variance of the Ornstein-Uhlenbeck process
        losses['ou_speed'] = outdict["sampled_s"][...,0].mean()
        losses['ou_mean'] = outdict["sampled_s"][...,1].mean()
        losses['ou_vola'] = outdict["sampled_s"][...,2].mean()
        # losses['ou_gamma'] = outdict["sampled_s"][...,3].mean()
        return losses
    
    
    def objective_function(
        self,
        feed_dict_idx, 
        idx,
    ):
        '''
        s_prior_dist: q_phi(s_t-1)
        '''
        ipdb.set_trace()
        users = feed_dict_idx['user_id']
        y_idx = feed_dict_idx['label_seq'][:, idx:idx+1]
        
        # log tilde_p_theta(s_t)
        s_tilde_dist = torch.distributions.MultivariateNormal(
            loc=self.pred_s_means[users,:,idx],
            scale_tril=torch.diag_embed(self.pred_s_vars[users,:,idx]),
        )
        s_infer_dist = torch.distributions.MultivariateNormal(
            loc=self.infer_s_means[users,:,idx],
            scale_tril=torch.diag_embed(self.infer_s_vars[users,:,idx]),
        )
        
        s_vp_sample = s_infer_dist.rsample((self.num_sample,)).transpose(0,1)[:,:,0] # [bs, num_sample, dim_s]
        log_prob_st = s_tilde_dist.log_prob(s_vp_sample).mean()
        
        # log tilde_p_theta(z_t | s_t) -> analytically? TODO
        z_tilde_dist = torch.distributions.MultivariateNormal(
            loc=self.pred_z_means[users,:,idx],
            scale_tril=torch.diag_embed(self.pred_z_vars[users,:,idx]),
        )
        z_infer_dist = torch.distributions.MultivariateNormal(
            loc=self.infer_z_means[users,:,idx],
            scale_tril=torch.diag_embed(self.infer_z_vars[users,:,idx]),
        )
        z_vp_sample = z_infer_dist.rsample((self.num_sample,)).transpose(0,1)[:,:,0]
        log_prob_zt = z_tilde_dist.log_prob(z_vp_sample).mean()
        
        # log p_theta(y_t | z_t)
        if self.dim_z > 1:
            sampled_scalar_z = sampled_scalar_z[:, 0] # [bsn, time, num_node]
            items = torch.tile(feed_dict_idx['skill_seq'], (self.num_sample,1)).unsqueeze(-1)
            sampled_scalar_z_item = torch.gather(sampled_scalar_z, -1, items) # [bsn, time, 1]
        else:
            sampled_scalar_z_item = z_vp_sample
        emission_prob = self.y_emit(sampled_scalar_z_item) # [bs, n, 1]
        emission_dist = torch.distributions.bernoulli.Bernoulli(probs=emission_prob)
        y_input = y_idx.unsqueeze(-1).repeat(1,self.num_sample,1).float()
        log_prob_yt = emission_dist.log_prob(y_input).mean()
        
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
            prediction=emission_prob,
            label=y_input,
        )
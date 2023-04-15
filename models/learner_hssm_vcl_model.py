import sys
sys.path.append('..')

import math 

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
        
        # ----- specify dimensions of all latents -----
        self.node_dim = 16
        self.emb_mean_var_dim = 16

        # ----- initialize graph parameters -----
        self.learned_graph = self.args.learned_graph
        if self.learned_graph == 'none' or self.num_node == 1:
            self.dim_s = 3
        else: 
            self.dim_s = 4
            if self.learned_graph == 'w_gt':
                pass
            elif self.learned_graph == 'no_gt':
                pass
            self.dim_s = 4 # [alpha, mu, sigma, gamma]
            self.adj = torch.tensor(nx_graph)
            assert(self.adj.shape[-1] >= num_node)
        
        self.node_dist = VarTransformation(device=self.device, num_nodes=self.num_node, tau_gumbel=1, dense_init = False, 
                        latent_prior_std=None, latent_dim=self.node_dim)
        self.edge_log_probs = self.node_dist.edge_log_probs()
        
        # ----- for parameters Theta -----
        # the initial distribution p(s0) p(z0), the transition distribution p(s|s') p(z|s,z'), the emission distribution p(y|s,z)
        # 1. initial distribution p(s0) p(z0): trainable mean and variance???
        self.vi_z_scalar, self.vi_z_embed = 0,1
        out_dim = self.dim_z if self.vi_z_scalar else self.node_dim
        self.gen_s0_mean, self.gen_s0_log_var = self._initialize_normal_mean_log_var(self.dim_s, False)
        self.gen_z0_mean, self.gen_z0_log_var = self._initialize_normal_mean_log_var(out_dim, False) # self.z0_scale is std
        
        # 2. transition distribution p(s|s') or p(s|s',y',c'); p(z|s,z') (OU)
        self.s_transit_w_slast = 0
        self.s_transit_w_slast_yc = 1 - self.s_transit_w_slast
        # TODO: should incoporate pe of s and time t
        if self.s_transit_w_slast: 
            self.gen_network_transition_s = build_dense_network( 
                self.dim_s,
                [self.dim_s, self.dim_s],
                [nn.ReLU(), None]
            )
        elif self.s_transit_w_slast_yc: 
            self.gen_network_transition_s = nn.LSTM(
                input_size=self.node_dim, 
                hidden_size=self.node_dim,  
                bidirectional = False, 
                batch_first = True,
            )
            self.gen_network_prior_mean_var_s = VAEEncoder(
                self.node_dim, self.emb_mean_var_dim, self.dim_s
            ) 
        else:
            raise NotImplementedError
        self.network_transition_z = None
        self.st_transition = self.st_transition_func
        self.zt_transition = self.zt_transition_func
        # 3. emission distribution p(y|z)
        self.yt_emission = torch.sigmoid # self.yt_emission_func
        
        
        # ----- for parameters Phi -----
        # the embedding network at each time step emb_t = f(y_t, c_t, t)
        # the variational posterior distribution q(s_1:t | y_1:t, c_1:t) and q(z_1:t | y_1:t, c_1:t) TODO could add structure later q(z_1:t | y_1:t, s_1:t)
        # 1. embedding network
        # TODO: it could be identity function; MLP; RNN (need to think about why bidirectional, is it helping causal?)
        # network_input_embedding = lambda x: x
        self.infer_network_emb = nn.LSTM( # TODO LSTM
            input_size=self.node_dim*2, 
            hidden_size=self.node_dim//2, 
            bidirectional=True, # TODO why bidirectional
            batch_first=True,
        )
        # 2. variational posterior distribution q(s_1:t | y_1:t, c_1:t) = q(s_1:t | emb_1:t)
        # TODO: could be RNN; MLP;
        self.transformer, self.rnn, self.explcit_rnn, self.implicit_rnn = 1,0,0,0
        if self.explcit_rnn:
            self.infer_network_posterior_s = build_rnn_cell(
                rnn_type="lstm", 
                hidden_dim_rnn=self.node_dim, 
                rnn_input_dim=self.node_dim*2
            )
        elif self.implicit_rnn:
            self.infer_network_posterior_s = nn.LSTM(
                input_size=self.node_dim*2, 
                hidden_size=self.node_dim,  
                bidirectional = False, 
                batch_first = True,
            )
        elif self.transformer: 
            self.infer_network_posterior_s = CausalTransformerModel(
                ntoken=self.num_node,
                ninp=self.infer_network_emb.hidden_size*2 if self.infer_network_emb.bidirectional else self.infer_network_emb.hidden_size,
                nhid=self.node_dim,
            )
        self.infer_network_posterior_mean_var_s = VAEEncoder(
            self.infer_network_posterior_s.hidden_dim, self.emb_mean_var_dim, self.dim_s
        )
        self.s_infer = self.s_transition_infer
        # 3. variational posterior distribution q(z_1:t | y_1:t, c_1:t)
        self.infer_network_posterior_z = CausalTransformerModel(
            ntoken=self.num_node,
            ninp=self.infer_network_emb.hidden_size*2 if self.infer_network_emb.bidirectional else self.infer_network_emb.hidden_size,
        )
        # TODO MoE; normalization unir sphere
        out_dim = self.dim_z if self.vi_z_scalar else self.node_dim
        self.infer_network_posterior_mean_var_z = VAEEncoder(
            self.infer_network_posterior_z.hidden_dim, self.emb_mean_var_dim, out_dim
        )
        self.z_infer = self.z_transition_infer
        
        # # ----- For old test -----
        # if self.fit_vi_transition_s: # why TODO
        #     dim = self.dim_s
        #     x0_mean = nn.init.xavier_uniform_(torch.empty(num_seq, 1, dim, device=self.device))
        #     self.s_trans_mean = nn.Parameter(x0_mean, requires_grad=True)
            
        #     m = nn.init.xavier_uniform_(torch.empty(num_seq, 1, int(dim * (dim + 1) / 2), device=self.device))
        #     x0_scale = torch.zeros((num_seq, 1, dim, dim), device=self.device)
        #     tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)
        #     x0_scale[:,:, tril_indices[0], tril_indices[1]] += m
        #     self.s_trans_scale = nn.Parameter(x0_scale, requires_grad=True)
            
        #     self.s_infer = self.s_transition_fit
        
        # elif self.infer_global_s:   
        #     self.infer_network_emb = nn.RNN(
        #         input_size=self.dim_y * 2, # input_y + input_time 2
        #         hidden_size=self.dim_y * 2 * 2,  # 4
        #         bidirectional = True,
        #         batch_first = True,
        #     )
        #     self.network_posterior_mean_mlp_s = build_dense_network(
        #         800,
        #         [64, self.dim_s * 2], 
        #         [nn.ReLU(), None]) # is it not allowed to add non-linear functions here?


    @staticmethod
    def positionalencoding1d(d_model, length, actual_time=None):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        https://github.com/wzlxjtu/PositionalEncoding2D
        """
        device = actual_time.device
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(actual_time.shape[0], length, d_model, device=device)
        if actual_time != None:
            position = actual_time.unsqueeze(-1) # [bs, times, 1]
        else:
            position = torch.arange(0, length).unsqueeze(1)

        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model))).reshape(1,1,-1).to(device)
        pe[..., 0::2] = torch.sin(position.float() * div_term)
        pe[..., 1::2] = torch.cos(position.float() * div_term)

        return pe
    
    
    def get_time_embedding(self, time, type='dt'):
        if type == 'dt':
            dt = torch.diff(time, dim=1) # ?
            t_pe = self.positionalencoding1d(self.node_dim, dt.shape[1], dt) # [bs, times, dim]
        elif type =='absolute':
            norm_t = time-time.min(1)[0].unsqueeze(1) # TODO
            t_pe = self.positionalencoding1d(self.node_dim, time.shape[1], norm_t) # [bs, times, dim]
        return t_pe
    
    
    def st_transition_func(self, sampled_s=None, dist_s=None, feed_dict=None):
        
        input_y = feed_dict['label_seq'] # [bs, times]
        items = feed_dict['skill_seq']
        bs, _ = input_y.shape
        num_sample = sampled_s.shape[0]//bs
        assert num_sample == self.num_sample, "num_sample should be equal to self.num_sample"
        
        t_pe = self.get_time_embedding(feed_dict['time_seq'], type='absolute')
    
        s_last = sampled_s[:,0,:-1] # [bsn, times-1, dim]
        
        if self.s_transit_w_slast:
            raise NotImplementedError
        
        elif self.s_transit_w_slast_yc:
            if self.num_node == 1:
                skill_value_emb = torch.zeros((bs, 1, self.node_dim), device=sampled_s.device)
            else:
                value_u = self.node_dist._get_node_embedding().to(sampled_s.device)# .clone().detach()
                skill_value_emb = value_u[items][:,:-1] # [bs, times-1, dim_node]
            
            self.gen_network_transition_s.flatten_parameters()
            
            label_emb = input_y.unsqueeze(-1)[:, :-1] # .to(sampled_s.device)[:, :-1]
            time_emb = t_pe.to(sampled_s.device)[:, :-1]
            rnn_input_emb = skill_value_emb + label_emb + time_emb # TODO is there any other way to concat?
            
            output, _ = self.gen_network_transition_s(rnn_input_emb)
            
            mean_div, log_var = self.gen_network_prior_mean_var_s(output) 
            mean = mean_div.repeat(num_sample, 1, 1) + s_last # TODO: can to add information of s_t-1 into the framework; how to constrain that s is not changing too much?
            cov_mat = torch.diag_embed(torch.exp(log_var) + EPS).tile(num_sample, 1, 1, 1)
            
            s_prior_dist = distributions.multivariate_normal.MultivariateNormal(
                    loc=mean, 
                    scale_tril=torch.tril(cov_mat)
                )
            # sampled_s_prior = dist_s.rsample() # [bsn, times-1, dim_s]
            
            # s_sampled = sampled_s_prior.reshape(num_sample * bs, 1, time_step-1, self.dim_s) 
            # s_entropy = dist_s.entropy() # [bs, times-1]
            # s_log_prob_q = dist_s.log_prob(sampled_s_prior)
            # rnn_states = output.reshape(1 * bs, time_step-1, output.shape[-1])
            # s_mean = mean
            # s_var = cov_mat
        self.register_buffer('output_prior_s_mean', mean.clone().detach())
        self.register_buffer('output_prior_s_empower', cov_mat.clone().detach( ))
        
        return s_prior_dist # , s_sampled, s_entropy, s_log_prob_q, rnn_states, s_mean, s_var
    
    
    def zt_transition_func(self, sampled_z_set, sampled_s, feed_dict):
        '''
        Compute the transition function of the scalar outcomes `z_t` in the ST-DKT model.

        Args:
            sampled_z_set (tuple): Tuple containing:
                - sampled_z (torch.Tensor): Sampled scalar outcomes `z_t` of shape [batch_size*num_sample, num_nodes, times].
                - sampled_scalar_z (torch.Tensor): Sampled scalar outcomes `z_t` of shape [batch_size*num_sample, num_nodes, times, 1].
            sampled_s (torch.Tensor): Sampled latent skills `s_t` of shape [batch_size*num_sample, num_nodes, times, dim_s].
            feed_dict (dict): Dictionary of input tensors containing the following keys:
                - time_seq (torch.Tensor): Sequence of time intervals of shape [batch_size, times].

        Returns:
            z_prior_dist (torch.distributions.MultivariateNormal): Multivariate normal distribution of `z_t`
                with mean and covariance matrix computed using the sampled latent skills `s_t`.
        '''
        if isinstance(sampled_z_set, list):
            sampled_z, sampled_scalar_z = sampled_z_set
        else:
            sampled_scalar_z = sampled_z_set
        input_t = feed_dict['time_seq']
        bs, num_steps = input_t.shape
        bsn, num_node, _, _ = sampled_s.shape
        assert(self.num_sample == int(bsn // bs))
        
        # ----- calculate time difference -----
        input_t = input_t.unsqueeze(1)
        dt = torch.diff(input_t, dim=-1)/T_SCALE + EPS  # [bs, 1, num_steps-1] # TODO: t scale would change if datasets change
        dt = dt.repeat(self.num_sample, num_node, 1).to(sampled_scalar_z.device)
        
        # ----- calculate the mean and variance of z -----
        z_last = sampled_scalar_z[:, 0].transpose(-1, -2).contiguous()[..., :-1] # [bsn, num_node, times-1]
        
        # TODO: need very careful design of the constraint of these interpretable parameters
        sampled_s = sampled_s[:, :, 1:] # [bsn, num_node, times-1, dim_s]
        sampled_alpha = torch.relu(sampled_s[..., 0]) + EPS*100 # TODO change
        decay = torch.exp(-sampled_alpha * dt)
        sampled_mean = sampled_s[..., 1]
        sampled_log_var = sampled_s[..., 2]
        sampled_var = torch.sigmoid(sampled_log_var) # torch.exp(sampled_log_var) * decay + EPS # TODO not constrained
        sampled_gamma = torch.sigmoid(sampled_s[..., 3])
        omega = 0.5
        
        # ----- Simulate the path of `z_t` -----
        # attn_output_weights = self.node_dist._get_atten_weights()
        # adj = attn_output_weights[-1].to(sampled_z.device)
        adj = torch.exp(self.node_dist.edge_log_probs()).to(sampled_scalar_z.device) 
        adj_t = adj # torch.transpose(adj, -1, -2).contiguous() # TODO test with multiple power of adj
        in_degree = adj_t.sum(dim=-1)
        ind = torch.where(in_degree == 0)[0] 
        w_adj_t = adj_t
        # adj = self.adj.float().to(sampled_s.device)
        # adj_t = torch.transpose(adj, -1, -2).contiguous() # TODO test with multiple power of adj
        # in_degree = adj_t.sum(dim=-1)
        # ind = torch.where(in_degree == 0)[0] # [284,]
        # attn_output_weights = self.node_dist._get_atten_weights()
        # w_adj_t = adj_t * attn_output_weights[-1].to(sampled_z.device) # .clone().detach()
        
        empower = torch.einsum('ij, ajm->aim', w_adj_t, z_last) # [bs*n, num_node, times-1]
        empower = (1 / (in_degree[None, :, None] + EPS)) * sampled_gamma * empower # [bs*n, num_node, 1]
        empower[:,ind] = 0
        stable = sampled_mean
        tmp_mean_level = omega * stable + (1-omega) * empower
        z_pred = z_last * decay + (1.0 - decay) * tmp_mean_level

        self.register_buffer('output_prior_z_decay', decay)
        self.register_buffer('output_prior_z_empower', empower)
        self.register_buffer('output_prior_z_tmp_mean_level', tmp_mean_level)
        
        z_mean = z_pred.reshape(bsn, num_node, num_steps-1, 1) # [bs*n, num_node, num_steps, 1]
        z_var = sampled_var.reshape(bsn, 1, num_steps-1, 1) # torch.cat([z0_var, sampled_var], -2) # [bs*n, num_node, num_steps, 1]
        z_var = torch.where(torch.isinf(z_var), torch.tensor(1e30, device=z_var.device, dtype=z_var.dtype), z_var)
        z_var += EPS
        
        z_prior_dist = distributions.multivariate_normal.MultivariateNormal(
            loc=z_mean, 
            scale_tril=torch.tril(torch.diag_embed(z_var))
        )
        # z_sampled = output_dist.rsample() # [bs*n, num_node, num_steps-1, 1]
        # z_entropy = output_dist.entropy().mean(-2) # [bs*n, num_steps-1]
        # z_log_prob_q = output_dist.log_prob(z_sampled).mean(-2) # [bs*n, num_steps-1]
        
        return z_prior_dist# , z_sampled, z_entropy, z_log_prob_q, z_mean, z_var
        
        
    def yt_emission_func(self, ):
        pass
    
    
    def s_transition_infer(
        self, 
        feed_dict, 
        num_sample=1, 
        emb_inputs=None,
        idx=None,
    ):
        if idx is not None:
            t_input = feed_dict['time_seq'][:, :idx]
        else: 
            t_input = feed_dict['time_seq'] # [bs, times]
        ipdb.set_trace()
        bs, time_step = t_input.shape
        emb_rnn_inputs = emb_inputs
        bsn = bs * self.num_sample
        
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
            
            # # Sample the latent skills `s_t` using Monte Carlo estimation
            # sampleds = dist_s.rsample((num_sample,))  # [num_sample, batch_size, time_step, dim_s]
            # s_sampled = sampleds.transpose(1, 0).reshape(bsn, 1, time_step, self.dim_s).contiguous() 
            
            # # Compute the entropy and log probability of the posterior distribution of `s_t`
            # s_entropy = dist_s.entropy()  # [batch_size, time_step]
            # s_log_prob_q = dist_s.log_prob(sampleds).mean(0)
            
            # # Store the posterior mean, log variance, and output states
            # s_mus = mean
            # s_log_var = log_var
            # s_posterior_states = output.reshape(bs * 1, time_step, output.shape[-1])

                
        elif self.rnn:
            raise NotImplementedError

        return dist_s
 

    def z_transition_infer(
        self, 
        feed_dict, 
        num_sample=1, 
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
        if idx is not None:
            t_input = feed_dict['time_seq'][:, :idx]
        else:
            t_input = feed_dict['time_seq'] # [bs, times]
        
        bs, time_step = t_input.shape

        # Embed the input sequence, if needed
        emb_rnn_inputs = emb_inputs
        # Compute the output of the posterior network
        output = self.infer_network_posterior_z(emb_rnn_inputs)
        
        # Compute the mean and covariance matrix of the posterior distribution of `z_t`
        output = output[:, -1:] # [bs, 1, out_dim]
        mean, log_var = self.infer_network_posterior_mean_var_z(output) # [bs, times, out_dim]
        cov_mat = torch.diag_embed(torch.exp(log_var) + EPS) 
        dist_z = distributions.multivariate_normal.MultivariateNormal(
            loc=mean, 
            scale_tril=torch.tril(cov_mat)
        )
        
        # # Sample the latent variable `z_t` using Monte Carlo estimation
        # samples = dist_z.rsample((num_sample,))  # [num_sample, batch_size, time_step, out_dim]
        # z_sampled = samples.transpose(1, 0).reshape(num_sample * bs, 1, time_step, -1).contiguous() 
        
        # # Compute the entropy and log probability of the posterior distribution of `z_t`
        # z_entropy = dist_z.entropy()  # [batch_size, time_step]
        # z_log_prob_q = dist_z.log_prob(samples).mean(0)
        
        # # Store the posterior mean, log variance, and output states
        # rnn_states = output.reshape(bs * 1, time_step, output.shape[-1])
        # z_mean = mean
        # z_log_var = cov_mat
        
        return dist_z, # z_sampled, z_entropy, z_log_prob_q, rnn_states, z_mean, z_log_var 
 
 
    def objective_function(
        self,
        s_prior_dist, z_prior_dist, 
        s_vp_dist, z_vp_dist,
        feed_dict_idx, 
    ):
        '''
        s_prior_dist: q_phi(s_t-1)
        '''

        # log tilde_p_theta(s_t)
        s_last_sample = s_prior_dist.rsample(self.num_sample)
        s_next_dist = self.st_transition_func(sampled_s=s_last_sample, feed_dict=feed_dict_idx)
        
        s_vp_sample = s_vp_dist.rsample(self.num_sample)
        log_prob_st = s_next_dist.log_prob(s_vp_sample).mean()
        
        # log tilde_p_theta(z_t | s_t) -> analytically? TODO
        z_last_sample = z_prior_dist.rsample(self.num_sample)
        z_next_dist = self.zt_transition_func(sampled_z=z_last_sample, sampled_s=s_vp_sample, feed_dict=feed_dict_idx)
        z_vp_sample = z_vp_dist.rsample(self.num_sample)
        log_prob_zt = z_next_dist.log_prob(z_vp_sample).mean()
        
        # log p_theta(y_t | z_t)
        if self.dim_z > 1:
            sampled_scalar_z = sampled_scalar_z[:, 0] # [bsn, time, num_node]
            items = torch.tile(feed_dict_idx['skill_seq'], (self.num_sample,1)).unsqueeze(-1)
            sampled_scalar_z_item = torch.gather(sampled_scalar_z, -1, items) # [bsn, time, 1]
        else:
            sampled_scalar_z_item = z_vp_sample
        emission_prob = self.y_emit(sampled_scalar_z_item)
        emission_dist = torch.distributions.bernoulli.Bernoulli(probs=emission_prob)
        y_input = torch.tile(feed_dict_idx['label_seq'].unsqueeze(-1), (self.num_sample, 1, 1)).float()
        log_prob_yt = emission_dist.log_prob(y_input).mean()
        
        st_entropy = s_vp_dist.entropy().mean()
        zt_entropy = z_vp_dist.entropy().mean()
        
        elbo = log_prob_st + log_prob_zt + log_prob_yt - st_entropy - zt_entropy
        iwae = None 
        
        
        return dict(
            elbo=elbo,
            iwae=iwae,
            sequence_likelihood=log_prob_yt+log_prob_zt+log_prob_st,
            st_entropy=st_entropy,
            zt_entropy=zt_entropy)


    def predictive_model(self, inputs, idx=None):
        ipdb.set_trace()
        y_idx = inputs['label_seq'][:, idx].unsqueeze(-1) # [bs, 1]
        device = y_idx.device
        bs = y_idx.shape[0]
        num_sample = self.num_sample
        bsn = bs * num_sample
        
        if idx == 0:
            y_idx = inputs['label_seq'][:, idx].unsqueeze(-1) # [bs, 1]
            
            self.s0_dist = distributions.MultivariateNormal(
                self.gen_s0_mean.to(device), 
                scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_s0_log_var.to(device)) + EPS))
            )
            self.z0_dist = distributions.MultivariateNormal(
                self.gen_z0_mean.to(device), 
                scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_z0_log_var.to(device)) + EPS))
            )
            z0_sampled = self.z0_dist.sample(self.num_sample)
            pred_y = self.y_emit(z0_sampled)
        else:
            time_idx = inputs['time_seq'][:, idx-1:idx+1]
            sampeld_s_last = inputs['sampeld_s_last']
            test_out_dict = self.st_transition_func(
                sampled_s=sampeld_s_last, dist_s=None, feed_dict=None
            )
        
        # ipdb.set_trace()
        pred_dict = {
            'prediction': recon_inputs_items,
            'label': label,
            'pred_y': recon_inputs,
            'pred_z': test_out_dict['sampled_z'][..., train_step:,:],
            'pred_s': test_out_dict['sampled_s'][..., train_step:,:],
            'mean_s': test_out_dict['mean_s'],
            'var_s': test_out_dict['var_s'],
            'mean_z': test_out_dict['mean_z'],
            'var_z': test_out_dict['var_z'],
        }
        
        return pred_dict
    
    
    def inference_model(self, feed_dict, idx=None):
        ipdb.set_trace()
        
        y_idx = feed_dict['label_seq'][:, :idx].unsqueeze(-1) # [bs, 1]
        t_idx = feed_dict['time_seq'][:, :idx].unsqueeze(-1) # [bs, 1]
        item_idx = feed_dict['skill_seq'][:, :idx].unsqueeze(-1) # [bs, 1]
    
        bs = y_idx.shape[0]
        device = y_idx.device
        
        # if idx == 0:
        #     h_emb_seq_last = [torch.zeros((bs, 1, self.node_dim)).to(device)] * 2
        # else:
        #     pass
            
        # self.s0_dist = distributions.MultivariateNormal(
        #     self.gen_s0_mean.to(device), 
        #     scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_s0_log_var.to(device)) + EPS))
        # )
        # self.z0_dist = distributions.MultivariateNormal(
        #     self.gen_z0_mean.to(device), 
        #     scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_z0_log_var.to(device)) + EPS))
        # )
        
        # ----- embedding -----
        t_pe = self.get_time_embedding(t_idx, type='absolute')
        y_pe = torch.tile(y_idx.unsqueeze(-1), (1,1, self.node_dim)) # TODO # [bs, times, dim]]   
        
        if self.num_node > 1:
            node_pe = self.node_dist._get_node_embedding()[items] 
            emb_input = torch.cat([node_pe, y_pe], dim=-1) # [bs, times, dim*4]
            self.infer_network_emb.flatten_parameters()
            emb_rnn_inputs, _ = self.infer_network_emb(emb_input) # [bs, times, dim*2(32)]
        else: 
            emb_rnn_inputs = y_pe
        emb_rnn_inputs = emb_rnn_inputs + t_pe

        # ----- Sample continuous hidden variable from `q(s[t] | y[1:t])' -----
        s_dist = self.s_infer(
            feed_dict, self.num_sample, emb_inputs=emb_rnn_inputs, idx=idx
        )

        # ----- Sample continuous hidden variable from `q(z[1:T] | y[1:T])' -----
        z_dist  = self.z_infer(
            feed_dict, num_sample=self.num_sample, emb_inputs=emb_rnn_inputs, idx=idx,
        )
        # if self.dim_z > 1:
        #     z_sampled_scalar = z_sampled @ self.node_dist._get_node_embedding().transpose(-1,-2).contiguous().to(z_sampled.device)# .clone().detach()
        # else:
        #     z_sampled_scalar = z_sampled
        
        
        # recon_inputs = torch.reshape(recon_inputs, [bs, self.num_sample, self.num_node, num_steps, -1])
        # recon_inputs_items = torch.reshape(recon_inputs_items, [bs, self.num_sample, 1, num_steps, -1])
        # z_sampled_scalar = torch.reshape(z_sampled_scalar, [bs, self.num_sample, num_steps, self.num_node, self.dim_z]).permute(0,1,3,2,4).contiguous()
        # s_sampled = torch.reshape(s_sampled,
        #                     [bs, self.num_sample, 1, num_steps, self.dim_s])
        
        # return_dict = self.get_objective_values(
        #     [log_prob_st, log_prob_zt, log_prob_yt], 
        #     [s_log_prob_q, z_log_prob_q],
        #     [s_entropy, z_entropy], 
        #     self.num_sample
        # )

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
        gt = torch.tile(gt[:,None, ...], (1,self.num_sample,1,1,1))
        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred, gt.float())
        losses['loss_bce'] = bceloss
        
        for key in ['elbo', 'initial_likelihood', 'sequence_likelihood', 
                    'st_entropy', 'zt_entropy',
                    'log_prob_yt', 'log_prob_zt', 'log_prob_st']:
            losses[key] = outdict[key].mean()
        losses['loss_total'] = -outdict['elbo'].mean()
        
        # Still NOT for optimization
        pred_att = torch.exp(self.edge_log_probs)# self.node_dist._get_atten_weights()[-1].clone().detach()
        gt_adj = self.adj.to(pred.device).transpose(-1,-2)
        losses['spasity'] = (pred_att >= 0.5).sum()
        losses['adj_0_att_1'] = (1 * (pred_att >= 0.5) * (1-gt_adj)).sum()
        losses['adj_1_att_0'] = (1 * (pred_att < 0.5) * gt_adj).sum()
        
        # Register output predictions
        self.register_buffer(name="output_predictions", tensor=pred)
        self.register_buffer(name="output_gt", tensor=gt)
        self.register_buffer(name="output_attention_weights", tensor=pred_att)
        self.register_buffer(name="output_gt_graph_weights", tensor=gt_adj)
        
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
        losses['ou_gamma'] = outdict["sampled_s"][...,3].mean()
        return losses
    
    
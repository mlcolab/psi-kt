import sys
sys.path.append('..')

import math, os, argparse
import numpy
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from utils.logger import Logger

import torch
from torch import nn, distributions
from torch.nn import functional as F

from collections import defaultdict

import ipdb

from models.modules import build_rnn_cell, build_dense_network
from models.BaseModel import BaseModel
from models.modules import CausalTransformerModel, VAEEncoder
from models.variational_distributions import VarTransformation

from enum import Enum

torch.autograd.set_detect_anomaly(True)

RANDOM_SEED = 131 
EPS = 1e-6
T_SCALE = 60*60*24


class HSSM(BaseModel):
    def __init__(
        self,
        mode: str = 'train',
        num_node: int = 1,
        num_seq: int = 1,
        args: argparse.Namespace=None,
        device: torch.device='cpu',
        logs=None,
        nx_graph=None,
    ):
        # self.fit_vi_global_s, self.fit_vi_transition_s, self.infer_global_s, self.infer_transition_s = 0,0,0,1
        self.user_time_dependent_covariance = 1
        self.diagonal_std, self.lower_tri_std = 1, 0
        
        self.num_node = num_node
        self.logs = logs
        self.device = device
        self.args = args
        self.num_seq = num_seq
        self.num_sample = args.num_sample

        # Set the device to use for computations
        self.device = args.device

        # Store the arguments and logs for later use
        self.args = args
        self.logs = logs
        BaseModel.__init__(self, model_path=os.path.join(args.log_path, 'Model/Model_{}_{}.pt'))
        

    @staticmethod	
    def normalize_timestamps(
        timestamps: torch.Tensor,
    ):	
        '''	
        timestamps: [bs, T, ...]
        '''	
        mean_val = torch.mean(timestamps, dim=1, keepdim=True)	
        std_val = torch.std(timestamps, dim=1, keepdim=True)	
        normalized_timestamps = (timestamps - mean_val) / std_val	
        return normalized_timestamps
    

    @staticmethod	
    def _construct_normal_from_mean_std(
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        """
        Construct a multivariate Gaussian distribution from a mean and covariance matrix.

        Parameters:
        - mean: a PyTorch parameter representing the mean of the Gaussian distribution
        - cov: a PyTorch parameter representing the covariance matrix of the Gaussian distribution

        Returns:
        - dist: a PyTorch distribution representing the multivariate Gaussian distribution
        """
        if std.shape[-1] == 1:
            # if the covariance matrix is a scalar, create a univariate normal distribution
            dist = distributions.MultivariateNormal(mean, std*std + EPS)
            return dist
        
        else:
            # check if the covariance matrix is positive definite
            
            # # way 1
            # raw_sigma_bias = 0.0
            # sigma_min = 1e-5
            # sigma_scale = 0.05
            # cov = (torch.maximum(F.softmax(cov + raw_sigma_bias, dim=-1),
            #                                 torch.tensor(sigma_min)) * sigma_scale)
            
            # # way 2
            # try:
            #     torch.linalg.cholesky(cov)
            # except RuntimeError:
            #     # if the covariance matrix is not positive definite, add a small positive constant to its diagonal
            #     cov = cov + torch.eye(cov.size(-1)) * EPS

            # way 3
            # use the torch.cholesky() function to compute the Cholesky decomposition of a matrix. 
            # The resulting lower triangular matrix is guaranteed to be positive definite
            cov_pdm = torch.matmul(std, std.transpose(-1,-2)) + EPS
            # try:
            #     torch.linalg.cholesky(cov_pdm)
            # except RuntimeError:
            #     # if the covariance matrix is not positive definite, add a small positive constant to its diagonal
            #     cov_pdm = cov_pdm + torch.eye(cov_pdm.size(-1)) * EPS
                
            # create a multivariate normal distribution
            dist = distributions.MultivariateNormal(mean, scale_tril=torch.tril(cov_pdm))
        return dist
    
    
    @staticmethod
    def _initialize_normal_mean_log_var(
        dim: int, 
        use_trainable_cov: bool, 
        num_sample: int = 1
    ):
        """
        Construct the initial mean and covariance matrix for the multivariate Gaussian distribution.

        Parameters:
        - dim: an integer representing the dimension of the Gaussian distribution
        - use_trainable_cov: a boolean indicating whether to use a trainable covariance matrix
        - num_sample: an integer representing the number of samples to generate

        Returns:
        - x0_mean: a PyTorch parameter representing the initial mean of the Gaussian distribution
        - x0_scale: a PyTorch parameter representing the initial covariance matrix of the Gaussian distribution
        """
        x0_mean = nn.init.xavier_uniform_(torch.empty(num_sample, dim))#, device=self.device))
        x0_mean = nn.Parameter(x0_mean, requires_grad=True)

        # m = nn.init.xavier_uniform_(torch.empty(num_sample, int(dim * (dim + 1) / 2), device=self.device))
        # x0_scale = torch.zeros((num_sample, dim, dim), device=self.device)
        # tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)
        # x0_scale[:, tril_indices[0], tril_indices[1]] += m
        # x0_scale = nn.Parameter(x0_scale, requires_grad=use_trainable_cov)
        x0_log_var = nn.init.xavier_uniform_(torch.empty(num_sample, dim))#, device=self.device))
        x0_log_var = nn.Parameter(x0_log_var, requires_grad=use_trainable_cov)
        
        return x0_mean, x0_log_var


    def get_objective_values(
        self,
        log_probs: List[torch.Tensor], 
        log_prob_q: torch.Tensor = None,
        posterior_entropies: List[torch.Tensor] = None,
        temperature: float = 1.0,
    ):
        [log_prob_st, log_prob_zt, log_prob_yt] = log_probs 
        
        # sequence_likelihood = log_prob_yt[:, 1:].mean(-1) # [bs,]
        # initial_likelihood = log_prob_yt[:, 0]
        sequence_likelihood = (log_prob_st[:, 1:] + log_prob_zt[:, 1:] + log_prob_yt[:, 1:]).mean(-1)/3 # [bs,]
        initial_likelihood = (log_prob_st[:, 0] + log_prob_zt[:, 0] + log_prob_yt[:, 0])/3

        t1_mean = torch.mean(sequence_likelihood, dim=0)
        t2_mean = torch.mean(initial_likelihood, dim=0)
        
        t3_mean = torch.mean(posterior_entropies[0]) 
        t4_mean = torch.mean(posterior_entropies[1]) 
        
        elbo = t1_mean + t2_mean + temperature * (t3_mean + t4_mean)
        
        iwae = None # TODO ???
        # iwae = self._get_iwae(sequence_likelihood, initial_likelihood, log_prob_q,
        #                     num_sample)
        return dict(
            elbo=elbo,
            iwae=iwae,
            initial_likelihood=t2_mean,
            sequence_likelihood=t1_mean,
            st_entropy=t3_mean,
            zt_entropy=t4_mean)


    def get_reconstruction(
        self,
        hidden_state_sequence: torch.Tensor,
        observation_shape: torch.Size = None,
        sample_for_reconstruction: bool = True,
        sample_hard: bool = False,
    ): 
        
        emission_dist = self.y_emit(hidden_state_sequence)
        mean = emission_dist

        if sample_for_reconstruction:
            # reconstructed_obs = emission_dist.sample() # NOTE: no gradient!
            probs = torch.cat([1-mean, mean], dim=-1)
            reconstructed_obs = F.gumbel_softmax(torch.log(probs + EPS), tau=1, hard=sample_hard, eps=1e-10, dim=-1)
            reconstructed_obs = reconstructed_obs[..., 1:]
        else:
            reconstructed_obs = mean

        if observation_shape is not None:
            reconstructed_obs = torch.reshape(reconstructed_obs, observation_shape)

        return reconstructed_obs
    
    
    def get_s_prior(
        self, 
        sampled_s: torch.Tensor,
        inputs: torch.Tensor = None
    ):
        """
        p(s[t] | s[t-1]) transition.
        """ 
        prior_distributions = self.st_transition_gen(sampled_s, inputs)
        
        sampled_s0 = sampled_s[:, :, 0] # [bsn, 1, s_dim]
        log_prob_s0 = self.s0_dist.log_prob(sampled_s0) # [bsn, 1]
        
        future_tensor = sampled_s[:, 0, 1:]
        log_prob_st = prior_distributions.log_prob(future_tensor)

        log_prob_st = torch.cat([log_prob_s0, log_prob_st], dim=-1) / self.dim_s

        self.register_buffer('output_s_prior_distributions_mean', prior_distributions.mean.clone().detach())
        self.register_buffer('output_s_prior_distributions_var', prior_distributions.variance.clone().detach())
          
        return log_prob_st
    
    
    def get_z_prior(
        self, 
        sampled_z_set: List[torch.Tensor],
        sampled_s: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
    ):
        """
        p(z[t] | z[t-1], s[t]) transition.
        z_sampled_scalar = z_sampled @ self.value_u.transpose(-1,-2)
        """
        sampled_z, sampled_scalar_z = sampled_z_set
        prior_distributions = self.zt_transition_gen(sampled_z_set, sampled_s, inputs)
        
        sampled_z0 = sampled_z[:, :, 0] # [bsn, 1, z_dim]
        log_prob_z0 = self.z0_dist.log_prob(sampled_z0) / sampled_z0.shape[-1] # [bs, 1]
        
        future_tensor = sampled_scalar_z[:, :, 1:]
        log_prob_zt = prior_distributions.log_prob(future_tensor).mean(1) # [bs*n, times-1]
        
        log_prob_zt = torch.cat([log_prob_z0, log_prob_zt], dim=-1)  # [bs*n, num_steps]
        
        self.register_buffer('output_z_prior_distributions_mean', prior_distributions.mean.clone().detach())
        self.register_buffer('output_z_prior_distributions_var', prior_distributions.variance.clone().detach())
        
        return log_prob_zt
    
    
    def loss(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        outdict: Dict[str, torch.Tensor],
        metrics: List[str] = None,
    ):
        losses = defaultdict(lambda: torch.zeros(()))#, device=self.device))
        
        gt = outdict["label"] 
        pred = outdict["prediction"]
        num_sample = pred.shape[1]
        gt = torch.tile(gt[:,None, ...], (1,num_sample,1,1,1))
        
        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred, gt.float())
        losses['loss_bce'] = bceloss
        
        for key in ['elbo', 'initial_likelihood', 'sequence_likelihood', 
                    'st_entropy', 'zt_entropy',
                    'log_prob_yt', 'log_prob_zt', 'log_prob_st']:
            losses[key] = outdict[key].mean()
        
        losses['loss_total'] = -outdict['elbo'].mean()
        
        if metrics != None:
            pred = pred.detach().cpu().data.numpy()
            gt = gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
        
        losses['ou_speed'] = outdict["sampled_s"][...,0].mean()
        losses['ou_mean'] = outdict["sampled_s"][...,1].mean()
        losses['ou_vola'] = outdict["sampled_s"][...,2].mean()

        return losses


class GraphHSSM(HSSM):
    def __init__(
        self,
        mode: str = 'ls_split_time',
        num_node: int = 1,
        num_seq: int = 1,
        args: argparse.Namespace = None,
        device: torch.device = torch.device('cpu'),
        logs: Logger = None,
        nx_graph: numpy.ndarray = None,
    ):
        # self.fit_vi_global_s, self.fit_vi_transition_s, self.infer_global_s, self.infer_transition_s = 0,0,0,1
        
        # ----- specify dimensions of all latents -----
        self.node_dim = 16
        self.emb_mean_var_dim = 16
        self.num_node = num_node

        # ----- initialize graph parameters -----
        self.learned_graph = args.learned_graph
        if self.learned_graph == 'none' or self.num_node == 1:
            self.dim_s = 3
            self.dim_z = 1
        else: 
            self.dim_s = 4
            self.dim_z = self.node_dim
            if self.learned_graph == 'w_gt': # TODO
                pass
            elif self.learned_graph == 'no_gt':
                pass
            self.adj = torch.tensor(nx_graph)
            assert(self.adj.shape[-1] >= num_node)
        
        self.var_log_max = torch.tensor(10) # TODO
        super().__init__(mode, num_node, num_seq, args, device, logs, nx_graph)


    def _init_weights(
        self
    ):
        """
        Initialize the weights of the model.

        This function creates the necessary layers of the model and initializes their weights.

        Returns:
            None
        """
        self.node_dist = VarTransformation(device=self.device, num_nodes=self.num_node, tau_gumbel=1, dense_init = False, 
                        latent_prior_std=None, latent_dim=self.node_dim)
        
        # ----- for parameters Theta -----
        # the initial distribution p(s0) p(z0), the transition distribution p(s|s') p(z|s,z'), the emission distribution p(y|s,z)
        # 1. initial distribution p(s0) p(z0): trainable mean and variance???
        self.gen_s0_mean, self.gen_s0_log_var = self._initialize_normal_mean_log_var(self.dim_s, False)
        self.gen_z0_mean, self.gen_z0_log_var = self._initialize_normal_mean_log_var(self.dim_z, False) # self.z0_scale is std
        
        # 2. transition distribution p(s|s') or p(s|s',y',c'); p(z|s,z') (OU)
        self.s_transit_w_slast = 0
        self.s_transit_w_slast_yc = 1 - self.s_transit_w_slast # TODO: should incoporate pe of s and time t
        if self.s_transit_w_slast: 
            self.gen_network_transition_s = build_dense_network( 
                self.dim_s,
                [self.dim_s, self.dim_s],
                [nn.ReLU(), None]
            )
        elif self.s_transit_w_slast_yc: 
            # self.gen_network_transition_s = build_dense_network(
            #     self.node_dim,
            #     [self.node_dim, self.dim_s], 
            #     [nn.ReLU(), None]
            # )
            # self.gen_network_transition_s = nn.LSTM(
            #     input_size=self.node_dim, 
            #     hidden_size=self.node_dim,  
            #     bidirectional = False, 
            #     batch_first = True,
            # )
            self.gen_network_prior_mean_var_s = VAEEncoder(
                self.node_dim, self.emb_mean_var_dim, self.dim_s
            ) 
        else:
            raise NotImplementedError
        # self.st_transition_gen
        # self.zt_transition_gen
        
        # 3. emission distribution p(y|z)
        self.y_emit = torch.nn.Sigmoid() 
        
        
        # ----- for parameters Phi -----
        # the embedding network at each time step emb_t = f(y_t, c_t, t)
        # the variational posterior distribution q(s_1:t | y_1:t, c_1:t) and q(z_1:t | y_1:t, c_1:t) TODO could add structure later q(z_1:t | y_1:t, s_1:t)
        # 1. embedding network
        # self.infer_network_emb = build_dense_network(
        #     self.node_dim*2,
        #     [self.node_dim, self.node_dim], 
        #     [nn.ReLU(), None]
        # )
        self.infer_network_emb = nn.LSTM( 
            input_size=self.node_dim*2, 
            hidden_size=self.node_dim//2, 
            bidirectional=True, # TODO why bidirectional
            batch_first=True,
        )
        # self.infer_network_posterior_s = CausalTransformerModel(
        #     ntoken=self.num_node,
        #     ninp=self.infer_network_emb.hidden_size*2 if self.infer_network_emb.bidirectional else self.infer_network_emb.hidden_size,
        #     nhid=self.node_dim,
        # )
        # 2. variational posterior distribution q(s_1:t | y_1:t, c_1:t) = q(s_1:t | emb_1:t)
        self.transformer, self.rnn, self.explcit_rnn, self.implicit_rnn = 0,1,0,1
        if self.explcit_rnn:
            self.infer_network_posterior_s = build_rnn_cell(
                rnn_type="lstm", 
                hidden_dim_rnn=self.node_dim, 
                rnn_input_dim=self.node_dim*2
            )
        elif self.implicit_rnn:
            self.infer_network_posterior_s = nn.LSTM(
                input_size=self.node_dim, # self.infer_network_emb.hidden_size*2 if self.infer_network_emb.bidirectional else self.infer_network_emb.hidden_size,
                hidden_size=self.node_dim,  
                bidirectional = False, 
                batch_first = True,
            )
        elif self.transformer: 
            self.infer_network_posterior_s = CausalTransformerModel(
                ntoken=self.num_node,
                ninp=self.node_dim, # self.infer_network_emb.hidden_size*2 if self.infer_network_emb.bidirectional else self.infer_network_emb.hidden_size,
                nhid=self.node_dim,
            )
        self.infer_network_posterior_mean_var_s = VAEEncoder(
            self.node_dim, self.emb_mean_var_dim, self.dim_s  # the first one should be # self.infer_network_posterior_s.hidden_dim
        )
        # self.st_transition_infer
        # 3. variational posterior distribution q(z_1:t | y_1:t, c_1:t)
        if self.rnn:
            self.infer_network_posterior_z = nn.LSTM(
                input_size=self.node_dim, # self.infer_network_emb.hidden_size*2 if self.infer_network_emb.bidirectional else self.infer_network_emb.hidden_size,
                hidden_size=self.node_dim,  
                bidirectional = False, 
                batch_first = True,
            )
        elif self.transformer: 
            self.infer_network_posterior_z = CausalTransformerModel(
                ntoken=self.num_node,
                ninp=self.node_dim, # self.infer_network_emb.hidden_size*2 if self.infer_network_emb.bidirectional else self.infer_network_emb.hidden_size,
                nhid=self.node_dim,
            )
        self.infer_network_posterior_mean_var_z = VAEEncoder(
            self.node_dim, self.emb_mean_var_dim, self.dim_z
        )
        

    @staticmethod
    def positionalencoding1d(
        d_model, 
        length, 
        actual_time=None):
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
    
    
    def get_time_embedding(
        self, 
        time: torch.Tensor,
        type: str = 'dt',
    ):
        if type == 'dt':
            dt = torch.diff(time, dim=1) # ?
            t_pe = self.positionalencoding1d(self.node_dim, dt.shape[1], dt) # [bs, times, dim]
        elif type =='absolute':
            norm_t = time # time-time.min(1)[0].unsqueeze(1) # TODO
            t_pe = self.positionalencoding1d(self.node_dim, time.shape[1], norm_t) # [bs, times, dim]
        return t_pe
    
    
    def st_transition_gen(
        self, 
        sampled_s: torch.Tensor,
        feed_dict: Dict[str, torch.Tensor],
        eval: bool = False,
    ):
        '''
        Compute the transition function of the latent skills `s_t` in the model.

        Args:
            sampled_s (torch.Tensor): [batch_size * num_sample, num_node (1), time_steps, dim_s], Sampled latent skills `s_t` of shape.
            feed_dict (dict): Dictionary of input tensors containing the following keys:
                - label_seq (torch.Tensor): Sequence of label embeddings of shape [batch_size, time].
                - skill_seq (torch.Tensor): Sequence of skill IDs of shape [batch_size, time].
                - time_seq (torch.Tensor): Sequence of time intervals of shape [batch_size, time].

        Returns:
            s_prior_dist (torch.distributions.MultivariateNormal): Multivariate normal distribution of `s_t`
                with mean and covariance matrix computed using a neural network.
        '''
        
        input_y = feed_dict['label_seq'] # [bs, times]
        items = feed_dict['skill_seq']
        bsn = items.shape[0] * self.num_sample
        
        time_emb = self.get_time_embedding(feed_dict['time_seq'], type='dt')
        value_u = self.node_dist._get_node_embedding().to(sampled_s.device)# .clone().detach()

        if sampled_s.shape[-2] == 1:
            s_last = sampled_s[:, 0]
            item_last = items
            y_last = input_y
        else:
            s_last = sampled_s[:, 0, :-1] # [bsn, times-1, dim]
            item_last = items[:, :-1]
            y_last = input_y[:, :-1]
            
            
        if self.s_transit_w_slast:
            raise NotImplementedError
        
        elif self.s_transit_w_slast_yc:
            input_emb = value_u[item_last] + y_last.unsqueeze(-1) + time_emb # [bs, times-1, dim_emb]
            # output = self.gen_network_transition_s(rnn_input_emb)
            mean_div, log_var = self.gen_network_prior_mean_var_s(input_emb) 
            
            mean = mean_div.unsqueeze(1).repeat(1,self.num_sample,1,1).reshape(bsn, mean_div.shape[-2], mean_div.shape[-1]) + s_last
            # mean = mean_div.repeat(self.num_sample, 1, 1) + s_last # TODO: can to add information of s_t-1 into the framework; how to constrain that s is not changing too much?
            
            log_var = torch.minimum(log_var, self.var_log_max.to(log_var.device))
            cov_mat = torch.diag_embed(torch.exp(log_var) + EPS).unsqueeze(1).repeat(1,self.num_sample,1,1,1).reshape(bsn, log_var.shape[-2], log_var.shape[-1], log_var.shape[-1])
            
            s_prior_dist = distributions.multivariate_normal.MultivariateNormal(
                    loc=mean, 
                    scale_tril=torch.tril(cov_mat)
                )
            
        if not eval:
            self.register_buffer('output_prior_s_mean', mean.clone().detach())
            self.register_buffer('output_prior_s_empower', cov_mat.clone().detach( ))
        
        return s_prior_dist # , s_sampled, s_entropy, s_log_prob_q, rnn_states, s_mean, s_var
    
    
    def zt_transition_gen(
        self, 
        sampled_z_set: Tuple[torch.Tensor, torch.Tensor],
        sampled_s: torch.Tensor,
        feed_dict: Dict[str, torch.Tensor],
        eval: bool = False,
    ):
        '''
        Compute the transition function of the scalar outcomes `z_t` in the model.

        Args:
            sampled_z_set (tuple): Tuple containing:
                - sampled_z (torch.Tensor): Sampled scalar outcomes `z_t` of shape [bsn, 1, time, dim_z]
                - sampled_scalar_z (torch.Tensor): Sampled scalar outcomes `z_t` of shape [bsn, num_node, time, dim_z (1)]
            sampled_s (torch.Tensor): Sampled latent skills `s_t` of shape [bsn, 1, time, dim_s]
            feed_dict (dict): Dictionary of input tensors containing the following keys:
                - time_seq (torch.Tensor): Sequence of time intervals of shape [batch_size, times].

        Returns:
            z_prior_dist (torch.distributions.MultivariateNormal): Multivariate normal distribution of `z_t`
                with mean and covariance matrix computed using the sampled latent skills `s_t`.
        '''
        
        input_t = feed_dict['time_seq']
        bs, num_steps = input_t.shape
        bsn = bs * self.num_sample
        
        # ----- calculate time difference -----
        # TODO: t scale would change if datasets change
        # ipdb.set_trace()
        dt = torch.diff(input_t.unsqueeze(1), dim=-1) + EPS# /T_SCALE + EPS  # [bs, 1, num_steps-1] 
        dt = dt.unsqueeze(1).repeat(1, self.num_sample, 1, 1).reshape(bsn, 1, num_steps-1) # [bsn, 1, num_steps-1]
        
        if num_steps == 2:
            sampled_st = sampled_s
            sampled_z = sampled_z_set[1][..., 0]
        else:
            sampled_st = sampled_s[:, :, 1:]
            sampled_z = sampled_z_set[1][:,:,:-1,0] # [bsn, num_node, times-1]
            
        # NOTE: need very careful design of the constraint of these interpretable parameters
        # sampled_alpha = torch.minimum(torch.relu(sampled_st[..., 0]) + EPS, torch.tensor(10.).to(sampled_st.device)) # *self.args.alpha_minimum # TODO # [bsn, 1, times-1]
        sampled_alpha = torch.sigmoid(sampled_st[..., 0]) # *self.args.alpha_minimum # TODO # [bsn, 1, times-1]
        decay = torch.exp(-sampled_alpha * dt)
        sampled_mean = sampled_st[..., 1]
        sampled_var = torch.sigmoid(sampled_st[..., 2]) # torch.exp(sampled_log_var) * decay + EPS # TODO not constrained
        sampled_gamma = torch.sigmoid(sampled_st[..., 3])
        
        # ----- Simulate the path of `z_t` -----
        # attn_output_weights = self.node_dist._get_atten_weights()
        # adj = attn_output_weights[-1].to(sampled_z.device)
        # TODO test with multiple power of adj
        # in_degree = self.adj.sum(dim=0)
        # ind = torch.where(in_degree == 0)[0] 
        adj = self.node_dist.sample_A(self.num_sample)[-1][:,0].mean(0).to(sampled_s.device) # [n, num_node, num_node] # adj_ij means i has influence on j
        # adj =  torch.exp(self.node_dist.edge_log_probs()[0]).to(sampled_s.device) # adj_ij means i has influence on j
        empower = torch.einsum('bin,ij->bjn', sampled_z, adj) / self.num_node * sampled_gamma # [bs*n, num_node, times-1]
        # torch.matmul(sampled_z, adj_t).transpose(-1,-2).contiguous()/self.num_node  * sampled_gamma
        # empower = (sampled_z.unsqueeze(-2) * adj_t).sum(-1).transpose(-1,-2).contiguous()/self.num_node  * sampled_gamma # [bs*n, num_node, times-1] 
        # empower = (1 / (in_degree[None, :, None] + EPS)) * sampled_gamma * empower # [bs*n, num_node, 1]
        # empower[:,ind] = 0
        z_pred = sampled_z * decay + (1.0 - decay) * (sampled_mean + empower)/2
        
        z_mean = z_pred.reshape(bsn, self.num_node, num_steps-1, 1) # [bs*n, num_node, num_steps, 1]
        z_var = sampled_var.reshape(bsn, 1, num_steps-1, 1) + EPS # torch.cat([z0_var, sampled_var], -2) # [bs*n, num_node, num_steps, 1]
        
        z_prior_dist = distributions.multivariate_normal.MultivariateNormal(
            loc=z_mean, 
            scale_tril=torch.tril(torch.diag_embed(z_var))
        )
        # z_sampled = output_dist.rsample() # [bs*n, num_node, num_steps-1, 1]
        # z_entropy = output_dist.entropy().mean(-2) # [bs*n, num_steps-1]
        # z_log_prob_q = output_dist.log_prob(z_sampled).mean(-2) # [bs*n, num_steps-1]
        
        if not eval:    
            self.register_buffer('output_prior_z_decay', decay.clone().detach())
            self.register_buffer('output_prior_z_empower', empower.clone().detach())
            self.register_buffer('output_prior_z_tmp_mean_level', ((sampled_mean + empower)/2).clone().detach())
        
        return z_prior_dist# , z_sampled, z_entropy, z_log_prob_q, z_mean, z_var
        
        
    def yt_emission_func(self, ):
        pass
    
    
    def st_transition_infer(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        num_sample: int = 1,
        emb_inputs: Optional[torch.Tensor] = None,
    ):

        """
        Recursively sample z[t] ~ q(z[t]|h[t]=f_RNN(h[t-1], z[t-1], h[t]^b)).

        Args:
        inputs:     a float `Tensor` of size [batch_size, num_steps, obs_dim], where each observation 
                    should be flattened.
        num_sample: an `int` scalar for number of samples per time-step, for posterior inference networks, 
                    `z[i] ~ q(z[1:T] | x[1:T])`.
        emb_inputs: [batch_size, num_steps, emb_dim], where `emb_dim` is the dimension of the embeddings

        Returns:
        s_sampled: [batch_size * num_sample, num_node (1), time_steps, dim_s]
        s_entropy: [batch_size, time_steps]
        s_log_prob_q: [batch_size, time_steps]
        s_posterior_states: [batch_size, time_steps, dim_emb]
        s_mus: [batch_size, time_steps, dim_s]
        s_log_vars: [batch_size, time_steps, dim_s]
        
        """
        
        t_input = feed_dict['time_seq'] # [bs, times]
        bs, time_step = t_input.shape
        emb_rnn_inputs = emb_inputs
        bsn = bs * self.num_sample
        
        if self.transformer:
            # Compute the output of the posterior network
            output = self.infer_network_posterior_s(emb_rnn_inputs)
            
            # Compute the mean and covariance matrix of the posterior distribution of `s_t`
            mean, log_var = self.infer_network_posterior_mean_var_s(output)  # [batch_size, time_step, dim_s]
            cov_mat = torch.diag_embed(torch.exp(log_var) + EPS)
            dist_s = distributions.multivariate_normal.MultivariateNormal(
                loc=mean, 
                scale_tril=torch.tril(cov_mat)
            )
            
            # Sample the latent skills `s_t` using Monte Carlo estimation
            samples = dist_s.rsample((num_sample,))  # [num_sample, batch_size, time_step, dim_s]
            s_sampled = samples.transpose(1, 0).reshape(bsn, 1, time_step, self.dim_s).contiguous() 
            
            # Compute the entropy and log probability of the posterior distribution of `s_t`
            s_entropy = dist_s.entropy()  # [batch_size, time_step]
            s_log_prob_q = dist_s.log_prob(samples).mean(0)
            
            # Store the posterior mean, log variance, and output states
            s_mus = mean
            s_log_var = log_var
            s_posterior_states = output.reshape(bs * 1, time_step, output.shape[-1])

                
        elif self.rnn:
            if self.implicit_rnn:
                output, _ = self.infer_network_posterior_s(emb_rnn_inputs) # [bs, times, dim*2(32)
                mean, log_var = self.infer_network_posterior_mean_var_s(output)
                # sampling epsilon https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb         
                # epsilon = torch.randn_like((bs, num_sample, time_step, self.dim_s))        
                # s_sampled = mean + torch.exp(0.5 * log_var)*epsilon

                log_var = torch.minimum(log_var, self.var_log_max.to(log_var.device))
                cov_mat = torch.diag_embed(torch.exp(log_var) + EPS) 
                dist_s = distributions.multivariate_normal.MultivariateNormal(
                    loc=mean, 
                    scale_tril=torch.tril(cov_mat)
                    )
                
                samples = dist_s.rsample((num_sample,)) # [n, bs, times, dim_s]
                s_sampled = samples.transpose(1,0).reshape(num_sample * bs, 1, time_step, self.dim_s).contiguous() 
                s_entropy = dist_s.entropy() # [bs, times]
                s_log_prob_q = dist_s.log_prob(samples).mean(0)
                
                s_posterior_states = output.reshape(1 * bs, time_step, output.shape[-1])
                s_mus = mean
                s_log_var = log_var
            
            elif self.explicit_rnn:
                latent_states = []
                rnn_states = []
                entropies = []
                log_probs = []
                mus = []
                covs = []
                for t in range(time_step):
                    current_input = emb_rnn_inputs[:, t] # [bs*n, 32] # contain the input (time, node, label)
                    
                    # rnn_input for p(h_t | [t, node_t, y_t], s_{t-1}, h_{t-1})
                    # rnn_out is h_t 
                    # rnn_state is c_t  
                    # TODO: do i need to concatenate: rnn_input = torch.concat([current_input, prev_latent_state], dim=-1)  # [bs, 64] 
                    rnn_input = current_input
                    rnn_out, rnn_cell_state = self.posterior_rnn(rnn_input, prev_rnn_state)  # [bs, 16]
                    # TODO: some papers sampled from rnn_out, rather than the mean_tensor below

                    # sample s_t from p(s_t | h_t)
                    mean_tensor = self.network_posterior_mean_mlp_s(rnn_out) # [bs*n, y_emission_net.out_dim]
                    cov_tensor = self.network_posterior_log_var_mlp_s(rnn_out) # TODO only predict diagonal covariance?
                    cov_mat = torch.diag_embed(cov_tensor + EPS) 
                    dist_s = distributions.multivariate_normal.MultivariateNormal(
                        loc=mean_tensor, scale_tril=torch.tril(cov_mat)
                    )
                    latent_state = dist_s.rsample((num_sample,))  # [n, bs, dim_s]
                    
                    mus.append(mean_tensor)
                    covs.append(cov_mat)
                    latent_states.append(latent_state)
                    rnn_states.append(rnn_out)
                    log_probs.append(dist_s.log_prob(latent_state)) 
                    entropies.append(dist_s.entropy()) 
                    # TODO no gradient now -> because the entropy of a multivariate Gaussian only depends on the 
                    # variance matrix; while the variance matrix now is not trainable -> even if it is trainable, 
                    # the variance is not depends on the input `rnn_out` (it is only a trainable parameter). 
                    # So the gradient will not flow back to the posterior estimator rnn
                    
                    prev_latent_state = rnn_out 
                    prev_rnn_state = [rnn_out, rnn_cell_state]
            
                sampled_s = torch.stack(latent_states, -2).reshape(num_sample * bs, 1, time_step, self.dim_s) 
                s_entropy = torch.stack(entropies, -1).reshape(1 * bs, time_step)
                s_log_prob_q = torch.stack(log_probs, -1).reshape(num_sample * bs, time_step)
                rnn_states = torch.stack(rnn_states, -2).reshape(1 * bs, time_step, self.posterior_rnn.hidden_size)
                mus = torch.stack(mus, -2).reshape(1 * bs, time_step, self.dim_s)
                covs = torch.stack(covs, -2).reshape(1 * bs, time_step, self.dim_s, self.dim_s)

        return s_sampled, s_entropy, s_log_prob_q, s_posterior_states, s_mus, s_log_var
 

    def zt_transition_infer(
        self, 
        inputs: Tuple[torch.Tensor, torch.Tensor],
        num_sample: int,
        emb_inputs: Optional[torch.Tensor] = None,
    ):
        '''
        Compute the posterior distribution of the latent variable `z_t` given the input and output sequences.

        Args:
            inputs (tuple): A tuple containing the feed dictionary and the sampled skills `s_t`.
                feed_dict: A dictionary containing the input and output sequences.
                sampled_s: [batch_size * num_sample, num_node (1), time_step, dim_s]
            num_sample (int): Number of samples for Monte Carlo estimation of the posterior distribution.
            emb_inputs (torch.Tensor): [batch_size, time_step, dim_emb] Optional embedded input sequence of shape.

        Returns:
            z_sampled (torch.Tensor): [batch_size * num_sample, num_node (1), time_steps, dim_z], Sampled latent variable `z_t` of shape.
            z_entropy (torch.Tensor): [batch_size, time_steps], Entropy of the posterior distribution of `z_t`.
            z_log_prob_q (torch.Tensor): [batch_size, time_steps], Log probability of the posterior distribution of `z_t`.
            z_posterior_states (torch.Tensor): [batch_size, time_steps, dim_emb], Output states of the posterior network.
            z_mean (torch.Tensor): [batch_size, time_steps, dim_z], Mean of the posterior distribution of `z_t`.
            z_log_var (torch.Tensor): [batch_size, time_steps, dim_z], Log variance of the posterior distribution of `z_t`.
        '''
        
        feed_dict, _ = inputs
        t_input = feed_dict['time_seq'] # [bs, times]
        bs, time_step = t_input.shape

        # Embed the input sequence, if needed
        emb_rnn_inputs = emb_inputs
        
        # Compute the output of the posterior network
        if self.transformer:
            output = self.infer_network_posterior_z(emb_rnn_inputs, None)
        else:
            output, _ = self.infer_network_posterior_z(emb_rnn_inputs, None)
        
        # Compute the mean and covariance matrix of the posterior distribution of `z_t`
        mean, log_var = self.infer_network_posterior_mean_var_z(output) # [bs, times, out_dim]

        log_var = torch.minimum(log_var, self.var_log_max.to(log_var.device))
        cov_mat = torch.diag_embed(torch.exp(log_var) + EPS) 
        dist_z = distributions.multivariate_normal.MultivariateNormal(
            loc=mean, 
            scale_tril=torch.tril(cov_mat)
        )
        
        # Sample the latent variable `z_t` using Monte Carlo estimation
        samples = dist_z.rsample((num_sample,))  # [num_sample, batch_size, time_step, out_dim]
        z_sampled = samples.transpose(1, 0).reshape(num_sample * bs, 1, time_step, -1).contiguous() 
        
        # Compute the entropy and log probability of the posterior distribution of `z_t`
        z_entropy = dist_z.entropy()  # [batch_size, time_step]
        z_log_prob_q = dist_z.log_prob(samples).mean(0)
        
        # Store the posterior mean, log variance, and output states
        z_posterior_states = output.reshape(bs, time_step, output.shape[-1])
        z_mean = mean
        z_log_var = cov_mat
        
        return z_sampled, z_entropy, z_log_prob_q, z_posterior_states, z_mean, z_log_var 
 
        # if self.dim_z > 1:
        #     # ipdb.set_trace()
        #     z_sampled_scalar = (z_sampled.unsqueeze(-2) * self.node_dist._get_node_embedding().to(z_sampled.device)).sum(-1) # [bsn, 1, time, num_node]
        #     z_sampled_scalar = z_sampled_scalar.permute(0,3,2,1).contiguous() # [bsn, num_node, time, dim_z_scalar]
            
            
    def calculate_likelihoods(
        self,
        feed_dict: Dict[str, torch.Tensor],
        sampled_s: torch.Tensor,
        sampled_z_set: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Calculates the log likelihood of the given inputs given the sampled s and z.

        Args:
            feed_dict (dict): A dictionary containing the input data, with keys 'label_seq' and 'skill_seq'.
            sampled_s (torch.Tensor): [batch_size * num_sample, num_node (1), time_steps, dim_s] The sampled s values.
            sampled_z_set (Tuple[torch.Tensor, torch.Tensor]): The sampled z values, which is a tuple containing two tensors:
                - sampled_z: [batch_size * num_sample, num_node (1), time_steps, dim_z_vector]
                - sampled_scalar_z: [batch_size * num_sample, num_node, time_steps, dim_z_scalar]
            temperature (float): The temperature for scaling the likelihoods.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The log likelihoods of the z values with shape [batch_size*num_sample, num_steps],
                - The log likelihoods of the s values with shape [batch_size*num_sample, num_steps],
                - The log likelihoods of the y values with shape [batch_size*num_sample, num_steps].
        """
        # Get the input data
        _, sampled_scalar_z = sampled_z_set
        bsn = sampled_scalar_z.shape[0]

        # Get log p(s[t] |s[t-1], x[t-1])
        log_prob_st = self.get_s_prior(sampled_s, feed_dict) # [bs*n, num_steps]
        
        # Get log p(z[t] | z[t-1], s[t])
        log_prob_zt = self.get_z_prior(sampled_z_set, sampled_s, feed_dict) # [bs*n, num_steps-1]
        
        # Get log p(y[t] | z[t])
        items = feed_dict['skill_seq'].unsqueeze(1).repeat(1, self.num_sample, 1).reshape(bsn, -1, 1)
        # items = torch.tile(feed_dict['skill_seq'], (self.num_sample,1)).unsqueeze(-1)
        sampled_scalar_z_item = torch.gather(sampled_scalar_z[..., 0], 1, items) # [bsn, time, 1]
        emission_prob = self.y_emit(sampled_scalar_z_item)
        emission_dist = torch.distributions.bernoulli.Bernoulli(probs=emission_prob)
        
        
        y_input = feed_dict['label_seq'].unsqueeze(1).repeat(1, self.num_sample, 1).reshape(bsn, -1, 1).float()
        # y_input = torch.tile(feed_dict['label_seq'][:, None, :, None], (1, self.num_sample, 1, 1)).float()
        # y_input = y_input.reshape(-1, y_input.shape[-2], y_input.shape[-1])
        # y_input = torch.tile(feed_dict['label_seq'].unsqueeze(-1), (self.num_sample, 1, 1)).float()
        log_prob_yt = emission_dist.log_prob(y_input) # [bsn, time, 1]
        log_prob_yt = log_prob_yt.squeeze(-1)

        return log_prob_yt, log_prob_zt, log_prob_st, emission_dist
    

    def predictive_model(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        single_step: bool = True, 
    ):
        
        with torch.no_grad():
            time_step = int(feed_dict['skill_seq'].shape[-1])
            train_step = int(time_step * self.args.train_time_ratio)
            test_step = int(time_step * self.args.test_time_ratio)
            val_step = time_step - train_step - test_step
            
            past_y = feed_dict['label_seq'][:, :train_step]
            past_t = feed_dict['time_seq'][:, :train_step]
            past_item = feed_dict['skill_seq'][:, :train_step]
            future_item = feed_dict['skill_seq'][:, train_step:]
            future_y = feed_dict['label_seq'][:, train_step:]
            
            bs = past_y.shape[0]
            num_sample = self.num_sample
            bsn = bs * num_sample
            
            # ------ inference based on past -----
            t_pe = self.get_time_embedding(past_t, 'absolute') # [bs, times, dim] 
            y_pe = torch.tile(past_y.unsqueeze(-1), (1,1, self.node_dim)) 
            node_pe = self.node_dist._get_node_embedding()[past_item]
            emb_input = torch.cat([node_pe, y_pe], dim=-1) # [bs, times, dim*4]
            emb_rnn_inputs, _ = self.infer_network_emb(emb_input, None) # [bs, times, dim*2(32)]
            emb_rnn_inputs = emb_rnn_inputs + t_pe
                
            if single_step:
                feed_dict_train = {
                    'time_seq': past_t,
                    'skill_seq': past_item,
                }
                s_sampled, _, _, _, s_mean, s_log_var = self.st_transition_infer(          
                    feed_dict_train, self.num_sample, emb_inputs=emb_rnn_inputs
                ) # s_sampled [bsn, 1, time, dim]
                z_sampled, _, _, _, z_mean, z_log_var  = self.zt_transition_infer(
                    [feed_dict_train, s_sampled],self.num_sample, emb_inputs=emb_rnn_inputs
                ) # z_sampled [bsn, 1, time, dim]
                z_sampled_scalar = (z_sampled.unsqueeze(-2) * self.node_dist._get_node_embedding().to(z_sampled.device)).sum(-1) # [bsn, 1, time, num_node]
                z_sampled_scalar = z_sampled_scalar.permute(0,3,2,1).contiguous() # [bsn, num_node, time, dim_z_scalar]
            
                # ----- generate based on inference -----
                last_s = s_sampled[:, :, -1:, :] # [bsn, 1, 1, dim]
                last_z = z_sampled_scalar[:, :, -1:, :] # [bsn, num_node, 1, dim]
                y_preds = []
                for i in range(time_step-train_step):
                    feed_dict_i = {
                        'label_seq': feed_dict['label_seq'][:, train_step-1+i:train_step+i],
                        'skill_seq': feed_dict['skill_seq'][:, train_step-1+i:train_step+i],
                        'time_seq': feed_dict['time_seq'][:, train_step-1+i:train_step+1+i],
                    }
                    s_dist = self.st_transition_gen(sampled_s=last_s, feed_dict=feed_dict_i, eval=True)
                    cur_s = s_dist.sample().unsqueeze(-2)
                    z_dist = self.zt_transition_gen(sampled_z_set=[None, last_z], sampled_s=cur_s, feed_dict=feed_dict_i, eval=True)
                    cur_z = z_dist.sample()
                    y_sampled = self.y_emit(cur_z)
                    last_s = cur_s
                    last_z = cur_z
                    y_preds.append(y_sampled)
            else: 
                s_sampled, _, _, _, s_mean, s_log_var = self.st_transition_infer(
                    feed_dict, 1, emb_inputs=emb_rnn_inputs
                )
                _, _, _, _, z_mean, z_log_var  = self.zt_transition_infer(
                    [feed_dict, s_sampled], num_sample=1, emb_inputs=emb_rnn_inputs
                )
                z_sampled_scalar = (z_mean.transpose(1,2) * self.node_dist._get_node_embedding().to(z_mean.device)).sum(-1) # [bs, time, num_node]
                
                # ----- generate based on inference -----
                last_label = feed_dict['label_seq'][:, train_step-1:train_step]
                last_s = s_mean[:, None, -1:, :]
                last_z = z_sampled_scalar[:, None, -1:, :]
                for i in range(time_step-train_step):
                    feed_dict_i = {
                        'label_seq': last_label,
                        'skill_seq': feed_dict['skill_seq'][:, train_step+i-1:train_step+i],
                    }
                    s_dist = self.st_transition_gen(sampled_s=last_s, feed_dict=feed_dict_i, eval=True)
                    z_dist = self.zt_transition_gen(sampled_z_set=[None, last_z], sampled_s=s_dist.sample(), inputs=feed_dict_i, eval=True)

        pred_all = torch.cat(y_preds,-2)[..., 0]
        future_item = future_item.unsqueeze(1).repeat(1, self.num_sample, 1).reshape(bsn, 1, -1)
        # future_item.unsqueeze(-2).repeat(self.num_sample,1,1) # [bsn, 1, future_time]
        pred_item = torch.gather(pred_all, 1, future_item).reshape(bs, self.num_sample, -1, 1) # [bs, 1, future_time, 1]
        label_item = future_y[:, None, :, None].repeat(1, self.num_sample, 1, 1)
        
        pred_dict = {
            'prediction': pred_item[:,:,-test_step:],
            'label': label_item[:,:,-test_step:],
            'prediction_val': pred_item[:,:,:val_step],
            'label_val': label_item[:,:,:val_step],
        }
    
        return pred_dict
    
    
    def forward(
        self, 
        feed_dict: Dict[str, torch.Tensor], 
    ):
        temperature = 0.5
        t_input = feed_dict['time_seq'] # [bs, times]
        y_input = feed_dict['label_seq']
        items = feed_dict['skill_seq']
        bs, num_steps = t_input.shape
        device = t_input.device
        
        self.s0_dist = distributions.MultivariateNormal(
            self.gen_s0_mean.to(device), 
            scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_s0_log_var.to(device)) + EPS))
        )
        self.z0_dist = distributions.MultivariateNormal(
            self.gen_z0_mean.to(device), 
            scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_z0_log_var.to(device)) + EPS))
        )
    
        # ----- embedding -----
        # It is possible that previous normalization makes float timestamp the same
        # which only have difference in Int!!
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506
        t_pe = self.get_time_embedding(t_input, 'absolute') # [bs, times, dim] 
        y_pe = torch.tile(y_input.unsqueeze(-1), (1,1, self.node_dim)) # + t_pe # TODO # [bs, times, dim]]   
        node_pe = self.node_dist._get_node_embedding()[items] # [bs, times, dim]      
        emb_input = torch.cat([node_pe, y_pe], dim=-1) # [bs, times, dim*4]
        emb_history = self.infer_network_emb(emb_input)
        # self.infer_network_emb.flatten_parameters()
        emb_history, _ = self.infer_network_emb(emb_input) # [bs, times, dim]
        emb_history = emb_history + t_pe

        # ipdb.set_trace()
        # ----- Sample continuous hidden variable from `q(s[1:T] | y[1:T])' -----
        s_sampled, s_entropy, s_log_prob_q, _, s_mean, s_log_var = self.st_transition_infer(
            feed_dict, self.num_sample, emb_inputs=emb_history
        )

        # ----- Sample continuous hidden variable from `q(z[1:T] | y[1:T])' -----
        z_sampled, z_entropy, z_log_prob_q, _, z_mean, z_log_var  = self.zt_transition_infer(
            [feed_dict, s_sampled], num_sample=self.num_sample, emb_inputs=emb_history
        ) 
        if self.dim_z > 1:
            # ipdb.set_trace()
            z_sampled_scalar = (z_sampled.unsqueeze(-2) * self.node_dist._get_node_embedding().to(z_sampled.device)).sum(-1) # [bsn, 1, time, num_node]
            z_sampled_scalar = z_sampled_scalar.permute(0,3,2,1).contiguous() # [bsn, num_node, time, dim_z_scalar]
            
        else:
            z_sampled_scalar = z_sampled
        
        # ----- joint log likelihood -----
        log_prob_yt, log_prob_zt, log_prob_st, emission_dist = self.calculate_likelihoods( 
            feed_dict, s_sampled, [z_sampled, z_sampled_scalar])
        recon_inputs = self.y_emit(z_sampled_scalar) # [bsn, num_node, time, dim_y]
        recon_inputs_items = emission_dist.sample()

        recon_inputs = recon_inputs.reshape(bs, self.num_sample, self.num_node, num_steps, -1)
        recon_inputs_items = recon_inputs_items.reshape(bs, self.num_sample, 1, num_steps, -1)
        z_sampled_scalar = z_sampled_scalar.reshape(bs, self.num_sample, self.num_node, num_steps, 1)
        s_sampled = s_sampled.reshape(bs, self.num_sample, 1, num_steps, self.dim_s)
        
        return_dict = self.get_objective_values(
            [log_prob_st, log_prob_zt, log_prob_yt], 
            [s_log_prob_q, z_log_prob_q],
            [s_entropy, z_entropy], 
            temperature=temperature,
        )

        return_dict["label"] = feed_dict['label_seq'][:, None, :, None]
        return_dict["prediction"] = recon_inputs_items   
        return_dict["sampled_s"] = s_sampled  
        
        self.register_buffer(name="output_mean_s", tensor=s_mean.clone().detach())
        self.register_buffer(name="output_var_s", tensor=s_log_var.clone().detach())
        self.register_buffer(name="output_mean_z", tensor=z_mean.clone().detach())
        self.register_buffer(name="output_var_z", tensor=z_log_var.clone().detach())
        self.register_buffer(name="output_emb_input", tensor=emb_input.clone().detach())
        self.register_buffer(name="output_sampled_z", tensor=z_sampled.clone().detach())
        self.register_buffer(name="output_sampled_z_scalar", tensor=z_sampled_scalar.clone().detach())
        self.register_buffer(name="output_sampled_y", tensor=recon_inputs.clone().detach())
        self.register_buffer(name="output_sampled_s", tensor=s_sampled.clone().detach())
        self.register_buffer(name="output_items", tensor=items.clone().detach())
        
        return_dict['log_prob_st'] = log_prob_st.mean()
        return_dict['log_prob_zt'] = log_prob_zt.mean()
        return_dict['log_prob_yt'] = log_prob_yt.mean()

        return return_dict
    
    
    def loss(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        outdict: Dict[str, torch.Tensor],
        metrics: List[str] = None
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
        gt = outdict["label"].repeat(1,self.num_sample,1,1) # .repeat(self.num_sample, 1) 
        pred = outdict["prediction"]
        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred.flatten(), gt.float().flatten())
        losses['loss_bce'] = bceloss
        
        for key in ['elbo', 'initial_likelihood', 'sequence_likelihood', 
                    'st_entropy', 'zt_entropy',
                    'log_prob_yt', 'log_prob_zt', 'log_prob_st']:
            losses[key] = outdict[key].mean()
        
        # Still NOT for optimization
        edge_log_probs = self.node_dist.edge_log_probs().to(pred.device)
        pred_att = torch.exp(edge_log_probs[0]).to(pred.device)
        pred_adj = torch.nn.functional.gumbel_softmax(edge_log_probs, hard=True, dim=0)[0].sum() * 1e-3

        losses['sparsity'] = (pred_att >= 0.5).sum()
        losses['loss_sparsity'] = pred_adj
        
        if 'junyi15' in self.args.dataset:
            gt_adj = self.adj.to(pred.device)
            losses['adj_0_att_1'] = (1 * (pred_att >= 0.5) * (1-gt_adj)).sum()
            losses['adj_1_att_0'] = (1 * (pred_att < 0.5) * gt_adj).sum()
            self.register_buffer(name="output_gt_graph_weights", tensor=gt_adj.clone().detach())
        
        losses['loss_total'] = -outdict['elbo'].mean() + losses['loss_spasity']
        
        # Register output predictions
        self.register_buffer(name="output_predictions", tensor=pred.clone().detach())
        self.register_buffer(name="output_gt", tensor=gt.clone().detach())
        self.register_buffer(name="output_attention_weights", tensor=pred_att.clone().detach())
        
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
        if self.dim_s == 4:
            losses['ou_gamma'] = outdict["sampled_s"][...,3].mean()
            
        return losses
    
    
    

    
    
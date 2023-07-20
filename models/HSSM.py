import sys
sys.path.append('..')

import math, os, argparse
import numpy
from enum import Enum
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import torch
from torch import nn, distributions
from torch.nn import functional as F

from models.modules import build_rnn_cell, build_dense_network
from models.modules import CausalTransformerModel, VAEEncoder
from models.hssm_graph_representation import VarTransformation, VarAttention
from models.gmvae import *
from utils.logger import Logger

from baseline.BaseModel import BaseModel, BaseLearnerModel

EPS = 1e-6
T_SCALE = 60


class HSSM(BaseModel):
    
    def __init__(
        self,
        mode: str = 'train',
        num_node: int = 1,
        num_seq: int = 1,
        nx_graph = None,
        device: torch.device = None,
        args: argparse.Namespace = None,
        logs = None,
    ):
        '''
        Args:
            mode: the training model. E.g., when mode=='ls_split_time', the model is trained with 
                    learner-specific parameters and the training data is split across time for each learner.
            num_node: the number of nodes in the graph.
            num_seq: the number of sequences in the dataset. This is an argument for the model when the mode
                        is 'synthetic' to generate synthetic data. 
            nx_graph: the graph adjacancy matrix. If mode=='synthetic', the graph is generated in advance. 
                        Otherwise, the ground-truth graph will be provided if the real-world dataset has one. 
        '''  
        self.num_node = num_node
        self.logs = logs
        self.device = device
        self.args = args
        self.num_seq = num_seq
        self.num_sample = args.num_sample
        self.node_dim = args.node_dim
        self.var_log_max = torch.tensor(args.var_log_max) 

        # Set the device to use for computations
        self.device = device if device != None else args.device

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
        x0_log_var = torch.ones((num_sample, dim)) * torch.log(torch.tensor(0.05))# nn.init.xavier_uniform_(torch.empty(num_sample, dim)) #, device=self.device))
        x0_log_var = nn.Parameter(x0_log_var, requires_grad=use_trainable_cov)
        
        return x0_mean, x0_log_var


    def get_objective_values(
        self,
        log_probs: List[torch.Tensor], 
        log_prob_q: torch.Tensor = None,
        posterior_entropies: List[torch.Tensor] = None,
        temperature: float = 1.0,
    ):
        temperature = 0.01
        w_s, w_z, w_y = 1.0, 1.0, 1.0
        [log_prob_st, log_prob_zt, log_prob_yt] = log_probs 
        
        # sequence_likelihood = log_prob_yt[:, 1:].mean(-1) # [bs,]
        # initial_likelihood = log_prob_yt[:, 0]
        sequence_likelihood = (w_s * log_prob_st[:, 1:] + w_z * log_prob_zt[:, 1:] + w_y * log_prob_yt[:, 1:]).mean(-1)/3 # [bs,]
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
        
        prior_distributions = self.st_transition_gen(sampled_s, inputs) # [bs, dim_s]
        
        # sampled_s0 = sampled_s[:, :, 0] # [bsn, 1, s_dim]
        # log_prob_s0 = self.s0_dist.log_prob(sampled_s0) # [bsn, 1]
        
        future_tensor = sampled_s # [bsn, 1,1,dim_s]
        if prior_distributions.mean.shape[0] != future_tensor.shape[0]:
            bs = prior_distributions.mean.shape[0]
            # future_tensor = future_tensor.reshape(bs, self.num_sample, -1, self.dim_s).transpose(0,1).contiguous()
            future_tensor = future_tensor.reshape(bs, self.num_sample, -1, self.dim_s).permute(1,2,0,3).contiguous()
            log_prob_st = prior_distributions.log_prob(future_tensor)
            # log_prob_st = log_prob_st.transpose(0,1).contiguous().reshape(bs*self.num_sample, -1)
            log_prob_st = log_prob_st.permute(2,0,1).contiguous().reshape(bs*self.num_sample, -1)
        else:
            log_prob_st = prior_distributions.log_prob(future_tensor)

        log_prob_s0 = torch.zeros((bs*self.num_sample,1)).to(log_prob_st.device)
        log_prob_st = torch.cat([log_prob_s0, log_prob_st], dim=-1) / self.dim_s

        self.register_buffer('output_s_prior_distributions_mean', prior_distributions.mean.clone().detach())
        self.register_buffer('output_s_prior_distributions_var', prior_distributions.variance.clone().detach())
          
        return log_prob_st
    
    
    def get_z_prior(
        self, 
        sampled_s: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
    ):
        """
        p(z[t] | z[t-1], s[t]) transition.
        z_sampled_scalar = z_sampled @ self.value_u.transpose(-1,-2)
        """
        
        prior_distributions = self.zt_transition_gen(sampled_s, inputs)
        
        self.register_buffer('output_z_prior_distributions_mean', prior_distributions.mean.clone().detach())
        self.register_buffer('output_z_prior_distributions_var', prior_distributions.variance.clone().detach())
        
        return prior_distributions
    
    
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
        self.node_dim = 8
        self.emb_mean_var_dim = 8
        self.num_node = num_node

        # ----- initialize graph parameters -----
        self.learned_graph = args.learned_graph
        if self.learned_graph == 'none' or self.num_node == 1:
            self.dim_s = 3
            self.dim_z = 1
        else: 
            self.dim_s = 4
            self.dim_z = 1# self.node_dim
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
        
        # --------------- for parameters Theta ---------------
        # the initial distribution p(s0) p(z0), the transition distribution p(s|s') p(z|s,z'), the emission distribution p(y|s,z)
        # ----- 1. initial distribution p(s0) p(z0): trainable mean and variance??? -----
        self.gen_s0_mean, self.gen_s0_log_var = self._initialize_normal_mean_log_var(self.dim_s, False)
        self.gen_z0_mean, self.gen_z0_log_var = self._initialize_normal_mean_log_var(self.dim_z, False) # self.z0_scale is std
        
        # ----- 2. transition distribution p(s|s') or p(s|s',y',c'); p(z|s,z') (OU) -----
        self.s_fit_gmvae = 1
        
        self.num_classes = 100
        self.gaussian_size = self.dim_s

        self.gen_network_transition_s = GenerativeNet(
            self.node_dim, self.dim_s, self.num_classes,
        )
        # ----- 3. emission distribution p(y|z) -----
        self.y_emit = torch.nn.Sigmoid() 
        
        
        # --------------- for parameters Phi ---------------
        # the embedding network at each time step emb_t = f(y_t, c_t, t)
        # the variational posterior distribution q(s_1:t | y_1:t, c_1:t) and q(z_1:t | y_1:t, c_1:t) TODO could add structure later q(z_1:t | y_1:t, s_1:t)
        # ----- 1. embedding network -----
        self.infer_network_emb = nn.LSTM( 
            input_size=self.node_dim*2, 
            hidden_size=self.node_dim, 
            bidirectional=False, # TODO why bidirectional
            batch_first=True,
        )
        time_step = 10
        # ----- 2. variational posterior distribution q(s_1:t | y_1:t, c_1:t) = q(s_1:t | emb_1:t) -----
        self.infer_network_posterior_s = InferenceNet(
            self.node_dim * time_step, self.dim_s, self.num_classes,
        )
        self.infer_network_posterior_mean_var_s = VAEEncoder(
            self.node_dim, self.emb_mean_var_dim, self.dim_s, tanh=False  
        )

        # self.st_transition_infer
        # 3. variational posterior distribution q(z_1:t | y_1:t, c_1:t)
        self.infer_network_posterior_z = nn.LSTM(
            input_size=self.node_dim, # self.infer_network_emb.hidden_size*2 if self.infer_network_emb.bidirectional else self.infer_network_emb.hidden_size,
            hidden_size=self.node_dim * 2,  
            bidirectional = False, 
            batch_first = True,
        )
        self.infer_network_posterior_mean_var_z = VAEEncoder(
            self.node_dim * 2, self.node_dim, self.num_node
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
        sampled_mean: torch.Tensor = None,
        sampled_log_var: torch.Tensor = None,
        s_latent=None, categ=None,
    ):
        '''
        Compute the transition function of the latent skills `s_t` in the model.

        Args:
            sampled_s (torch.Tensor): [batch_size * num_sample, num_node (1), 1, dim_s], Sampled latent skills `s_t` of shape.
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
        bs = items.shape[0]
        bsn = bs * self.num_sample
        
        time_emb = self.get_time_embedding(feed_dict['time_seq'], type='dt')
        value_u = self.node_dist._get_node_embedding().to(sampled_s.device)# .clone().detach()

        s_category = self.s_category

        if eval:
            s_last = sampled_s[:, 0, :1]
            item_last = items[:, :-1] # TODO
            y_last = input_y[:, :-1] # TODO
            # h0 = torch.cat([sampled_mean[:,-1], sampled_log_var[:,-1]], dim=-1).unsqueeze(0) 
        else:
            s_last = sampled_s[:, 0, :1] # [bsn, times-1, dim]
            item_last = items[:, :-1]
            y_last = input_y[:, :-1]
            h0 = torch.cat([self.gen_s0_mean, self.gen_s0_log_var], dim=-1).unsqueeze(1).repeat(1,bs,1)
            

        out_gen = self.gen_network_transition_s(s_category) 
        mean = out_gen['y_mean'] # [bs, dim_s]
        var = out_gen['y_var'] # [bs, dim_s]
        
        cov_mat = torch.diag_embed(var) + EPS
        
        s_prior_dist = distributions.multivariate_normal.MultivariateNormal(
                loc=mean, 
                scale_tril=torch.tril(cov_mat)
                )
            
        if not eval:
            self.register_buffer('output_prior_s_mean', mean.clone().detach())
            self.register_buffer('output_prior_s_var', cov_mat.clone().detach( ))
        
        return s_prior_dist
    
    
    def zt_transition_gen(
        self, 
        sampled_s: torch.Tensor,
        feed_dict: Dict[str, torch.Tensor],
        sampled_z: torch.Tensor = None,
        eval: bool = False,
    ):
        '''
        Compute the transition function of the scalar outcomes `z_t` in the model.

        Args:
            sampled_z_set (tuple): Tuple containing:
                - sampled_z (torch.Tensor): Sampled scalar outcomes `z_t` of shape [bsn, 1, time, dim_z]
                - sampled_scalar_z (torch.Tensor): Sampled scalar outcomes `z_t` of shape [bsn, num_node, time, dim_z (1)]
            sampled_s (torch.Tensor): Sampled latent skills `s_t` of shape [bsn, 1, 1, dim_s]
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
        dt = torch.diff(input_t.unsqueeze(1), dim=-1) /T_SCALE + EPS  # [bs, 1, num_steps-1] 
        dt = dt.unsqueeze(1).repeat(1, self.num_sample, 1, 1).reshape(bsn, 1, num_steps-1) # [bsn, 1, num_steps-1]
        
        z0 = self.z0_dist.rsample((self.num_sample,)).repeat(bs, self.num_node, 1).to(sampled_s.device) # [bsn, num_node, 1]
        z0_var = (torch.exp(self.gen_z0_log_var.to(dt.device)) + EPS).reshape(1, 1, 1, 1).repeat(bsn,1,1,1)
        sampled_st = sampled_s

        sampled_alpha = torch.relu(sampled_st[..., 0]) + EPS # *self.args.alpha_minimum # TODO # [bsn, 1, 1]
        decay = torch.exp(-sampled_alpha * dt)
        sampled_mean = sampled_st[..., 1]
        sampled_gamma = torch.sigmoid(sampled_st[..., 3])
        sampled_log_var = torch.minimum(sampled_st[..., 2], self.var_log_max.to(sampled_st.device)) # torch.exp(sampled_log_var) * decay + EPS # TODO not constrained
        sampled_var = torch.exp(sampled_log_var) * decay + EPS 
        
        # adj = self.node_dist.sample_A(self.num_sample)[-1][:,0].mean(0).to(sampled_s.device) # [n, num_node, num_node] # adj_ij means i has influence on j
        adj =  torch.exp(self.node_dist.edge_log_probs()[0]).to(sampled_s.device) # adj_ij means i has influence on j
        
        # ----- Simulate the path of `z_t` -----
        z_preds = [z0]
        for i in range(0, dt.shape[-1]):
            empower =  torch.einsum('bin,ij->bjn', z_preds[-1], adj) / self.num_node * sampled_gamma 
            
            stable = sampled_mean 
            tmp_mean_level = (empower + stable) / 2
            
            z_mean = z_preds[-1] * decay[..., i:i+1]  + tmp_mean_level * (1 - decay[..., i:i+1])
            z_preds.append(z_mean)

        z_var = torch.cat([z0_var, sampled_var.reshape(bsn, 1, num_steps-1, 1)], dim=-2)
        z_mean = torch.stack(z_preds, -2)
        
        z_prior_dist = distributions.multivariate_normal.MultivariateNormal(
            loc=z_mean, 
            scale_tril=torch.tril(torch.diag_embed(z_var))
        )
        
        self.register_buffer('output_prior_z_decay', decay.clone().detach())
        self.register_buffer('output_prior_z_empower', empower.clone().detach())
        self.register_buffer('output_prior_z_tmp_mean_level', ((sampled_mean + empower)/2).clone().detach())
        
        self.register_buffer(name="output_prior_mean_z", tensor=z_mean.clone().detach())
        self.register_buffer(name="output_prior_var_z", tensor=z_var.clone().detach())
            
        
        return z_prior_dist 
        
        
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
        bs, time_step, _ = emb_inputs.shape # train: [bs, time (10), dim_emb]
        bsn = bs * self.num_sample
        
        temperature = 1.0
        hard = 1

        out_inf = self.infer_network_posterior_s(emb_inputs, temperature, hard) 

        _, s_category = out_inf['gaussian'], out_inf['categorical'] # s_latent not time-dependent?
        self.s_catoegory = s_category # [bs, num_cat]
        self.logits = out_inf['logits'] # [bs, num_cat]
        self.prob_cat = out_inf['prob_cat'] # [bs, num_cat]
        
        self.register_buffer('s_category', s_category.clone().detach())

        mean = out_inf['mean'] # [bs, dim_s]
        var = out_inf['var'] # [bs, dim_s]
        cov_mat = torch.diag_embed(var + EPS) 

        dist_s = distributions.multivariate_normal.MultivariateNormal(
            loc=mean, 
            scale_tril=torch.tril(cov_mat)
            )
        
        samples = dist_s.rsample((num_sample,)) # [n, bs, 1, dim_s] 
        s_sampled = samples.transpose(1,0).reshape(bsn, 1, 1, self.dim_s) 
        s_entropy = dist_s.entropy() # [bs, times]
        s_log_prob_q = dist_s.log_prob(samples).mean(0)
        
        s_posterior_states = None  
        s_mus = mean
        s_log_var = torch.log(var)

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
        bs, time_step, _ = emb_inputs.shape
        
        # Compute the output of the posterior network
        self.infer_network_posterior_z.flatten_parameters()
        output, _ = self.infer_network_posterior_z(emb_inputs, None)
        
        # Compute the mean and covariance matrix of the posterior distribution of `z_t`
        mean, log_var = self.infer_network_posterior_mean_var_z(output) # [bs, times, out_dim]

        # ipdb.set_trace()
        log_var = torch.minimum(log_var, self.var_log_max.to(log_var.device))
        cov_mat = torch.diag_embed(torch.exp(log_var) + EPS) 
        dist_z = distributions.multivariate_normal.MultivariateNormal(
            loc=mean, 
            scale_tril=torch.tril(cov_mat)
        )
        
        # Sample the latent variable `z_t` using Monte Carlo estimation
        samples = dist_z.rsample((self.num_sample,))  # [num_sample, batch_size, time_step, out_dim]
        z_sampled = samples.transpose(1, 0).reshape(self.num_sample * bs, 1, time_step, -1).contiguous() 
        
        # Compute the entropy and log probability of the posterior distribution of `z_t`
        z_entropy = dist_z.entropy()  # [batch_size, time_step]
        z_log_prob_q = dist_z.log_prob(samples).mean(0)
        
        # Store the posterior mean, log variance, and output states
        z_posterior_states = None # output.reshape(bs, time_step, output.shape[-1])
        z_mean = mean
        z_log_var = cov_mat
        
        return z_sampled, z_entropy, z_log_prob_q, z_posterior_states, z_mean, z_log_var 
    
    
    def forward(
        self, 
        feed_dict: Dict[str, torch.Tensor], 
    ):
        time_step = int(feed_dict['skill_seq'].shape[-1])
        train_step = int(time_step * self.args.train_time_ratio)
        test_step = int(time_step * self.args.test_time_ratio)
        val_step = time_step - train_step - test_step

        t_train = feed_dict['time_seq'][:, :train_step] # [bs, times]
        y_train = feed_dict['label_seq'][:, :train_step]
        item_train = feed_dict['skill_seq'][:, :train_step]

        y_val = feed_dict['label_seq'][:, train_step:train_step+val_step]
        item_train_val = feed_dict['skill_seq'][:, :train_step+val_step]
        
        bs, _ = t_train.shape
        device = t_train.device
        
        self.z0_dist = distributions.MultivariateNormal(
            self.gen_z0_mean.to(device), 
            scale_tril=torch.tril(torch.diag_embed(torch.exp(self.gen_z0_log_var.to(device)) + EPS))
        )
        
        t_pe = self.get_time_embedding(t_train, 'absolute') # [bs, times, dim] 
        y_pe = torch.tile(y_train.unsqueeze(-1), (1,1, self.node_dim)) # + t_pe # TODO # [bs, times, dim]]   
        node_pe = self.node_dist._get_node_embedding()[item_train] # [bs, times, dim]      
        emb_input = torch.cat([node_pe, y_pe], dim=-1) # [bs, times, dim*4]
        self.infer_network_emb.flatten_parameters()
        emb_history, _ = self.infer_network_emb(emb_input) # [bs, trian_t, dim]
        emb_history = emb_history + t_pe

        # ----- Sample continuous hidden variable from `q(s[1:T] | y[1:T])' -----
        s_sampled, s_entropy, s_log_prob_q, _, s_mean, s_log_var = self.st_transition_infer(
            feed_dict, self.num_sample, emb_inputs=emb_history
        )

        # ----- joint log likelihood -----
        log_prob_st = self.get_s_prior(s_sampled, feed_dict) # [bs*n, num_steps]
        
        # Get log p(z[t] | z[t-1], s[t])
        feed_dict_train_val = {
            'time_seq': feed_dict['time_seq'][:, :train_step+val_step],
            'skill_seq': feed_dict['skill_seq'][:, :train_step+val_step],
            'label_seq': feed_dict['label_seq'][:, :train_step+val_step],
        }
        prior_z_dist = self.get_z_prior(s_sampled, feed_dict_train_val) # [bs*n, num_steps-1]
        log_prob_zt = torch.zeros_like(log_prob_st)
        
        # Get log p(y[t] | z[t])
        sampled_scalar_z = prior_z_dist.rsample()
        bsn = sampled_scalar_z.shape[0]
        items = item_train_val.unsqueeze(1).repeat(1, self.num_sample, 1).reshape(bsn, 1, -1) # [bsn, 1, time]
        sampled_scalar_z_item = torch.gather(sampled_scalar_z[..., 0], 1, items).transpose(-1,-2).contiguous() # [bsn, time, 1]
        
        y_prob_train = self.y_emit(sampled_scalar_z_item[:, :train_step])
        y_dist_train = torch.distributions.bernoulli.Bernoulli(probs=y_prob_train)
        y_train_mc = y_train.unsqueeze(1).repeat(1, self.num_sample, 1).reshape(bsn, -1, 1).float()
        log_prob_yt = y_dist_train.log_prob(y_train_mc) # [bsn, time, 1]
        log_prob_yt = log_prob_yt.squeeze(-1)

        y_prob_val = self.y_emit(sampled_scalar_z_item[:, train_step:])
        y_dist_val = torch.distributions.bernoulli.Bernoulli(probs=y_prob_val)
        y_val_mc_sample = y_dist_val.sample() # ??
        
        recon_val_inputs_items = y_val_mc_sample.reshape(bs, self.num_sample, 1, val_step, -1)

        recon_inputs = self.y_emit(sampled_scalar_z) # [bsn, num_node, time, dim_y]
        recon_inputs = recon_inputs.reshape(bs, self.num_sample, self.num_node, train_step+val_step, -1)
        recon_inputs_items = y_prob_train.reshape(bs, self.num_sample, 1, train_step, -1)

        z_sampled_scalar = sampled_scalar_z.reshape(bs, self.num_sample, self.num_node, train_step+val_step, 1)
        s_sampled = s_sampled.reshape(bs, self.num_sample, 1, 1, self.dim_s)
        
        return_dict = self.get_objective_values(
            [log_prob_st, log_prob_zt, log_prob_yt], 
            [s_log_prob_q, None],
            [s_entropy, torch.zeros_like(s_entropy)], 
        )

        return_dict["label"] = y_train[:, None, :, None]
        return_dict["prediction"] = recon_inputs_items   
        return_dict["sampled_s"] = s_sampled  
        return_dict['val_label'] = y_val[:, None, :, None]
        return_dict['val_prediction'] = recon_val_inputs_items
        
        self.register_buffer(name="output_mean_s", tensor=s_mean.clone().detach())
        self.register_buffer(name="output_var_s", tensor=s_log_var.clone().detach())
        self.register_buffer(name="output_emb_input", tensor=emb_input.clone().detach())
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
        val_gt = outdict["val_label"].repeat(1,self.num_sample,1,1) # .repeat(self.num_sample, 1) 
        val_pred = outdict["val_prediction"]
        
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
        pred_adj = torch.nn.functional.gumbel_softmax(edge_log_probs, hard=True, dim=0)[0].sum() * 1e-6

        losses['sparsity'] = (pred_att >= 0.5).sum()
        losses['loss_sparsity'] = pred_adj
        
        if 'junyi15' in self.args.dataset:
            gt_adj = self.adj.to(pred.device)
            losses['adj_0_att_1'] = (1 * (pred_att >= 0.5) * (1-gt_adj)).sum()
            losses['adj_1_att_0'] = (1 * (pred_att < 0.5) * gt_adj).sum()
            self.register_buffer(name="output_gt_graph_weights", tensor=gt_adj.clone().detach())
        
        gmvae_loss = LossFunctions()
        loss_cat = -gmvae_loss.entropy(self.logits, self.prob_cat) - numpy.log(0.1)
        losses['loss_cat'] = loss_cat
        
        loss_cat_in_entropy = gmvae_loss.prior_entropy(self.num_classes, self.gen_network_transition_s, self.device)
        losses['loss_cat_in_entropy'] = loss_cat_in_entropy
        
        losses['loss_total'] = -outdict['elbo'].mean() + losses['loss_cat'] + \
                                + losses['loss_cat_in_entropy'] + losses['loss_sparsity']
        
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
            val_pred = val_pred.detach().cpu().data.numpy()
            val_gt = val_gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(val_pred, val_gt, metrics)
            for key in evaluations.keys():
                losses['val_'+key] = evaluations[key]
            
        # Calculate mean and variance of the Ornstein-Uhlenbeck process
        losses['ou_speed'] = outdict["sampled_s"][...,0].mean()
        losses['ou_mean'] = outdict["sampled_s"][...,1].mean()
        losses['ou_vola'] = outdict["sampled_s"][...,2].mean()
        if self.dim_s == 4:
            losses['ou_gamma'] = outdict["sampled_s"][...,3].mean()
            
        return losses
    
    
    

    
    
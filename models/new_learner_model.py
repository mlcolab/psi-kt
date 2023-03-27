# -*- coding: UTF-8 -*-
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Amortized Variational Inference for Hierachical Non-linear SSM.

1. an inference network, with Bidirectional-RNN for input embedding, and a
   forward RNN to get the posterior distribution of `q(z[1:T] | y[1:T])`.
2. a continuous state transition network, `p(z[t] | z[t-1], s[t])`.
3. a continuous state transition network that conditioned on the input,
   `p(s[t] | s[t-1], y[t-1])`.
4. an emission network conditioned on the continuous hidden dynamics,
   `p(y[t] | z[t])`.

"""

import sys
sys.path.append('..')


import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import distributions

import math
from torch.nn import init
from functools import partial
from torch.nn.modules.utils import _pair

import collections
namedtuple = collections.namedtuple
RANDOM_SEED = 131
from collections import defaultdict
import ipdb

from models.learner_model import BaseLearnerModel
from utils.utils import ConfigDict
from models.BaseModel import BaseModel

torch.autograd.set_detect_anomaly(True)




class ContinuousStateTransition(nn.Module):
    def __init__(self,
                transition_mean_networks,
                distribution_dim,
                transition_cov_networks=None,
                config=None,
                device='cpu',):
        """Construct a `ContinuousStateTransition` instance.

        Args:
        transition_mean_networks: a list of `callable` networks,  
                                    Each one of the networks will take previous step hidden state, `z[t-1]`, 
                                    and returns the mean of transition distribution, `p(z[t] | z[t-1], s[t]=i)` 
                                    for each discrete state `i`.
        distribution_dim: an `int` scalar for dimension of continuous hidden states, `z`.
        cov_mat:          an optional `float` Tensor for predefined covariance matrix. Default to `None`, in which c
                            ase, a `cov` variable will be created.
                                Otherwise, a diagonal covariance matrices and `tfp.distributions.MultivariateNormalDiag` will be used.
        use_trainable_cov:  a `bool` scalar indicating whether the scale of the distribution is trainable. 
                                Default to False.
        raw_sigma_bias:     a `float` scalar to be added to the raw sigma, which is standard deviation of the 
                                distribution. Default to `0.`.
        sigma_min:          a `float` scalar for minimal level of sigma to prevent underflow. Default to `1e-5`.
        sigma_scale:        a `float` scalar for scaling the sigma. Default to `0.05`. The above three arguments are used as
                                `sigma_scale * max(softmax(raw_sigma + raw_sigma_bias), sigma_min))`.
        """
        super(ContinuousStateTransition, self).__init__()

        cov_mat=config['cov_mat'],
        use_trainable_cov=config['use_trainable_cov'],
        raw_sigma_bias=config['raw_sigma_bias'],
        sigma_min=config['sigma_min'],
        sigma_scale=config['sigma_scale'],
        
        self.latent_trans_mean = transition_mean_networks
        self.latent_trans_cov = transition_cov_networks
        
        self.dist_dim = distribution_dim
        self.device = device
        self.use_trainable_cov = use_trainable_cov[0]

        if cov_mat[0]:
            self.cov_mat = cov_mat
        else:
            m = torch.empty(1, int(self.dist_dim * (self.dist_dim + 1) / 2), device=device)
            m = torch.nn.init.uniform_(m)[0]
            cov_mat = torch.zeros((self.dist_dim, self.dist_dim), device=device)
            tril_indices = torch.tril_indices(row=self.dist_dim, col=self.dist_dim, offset=0)
            cov_mat[tril_indices[0], tril_indices[1]] += m
            cov_mat = Parameter(cov_mat, requires_grad=use_trainable_cov[0])
            
            if self.dist_dim == 1:
                self.cov_mat = (torch.maximum((cov_mat + raw_sigma_bias[0]), 
                                    torch.tensor(sigma_min[0])) * sigma_scale[0])
            else: 
                self.cov_mat = (torch.maximum(F.softmax(cov_mat + raw_sigma_bias[0], dim=-1),
                                    torch.tensor(sigma_min[0])) * sigma_scale[0])
        
    def forward(self, inputs, additions=None, dynamic_model='Gaussian'):
        '''
        Args:
            inputs: [bs, num_steps, dist_dim]
            additions: 
            dynamic_model: ['Gaussian', 'nonlinear_input', 'OU']
        '''
        
        bs, num_steps, dist_dim = inputs.shape
        eps = 1e-6
        if 'OU' in dynamic_model:
            assert(additions != None)
            sampled_s, time_seq = additions
            time_diff = torch.diff(time_seq, dim=1)/60/60/24 + eps # [bs*n, num_steps, 1]
            sampled_s = sampled_s[:, 1:] if num_steps > 1 else sampled_s# [bs*n, num_steps, 1]
            sampled_alpha = torch.relu(sampled_s[..., 0:1]) + eps 
            sampled_mean = sampled_s[..., 1:2]
            sampled_vola = torch.relu(sampled_s[..., 2:3]) + eps
          
            mean_tensor = inputs * torch.exp(- sampled_alpha * time_diff) + \
                                (1.0 - torch.exp(- sampled_alpha * time_diff)) * sampled_mean
            self.mean = mean_tensor
                                
            if 'vola' in dynamic_model:
                # if the mean_tensor has shape [bs, time_steps], variance matrix has shape [bs, t, t]
                # given data with shape [bs, t], it will only output a log likelihood with shape [bs]
                # by averaging out the time steps.
                # sampled_vola = sampled_vola.tile((1,1,num_steps))
                # mask = torch.eye(num_steps, device=self.device).unsqueeze(0)
                # self.cov_mat = sampled_vola * mask
                self.cov_mat = sampled_vola[..., None]
                output_dist = distributions.multivariate_normal.MultivariateNormal(
                                        loc = mean_tensor, scale_tril=torch.tril(self.cov_mat))
            else: 
                output_dist = distributions.multivariate_normal.MultivariateNormal(
                    loc = mean_tensor, scale_tril=torch.tril(self.cov_mat))
            return output_dist

        elif dynamic_model == 'nonlinear_input':  
            # TODO should it combine with input_y???
            eps = 1e-6
            
            input_y = additions[0][:, :-1] if additions[0].shape[-2] > 1 else additions[0]
            input_y = input_y.float()
            input_y = torch.tile(input_y.float(), (1,1,3))
            
            # mean_tensor = self.latent_trans_mean(input_y) 
            # ipdb.set_trace()
            input_sy = torch.cat([inputs, input_y], -1)
            mean_tensor = self.latent_trans_mean(input_sy) 
            self.mean = mean_tensor
            
            # ipdb.set_trace()
            if self.use_trainable_cov:
                cov_tensor = self.latent_trans_cov(input_sy)
                self.cov_mat = torch.diag_embed(torch.relu(cov_tensor) + 1e-6) 

        output_dist = distributions.multivariate_normal.MultivariateNormal(
                loc = mean_tensor, scale_tril=torch.tril(self.cov_mat))    

        return output_dist

    @property
    def output_event_dims(self):
        return self.dist_dim


class GaussianDistributionFromMean(nn.Module):
    """
    """
    def __init__(self,
                mean_network,
                distribution_dim,
                cov_network=None,
                config=None,
                device='cpu'):
        """
        Construct a `GaussianDistributionFromMean` instance.
        Args:
        mean_network: a `callable` network taking continuous hidden states, `z[t]`, and returning the
                                mean of emission distribution, `p(y[t] | z[t])`.
                                
        distribution_dim:    an `int` scalar for dimension of observations, `x`.
        """
        super(GaussianDistributionFromMean, self).__init__()
        cov_mat=config['cov_mat'],
        use_trainable_cov=config['use_trainable_cov'],
        raw_sigma_bias=config['raw_sigma_bias'],
        sigma_min=config['sigma_min'],
        sigma_scale=config['sigma_scale'],
        
        self.dist_dim = distribution_dim
        self.mean_net = mean_network
        self.cov_net = cov_network
        self.cov_mat = cov_mat[0]
        self.use_trainable_cov = use_trainable_cov[0]
        
        if cov_mat[0]:
            self.cov_mat = cov_mat
        else:
            m = torch.empty(1, int(self.dist_dim * (self.dist_dim + 1) / 2), device=device)
            m = torch.nn.init.uniform_(m)[0]
            cov_mat = torch.zeros((self.dist_dim, self.dist_dim), device=device)
            tril_indices = torch.tril_indices(row=self.dist_dim, col=self.dist_dim, offset=0)
            cov_mat[tril_indices[0], tril_indices[1]] += m
            cov_mat = Parameter(cov_mat, requires_grad=use_trainable_cov[0])
            
            if self.dist_dim == 1:
                self.cov_mat = (torch.maximum((cov_mat + raw_sigma_bias[0]), 
                                    torch.tensor(sigma_min[0])) * sigma_scale[0])
            else: 
                self.cov_mat = (torch.maximum(F.softmax(cov_mat + raw_sigma_bias[0], dim=-1),
                                    torch.tensor(sigma_min[0])) * sigma_scale[0])

            
    def forward(self, inputs):
        eps = 1e-6
        
        mean_tensor = self.mean_net(inputs) # [bs*n, y_emission_net.out_dim]
        self.mean = mean_tensor
        
        if self.use_trainable_cov:
            cov_tensor = self.cov_net(inputs)
            self.cov_mat = torch.diag_embed(cov_tensor + eps) 
        
        output_dist = distributions.multivariate_normal.MultivariateNormal(
            loc=mean_tensor, scale_tril=torch.tril(self.cov_mat)
        )
        return output_dist

    @property
    def output_event_dims(self):
        return self.dist_dim


class NonlinearEmissionDistribution(nn.Module):
    """
    """
    def __init__(self,
                emission_mean_network,
                observation_dim,
                config=None,
                device='cpu'):
        """
        Args:
        emission_network: a `callable` network taking continuous hidden states, `z[t]`, and returning the
                                mean of emission distribution, `p(y[t] | z[t])`.
                                
        observation_dim:    an `int` scalar for dimension of observations, `y`.
        """
        super(NonlinearEmissionDistribution, self).__init__()

        cov_mat=config['cov_mat'],
        use_trainable_cov=config['use_trainable_cov'],
        raw_sigma_bias=config['raw_sigma_bias'],
        sigma_min=config['sigma_min'],
        sigma_scale=config['sigma_scale'],
        
        self.ob_dim = observation_dim
        self.y_emission_net = emission_mean_network
        
        if cov_mat[0]:
            self.cov_mat = cov_mat
        else:
            m = torch.empty(1, int(self.ob_dim * (self.ob_dim + 1) / 2), device=device)
            m = torch.nn.init.uniform_(m)[0]
            cov_mat = torch.zeros((self.ob_dim, self.ob_dim), device=device)
            tril_indices = torch.tril_indices(row=self.ob_dim, col=self.ob_dim, offset=0)
            cov_mat[tril_indices[0], tril_indices[1]] += m
            cov_mat = Parameter(cov_mat, requires_grad=use_trainable_cov[0])
            
            if self.ob_dim == 1:
                self.cov_mat = (torch.maximum((cov_mat + raw_sigma_bias[0]), 
                                    torch.tensor(sigma_min[0])) * sigma_scale[0])
            else: 
                self.cov_mat = (torch.maximum(F.softmax(cov_mat + raw_sigma_bias[0], dim=-1),
                                    torch.tensor(sigma_min[0])) * sigma_scale[0])


    def forward(self, inputs, nonlinear='Bernoulli'):
        # TODO
        # mean_tensor = self.y_emission_net(inputs) # [bs*n, y_emission_net.out_dim]
        mean_tensor = torch.sigmoid(inputs)
        self.mean = mean_tensor
        
        if nonlinear == 'Bernoulli':
            output_dist = distributions.bernoulli.Bernoulli(probs=self.mean)
        # check https://github.com/pytorch/pytorch/issues/7857
        # https://pytorch.org/docs/stable/distributions.html#logitrelaxedbernoulli

        elif nonlinear == 'Gaussian': 
            output_dist = distributions.multivariate_normal.MultivariateNormal(
                loc = mean_tensor, scale_tril=torch.tril(self.cov_mat)
            )

        return output_dist

    @property
    def output_event_dims(self):
        return self.ob_dim
    
    
class RnnInferenceNetwork(nn.Module):
    """
    Inference network for posterior q(z[1:T] | x[1:T]) or q(s[1:T] | x[1:T]). 
    """

    def __init__(self,
                posterior_rnn,
                posterior_dist,
                latent_dim,
                embedding_network=None,
                device='cpu'):
        """
        Construct a `RnnInferenceNetwork` instance.

        Args:
        posterior_rnn:     a RNN cell, `h[t]=f_RNN(h[t-1], z[t-1], input[t])`, which recursively takes previous 
                            step RNN states `h`, previous step sampled dynamical state `z[t-1]`, and conditioned 
                            input `input[t]`.
        posterior_dist:    a distribution instance for `p(z[t] | h[t])`, where h[t] is the output of `posterior_rnn`.
        embedding_network: an optional network to embed the observations, `y[t]`.
                            Default to `None`, in which case, no embedding is applied.
                            
        latent_dim: an `int` scalar for dimension of continuous hidden states.

        """
        super(RnnInferenceNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.posterior_rnn = posterior_rnn
        self.posterior_dist = posterior_dist
    
        self.device = device

        if embedding_network is None:
            self.embedding_network = lambda x: x
        self.embedding_network = embedding_network


    def forward(
            self,
            inputs,
            num_samples=20,
            random_seed=RANDOM_SEED,
        ):
        """
        Recursively sample z[t] ~ q(z[t]|h[t]=f_RNN(h[t-1], z[t-1], h[t]^b)).

        Args:
        inputs:              a float `Tensor` of size [batch_size, num_steps, obs_dim], where each observation 
                                should be flattened.
        num_samples:         an `int` scalar for number of samples per time-step, for posterior inference networks, 
                                `z[i] ~ q(z[1:T] | x[1:T])`.
        random_seed:         an `Int` as the seed for random number generator.
        parallel_iterations: a positive `Int` indicates the number of iterations
            allowed to run in parallel in `torch.while_loop`, where `torch.while_loop`
            defaults it to be 10.

        Returns:
        sampled_z: a float 3-D `Tensor` of size [num_samples, batch_size,
        num_steps, latent_dim], which stores the z_t sampled from posterior.
        entropies: a float 2-D `Tensor` of size [num_samples, batch_size,
        num_steps], which stores the entropies of posterior distributions.
        log_probs: a float 2-D `Tensor` of size [num_samples. batch_size,
        num_steps], which stores the log posterior probabilities.
        """
        # ipdb.set_trace()
        bs, num_steps, _ = inputs[0].shape
        
        # TODO how to normalize the time
        t_inputs = (inputs[1]-inputs[1][:,0:1])/(inputs[1][:,-1:]-inputs[1][:,0:1])
        mc_t_inputs = torch.tile(t_inputs, (num_samples, 1,1)).float()
        mc_y_inputs = torch.tile(inputs[0], (num_samples, 1,1)).float() # [bs, times, 1]
        mc_yt_inputs = torch.cat([mc_y_inputs, mc_t_inputs], -1) # [bs, times, 1]
        
        latent_dim = self.latent_dim
        
        mc_yt_rnn_inputs, _ = self.embedding_network(mc_yt_inputs)
        emb_dim_yt = mc_yt_rnn_inputs.shape[-1]
        # -> the same for z and s now

        ## passing through forward RNN
        if isinstance(self.posterior_rnn, nn.LSTMCell): 
            prev_rnn_state = [torch.zeros((bs * num_samples, self.posterior_rnn.hidden_size), 
                                         device=self.device)] * 2
        else:   
            prev_rnn_state = [torch.zeros((bs * num_samples, self.posterior_rnn.hidden_size), 
                                         device=self.device)]
            
        if latent_dim == 3:
            prev_latent_state = torch.zeros((bs * num_samples, latent_dim*2),  device=self.device)
        if latent_dim == 1:
            prev_latent_state = torch.zeros((bs * num_samples, latent_dim*8),  device=self.device)
        # prev_latent_state = torch.zeros((bs * num_samples, emb_dim_yt), device=self.device)
    
        latent_states = []
        rnn_states = []
        entropies = []
        log_probs = []
        for t in range(num_steps):
            # Duplicate current observation to sample multiple trajectories.
            current_input = mc_yt_rnn_inputs[:, t] # [bs*n, 1]
            rnn_input = torch.concat([current_input, prev_latent_state], dim=-1)  # [bs*n latent_dim+input_dim], 
            
            rnn_out, rnn_state = self.posterior_rnn(rnn_input, prev_rnn_state) 
                # rnn_out [bs*n, rnn_hid_dim]
                # rnn_state [bs*n, rnn_hid_dim]
            dist = self.posterior_dist(rnn_out)
            
            # # 1. dist.sample() no gradient!!!
            # latent_state = dist.sample() 
            # # 2. what if using reparam trick? -> it has gradient
            # std = torch.sqrt(torch.diagonal(self.posterior_dist.cov_mat))
            # eps = torch.randn_like(std) 
            # latent_state = self.posterior_dist.mean + eps*std 
            # # 3. rsample() it also has gradient! 
            latent_state = dist.rsample() 
            
            latent_states.append(latent_state)
            rnn_states.append(rnn_state)
            log_probs.append(dist.log_prob(latent_state)) 
            entropies.append(dist.entropy()) 
            # TODO no gradient now -> because the entropy of a multivariate Gaussian only depends on the 
            # variance matrix; while the variance matrix now is not trainable -> even if it is trainable, 
            # the variance is not depends on the input `rnn_out` (it is only a trainable parameter). 
            # So the gradient will not flow back to the posterior estimator rnn
            
            if latent_dim == 3:
                prev_latent_state = torch.tile(latent_state, (1,2))
            else:
                prev_latent_state = torch.tile(latent_state, (1,8))
            prev_rnn_state = [rnn_out, rnn_state]
            # TODO is it reasonable???
            # prev_latent_state = latent_state.detach()
            # prev_rnn_state = [rnn_out.detach(), rnn_state.detach()]
            
        # TODO check when multiple samples 
        sampled_s = torch.stack(latent_states, -2).reshape(num_samples, bs, num_steps, latent_dim) 
        entropies = torch.stack(entropies, -1).reshape(num_samples, bs, -1)
        log_probs = torch.stack(log_probs, -1).reshape(num_samples, bs, -1)
        rnn_states = torch.stack(rnn_states, -1).reshape(num_samples, bs, num_steps, self.posterior_rnn.hidden_size)

        return sampled_s, entropies, log_probs, rnn_states




"""Hierachical non-linear dynamical systems."""

class HierachicalSSM(BaseLearnerModel):
    """Hierachical NonLinear Dynamical Systems base model.

    This provides the implementation of the core algorithm for
    collapsed variational inference in hierachical SSM. The configurable
    subnetworks include the continuous dynamic transition network
    p(z[t] | z[t-1], ...), discrete state transition network
    p(s[t] | s[t-1], ...), emission network p(x[t] | z[t]), and inference
    network p(z[1:T] | x[1:T]) etc. 
    For more details, please check the `__init__` function.

    References:
        [1] Dong, Zhe and Seybold, Bryan A. and Murphy, Kevin P., and Bui,
            Hung H.. Collapsed Amortized Variational Inference for Switching
            Nonlinear Dynamical Systems. 2019. https://arxiv.org/abs/1910.09588.
    """

    def __init__(
        self,
        z_transition_network,
        s_transition_network,
        emission_network,
        inference_network,
        # initial_distribution,
        num_samples=1,
        args=None,
        device='cpu',
        logs=None,
    ):
        """Constructor of Switching Non-Linear Dynamical System.

        The model framework, as described in Dong et al. (2019)[1].

        Args:
        continuous_transition_network:  a `callable` with its `call` function taking batched sequences of continuous 
                                            hidden states, `z[t-1]`, with shape [batch_size, num_steps, hidden_states], 
                                            and returning a distribution with its `log_prob` function implemented. 
                                            The `log_prob` function takes continuous hidden states, `z[t]`, and 
                                            returns their likelihood, `p(z[t] | z[t-1], s[t])`.
        discrete_transition_network:    a `callable` with its `call` function taking batch conditional inputs, 
                                            `x[t-1]`, and returning the discrete state transition matrices, 
                                            `log p(s[t] |s[t-1], x[t-1])`.
        emission_network:               a `callable` with its `call` function taking continuous hidden states, 
                                            `z[t]`, and returning a distribution, `p(x[t] | z[t])`. 
                                            The distribution should have `mean` and `sample` function, 
                                            similar as the classes in `tfp.distributions`.
        inference_network:              inference network should be a class that has `sample` function, which takes 
                                            input observations, `x[1:T]`, and outputs the sampled hidden states 
                                            sequence of `q(z[1:T] | x[1:T])` and the entropy of the distribution.
        initial_distribution:           a initial state distribution for continuous variables, `p(z[0])`.
        discrete_state_prior:           a `float` Tensor, indicating the prior of discrete state distribution, 
                                            `p[k] = p(s[t]=k)`. This is used by cross entropy regularizer, which 
                                            tries to minize the difference between discrete_state_prior and the 
                                            smoothed likelihood of the discrete states, `p(s[t] | x[1:T], z[1:T])`.
                                            
        continuous_state_dim: number of continuous hidden units, `z[t]`.

        Reference:
        [1] Dong, Zhe and Seybold, Bryan A. and Murphy, Kevin P., and Bui,
            Hung H.. Collapsed Amortized Variational Inference for Switching
            Nonlinear Dynamical Systems. 2019. https://arxiv.org/abs/1910.09588.
        """
        super().__init__(mode='train', device=device, logs=logs)

        self.z_tran = z_transition_network
        self.s_tran = s_transition_network
        self.y_emit = emission_network
        
        self.s_infer, self.z_infer = inference_network
        
        # self.s0_dist, self.z0_dist = initial_distribution
        self.s0_mean, self.s0_scale = self._construct_initial_mean_cov(3, False)
        self.z0_mean, self.z0_scale = self._construct_initial_mean_cov(1, False)

        # self.discrete_prior = self.s0_dist
        self.observation_dim = self.y_emit.output_event_dims
        
        self.num_samples = num_samples
        self.logs = logs
        self.device = device
        self.args = args

        
    def _construct_initial_mean_cov(self, dim, use_trainable_cov):
        x0_mean = torch.empty(1, dim, device=self.device)
        x0_mean = torch.nn.init.xavier_uniform_(x0_mean)[0]
        x0_mean = Parameter(x0_mean, requires_grad=True)
        
        m = torch.empty(int(dim * (dim + 1) / 2), 1, device=self.device)
        m = torch.nn.init.xavier_uniform_(m)
        x0_scale = torch.zeros((dim, dim), device=self.device)
        tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)
        x0_scale[tril_indices[0], tril_indices[1]] += m[:, 0]
        x0_scale = Parameter(x0_scale, requires_grad=use_trainable_cov)
        
        return x0_mean, x0_scale
        
    def _construct_initial_state_distribution(
        self,
        latent_dim,
        z0_mean=None,
        z0_scale=None,
        config=None,
        device='cpu'):
        """
        Construct the initial state distribution, `p(s_0) or p(z_0)`.
        Args:
            latent_dim:  an `int` scalar for dimension of continuous hidden states, `z`.
            num_categ:   an `int` scalar for number of discrete states, `s`.
            
            use_trainable_cov:  a `bool` scalar indicating whether the scale of `p(z[0])` is trainable. Default to False.
            raw_sigma_bias:     a `float` scalar to be added to the raw sigma, which is standard deviation of the 
                                    distribution. Default to `0.`.
            sigma_min:          a `float` scalar for minimal level of sigma to prevent underflow. Default to `1e-5`.
            sigma_scale:        a `float` scalar for scaling the sigma. Default to `0.05`. The above three arguments 
                                    are used as `sigma_scale * max(softmax(raw_sigma + raw_sigma_bias), sigma_min))`.
                                    
            dtype: data type for variables within the scope. Default to `torch.float32`.
            name: a `str` to construct names of variables.

        Returns:
            return_dist: a `tfp.distributions` instance for the initial state
            distribution, `p(z[0])`.
        """
        use_trainable_cov = False
        raw_sigma_bias = 0.0
        sigma_min = 1e-5
        sigma_scale = 0.05
        
        if z0_mean == None:
            z0_mean = torch.empty(1, latent_dim, device=self.device)
            z0_mean = torch.nn.init.xavier_uniform_(z0_mean)[0]
            z0_mean = Parameter(z0_mean, requires_grad=True)

        if z0_scale == None:
            m = torch.empty(int(latent_dim * (latent_dim + 1) / 2), 1, device=self.device)
            m = torch.nn.init.xavier_uniform_(m)
            z0_scale = torch.zeros((latent_dim, latent_dim), device=self.device)
            tril_indices = torch.tril_indices(row=latent_dim, col=latent_dim, offset=0)

            z0_scale[tril_indices[0], tril_indices[1]] += m[:, 0]
            z0_scale = Parameter(z0_scale, requires_grad=use_trainable_cov)
        
        if latent_dim == 1:
            z0_scale = (torch.maximum((z0_scale + raw_sigma_bias), # TODO is this correct?
                                torch.tensor(sigma_min)) * sigma_scale)
            dist = distributions.multivariate_normal.MultivariateNormal(
                loc = z0_mean, covariance_matrix=z0_scale)
        else: 
            z0_scale = (torch.maximum(F.softmax(z0_scale + raw_sigma_bias, dim=-1),
                                torch.tensor(sigma_min)) * sigma_scale)
            dist = distributions.multivariate_normal.MultivariateNormal(
                loc = z0_mean, scale_tril=torch.tril(z0_scale)
                )

        return dist
    
    
    # TODO
    def _get_iwae(self, sequence_likelihood, initial_likelihood, log_prob_q,
                    num_samples):
        r"""Computes the IWAE bound given the pre-computed log-probabilities.

        The IWAE Bound is given by:
        E_{z^i~q(z^i|x)}[ log 1/k \sum_i \frac{p(x, z^i)}{q(z^i | x)} ]
        where z^i and x are complete trajectories z_{1:T}^i and x_{1:T}. The {1:T}
        is omitted for simplicity of notation.

        log p(x, z) is given by E_s[log p(s, x, z)]

        Args:
        sequence_likelihood: E_s[log p(s[1:T], v[1:T] | s[0], v[0])],
            size [num_samples * batch_size]
        initial_likelihood: E_s[log p(s[0], v[0])],
            size [num_samples * batch_size]
        log_prob_q: log q(z[t]| x[1:T], z[1:t-1]),
            size [num_samples * batch_size, T]
        num_samples: number of samples per trajectory.

        Returns:
        torch.Tensor, the estimated IWAE bound.
        """
        log_likelihood = sequence_likelihood + initial_likelihood
        log_surrogate_posterior = torch.reduce_sum(log_prob_q, axis=-1)

        # Reshape likelihoods to [num_samples, batch_size]
        log_likelihood = torch.reshape(log_likelihood, [num_samples, -1])
        log_surrogate_posterior = torch.reshape(log_surrogate_posterior,
                                            [num_samples, -1])

        iwae_bound = torch.reduce_logsumexp(
            log_likelihood - log_surrogate_posterior,
            axis=0) - torch.math.log(torch.cast(num_samples, torch.float32))
        iwae_bound_mean = torch.reduce_mean(iwae_bound)
        return iwae_bound_mean


    def get_objective_values(self,
                            log_probs, 
                            #log_init, log_gamma1, log_gamma2,
                            log_prob_q,
                            posterior_entropies,
                            num_samples):
        """Given all precalculated probabilities, return ELBO.
        Args:
            log_a: [bs, times]
        """
        # All the sequences should be of the shape
        # [batch_size, num_steps, (data_dim)]
        [log_prob_st, log_prob_zt, log_prob_yt] = log_probs 
        
        sequence_likelihood = (log_prob_st[:, 1:] + log_prob_zt[:, 1:] + log_prob_yt[:, 1:]).mean(-1) # [bs,]
        initial_likelihood = log_prob_st[:, 0] + log_prob_zt[:, 0] + log_prob_yt[:, 0]

        t1_mean = torch.mean(sequence_likelihood, dim=0)
        t2_mean = torch.mean(initial_likelihood, dim=0)
        
        t3 = torch.sum(posterior_entropies[0], dim=-1) # no gradient now  TODO
        t3_mean = torch.mean(t3, dim=0)
        
        t4 = torch.sum(posterior_entropies[1], dim=-1) # no gradient now  TODO
        t4_mean = torch.mean(t4, dim=0)
        
        elbo = t1_mean + t2_mean + t3_mean + t4_mean
        
        iwae = None # TODO ???
        # iwae = self._get_iwae(sequence_likelihood, initial_likelihood, log_prob_q,
        #                     num_samples)
        return dict(
            elbo=elbo,
            iwae=iwae,
            initial_likelihood=t2_mean,
            sequence_likelihood=t1_mean,
            st_entropy=t3_mean,
            zt_entropy=t4_mean)
        

    def calculate_likelihoods(self,
                                inputs,
                                sampled_s,
                                sampled_z,
                                switching_conditional_inputs=None,
                                temperature=1.0):
        """
        Calculate the probability by p network, `p_theta(x,z,s)`.

        Args:
        inputs:      a float 3-D `Tensor` of shape [batch_size, num_steps, obs_dim], containing the 
                        observation time series of the model.
        sampled_s:   same as sampled_z
        sampled_z:   a float 3-D `Tensor` of shape [batch_size, num_steps, latent_dim] for continuous 
                        hidden states, which are sampled from inference networks, `q(z[1:T] | x[1:T])`.
        temperature: a float scalar `Tensor`, indicates the temperature for transition probability, 
                        `p(s[t] | s[t-1], x[t-1])`.
        switching_conditional_inputs: a float 3-D `Tensor` of shape [batch_size, num_steps, encoded_dim], 
                                        which is the conditional input for discrete state transition 
                                        probability, `p(s[t] | s[t-1], x[t-1])`.
                                        Default to `None`, when `inputs` will be used.
            
        Returns:
        log_xt_zt: a float `Tensor` of size [batch_size, num_steps, num_categ]
            indicates the distribution, `log(p(x_t | z_t) p(z_t | z_t-1, s_t))`.
        reconstruced_inputs: a float `Tensor` of size [batch_size, num_steps,
            obs_dim] for reconstructed inputs.
        """
        # ipdb.set_trace()
        y_input = inputs['label_seq'] 
        bs, num_steps = y_input.shape
        num_sample = int(sampled_s.shape[0] / bs)
        
        y_input = torch.tile(inputs['label_seq'].unsqueeze(-1), (num_sample, 1, 1))
        time_input = torch.tile(inputs['time_seq'].unsqueeze(-1), (num_sample, 1, 1))

        ########################################
        ## getting log p(z[t] | z[t-1], s[t])
        ########################################

        # Broadcasting rules of TFP dictate that: if the samples_z0 of dimension
        # [batch_size, 1, event_size], z0_dist is of [num_categ, event_size].
        # z0_dist.log_prob(samples_z0[:, None, :]) is of [batch_size, num_categ].
        self.z0_dist = self._construct_initial_state_distribution(
            latent_dim=self.z0_mean.shape[0], 
            z0_mean=self.z0_mean, 
            z0_scale=self.z0_scale
        )
        sampled_z0 = sampled_z[:, 0, :]
        log_prob_z0 = self.z0_dist.log_prob(sampled_z0[:, None, :]) # [bs, 1]
        
        # `log_prob_zt` should be of the shape [batch_size, num_steps, self.z_dim]
        log_prob_zt = self.get_z_prior(sampled_z, [sampled_s, time_input], log_prob_z0) # [bs*n, num_steps]


        ########################################
        ## getting log p(s[t] |s[t-1], x[t-1])
        ########################################
        self.s0_dist = self._construct_initial_state_distribution(
            latent_dim=self.s0_mean.shape[0], 
            z0_mean=self.s0_mean, 
            z0_scale=self.s0_scale
        )
        if switching_conditional_inputs is None:
            switching_conditional_inputs = y_input
        sampled_s0 = sampled_s[:, 0, :]
        
        log_prob_s0 = self.s0_dist.log_prob(sampled_s0[:, None, :]) # [bs, 1]
            
        log_prob_st = self.get_s_prior(sampled_s, [switching_conditional_inputs], log_prob_s0) # [bs*n, num_steps]
        
        # log_prob_st_stm1 = torch.reshape(
        #     self.s_tran(switching_conditional_inputs[:, :-1, :]),
        #     [batch_size, num_steps-1, self.num_categ, self.num_categ])
        # # by normalizing the 3rd axis (axis=-2), we restrict A[:,:,i,j] to be
        # # transiting from s[t-1]=j -> s[t]=i
        # log_prob_st_stm1 = utils.normalize_logprob(
        #     log_prob_st_stm1, axis=-2, temperature=temperature)[0]

        # log_prob_st_stm1 = torch.concat(
        #     [torch.eye(self.num_categ, self.num_categ, batch_shape=[batch_size, 1],
        #             dtype=torch.float32, name="concat_likelihoods"),
        #     log_prob_st_stm1], axis=1)
        
        
        ########################################
        ## getting log p(x[t] | z[t])
        ########################################
        emission_dist = self.y_emit(sampled_z)

        # `emission_dist' should have the same event shape as `inputs',
        # by broadcasting rule, the `log_prob_xt' should be of the shape
        # [batch_size, num_steps],
        log_prob_yt = emission_dist.log_prob(y_input.float())[..., 0] # [bs, t]

        # log ( p(x_t | z_t) p(z_t | z_t-1, s_t) )
        # log_xt_zt = log_prob_yt + log_prob_zt
        return log_prob_yt, log_prob_zt, log_prob_st


    def get_reconstruction(self,
                            hidden_state_sequence,
                            observation_shape=None,
                            sample_for_reconstruction=True,
                            sample_hard=False,
                            num_samples=1): # TODO
        """Generate reconstructed inputs from emission distribution, `p(x[t]|z[t])`.

        Args:
        hidden_state_sequence: a `float` `Tensor` of the shape [batch_size, num_steps, hidden_dims], 
                                containing batched continuous hidden variable `z[t]`.
        observation_shape: the shape of sampled `y[t]` to reshape reconstructed inputs. Default to `None`, 
                            in which case the output of `mean` or `sample` function for emission distribution 
                            will be returned directly, without reshape.
        sample_for_reconstruction: a `bool` scalar. When `True`, it will will use `emission_distribution.sample()` 
                                    to generate reconstructions.

        Returns:
        reconstructed_obs: a `float` `Tensor` of the shape [batch_size, num_steps,
            observation_dims], containing reconstructed observations.
        """
        eps = 1e-6
        emission_dist = self.y_emit(hidden_state_sequence)

        if sample_for_reconstruction:
            # reconstructed_obs = emission_dist.sample() # NOTE: no gradient!
            probs = torch.cat([1-emission_dist.mean, emission_dist.mean], dim=-1)
            reconstructed_obs = F.gumbel_softmax(torch.log(probs + eps), tau=1, hard=sample_hard, eps=1e-10, dim=-1)
            reconstructed_obs = reconstructed_obs[..., 1:]
        else:
            reconstructed_obs = emission_dist.mean

        if observation_shape is not None:
            reconstructed_obs = torch.reshape(reconstructed_obs, observation_shape)

        return reconstructed_obs


    def get_s_prior(self, sampled_s, additional_input, log_prob_s0):
        """
        p(s[t] | s[t-1]) transition.
        """ 
        
        prior_distributions = self.s_tran(sampled_s[:, :-1, :], additional_input, 'nonlinear_input')

        future_tensor = sampled_s[:, 1:, :]
        log_prob_st = prior_distributions.log_prob(future_tensor)

        log_prob_st = torch.concat([log_prob_s0, log_prob_st], dim=-1)
        return log_prob_st
    
    
    def get_z_prior(self, sampled_z, additional_input, log_prob_z0):
        """
        p(z[t] | z[t-1], s[t]) transition.
        """
        prior_distributions = self.z_tran(sampled_z[:, :-1, :], additional_input, 'OU_vola') # or 'OU'

        future_tensor = sampled_z[:, 1:]
        log_prob_zt = prior_distributions.log_prob(future_tensor) # [bs*n, times-1]
        
        log_prob_zt = torch.concat([log_prob_z0, log_prob_zt], dim=-1)
        return log_prob_zt
    

    def forward(self, inputs, temperature=1.0, num_samples=50):
        """
        Inference call of SNLDS.

        Args:
        inputs:      a `float` Tensor of shape `[batch_size, num_steps, event_size]`, containing the observation time series of the model.
        temperature: a `float` Scalar for temperature used to estimate discrete
                        state transition `p(s[t] | s[t-1], y[t-1])` as described in Dong et al.
                        (2019). Increasing temperature increase the uncertainty about each
                        discrete states.
                        Default to 1. For ''temperature annealing'', the temperature is set
                        to large value initially, and decay to a smaller one. A temperature
                        should be positive, but could be smaller than `1.`.
        num_samples: an `int` scalar for number of samples per time-step, for
                        posterior inference networks, `z[i] ~ q(z[1:T] | x[1:T])`.

        Returns:
        return_dict: a python `dict` contains all the `Tensor`s for inference results. Including the following keys:
            elbo:   Evidence Lower Bound
            iwae:   IWAE Bound
            initial_likelihood:  the likelihood of `p(s[0], z[0], x[0])`.
            sequence_likelihood: the likelihood of `p(s[1:T], z[1:T], x[0:T])`
            zt_entropy:     the entropy of posterior distribution `H(q(z[t] | x[1:T])`
            reconstruction: the reconstructed inputs, returned by `get_reconstruction` function.
            posterior_llk:  the posterior likelihood, `p(s[t] | x[1:T], z[1:T])`, returned by `forward_backward_algo.forward_backward` function.
            sampled_z:      the sampled z[1:T] from the approximate posterior.
            cross_entropy:  batched cross entropy between discrete state posterior likelihood and its prior distribution.
        """
        eps = 1e-6
        
        input_y = inputs['label_seq'].unsqueeze(-1)  # [bs, times] -> [bs, times, 1]
        input_t = inputs['time_seq'].unsqueeze(-1)
        # ipdb.set_trace()
        
        # ----- Sample continuous hidden variable from `q(s[1:T] | y[1:T])' -----
        s_sampled, s_entropy, s_log_prob_q, _ = self.s_infer(
            [input_y, input_t], num_samples=num_samples
        )
        
        _, bs, num_steps, s_dim = s_sampled.shape

        s_sampled = torch.reshape(
            s_sampled, [num_samples * bs, num_steps, s_dim]
        )
        s_entropy = torch.reshape(s_entropy, [num_samples * bs, num_steps])
        s_log_prob_q = torch.reshape(s_log_prob_q, [num_samples * bs, num_steps])

        
        # ----- Sample continuous hidden variable from `q(z[1:T] | y[1:T])' -----
        z_sampled, z_entropy, z_log_prob_q, _ = self.z_infer(
            [input_y, input_t], num_samples=num_samples)
        
        _, _, _, z_dim = z_sampled.shape

        z_sampled = torch.reshape(z_sampled,
                            [num_samples * bs, num_steps, z_dim])
        z_entropy = torch.reshape(z_entropy, [num_samples * bs, num_steps])
        z_log_prob_q = torch.reshape(z_log_prob_q, [num_samples * bs, num_steps])

        # ----- joint log likelihood -----
        log_prob_yt, log_prob_zt, log_prob_st = self.calculate_likelihoods( 
            inputs, s_sampled, z_sampled, temperature=temperature)
        
        recon_inputs = self.get_reconstruction(
            z_sampled,
            observation_shape=z_sampled.shape,
            sample_for_reconstruction=True, # TODO
        )

        # Calculate Evidence Lower Bound with components.
        # The return_dict currently support the following items:
        #   elbo: Evidence Lower Bound.
        #   iwae: IWAE Lower Bound.
        #   initial_likelihood: likelihood of p(s[0], z[0], x[0]).
        #   sequence_likelihood: likelihood of p(s[1:T], z[1:T], x[0:T]).
        #   zt_entropy: the entropy of posterior distribution.
        return_dict = self.get_objective_values([log_prob_st, log_prob_zt, log_prob_yt], 
                                                [s_log_prob_q, z_log_prob_q],
                                                [s_entropy, z_entropy], num_samples)

        # TODO do we need this consistency?
        # # Estimate the cross entropy between state prior and posterior likelihoods.
        # state_crossentropy = utils.get_posterior_crossentropy(
        #     log_gamma1,
        #     prior_probs=self.discrete_prior)
        # state_crossentropy = torch.reduce_mean(state_crossentropy, axis=0)

        recon_inputs = torch.reshape(recon_inputs, [num_samples, bs, num_steps, -1])
        z_sampled = torch.reshape(z_sampled,
                            [num_samples, bs, num_steps, z_dim])
        s_sampled = torch.reshape(s_sampled,
                            [num_samples, bs, num_steps, s_dim])

        return_dict["label"] = input_y
        return_dict["prediction"] = recon_inputs
        return_dict["sampled_z"] = z_sampled[0]
        return_dict["sampled_s"] = s_sampled[0]
        # return_dict["cross_entropy"] = state_crossentropy
        # ipdb.set_trace()
        return return_dict
    
    
    def inference_model(self):
        pass
    
    
    def generative_model(self, past_states, future_timestamps, steps, num_samples=1):
        '''
        Args:
            past_states:
            
        '''
        outputs = []
        s_last, z_last, y_last = past_states
        for t in range(0, steps):
            s_next = self.s_tran(s_last, [y_last], 'nonlinear_input').mean # or sample  # [bs, 1, 3]
            z_next = self.z_tran(z_last, [s_next, future_timestamps[:, t:t+2]], 'OU_vola').mean
            
            y_next = self.get_reconstruction(
                z_next,
                observation_shape=z_next.shape,
                sample_for_reconstruction=True,
                sample_hard=True,
            )

            y_last = y_next
            z_last = z_next
            s_last = s_next
            
            outputs.append((s_next, z_next, y_next))

        pred_y = torch.cat([outputs[i][2] for i in range(len(outputs))], 1) # [n, pred_step, 1]
        pred_z = torch.cat([outputs[i][1] for i in range(len(outputs))], 1)
        pred_s = torch.cat([outputs[i][0] for i in range(len(outputs))], 1)
        # ipdb.set_trace()
        return [pred_s, pred_z, pred_y]
    
    
    def predictive_model(self, inputs, num_samples=50):
        # TODO it depends on the training mode, for now, it is only for splitting time 
        '''
        p(s_t+1, z_t+1 | y_1:t)
        Args:
            inputs: 
            num_samples
        '''
        time_step = int(inputs['skill_seq'].shape[-1])
        train_step = int(time_step * self.args.train_time_ratio)
        test_step = int(time_step * self.args.test_time_ratio)
        
        # ipdb.set_trace()
        past_y = inputs['label_seq'][:, :train_step].unsqueeze(-1)
        past_timestamps = inputs['time_seq'][:, :train_step].unsqueeze(-1)
        future_timestamps = inputs['time_seq'][:, train_step-1:].unsqueeze(-1)
        future_timestamps = torch.tile(future_timestamps, (num_samples,1,1))
        
        # TODO which way to sample? from q(z) or from q(s)? or prior distribution?
        # ideally they should be the same, but currently because the 
        s_sampled, _, _, _ = self.s_infer([past_y, past_timestamps], num_samples)
        z_sampled, _, _, _ = self.z_infer([past_y, past_timestamps], num_samples)
        
        
        s_last = s_sampled[:,:,-1:].reshape(-1, 1,3) # [n_samples, train_total, train_time, 3] -> [total, 1. 3]
        z_last = z_sampled[:,:,-1:].reshape(-1, 1,1)
        y_last = torch.tile(past_y[:, -1:], (num_samples,1,1))
        
        # s_last = s_sampled[:,:,-2:-1].reshape(-1, 1,3)
        # y_last = torch.tile(past_y[:, -2:-1], (num_samples,1,1))
        
        # TODO for the s_tran, it needs the input of label p(s_t | s_t-1, y_t-1)
        # two ways: 
        # 1. we make the x_t-1 as hat(x_t-1) which is the prediction -> easier to compare with others, but will the gradient be accumulated?
        # 2. use the ground truth x_t-1 when it is observed -> it is online learning, but it makes difficult to compare with other methods  

        pred_s, pred_z, pred_y = self.generative_model([s_last, z_last, y_last], future_timestamps, 
                                  self.args.max_step-train_step, )
        # ipdb.set_trace()
        pred_dict = {
            'prediction': pred_y[:, -test_step:], 
            'label': inputs['label_seq'][:, -test_step:],
            'pred_y': pred_y, 
            'pred_z': pred_z,
            'pred_s': pred_s
        }
        
        return pred_dict
    
    
    def loss(self, feed_dict, outdict, metrics=None):
        losses = defaultdict(lambda: torch.zeros((), device=self.device))
        
        gt = outdict["label"] 
        pred = outdict["prediction"]
        num_sample = pred.shape[0]
        gt = torch.tile(gt[None, ...], (num_sample,1,1,1))
        
        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred, gt.float())
        losses['loss_bce'] = bceloss

        elbo = outdict['elbo']
        losses['loss_total'] = -elbo + bceloss
        losses['loss_elbo'] = elbo
        
        if metrics != None:
            pred = pred.detach().cpu().data.numpy()
            gt = gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
            
        # losses['prediction'] = recon_inputs
        # losses['label'] = input_y
        return losses




############################## create base models start ##############################

def get_default_distribution_config():
    config = ConfigDict()
    config.cov_mat = None
    config.use_trainable_cov = False
    config.raw_sigma_bias = 0.0
    config.sigma_min = 1e-5
    config.sigma_scale = 0.05
    return config
def build_dense_network(
        input_size,
        layer_sizes,
        layer_activations,
    ):
    """
    Helper function for building a multi-layer network.
    """
    modules = []
    for lsize, activation in zip(layer_sizes, layer_activations):
        modules.append(nn.Linear(input_size, lsize)) 
        if activation != None:
            modules.append(activation)
        input_size = lsize
    nets = nn.Sequential(*modules)
    return nets
def build_rnn_cell(rnn_type, hidden_dim_rnn, rnn_input_dim):
    """
    Helper function for building RNN cells.
    """
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        rnn_cell = nn.GRUCell(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim_rnn) 
    elif rnn_type == "lstm":
        rnn_cell = nn.LSTMCell(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim_rnn)
    elif rnn_type == "simplernn":
        rnn_cell = nn.RNNCell(  # what is difference of simplernn in tf vs. rnn in torch???
            input_size=rnn_input_dim,
            hidden_size=hidden_dim_rnn)
    return rnn_cell
    
    
def create_model(dim_s,
                 dim_z,
                 dim_y,

                 config_emission=get_default_distribution_config(),
                 config_inference=get_default_distribution_config(),
                 config_s_initial=get_default_distribution_config(),
                 config_s_transition=get_default_distribution_config(),
                 config_z_initial=get_default_distribution_config(),
                 config_z_transition=get_default_distribution_config(),
                 
                 hidden_dim_rnn=8, # args TODO
                 args=None, # TODO figure out *args means
                 device='cpu',
                 logs=None,
        ):
    """
    Construct SNLDS model.
    Args:
        dim_s:     an `int` scalar for dimension of continuous hidden states, `s`.
        dim_z:     an `int` scalar for dimension of continuous hidden states, `z`.
        dim_y:  an `int` scalar for dimension of observations, `y`.
        
        config_emission:     a `dict` for configuring emission distribution,
                                `p(x[t] | z[t])`.
        config_inference:    a `dict` for configuring the posterior distribution,
                                `q(z[t]|h[t]=f_RNN(h[t-1], z[t-1], h[t]^b))`.
        config_z_initial:    a `dict` for configuring the initial distribution of continuous hidden state, 
                                `p(z[0])`.
        config_z_transition: a `dict` for configuring the transition distribution
                                `p(z[t] | z[t-1], s[t])`.
                                
        network_emission:        a `callable` network taking continuous hidden states, `z[t]`, 
                                    and returning the mean of emission distribution, `p(y[t] | z[t])`.
        network_input_embedding: a `callable` network to embed the observations,
                                    `y[t]`. E.g. a bidirectional RNN to embedding `y[1:T]`.
        network_posterior_rnn:   a RNN cell, `h[t]=f_RNN(h[t-1], z[t-1], input[t])`, which recursively takes 
                                    previous step RNN states `h`, previous step sampled dynamical state `z[t-1]`, 
                                    and conditioned input `input[t]`.
        network_s_transition:    a `callable` network taking batch conditional inputs, `x[t-1]`, and returning 
                                    the discrete state transition matrices, `log p(s[t] |s[t-1], x[t-1])`.
        networks_z_transition:   a list of `callable` networks, 
                                    Each one of the networks will take previous step hidden state, `z[t-1]`, and 
                                    returns the mean of transition distribution, `p(z[t] | z[t-1], s[t]=i)` for 
                                    each discrete state `i`.
        network_posterior_mlp:   an optional network to embedding the output of inference RNN networks, before 
                                    passing into the distribution as mean, `q(z[t] | mlp( h[t] ))`. Default to 
                                    identity mapping.
                                    
        name: a `str` to construct names of variables.

    Returns:
        An instance of instantiated `model_base.HierachicalSSM` model.
    """


    ############################## For latent states S ##############################
    # -- initialization p(s_0 ; s_0_mean, s_0_var) -- 
    # initial_distribution_s = construct_initial_state_distribution(
    #     latent_dim=dim_s,
    #     config=config_s_initial,
    #     device=device,
    # )
    # -- transition p(s_t | s_t-1 ; s_t_mean, s_t_var) -- 
    network_transition_s = build_dense_network(
            dim_y *dim_s + dim_s,
            [4 * dim_s, dim_s ], # TODO
            [nn.ReLU(), None]
    )
    transition_s = ContinuousStateTransition(
        transition_mean_networks=network_transition_s,
        distribution_dim=dim_s,
        config=config_s_transition,
        device=device,
    )
    
    
    ############################## For latent states Z ##############################
    # -- initialization p(z_0 ; z_0_mean, z_0_var) -- 
    # initial_distribution_z = construct_initial_state_distribution(
    #     latent_dim=1,
    #     config=config_z_initial,
    #     device=device,
    # )
    # -- transition p(z_t | z_t-1, s_t ; z_t_var) -- only variance 
    networks_transition_z = build_dense_network(
            dim_y, 
            [3*dim_z, dim_z],  # TODO
            [nn.ReLU(), None]
    )
    transition_z = ContinuousStateTransition(
        transition_mean_networks=networks_transition_z,
        distribution_dim=1,
        config=config_z_transition, 
        device=device,
    )
    
    
    ############################## For observations Y ##############################
    # -- emission p(y_t | z_t ; ??? ) -- 
    # network_emission = build_dense_network(
    #     dim_z, 
    #     [4 * dim_y, dim_y],
    #     [nn.ReLU(), nn.Sigmoid()]
    # )
    network_emission = build_dense_network(
        dim_z, 
        [dim_y],
        [nn.Sigmoid()]
    )
    emission_network = NonlinearEmissionDistribution(
        emission_mean_network=network_emission,
        observation_dim=dim_y,
        config=config_emission, 
        device=device,
    )

    ############################## For posterior Z^hat or S^hat ##############################
    # -- posterior p(???) -- 
    # network_input_embedding = lambda x: x
    # network_input_embedding = build_dense_network(
    #         dim_y * 2,
    #         [dim_y * 2 * 4, dim_y * 2 * 8], 
    #         [nn.ReLU, None]
    # )
    network_input_embedding = nn.RNN(
        input_size=dim_y * 2, # input_y + input_time 2
        hidden_size=dim_y * 2 * 2,  # 4
        bidirectional = True,
        batch_first = True,
    )
    
    emb_dim_yt = dim_y * 2 * 2 * 2 # 8 because it is bidirectional
    emb_mean_dim_s = emb_dim_yt
    emb_cov_dim_s = emb_dim_yt
    hidden_dim_rnn = emb_dim_yt * 2 # 16
    
    network_posterior_mean_mlp_s = build_dense_network(
            hidden_dim_rnn,
            [emb_mean_dim_s, dim_s], 
            [nn.ReLU(), None]) # is it not allowed to add non-linear functions here?
    network_posterior_cov_mlp_s = build_dense_network(
            hidden_dim_rnn,
            [emb_cov_dim_s, dim_s], 
            [nn.ReLU(), nn.ReLU()]) # is it not allowed to add non-linear functions here?
    posterior_distribution_s = GaussianDistributionFromMean(
        mean_network=network_posterior_mean_mlp_s,
        cov_network=network_posterior_cov_mlp_s,
        distribution_dim=dim_s,
        config=config_inference,        
        device=device,
    )
    network_posterior_rnn_s = build_rnn_cell(
        rnn_type="lstm", hidden_dim_rnn=hidden_dim_rnn, rnn_input_dim=emb_dim_yt+dim_s*2
    )
    posterior_network_s = RnnInferenceNetwork(
        posterior_rnn=network_posterior_rnn_s,
        posterior_dist=posterior_distribution_s,
        latent_dim=dim_s,
        embedding_network=network_input_embedding,
        device=device)
    
    
    network_posterior_mean_mlp_z = build_dense_network(
            hidden_dim_rnn,
            [emb_mean_dim_s, dim_z], 
            [nn.ReLU(), None])
    network_posterior_cov_mlp_z = build_dense_network(
            hidden_dim_rnn,
            [emb_mean_dim_s, dim_z], 
            [nn.ReLU(), None])
    posterior_distribution_z = GaussianDistributionFromMean(
        mean_network=network_posterior_mean_mlp_z,
        cov_network=network_posterior_cov_mlp_z, 
        distribution_dim=dim_z,
        config=config_inference,
        device=device,
    )
    network_posterior_rnn_z = build_rnn_cell(
        rnn_type="lstm", hidden_dim_rnn=hidden_dim_rnn, rnn_input_dim=emb_dim_yt+dim_z*8
    )
    posterior_network_z = RnnInferenceNetwork(
        posterior_rnn=network_posterior_rnn_z,
        posterior_dist=posterior_distribution_z,
        latent_dim=dim_z,
        embedding_network=network_input_embedding,
        device=device)


    snlds_model = HierachicalSSM(
        z_transition_network=transition_z,
        s_transition_network=transition_s,
        emission_network=emission_network,
        inference_network=[posterior_network_s, posterior_network_z],
        num_samples=1,
        args=args,
        device=device,
        logs=logs,)

    return snlds_model

############################## create base models END ##############################
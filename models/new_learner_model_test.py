import sys
sys.path.append('..')

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import distributions

import collections
namedtuple = collections.namedtuple

from collections import defaultdict
import ipdb

from models.learner_model import BaseLearnerModel
from models.new_learner_model import build_dense_network
from utils.utils import ConfigDict
from models.BaseModel import BaseModel

torch.autograd.set_detect_anomaly(True)

RANDOM_SEED = 131
EPS = 1e-6
T_SCALE = 60*60*24


class TestHierachicalSSM(BaseLearnerModel):
    
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
        
        super().__init__(mode='train', device=device, logs=logs)
        self.dim_y = 1
        self.dim_z = 1
        self.dim_s = 3
        self.num_sample = 100 
        
        self.directly_fit_vi, self.infer_global_s, self.infer_transition_s = 1, 0, 0
        # options for self.directly_fit_vi
        # 1. trainable covariance matrix for p(s_0), p(z_0), p(s_t)
        # 2. covariance matrix is a time-dependent

        self.s0_mean, self.s0_scale = self._initialize_normal_mean_cov(self.dim_s, False)
        self.z0_mean, self.z0_scale = self._initialize_normal_mean_cov(self.dim_z, False) # self.z0_scale is std
        
        
        # FLAG
        # self.s_infer, self.z_infer = inference_network
        if self.directly_fit_vi:
            self.s_trans_mean, self.s_trans_scale = self._initialize_normal_mean_cov(self.dim_s, True, num_sample=num_seq) 
        
        elif self.infer_global_s:   
            self.embedding_network = nn.RNN(
                input_size=self.dim_y * 2, # input_y + input_time 2
                hidden_size=self.dim_y * 2 * 2,  # 4
                bidirectional = True,
                batch_first = True,
            )
            self.network_posterior_mean_mlp_s = build_dense_network(
                800,
                [64, self.dim_s * 2], 
                [nn.ReLU(), None]) # is it not allowed to add non-linear functions here?
            
        # FLAG
        # self.y_emit = emission_network
        self.y_emit = torch.sigmoid
        
        # self.z_tran = z_transition_network
        # self.s_tran = s_transition_network
        
        # self.discrete_prior = self.s0_dist
        self.observation_dim = self.dim_y
        
        self.logs = logs
        self.device = device
        self.args = args

    @staticmethod	
    def normalize_timestamps(timestamps):	
        '''	
        timestamps: [bs, T, ...]?	
        '''	
        mean_val = torch.mean(timestamps, dim=1, keepdim=True)	
        std_val = torch.std(timestamps, dim=1, keepdim=True)	
        normalized_timestamps = (timestamps - mean_val) / std_val	
        return normalized_timestamps
    
    
    def _construct_normal_from_mean_cov(self, mean, cov):
        """
        Construct a multivariate Gaussian distribution from a mean and covariance matrix.

        Parameters:
        - mean: a PyTorch parameter representing the mean of the Gaussian distribution
        - cov: a PyTorch parameter representing the covariance matrix of the Gaussian distribution

        Returns:
        - dist: a PyTorch distribution representing the multivariate Gaussian distribution
        """
        if cov.shape[-1] == 1:
            # if the covariance matrix is a scalar, create a univariate normal distribution
            dist = distributions.MultivariateNormal(mean, cov*cov + EPS)
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
            cov_pdm = torch.matmul(cov, cov.transpose(-1,-2)) + EPS
            # try:
            #     torch.linalg.cholesky(cov_pdm)
            # except RuntimeError:
            #     # if the covariance matrix is not positive definite, add a small positive constant to its diagonal
            #     cov_pdm = cov_pdm + torch.eye(cov_pdm.size(-1)) * EPS
                
            # create a multivariate normal distribution
            dist = distributions.MultivariateNormal(mean, scale_tril=torch.tril(cov_pdm))
        return dist
    
    
    def _initialize_normal_mean_cov(self, dim: int, use_trainable_cov: bool, num_sample: int = 1):
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
        x0_mean = nn.init.xavier_uniform_(torch.empty(num_sample, dim, device=self.device))
        x0_mean = nn.Parameter(x0_mean, requires_grad=True)

        x0_scale = nn.init.xavier_uniform_(torch.empty(num_sample, dim, device=self.device))
        x0_scale = nn.Parameter(x0_scale, requires_grad=use_trainable_cov)

        x0_scale = torch.diag_embed(x0_scale)
        # m = nn.init.xavier_uniform_(torch.empty(num_sample, int(dim * (dim + 1) / 2), device=self.device))
        # x0_scale = torch.zeros((num_sample, dim, dim), device=self.device)
        # tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)
        # x0_scale[:, tril_indices[0], tril_indices[1]] += m
        # x0_scale = nn.Parameter(x0_scale, requires_grad=use_trainable_cov)
        
        return x0_mean, x0_scale


    def get_objective_values(self,
                            log_probs, 
                            log_prob_q,
                            posterior_entropies,
                            num_samples):
        
        # All the sequences should be of the shape
        # [batch_size, num_steps, (data_dim)]
        [log_prob_st, log_prob_zt, log_prob_yt] = log_probs 
        
        sequence_likelihood = (log_prob_st[:, 1:] + log_prob_zt[:, 1:] + log_prob_yt[:, 1:]).mean(-1) # [bs,]
        initial_likelihood = log_prob_st[:, 0] + log_prob_zt[:, 0] + log_prob_yt[:, 0]

        t1_mean = torch.mean(sequence_likelihood, dim=0)
        t2_mean = torch.mean(initial_likelihood, dim=0)
        
        t3 = torch.sum(posterior_entropies[0], dim=-1) # no gradient now  TODO entropy should be minimized? in ELBO?
        t3_mean = torch.mean(t3, dim=0)
        
        t4 = torch.sum(posterior_entropies[1], dim=-1) # no gradient now  TODO
        t4_mean = torch.mean(t4, dim=0)
        
        elbo = t1_mean + t2_mean - t3_mean - t4_mean
        
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
        
        y_input = inputs['label_seq']  # [bs, times]
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
        
        sampled_z0 = sampled_z[:, 0] # [bs*n, z_dim]
        log_prob_z0 = self.z0_dist.log_prob(sampled_z0[:, None, :]) # [bs, 1]
        
        # `log_prob_zt` should be of the shape [batch_size, num_steps, self.z_dim]
        log_prob_zt = self.get_z_prior(sampled_z, [sampled_s, time_input], log_prob_z0) # [bs*n, num_steps]


        ########################################
        ## getting log p(s[t] |s[t-1], x[t-1])
        ########################################

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
        
        
        eps = 1e-6
        emission_dist = self.y_emit(hidden_state_sequence)
        mean = emission_dist
        # ipdb.set_trace()

        if sample_for_reconstruction:
            # reconstructed_obs = emission_dist.sample() # NOTE: no gradient!
            probs = torch.cat([1-mean, mean], dim=-1)
            reconstructed_obs = F.gumbel_softmax(torch.log(probs + eps), tau=1, hard=sample_hard, eps=1e-10, dim=-1)
            reconstructed_obs = reconstructed_obs[..., 1:]
        else:
            reconstructed_obs = mean

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
    
	
    def s_infer(self, inputs, num_samples=1): # FLAG
        '''
        Args:
            inputs: [y, t, user_id]
            num_samples: number of samples for each time step
        Return:

        '''
        input_y, input_t, user_id = inputs	
        bs, num_steps, _ = input_y.shape	
        ipdb.set_trace()
        
        time_diff = torch.diff(input_t, dim=1)/T_SCALE + EPS  
        # TODO add time information p(s_t | s_t-1) variance should based on time difference

        self.s0_dist = self._construct_normal_from_mean_cov(
            mean=self.s0_mean, 
            cov=self.s0_scale
        )
        
        # direcly fit 
        if self.directly_fit_vi:
            # initial samples
            s_trans_mean_user_wise = self.s_trans_mean[user_id][None, ...] # [bs, dim_s]
            s_trans_cov_user_wise = self.s_trans_scale[user_id][None, ...]
            s0_sample = self.s0_dist.rsample((num_samples,)) # [n, 1, dim_s]
            s0_sample = torch.tile(s0_sample, (1, bs, 1)) # [n, bs, dim_s]
            
            # set the list to store the samples
            samples = [s0_sample]
            
            # set the list to store the mean and covariance matrix
            mus = [self.s0_dist.mean[None,...].repeat(num_samples, bs, 1)]
            covs = [self.s0_dist.covariance_matrix[None,...].repeat(num_samples, bs, 1, 1)]
            entropies = [self.s0_dist.entropy()[None,...].repeat(num_samples, bs)]
            log_probs = [self.s0_dist.log_prob(s0_sample)]
            
            # sequentially sample from the multivariate Normal distribution
            s_last = s0_sample
            for step in range(num_steps-1):
                # mu is elementwise multiplication of self.s_trans_mean and s_last 
                mu = s_trans_mean_user_wise * s_last
                cov =  s_trans_cov_user_wise # * time_diff[:, step] # TODO
                cov = torch.tile(cov, (num_samples, 1,1,1)) # [n, bs, dim_s, dim_s]
                # ipdb.set_trace()
                dist = self._construct_normal_from_mean_cov(mu, cov)
                sample = dist.rsample()
                
                # append the sample, mean and covariance matrix to the list
                s_last = sample
                samples.append(sample)
                mus.append(mu)
                covs.append(cov)
                entropies.append(dist.entropy()) # [n, bs]
                log_probs.append(dist.log_prob(sample)) # [n, bs]
            
            # convert the list to the tensor
            s_sampled = torch.stack(samples, -2) # [n, bs, times, dim_s]
            mus = torch.stack(mus, -2) # [n, bs, times, dim_s]
            covs = torch.stack(covs, -3) # [n, bs, times, dim_s, dim_s]
            s_entropy = torch.stack(entropies, -1) # [n, bs, times]
            s_log_probs = torch.stack(log_probs, -1) # [n, bs, times]
            
            rnn_states = None
            
            
        elif self.infer_global_s:
            t_inputs = (inputs[1]-inputs[1][:,0:1])/(inputs[1][:,-1:]-inputs[1][:,0:1])
            mc_t_inputs = torch.tile(t_inputs, (num_samples, 1,1)).float()
            mc_y_inputs = torch.tile(inputs[0], (num_samples, 1,1)).float() # [bs, times, 1]
            mc_yt_inputs = torch.cat([mc_y_inputs, mc_t_inputs], -1) # [bs, times, 1]
            
            out1, out2 = self.embedding_network(mc_yt_inputs) # out1: [bs, time, rnn_hid_dim*2]
            out1 = torch.reshape(out1, (bs*num_samples, 1, -1))
            
            dist = self.network_posterior_mean_mlp_s(out1)
            mus = dist[..., :3]
            covs = torch.pow(dist[..., 3:], 2) + EPS
        
            output_dist = distributions.multivariate_normal.MultivariateNormal(
                    loc = mus, scale_tril=torch.tril(torch.diag_embed(covs)))  
            
            s_sampled = output_dist.rsample()
            s_entropy = output_dist.entropy()
            s_log_probs = output_dist.log_prob(s_sampled)
            
            s_sampled = s_sampled.reshape(num_samples, bs, 1, self.dim_s) 
            s_entropy = s_entropy.reshape(num_samples, bs, -1)
            s_log_probs = s_log_probs.reshape(num_samples, bs, -1)
            rnn_states = out1.reshape(num_samples, bs, num_steps, -1)
            
        return s_sampled, s_entropy, s_log_probs, rnn_states, mus, covs
    
    
    def z_infer(self, inputs, num_samples=1): # FLAG
        '''
        Args:
            inputs: [t, s]
            s_sampled: [num_samples * bs, num_steps, s_dim]
            input_y, input_t: [bs, times, 1]
        '''
        
        ipdb.set_trace()
        input_t, s_sampled = inputs 
        bs, num_steps, _ = input_t.shape
        bsn, num_s_steps, _ = s_sampled.shape
        assert(num_samples == int(bsn // bs))
        
        self.z0_dist = self._construct_normal_from_mean_cov(mean=self.z0_mean, cov=self.z0_scale)
        
        time_diff = torch.diff(input_t, dim=1)/T_SCALE + EPS 
        time_diff = torch.tile(time_diff, (num_samples, 1,1)) # [bs*n, num_steps, 1]
        s_sampled = s_sampled[:, 1:] if num_s_steps > 1 else s_sampled # [bs*n, num_steps, 1]
        
        sampled_alpha = torch.relu(s_sampled[..., 0:1]) + EPS 
        sampled_mean = s_sampled[..., 1:2]
        sampled_vola = s_sampled[..., 2:3]
        sampled_var = torch.pow(sampled_vola, 2) * torch.exp(- sampled_alpha * time_diff) + EPS
        # sampled_var = torch.pow(torch.sigmoid(sampled_vola)) + EPS # TODO whether to use sigmoid
        # torch.sigmoid(torch.pow(sampled_vola, 2)) + EPS 
        
        z0_var = torch.tile(self.z0_dist.covariance_matrix, (bsn, 1, 1))
        z_var = torch.cat([z0_var, sampled_var], -2) # [bs*n, num_steps, 1]
        
        z0 = self.z0_dist.rsample((num_samples, )) # torch.zeros((bsn, 1, 1), device=self.device)
        z0 = torch.tile(z0, (bs, 1, 1))
        z_last = z0
        
        z_mean = [z0]
        
        for i in range(1, num_steps):
            decay = torch.exp(- sampled_alpha[:,i-1:i] * time_diff[:, i-1:i])
            mean = z_last * decay + (1.0 - decay) * sampled_mean[:,i-1:i]     
            z_mean.append(mean)
            
        z_mean = torch.cat(z_mean, -2) # [bs*n, num_steps, 1]       
        output_dist = distributions.multivariate_normal.MultivariateNormal(
                                loc = z_mean, scale_tril=torch.tril(torch.diag_embed(z_var)))
        
        z_sampled = output_dist.rsample() # [bs*n, num_steps, 1]
        z_entropy = output_dist.entropy() # [bs*n, num_steps]
        log_prob = output_dist.log_prob(z_sampled) # [bs*n, num_steps]
        
        z_sampled = z_sampled.reshape(num_samples, bs, num_steps, self.dim_z) 
        z_entropy = z_entropy.reshape(num_samples, bs, -1)
        log_prob = log_prob.reshape(num_samples, bs, -1)
        z_mean = z_mean.reshape(num_samples, bs, num_steps, self.dim_z)
        z_var = z_var.reshape(num_samples, bs, num_steps, self.dim_z, self.dim_z)

        return z_sampled, z_entropy, log_prob, z_mean, z_var
        
        
        

    def forward(self, inputs, temperature=1.0):
        s_dim = self.dim_s
        z_dim = self.dim_z
        num_samples = self.num_sample
        
        input_y = inputs['label_seq'].unsqueeze(-1)  # [bs, times] -> [bs, times, 1]
        input_t = inputs['time_seq'].unsqueeze(-1)
        user_id = inputs['user_id']
        num_steps = input_t.shape[1]
        
        # ----- Sample continuous hidden variable from `q(s[1:T] | y[1:T])' -----
        s_sampled, s_entropy, s_log_prob_q, _, s_mean, s_var = self.s_infer(
            [input_y, input_t, user_id], num_samples=num_samples
        )
        
        _, bs, num_s_steps, _ = s_sampled.shape

        s_sampled = torch.reshape(s_sampled, [num_samples * bs, num_s_steps, s_dim])
        s_entropy = torch.reshape(s_entropy, [num_samples * bs, num_s_steps])
        s_log_prob_q = torch.reshape(s_log_prob_q, [num_samples * bs, num_s_steps])

        # ----- Sample continuous hidden variable from `q(z[1:T] | y[1:T])' -----
        z_sampled, z_entropy, z_log_prob_q, z_mean, z_var = self.z_infer(
            [input_t, s_sampled], num_samples=num_samples
        )
        
        z_sampled = torch.reshape(z_sampled, [num_samples * bs, num_steps, z_dim])
        z_entropy = torch.reshape(z_entropy, [num_samples * bs, num_steps])
        z_log_prob_q = torch.reshape(z_log_prob_q, [num_samples * bs, num_steps])

        # ipdb.set_trace()
        # ----- joint log likelihood -----
        if self.directly_fit_vi:
            # ipdb.set_trace()
            probs = self.y_emit(z_sampled)
            emission_dist = torch.distributions.bernoulli.Bernoulli(probs=probs)
            log_prob_yt = emission_dist.log_prob(torch.tile(input_y.float(), (num_samples, 1, 1)))[..., 0]
            log_prob_zt = z_log_prob_q
            log_prob_st = s_log_prob_q
        else:
            log_prob_yt, log_prob_zt, log_prob_st = self.calculate_likelihoods( 
                inputs, s_sampled, z_sampled, temperature=temperature)
        
        recon_inputs = self.get_reconstruction(
            z_sampled,
            observation_shape=z_sampled.shape,
            sample_for_reconstruction=False, # TODO
        )
        
        recon_inputs = torch.reshape(recon_inputs, [num_samples, bs, num_steps, -1])
        z_sampled = torch.reshape(z_sampled,
                            [num_samples, bs, num_steps, z_dim])
        s_sampled = torch.reshape(s_sampled,
                            [num_samples, bs, num_s_steps, s_dim])
        
        return_dict = self.get_objective_values([log_prob_st, log_prob_zt, log_prob_yt], 
                                                [s_log_prob_q, z_log_prob_q],
                                                [s_entropy, z_entropy], num_samples)

        # ipdb.set_trace()
        return_dict["label"] = input_y # [bs, times, 1]
        return_dict["prediction"] = recon_inputs  # [n, bs, times, 1]
        return_dict["sampled_z"] = z_sampled # [n, bs, times, 1]
        return_dict["sampled_s"] = s_sampled # [n, bs, times, 3]
        return_dict["sampled_y"] = recon_inputs # [n, bs, times, 1]
        return_dict['mean_s'] = s_mean # [n, bs, times, 3]
        return_dict['var_s'] = s_var # [n, bs, times, 3, 3]
        return_dict['mean_z'] = z_mean # [n, bs, times, 1]
        return_dict['var_z'] = z_var # [n, bs, times, 1, 1]
        # ipdb.set_trace()
        # test
        return_dict['log_prob_st'] = log_prob_st.mean()
        return_dict['log_prob_zt'] = log_prob_zt.mean()
        return_dict['log_prob_yt'] = log_prob_yt.mean()

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
    
    
    def predictive_model(self, inputs):
        # TODO it depends on the training mode, for now, it is only for splitting time 
        '''
        p(s_t+1, z_t+1 | y_1:t)
        Args:
            inputs: 
            num_samples
        '''
        num_samples = self.num_sample
        # ipdb.set_trace()
        time_step = int(inputs['skill_seq'].shape[-1])
        train_step = int(time_step * self.args.train_time_ratio)
        test_step = int(time_step * self.args.test_time_ratio)
        
        past_y = inputs['label_seq'][:, :train_step].unsqueeze(-1)
        past_t = inputs['time_seq'][:, :train_step].unsqueeze(-1)
        
        future_t = inputs['time_seq'][:, train_step-1:].unsqueeze(-1)
        future_t = torch.tile(future_t, (num_samples,1,1))
        bs = past_y.shape[0]
        
        # TODO which way to sample? from q(z) or from q(s)? or prior distribution?
        # ideally they should be the same, but currently because the 
        
        if self.directly_fit_vi:
            s_sampled, _, _, _, s_mean, s_var = self.s_infer(
                [inputs['label_seq'].unsqueeze(-1), inputs['time_seq'].unsqueeze(-1), inputs['user_id']], 
                num_samples=num_samples
            )
        else:
            s_sampled, _, _, _, s_mean, s_var = self.s_infer([past_y, past_t, inputs['user_id']], num_samples)
        
        s_step = s_sampled.shape[-2]
        s_sampled = s_sampled.reshape(-1, s_step, self.dim_s)
        
        
        z_sampled, _, _, z_mean, z_var = self.z_infer(
                [inputs['time_seq'].unsqueeze(-1), s_sampled], 
                num_samples=num_samples
            )
        z_step = z_sampled.shape[-2]
        z_sampled = torch.reshape(z_sampled, [-1, z_step, self.dim_z])
        
        recon_inputs = self.get_reconstruction(
            z_sampled,
            observation_shape=z_sampled.shape,
            sample_for_reconstruction=False, # TODO
        )
        # s_last = s_sampled[:,:,-1:].reshape(-1, 1,3) # [n_samples, train_total, train_time, 3] -> [total, 1. 3]
        # z_last = z_sampled[:,:,-1:].reshape(-1, 1,1)
        # y_last = torch.tile(past_y[:, -1:], (num_samples,1,1))
        
        
        # pred_s, pred_z, pred_y = self.generative_model([s_last, z_last, y_last], future_timestamps, 
        #                           self.args.max_step-train_step, )
        # ipdb.set_trace()
        pred_dict = {
            'prediction': recon_inputs[:, -test_step:],  # [bs, 80, 1]
            'label': inputs['label_seq'][:, -test_step:].unsqueeze(-1),
            'pred_y': recon_inputs.reshape(num_samples, bs, time_step, self.dim_y),  # [bs, 200, 1]
            'pred_z': z_sampled.reshape(num_samples, bs, z_step, self.dim_z), # [bsn, 200, 1]
            'pred_s': s_sampled.reshape(num_samples, bs, s_step, self.dim_s), # [bsn, 1, 3]
            'mean_s': s_mean.reshape(num_samples, bs, s_step, self.dim_s), # [bsn, 3]
            'var_s': s_var.reshape(num_samples, bs, s_step, self.dim_s, self.dim_s), # [bsn, 3]
            'mean_z': z_mean.reshape(num_samples, bs, z_step, self.dim_z), # [bsn, time, 1]
            'var_z': z_var.reshape(num_samples, bs, z_step, self.dim_z), # [bsn, 1, 1]
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
        
        for key in ['elbo', 'initial_likelihood', 'sequence_likelihood', 
                    'st_entropy', 'zt_entropy',
                    'log_prob_yt', 'log_prob_zt', 'log_prob_st']:
            losses[key] = outdict[key]
        
        losses['loss_total'] = -outdict['elbo']
        # losses['loss_total'] = -elbo + bceloss
        
        if metrics != None:
            pred = pred.detach().cpu().data.numpy()
            gt = gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
            
        losses['ou_speed'] = outdict["sampled_s"][0,0,0,0]
        losses['ou_mean'] = outdict["sampled_s"][0,0,0,1]
        losses['ou_vola'] = outdict["sampled_s"][0,0,0,2]

        return losses









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
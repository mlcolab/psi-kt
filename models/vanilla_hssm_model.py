import sys
sys.path.append('..')

import math 

import torch
from torch import nn, distributions
from torch.nn import functional as F

from collections import defaultdict

import ipdb

from models.learner_hssm_model import HSSM
from models.modules import build_rnn_cell
from models.BaseModel import BaseModel
from models.learner_model import BaseLearnerModel
from models.new_learner_model import build_dense_network
from models.modules import CausalTransformerModel, VAEEncoder
from models.variational_distributions import VarDIBS, VarTransformation, VarAttention

from enum import Enum


class VanillaHSSM(HSSM):
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
        
        self.dim_s = 3 # [alpha, mu, sigma]
        self.adj = None
        
        self.fit_vi_global_s, self.fit_vi_transition_s, self.infer_global_s, self.infer_transition_s = 1,0,0,0

        # ----- initialize initial parameter distribution -----
        self.z0_mean, self.z0_scale = self._initialize_normal_mean_log_var(self.dim_z, False) # self.z0_scale is std
        self.z0_dist = self._construct_normal_from_mean_std(mean=self.z0_mean, std=self.z0_scale)

        if 'simple' in self.mode or 'ns_' in self.mode:
            num_s = 1
        elif 'ls_' in self.mode or 'ln_' in self.mode:
            num_s = num_seq
            
        # ----- FLAG -----
        if self.fit_vi_global_s:
            self.s_trans_mean, self.s_trans_scale = self._initialize_normal_mean_log_var(
                self.dim_s, True, num_sample=num_s)
            self.s_infer = self.s_global_fit
            # the s_s_trans_mean is actually mean of s, rather than mean of transition of s
            
        elif self.fit_vi_transition_s:
            self.s_trans_mean, self.s_trans_scale = self._initialize_normal_mean_log_var(
                self.dim_s, True, num_sample=num_s)
            
            self.s0_mean, self.s0_scale = self._initialize_normal_mean_log_var(self.dim_s, False)
            self.s0_dist = self._construct_normal_from_mean_std(
                        mean=self.s0_mean, 
                        std=self.s0_scale
                    )
            self.s_infer = self.s_transition_fit
            
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
            
            
    def s_global_fit(self, feed_dict, num_samples=1): # FLAG 
        # [bs, num_samples, num_node, num_steps, dim]
        user_id = feed_dict['user_id']
        bs, time_step = feed_dict['time_seq'].shape
        bsn = bs * self.num_sample
        
        # ----- initial samples -----
        # s0_sample = self.s0_dist.rsample((num_samples,)) # [n, 1, dim_s]
        # s0_sample = torch.tile(s0_sample[None, ], (bs,1,1,1)) # [bs, n, 1, dim_s]
        
        # ----- calculate time difference -----
        input_t = feed_dict['time_seq'].unsqueeze(1)
        dt = torch.diff(input_t, dim=-1)/T_SCALE + EPS  # [bs, 1, num_steps-1]
        if self.user_time_dependent_covariance:
            cov_scale = dt / torch.min(dt, dim=-1, keepdim=True)[0] # TODO
        else:
            cov_scale = torch.ones_like(dt)
        cov_scale = cov_scale.unsqueeze(1) # [bs,1,1,num_steps-1]
        cov_time_scale = 1/(cov_scale + EPS)
        cov_time_scale = torch.cat([torch.zeros_like(cov_time_scale[...,:1]), cov_time_scale], dim=-1) # [bs,1,1,num_steps]
        
        # ----- direcly fit -----
        s_mean = self.s_trans_mean[user_id].reshape(bs,1,1,1,-1) # [1, 1, dim_s]
        s_std = self.s_trans_scale[user_id].reshape(bs,1,1,1,self.dim_s,self.dim_s) # [bs, 1, 1, dim_s, dim_s]

        s_device = s_mean.device
        s_std = s_std.to(s_device)
        cov_time_scale = cov_time_scale.to(s_device)
        s_std_rescale = s_std * cov_time_scale[..., None, None] # [bs, 1, 1, dim_s, dim_s]
        s_dist = self._construct_normal_from_mean_std(s_mean, s_std_rescale)
        
        s_sampled = s_dist.rsample((num_samples,)).transpose(1,0).squeeze(2)  # [bs, n, 1, time_step, dim_s]
        s_entropy = s_dist.entropy().reshape((bs, time_step)) # [
        s_log_prob_q  = s_dist.log_prob(s_sampled).reshape((bsn, time_step)) # [bsn, time_step]
        mus = s_mean
        stds = s_std
        
        s_sampled = s_sampled.reshape((bsn, 1, time_step, self.dim_s)) # [bs, n, 1, 1, time_step dim_s]
        rnn_states = None
        
        return s_sampled, s_entropy, s_log_prob_q, rnn_states, mus, stds
        
        
        
        
    def s_transition_infer(self, feed_dict, num_samples=1): # FLAG 
        '''
        '''
        
        user_id = feed_dict['user_id']
        bs, time_step = feed_dict['time_seq'].shape
        
        # ----- initial samples -----
        self.s0_dist = self._construct_normal_from_mean_std(
                    mean=self.s0_mean, 
                    std=self.s0_scale
                )
        s0_sample = self.s0_dist.rsample((num_samples,)) # [n, 1, dim_s]
        s0_sample = torch.tile(s0_sample, (1, bs, 1)) # [n, bs, dim_s]
        
        # ----- calculate time difference -----
        input_t = feed_dict['time_seq'].unsqueeze(1)
        dt = torch.diff(input_t, dim=-1)/T_SCALE + EPS  # [bs, 1, num_steps-1]
        if self.user_time_dependent_covariance:
            cov_scale = dt / torch.min(dt,1,True)[0] # TODO
        else:
            cov_scale = torch.ones_like(dt)
        cov_scale = cov_scale.unsqueeze(0) # [1, bs, num_steps-1]
        cov_time_scale = 1/(cov_scale + EPS)
        
        # ----- direcly fit -----
        if self.fit_vi_transition_s:
            # ----- adapt to different modes -----
            if 'simple' in self.mode:
                user_id = Ellipsis
            elif 'ls_' in self.mode or 'ln_' in self.mode:
                user_id = user_id
            
            s_trans_mean_user_wise = self.s_trans_mean[user_id][None,...] # [1, bs, dim_s]
            s_trans_std_user_wise = self.s_trans_scale[user_id][None,...]

            # set the list to store the samples
            samples = [s0_sample]
            mus = [self.s0_mean.reshape(1,1,self.dim_s).repeat(num_samples, bs, 1)]
            stds = [self.s0_scale.reshape(1,1,self.dim_s,self.dim_s).repeat(num_samples, bs, 1, 1)]
            entropies = [self.s0_dist.entropy()[None,...].repeat(num_samples, bs)] # what size should it be? for now [n, bs]
            log_probs = [self.s0_dist.log_prob(s0_sample)] # [n, bs]
            
            # sequentially sample from the multivariate Normal distribution
            s_last = s0_sample
            for step in range(time_step-1):
                mu = s_trans_mean_user_wise * s_last
                std =  s_trans_std_user_wise * cov_time_scale[...,step:step+1] # TODO
                std = torch.tile(std, (num_samples, 1,1,1)) # [n, bs, dim_s, dim_s]

                dist = self._construct_normal_from_mean_std(mu, std)
                sample = dist.rsample() # [n, bs, dim_s]
                
                # append the sample, mean and covariance matrix to the list
                s_last = sample
                samples.append(sample)
                mus.append(mu)
                stds.append(std)
                entropies.append(dist.entropy()) # [n, bs]
                log_probs.append(dist.log_prob(sample)) # [n, bs]
            
            # convert the list to the tensor
            s_sampled = torch.stack(samples, -2) # [n, bs, times, dim_s]
            mus = torch.stack(mus, -2) # [n, bs, times, dim_s]
            stds = torch.stack(stds, -3) # [n, bs, times, dim_s, dim_s]
            s_entropy = torch.stack(entropies, -1) # [n, bs, times]
            s_log_prob_q = torch.stack(log_probs, -1) # [n, bs, times]
            
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

        num_s_steps= s_sampled.shape[-2]
        s_sampled = torch.reshape(s_sampled, [self.num_sample * bs, num_s_steps, self.dim_s])
        s_entropy = torch.reshape(s_entropy, [self.num_sample * bs, num_s_steps])
        s_log_prob_q = torch.reshape(s_log_prob_q, [self.num_sample * bs, num_s_steps])
        mus = torch.reshape(mus, [self.num_sample * bs, num_s_steps, self.dim_s])
        stds = torch.reshape(stds, [self.num_sample * bs, num_s_steps, self.dim_s, self.dim_s])
        
        return s_sampled, s_entropy, s_log_prob_q, rnn_states, mus, stds
    

    def s_transition_fit(self, feed_dict, num_samples=1): # FLAG 
        '''
        Args:
            inputs: [y, t, user_id]
            num_samples: number of samples for each time step
        Return:

        '''
        user_id = feed_dict['user_id']
        bs, time_step = feed_dict['time_seq'].shape
        bsn = bs * self.num_sample
        # num_node = 1 # self.num_node
    
        
        # ----- initial samples -----
        self.s0_dist = self._construct_normal_from_mean_std(
                    mean=self.s0_mean, 
                    std=self.s0_scale
                )
        s0_sample = self.s0_dist.rsample((num_samples,)).unsqueeze(0) # [1, n, 1, dim_s]
        s0_sample = torch.tile(s0_sample, (bs, 1, 1, 1)) # [bs, n, num_node, dim_s]
        
        # ----- calculate time difference -----
        input_t = feed_dict['time_seq'].unsqueeze(1)
        dt = torch.diff(input_t, dim=-1)/T_SCALE + EPS  # [bs, 1, num_steps-1]
        if self.user_time_dependent_covariance:
            cov_scale = dt / torch.min(dt, dim=-1, keepdim=True)[0] # TODO
        else:
            cov_scale = torch.ones_like(dt)
        cov_scale = cov_scale.unsqueeze(1) # [bs,1,1,num_steps-1]
        cov_time_scale = 1/(cov_scale + EPS)
        cov_time_scale = torch.cat([torch.zeros_like(cov_time_scale[...,:1]), cov_time_scale], dim=-1) # [bs,1,1,num_steps]
        
        # ----- adapt to different modes -----
        if 'simple' in self.mode:
            user_id = Ellipsis
        elif 'ls_' in self.mode or 'ln_' in self.mode:
            user_id = user_id
        
        s_trans_mean_user_wise = self.s_trans_mean[user_id].reshape(bs,1,1,self.dim_s) # [bs, 1, num_node, dim_s]
        s_trans_std_user_wise = self.s_trans_scale[user_id].reshape(bs,1,1,self.dim_s,self.dim_s)

        # set the list to store the samples
        samples = [s0_sample]
        mus = [self.s0_mean.reshape(1,1,1,self.dim_s).repeat(bs, num_samples, 1, 1)]
        stds = [self.s0_scale.reshape(1,1,1,self.dim_s,self.dim_s).repeat(bs, num_samples, 1, 1, 1)]
        entropies = [self.s0_dist.entropy()[None,...].repeat(bs, num_samples)]
        log_probs = [self.s0_dist.log_prob(s0_sample)[:,:,0]] 
        
        # sequentially sample from the multivariate Normal distribution
        s_last = s0_sample
        for step in range(time_step-1):
            mu = s_trans_mean_user_wise * s_last
            std =  s_trans_std_user_wise * cov_time_scale[...,step:step+1,None] # TODO
            std = torch.tile(std, (num_samples, 1,1,1,1)) # [n, bs, dim_s, dim_s]

            dist = self._construct_normal_from_mean_std(mu, std)
            sample = dist.rsample()
            
            # append the sample, mean and covariance matrix to the list
            s_last = sample
            samples.append(sample)
            mus.append(mu)
            stds.append(std)
            entropies.append(dist.entropy().mean(-1)) # [n, bs]
            log_probs.append(dist.log_prob(sample).mean(-1)) # [n, bs]
        
        # convert the list to the tensor
        s_sampled = torch.stack(samples, -2) # [n, bs, num_node, times, dim_s]
        mus = torch.stack(mus, -2) # [n, bs, num_node, times, dim_s]
        stds = torch.stack(stds, -3) # [n, bs, num_node, times, dim_s, dim_s]
        s_entropy = torch.stack(entropies, -1) # [n, bs, num_node, times]
        s_log_prob_q = torch.stack(log_probs, -1) # [n, bs, num_node, times]
        
        rnn_states = None
        num_s_steps= s_sampled.shape[-2]
        s_sampled = torch.reshape(s_sampled, [self.num_sample * bs, num_node, num_s_steps, self.dim_s])
        s_entropy = torch.reshape(s_entropy, [self.num_sample * bs, num_s_steps])
        s_log_prob_q = torch.reshape(s_log_prob_q, [self.num_sample * bs, num_s_steps])
        mus = torch.reshape(mus, [self.num_sample * bs, num_node, num_s_steps, self.dim_s])
        stds = torch.reshape(stds, [self.num_sample * bs, num_node, num_s_steps, self.dim_s, self.dim_s])
        
        return s_sampled, s_entropy, s_log_prob_q, rnn_states, mus, stds
    
    

    def predictive_model(self, inputs):
        # TODO it depends on the training mode, for now, it is only for splitting time 
        '''
        p(s_t+1, z_t+1 | y_1:t)
        Args:
            inputs: 
            num_sample
        '''
        
        time_step = int(inputs['skill_seq'].shape[-1])
        train_step = int(time_step * self.args.train_time_ratio)
        test_step = int(time_step * self.args.test_time_ratio)
        
        past_y = inputs['label_seq'][:, :train_step].unsqueeze(-1)
        past_t = inputs['time_seq'][:, :train_step].unsqueeze(-1)
        
        # future_t = inputs['time_seq'][:, train_step-1:].unsqueeze(-1)
        # future_t = torch.tile(future_t, (num_sample,1,1))
        
        bs = past_y.shape[0]
        num_sample = self.num_sample
        bsn = bs * num_sample
        
        # TODO which way to sample? from q(z) or from q(s)? or prior distribution?
        # ideally they should be the same, but currently because the 
        if self.fit_vi_global_s:
            test_out_dict = self.forward(inputs)
            recon_inputs = test_out_dict['sampled_y'][...,train_step:,:]
            
            tmp_recon_inputs = recon_inputs.flatten(0,1)
            future_item = inputs['skill_seq'][:, train_step:].tile(num_sample,1)
            recon_inputs_items = torch.cat(
                [tmp_recon_inputs[torch.arange(bsn), future_item[:,i], i] for i in range(future_item.shape[-1])], -1
            )
            recon_inputs_items = recon_inputs_items.reshape(bs, num_sample, 1, -1, 1)
            label = inputs['label_seq'][:, train_step:].reshape(bs, 1, 1, -1, 1).tile(1,num_sample,1,1,1)
            
            
        elif self.fit_vi_transition_s:
            s_sampled, _, _, _, s_mean, s_var = self.s_infer(
                [inputs['label_seq'].unsqueeze(-1), inputs['time_seq'].unsqueeze(-1), inputs['user_id']], 
                num_sample=num_sample
            )
            s_step = s_sampled.shape[-2]
            s_sampled = s_sampled.reshape(-1, s_step, self.dim_s)
            
            z_sampled, _, _, z_mean, z_var = self.z_infer(
                    [inputs['time_seq'].unsqueeze(-1), s_sampled], 
                    num_sample=num_sample
                )
            z_step = z_sampled.shape[-2]
            z_sampled = torch.reshape(z_sampled, [-1, z_step, self.dim_z])
            
            recon_inputs = self.get_reconstruction(
                z_sampled,
                observation_shape=z_sampled.shape,
                sample_for_reconstruction=False, 
            )
            
        elif self.fit_vi_global_s:
            test_out_dict = self.forward(inputs)
            recon_inputs = test_out_dict['sampled_y'][...,train_step:,:]
            
            tmp_recon_inputs = recon_inputs.flatten(0,1)
            future_item = inputs['skill_seq'][:, train_step:].tile(num_sample,1)
            recon_inputs_items = torch.cat(
                [tmp_recon_inputs[torch.arange(bsn), future_item[:,i], i] for i in range(future_item.shape[-1])], -1
            )
            recon_inputs_items = recon_inputs_items.reshape(bs, num_sample, 1, -1, 1)
            label = inputs['label_seq'][:, train_step:].reshape(bs, 1, 1, -1, 1).tile(1,num_sample,1,1,1)
        
        elif self.infer_transition_s:
            test_out_dict = self.forward(inputs)
            recon_inputs = test_out_dict['sampled_y'][...,train_step:,:]
            
            tmp_recon_inputs = recon_inputs.flatten(0,1)
            future_item = inputs['skill_seq'][:, train_step:].tile(num_sample,1)
            recon_inputs_items = torch.cat(
                [tmp_recon_inputs[torch.arange(bsn), future_item[:,i], i] for i in range(future_item.shape[-1])], -1
            )
            recon_inputs_items = recon_inputs_items.reshape(bs, num_sample, 1, -1, 1)
            label = inputs['label_seq'][:, train_step:].reshape(bs, 1, 1, -1, 1).tile(1,num_sample,1,1,1)
            
        else:
            s_sampled, _, _, _, s_mean, s_var = self.s_infer([past_y, past_t, inputs['user_id']], num_sample)
        

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
    
    
    def z_infer(self, inputs, num_samples=1): # FLAG
        '''
        [bs, num_samples, num_node, num_steps, dim]
        '''
        feed_dict, s_sampled = inputs 
        input_t = feed_dict['time_seq']
        items = feed_dict['skill_seq']
        bs, num_steps = input_t.shape
        num_seq = bs
        bsn, num_node, num_s_steps, _ = s_sampled.shape
        assert(num_samples == int(bsn // bs))
        
        # ----- compute the stats of history -----
        stats = torch.stack([feed_dict['num_history'], feed_dict['num_success'], feed_dict['num_failure']], dim=-1)
        stats = stats.unsqueeze(1)
        
        all_feature = self._compute_all_features(
            num_seq, self.num_node, input_t, self.device, False, items, stats
        )
        whole_stats, whole_last_time = self._find_whole_stats(
            all_feature, input_t, items, self.num_node
        ) # [bs, num_node, times, 3]; [bs, num_node, times+1]
        
        z_device = s_sampled.device
        input_t = input_t.to(z_device)
        whole_last_time = whole_last_time.to(z_device)
        
        # ----- calculate time difference -----
        input_t = input_t.unsqueeze(1) # 
        dt = torch.diff(input_t, dim=-1)/T_SCALE + EPS  # [bs, 1, num_steps-1]
        dt = torch.tile(dt, (num_samples,1,1)).unsqueeze(-1) # [bsn, 1, num_steps-1,1]
        
        # ----- calculate the mean and variance of z -----
        s_sampled = s_sampled[:,:,1:]
        sampled_alpha = torch.relu(s_sampled[..., 0:1]) + EPS  
        sampled_mean = s_sampled[..., 1:2]
        sampled_vola = s_sampled[..., 2:3] # [bsn, 1, num_steps-1, 1]
        sampled_var = torch.pow(sampled_vola, 2) * torch.exp(- sampled_alpha * dt) + EPS

        # ----- initial samples -----
        z0 = self.z0_dist.rsample((num_samples, )) # [n, 1, 1]
        z0 = torch.tile(z0, (bs, 1, 1))
        z_mean = [torch.tile(z0, (1, self.num_node, 1))] # [bsn, num_node, 1]
        z0_var = torch.tile(self.z0_dist.covariance_matrix, (bsn, 1, 1)) # [bsn, 1, 1]
        z_var = torch.cat([z0_var.unsqueeze(-1), sampled_var], -2) # [bsn, 1, num_steps, 1]

        # ----- simulate path -----
        z_last = z0.tile((1,self.num_node,1)) # [bs*n, num_node, 1]
        for i in range(1, num_steps):
            cur_dt = (input_t[...,i:i+1] - whole_last_time[..., i:i+1])/T_SCALE + EPS # [bs, num_node, 1]
            cur_dt = torch.tile(cur_dt, (num_samples, 1, 1)) # [nbs, num_node, 1]
            
            decay = torch.exp(- sampled_alpha[:,:,i-1] * cur_dt)  
            mean = z_last * decay + (1.0 - decay) * sampled_mean[:,:,i-1]
        
            z_mean.append(mean) # [bs*n, num_node, 1]
            
        z_mean = torch.stack(z_mean, -2) # [bs*n, num_node, num_steps, 1]
        # z_var = [bs*n, 1, num_steps, 1]
        output_dist = distributions.multivariate_normal.MultivariateNormal(loc=z_mean, scale_tril=torch.tril(torch.diag_embed(z_var)))
        z_sampled = output_dist.rsample() # [bs*n, num_node, num_steps, 1]
        z_entropy = output_dist.entropy().mean(-2) # [bs*n, num_steps]
        z_log_prob_q = output_dist.log_prob(z_sampled).mean(-2) # [bs*n, num_steps]
        
        return z_sampled, z_entropy, z_log_prob_q, z_mean, z_var


    
    
import sys
sys.path.append('..')

import math, os, argparse
import numpy
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import torch
from torch import nn, distributions
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F

from models.modules import build_rnn_cell, build_dense_network
from models.modules import CausalTransformerModel, VAEEncoder
from models.groupkt_graph_representation import VarTransformation, VarAttention
from models.gmvae import *
from utils.logger import Logger
from groupkt import * 

from baseline.BaseModel import BaseModel



class GroupKT(BaseModel):
    
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
        self.logs = logs
        self.device = device
        self.args = args
        self.num_seq = num_seq
        self.num_sample = args.num_sample
        self.var_log_max = torch.tensor(args.var_log_max) 

        # Set the device to use for computations
        self.device = device if device != None else args.device

        # Store the arguments and logs for later use
        self.args = args
        self.logs = logs
        
        BaseModel.__init__(self, model_path=os.path.join(args.log_path, 'Model/Model_{}_{}.pt'))
        

    @staticmethod	
    def _normalize_timestamps(
        timestamps: torch.Tensor,
    ):	
        '''	
        Normalizes timestamps by subtracting the mean and dividing by the standard deviation.
        Args:
            timestamps (torch.Tensor): Input timestamps of shape [bs, T, ...].

        Returns:
            torch.Tensor: Normalized timestamps of the same shape as the input.
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
        x0_log_var = torch.ones((num_sample, dim)) * torch.log(torch.tensor(COV_MIN))
        x0_log_var = nn.Parameter(x0_log_var, requires_grad=use_trainable_cov)
        
        return x0_mean, x0_log_var

    @staticmethod
    def _positional_encoding1d(
        d_model: int,
        length: int,
        actual_time=None
    ):
        """
        Modified based on https://github.com/wzlxjtu/PositionalEncoding2D
        Args:
            d_model: dimension of the model
            length: length of positions
        
        Returns:
            length*d_model position matrix
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
        """
        Get time embeddings based on the given time tensor.
        Args:
            time (torch.Tensor): Input time tensor of shape [bs, times, ...].
            type (str): Type of time embedding to compute. Can be 'dt' for time differences
                        or 'absolute' for absolute time values. Defaults to 'dt'.

        Returns:
            torch.Tensor: Time embeddings of shape [bs, times, dim].

        Note:
            The 'dt' option computes time differences by taking the differences between consecutive
            time steps. The 'absolute' option uses the original time values as time embeddings.
        
        """
    
        if type == 'dt':
            dt = torch.diff(time, dim=1) 
            t_pe = self._positional_encoding1d(self.node_dim, dt.shape[1], dt) # [bs, times, dim]
        elif type =='absolute':
            norm_t = time 
            t_pe = self._positional_encoding1d(self.node_dim, time.shape[1], norm_t) # [bs, times, dim]
        return t_pe
    
    def get_reconstruction(
        self,
        hidden_state_sequence: torch.Tensor,
        observation_shape: torch.Size = None,
        sample_for_reconstruction: bool = True,
        sample_hard: bool = False,
    ):
        """
        Generate reconstructed observations based on hidden state sequence.

        Args:
            hidden_state_sequence (torch.Tensor): Hidden state sequence used for reconstruction.
            observation_shape (torch.Size, optional): Desired shape of the reconstructed observation tensor.
            sample_for_reconstruction (bool, optional): Whether to sample for reconstruction.
            sample_hard (bool, optional): Whether to apply hard Gumbel softmax during sampling.

        Returns:
            torch.Tensor: Reconstructed observations.
        """
        emission_dist = self.y_emit(hidden_state_sequence)
        mean = emission_dist

        if sample_for_reconstruction:
            probs = torch.cat([1 - mean, mean], dim=-1)
            reconstructed_obs = F.gumbel_softmax(
                torch.log(probs + EPS), tau=1, hard=sample_hard, eps=1e-10, dim=-1
            )
            reconstructed_obs = reconstructed_obs[..., 1:]
        else:
            reconstructed_obs = mean

        if observation_shape is not None:
            reconstructed_obs = torch.reshape(reconstructed_obs, observation_shape)

        return reconstructed_obs


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


class AmortizedGroupKT(GroupKT):
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
    

    
    def st_transition_gen(
        self, 
        qs_dist: MultivariateNormal,
        eval: bool = False,
    ) -> MultivariateNormal:
        
        qs_mean = qs_dist.mean # [bs, 1, time, dim_s]
        qs_cov_mat = qs_dist.covariance_matrix # [bs, 1, time, dim_s, dim_s]
        device = qs_mean.device
        bs = qs_mean.shape[0]
        
        # -- 1. the prior from generative model of GMVAE --
        
        # -- 2. prior of single step of H, R -- 
        
        # # -- 3. multi-step transition --
        # # -- prior of multiple steps of H, R --
    
    def zt_transition_gen(
        self, 
        qs_sampled: torch.Tensor,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = 0,
        sampled_z: torch.Tensor = None,
        eval: bool = False,
    ):
        pass
        # return pz_dist 
        
        
    def yt_emission_func(self, ):
        pass
    
    
    def st_transition_infer(
        self, 
        emb_inputs: torch.Tensor,
        num_sample: int = 0,
        eval: bool = False,
    ):
        
        num_sample = self.num_sample if num_sample == 0 else num_sample
        
        qs_out_inf = self.infer_network_posterior_s(
            emb_inputs, 
            self.qs_temperature, 
            self.qs_hard, 
            self.time_dependent_s,
        ) 

        s_category = qs_out_inf['categorical'] # [bs, 1, num_cat]
        s_mean = qs_out_inf['s_mu_infer'] # [bs, time, dim_s]
        s_var = qs_out_inf['s_var_infer'] # [bs, time, dim_s]

        s_var_mat = torch.diag_embed(s_var + EPS)   # [bs, time, dim_s, dim_s]
        qs_dist = MultivariateNormal(
            loc=s_mean, 
            scale_tril=torch.tril(s_var_mat)
        )

        # NOTE: For debug use
        if not eval:
            self.register_buffer('qs_category_logits', qs_out_inf['logits'].clone().detach())
            self.register_buffer(name="qs_mean", tensor=s_mean.clone().detach())
            self.register_buffer(name="qs_var", tensor=s_var.clone().detach())
            self.logits = qs_out_inf['logits']
            self.probs = qs_out_inf['prob_cat']
            self.s_category = s_category
        self.register_buffer('qs_category', s_category.clone().detach())
        
        return qs_dist


        
    
    def generative_process(
        self,
        qs_dist: distributions.MultivariateNormal,
        qz_dist: distributions.MultivariateNormal,
        feed_dict: Dict[str, torch.Tensor] = None,
        eval: bool = False,
    ):
        # generative model for s (Karman filter)
        ps_dist = self.st_transition_gen(qs_dist, eval=eval) 
        
        # generative model for z (OU process)
        pz_dist = self.zt_transition_gen(qs_dist, feed_dict, eval=eval)

        return ps_dist, pz_dist 
    
    
    def predictive_model(
        self,
    ):
        t_train = feed_dict['time_seq']
        y_train = feed_dict['label_seq']
        item_train = feed_dict['skill_seq']
        bs, time_step = t_train.shape
        
        emb_history = self.embedding_process(time=t_train, label=y_train, item=item_train)
        
        s_sampled, z_sampled_scalar, s_entropy = self.inference_process(feed_dict, emb_history)
        
        [log_prob_st, log_prob_zt, log_prob_yt], recon_inputs_items = self.generative_process(
            feed_dict, s_sampled, z_sampled_scalar)

        return return_dict
    

    def embedding_process(self):
        pass


    def inference_process(self):
        pass


    def generative_process(self):
        pass









    
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
    

    def embedding_process(
        self,
        time: torch.Tensor, 
        label: torch.Tensor,
        item: torch.Tensor,
    ):
        """
        Process the input features to create the embedding history.

        Args:
            time (torch.Tensor): Time information tensor of shape [batch_size, times].
            label (torch.Tensor): Label information tensor of shape [batch_size, times].
            item (torch.Tensor): Item information tensor of shape [batch_size].

        Returns:
            torch.Tensor: The embedding history tensor of shape [batch_size, trian_t, dim].
        """
        
        # Concatenate three kinds of input features: KC embedding + time info + label info
        # TODO: the simplest way is to concatenate them and input into an NN, 
        #       in Transformer architecture, the time info is added to the input embedding,
        #       not sure which one is better
        
        if isinstance(self.infer_network_emb, nn.LSTM):
            t_pe = self.get_time_embedding(time, 'absolute') # [bs, times, dim] 
            y_pe = torch.tile(label.unsqueeze(-1), (1,1, self.node_dim)) # + t_pe 
            node_pe = self.node_dist._get_node_embedding()[item] # [bs, times, dim]      
            emb_input = torch.cat([node_pe, y_pe], dim=-1) # [bs, times, dim*2]
            self.infer_network_emb.flatten_parameters()
            emb_history, _ = self.infer_network_emb(emb_input) # [bs, trian_t, dim]
            emb_history = emb_history + t_pe
            
        else:
            t_emb = self.get_time_embedding(time, 'absolute') # [bs, times, dim] 
            y_emb = torch.tile(label.unsqueeze(-1), (1,1, self.node_dim)) + t_emb 
            node_emb = self.node_dist._get_node_embedding()[item] # [bs, times, dim]      
            emb_input = torch.cat([node_emb, y_emb], dim=-1) # [bs, times, dim*2]
            emb_history = self.infer_network_emb(emb_input)
        
        return emb_history
    
    
    def inference_process(
        self,
        emb_history: torch.Tensor,
        feed_dict: Dict[str, torch.Tensor] = None,
        eval: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the inference process to sample continuous hidden variables.

        Args:
            emb_history (torch.Tensor): The embedding history tensor of shape [batch_size, trian_t, dim].
            feed_dict (Optional[Dict[str, torch.Tensor]], optional): A dictionary containing additional input tensors.
                Defaults to None.
            eval (bool, optional): A boolean flag indicating whether to run in evaluation mode. 
                Evaluation mode may change the behavior of certain operations like dropout. 
                Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sampled continuous hidden variables
            `qs_dist` (shape [batch_size, trian_t, dim]) and `qz_dist` (shape [batch_size, trian_t, dim]).
        """
        
        # sample continuous hidden variable from `q(s[1:T] | y[1:T])' 
        qs_dist = self.st_transition_infer(emb_inputs=emb_history, eval=eval)
        
        # sample continuous hidden variable from `q(z[1:T] | y[1:T])'
        qz_dist  = self.zt_transition_infer(
            feed_dict=feed_dict, emb_inputs=emb_history, eval=eval
        ) 
        
        return qs_dist, qz_dist

        
    def zt_transition_infer(
        self, 
        inputs: Tuple[torch.Tensor, torch.Tensor],
        num_sample: int,
        emb_inputs: Optional[torch.Tensor] = None,
    ):
        """
        Compute the posterior distribution of `z_t` using an inference network.

        Args:
            feed_dict (Tuple[torch.Tensor, torch.Tensor]): A tuple of tensors containing the input data.
                The first tensor is of shape [batch_size, times], and the second tensor is of shape [batch_size, times].
            emb_inputs (Optional[torch.Tensor], optional): An optional tensor representing additional embeddings.
                Defaults to None.
            eval (bool, optional): A boolean flag indicating whether to run in evaluation mode. 
                Evaluation mode may change the behavior of certain operations like dropout. 
                Defaults to False.

        Returns:
            torch.distributions.MultivariateNormal: The posterior distribution of `z_t` with mean and covariance matrix.
                The mean has shape [batch_size, times, num_node], and the covariance matrix has shape [batch_size, times, num_node, num_node].
        """
        
        # Compute the output of the posterior network
        self.infer_network_posterior_z.flatten_parameters() # useful when using DistributedDataParallel (DDP)
        qz_emb_out, _ = self.infer_network_posterior_z(emb_inputs, None) # [bs, times, dim*2]
        
        # Compute the mean and covariance matrix of the posterior distribution of `z_t`
        qz_mean, qz_log_var = self.infer_network_posterior_mean_var_z(qz_emb_out)

        qz_log_var = torch.minimum(qz_log_var, self.var_log_max.to(qz_log_var.device))
        qz_cov_mat = torch.diag_embed(torch.exp(qz_log_var) + EPS)  # [bs, times, num_node, num_node]
        qz_dist = MultivariateNormal(
            loc=qz_mean, 
            scale_tril=torch.tril(qz_cov_mat)
        ) # [bs, times, num_node]; [bs, times, num_node, num_node]
        
        if not eval:
            self.register_buffer(name="qz_mean", tensor=qz_mean.clone().detach())
            self.register_buffer(name="qz_var", tensor=torch.exp(qz_log_var).clone().detach())
            
        return qz_dist
    
    
    def forward(
        self, 
        feed_dict: Dict[str, torch.Tensor], 
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input tensors, 
                including 'time_seq' (shape [batch_size, times]), 'label_seq' (shape [batch_size, times]), 
                and 'skill_seq' (shape [batch_size]).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the computed objective values.
        """
        
        # Embed the input sequence
        t_train = feed_dict['time_seq']
        y_train = feed_dict['label_seq']
        item_train = feed_dict['skill_seq']
        emb_history = self.embedding_process(time=t_train, label=y_train, item=item_train)
        
        # Compute the posterior distribution of `s_t` and `z_t`
        qs_dist, qz_dist = self.inference_process(emb_history, feed_dict)
        
        # Compute the prior distribution of `s_t` and `z_t`
        ps_dist, pz_dist = self.generative_process(qs_dist, qz_dist, feed_dict)

        return_dict = self.get_objective_values(
            [qs_dist, qz_dist],
            [ps_dist, pz_dist], 
            feed_dict,
        )

        self.register_buffer(name="output_emb_input", tensor=emb_history.clone().detach())

        return return_dict
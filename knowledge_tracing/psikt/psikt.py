import math, argparse
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy
from collections import defaultdict

import torch
from torch import nn, distributions
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F

from knowledge_tracing.psikt import T_SCALE, EPS, COV_MIN
from knowledge_tracing.psikt.modules import build_dense_network, VAEEncoder
from knowledge_tracing.psikt.psikt_graph_representation import VarTransformation
from knowledge_tracing.psikt.GMVAE.gmvae import *
from knowledge_tracing.utils.logger import Logger
from knowledge_tracing.baseline.basemodel import BaseModel


class PSIKT(BaseModel):
    """
    PSI Knowledge Tracing Model.

    This class represents a PSI Knowledge Tracing model used for training with different modes.

    Args:
        mode (str): The training mode. Examples include 'train' and 'ls_split_time'.
        num_node (int): The number of nodes in the graph.
        nx_graph (nx.Graph or None): The graph adjacency matrix (None for synthetic mode).
        device (torch.device or None): The device for computations (None for default).
        args (argparse.Namespace or None): Command-line arguments (None for default).
        logs: Logs for the model.

    Usage:
    >>> psi_kt_model = PSIKT(mode, num_node, nx_graph, device, args, logs)
    """

    def __init__(
        self,
        mode: str = "train",
        num_node: int = 1,
        nx_graph=None,
        device: torch.device = None,
        args: argparse.Namespace = None,
        logs=None,
    ):
        self.logs = logs
        self.device = device
        self.args = args
        self.num_sample = args.num_sample
        self.var_log_max = torch.tensor(args.var_log_max)

        # Set the device to use for computations
        self.device = device if device != None else args.device

        # Store the arguments and logs for later use
        self.args = args
        self.logs = logs

        BaseModel.__init__(self, model_path=Path(args.log_path, "Model"))

    @staticmethod
    def _normalize_timestamps(
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalizes timestamps by subtracting the mean and dividing by the standard deviation.
        Args:
            timestamps (torch.Tensor): Input timestamps of shape [bs, T, ...].

        Returns:
            torch.Tensor: Normalized timestamps of the same shape as the input.
        """
        mean_val = torch.mean(timestamps, dim=1, keepdim=True)
        std_val = torch.std(timestamps, dim=1, keepdim=True)
        normalized_timestamps = (timestamps - mean_val) / std_val
        return normalized_timestamps

    @staticmethod
    def _construct_normal_from_mean_std(
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> distributions.MultivariateNormal:
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
            dist = distributions.MultivariateNormal(mean, std * std + EPS)
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
            cov_pdm = torch.matmul(std, std.transpose(-1, -2)) + EPS
            # try:
            #     torch.linalg.cholesky(cov_pdm)
            # except RuntimeError:
            #     # if the covariance matrix is not positive definite, add a small positive constant to its diagonal
            #     cov_pdm = cov_pdm + torch.eye(cov_pdm.size(-1)) * EPS

            # create a multivariate normal distribution
            dist = distributions.MultivariateNormal(
                mean, scale_tril=torch.tril(cov_pdm)
            )
        return dist

    @staticmethod
    def _positional_encoding1d(dim: int, length: int, actual_time=None) -> torch.Tensor:
        """
        Modified based on https://github.com/wzlxjtu/PositionalEncoding2D
        Args:
            d_model: dimension of the model
            length: length of positions

        Returns:
            length*d_model position matrix
        """
        if actual_time == None:
            position = torch.arange(0, length).unsqueeze(1)

        else:
            device = actual_time.device
            if dim % 2 != 0:
                raise ValueError(
                    "Cannot use sin/cos positional encoding with "
                    "odd dim (got dim={:d})".format(dim)
                )
            pe = torch.zeros(actual_time.shape[0], length, dim, device=device)
            position = actual_time.unsqueeze(-1)  # [bs, times, 1]

        div_term = (
            torch.exp(
                (
                    torch.arange(0, dim, 2, dtype=torch.float)
                    * -(math.log(10000.0) / dim)
                )
            )
            .reshape(1, 1, -1)
            .to(device)
        )
        pe[..., 0::2] = torch.sin(position.float() * div_term)
        pe[..., 1::2] = torch.cos(position.float() * div_term)

        return pe

    def get_time_embedding(
        self,
        time: torch.Tensor,
        type: str = "dt",
    ) -> torch.Tensor:
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
        if type == "dt":
            dt = torch.diff(time, dim=1)
            t_pe = self._positional_encoding1d(
                self.node_dim, dt.shape[1], dt
            )  # [bs, times, dim]
        elif type == "absolute":
            norm_t = time
            t_pe = self._positional_encoding1d(
                self.node_dim, time.shape[1], norm_t
            )  # [bs, times, dim]
        return t_pe

    def get_reconstruction(
        self,
        hidden_state_sequence: torch.Tensor,
        observation_shape: torch.Size = None,
        sample_for_reconstruction: bool = True,
        sample_hard: bool = False,
    ) -> torch.Tensor:
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
    ) -> dict:
        """
        Calculate objective values for the model.

        Args:
            log_probs (List[torch.Tensor]): List of log probabilities for different components.
            log_prob_q (torch.Tensor, optional): Log probability q.
            posterior_entropies (List[torch.Tensor], optional): List of posterior entropies.
            temperature (float, optional): Temperature parameter.

        Returns:
            dict: A dictionary containing objective values including ELBO and others.
        """
        temperature = 0.01
        w_s, w_z, w_y = 1.0, 1.0, 1.0
        [log_prob_st, log_prob_zt, log_prob_yt] = log_probs

        sequence_likelihood = (
            w_s * log_prob_st[:, 1:]
            + w_z * log_prob_zt[:, 1:]
            + w_y * log_prob_yt[:, 1:]
        ).mean(
            -1
        ) / 3  # [bs,]
        initial_likelihood = (
            log_prob_st[:, 0] + log_prob_zt[:, 0] + log_prob_yt[:, 0]
        ) / 3

        t1_mean = torch.mean(sequence_likelihood, dim=0)
        t2_mean = torch.mean(initial_likelihood, dim=0)

        t3_mean = torch.mean(posterior_entropies[0])
        t4_mean = torch.mean(posterior_entropies[1])

        elbo = t1_mean + t2_mean + temperature * (t3_mean + t4_mean)

        return dict(
            elbo=elbo,
            initial_likelihood=t2_mean,
            sequence_likelihood=t1_mean,
            st_entropy=t3_mean,
            zt_entropy=t4_mean,
        )

    def loss(
        self,
        feed_dict: Dict[str, torch.Tensor],
        outdict: Dict[str, torch.Tensor],
        metrics: List[str] = None,
    ):
        pass


class AmortizedPSIKT(PSIKT):
    """
    An instance of AmortizedPSIKT.

    Args:
        mode (str): The mode for initialization (default: "ls_split_time").
        num_node (int): The number of nodes (default: 1).
        args (argparse.Namespace): Command-line arguments (default: None).
        device (torch.device): The device to use (default: torch.device("cpu")).
        logs (Logger): Logger object for logging (default: None).
        nx_graph (np.ndarray): NumPy array for the graph (default: None).
    """

    def __init__(
        self,
        mode: str = "ls_split_time",
        num_node: int = 1,
        args: argparse.Namespace = None,
        device: torch.device = torch.device("cpu"),
        logs: Logger = None,
        nx_graph: numpy.ndarray = None,
    ):
        self.num_node = num_node

        # specify dimensions of all latents
        self.node_dim = args.node_dim
        self.emb_mean_var_dim = 16

        self.var_log_max = torch.tensor(args.var_log_max)
        self.num_category = args.num_category
        self.learned_graph = args.learned_graph

        # initialize graph parameters
        if self.learned_graph == "none" or self.num_node == 1:
            self.dim_s, self.dim_z = 3, 1
        else:
            self.dim_s, self.dim_z = 4, 1
            self.adj = torch.tensor(nx_graph)
            assert self.adj.shape[-1] >= num_node

        self.qs_temperature = 1.0
        self.qs_hard = 0

        super().__init__(mode, num_node, nx_graph, device, args, logs)

    def _init_params(self, shape) -> None:
        """
        Initialize the parameters of the model.

        This function creates the necessary layers of the model and initializes their parameters.

        Returns:
            None
        """
        param = nn.init.xavier_uniform_(torch.empty(shape))
        return nn.Parameter(param, requires_grad=True)

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model.

        This function creates the necessary layers of the model and initializes their weights.

        Returns:
            None
        """
        self.node_dist = VarTransformation(
            device=self.device,
            num_nodes=self.num_node,
            latent_dim=self.node_dim,
            tau_gumbel=1,
            dense_init=False,
        )

        # --------------- for parameters Theta ---------------
        # the initial distribution p(s0) p(z0), the transition distribution p(s|s') p(z|s,z'), the emission distribution p(y|s,z)
        # ----- 1. initial distribution p(s0) p(z0): trainable mean and variance??? -----
        self.gen_s0_mean, self.gen_s0_log_var = self._initialize_gaussian_mean_log_var(
            self.dim_s, True
        )
        self.gen_z0_mean, self.gen_z0_log_var = self._initialize_gaussian_mean_log_var(
            self.dim_z, True
        )

        # ----- 2. transition distribution p(s|s') or p(s|s',y',c'); p(z|s,z') (OU) -----
        gen_st_h = nn.init.xavier_uniform_(torch.empty(1, self.dim_s))
        self.gen_st_h = nn.Parameter(torch.diag_embed(gen_st_h)[0], requires_grad=True)
        self.gen_st_log_r = self._init_params((1, self.dim_s))

        # ----- 3. emission distribution p(y|z) -----
        self.y_emit = torch.nn.Sigmoid()

        # --------------- for parameters Phi ---------------
        # the embedding network at each time step emb_t = f(y_t, c_t, t)
        # the variational posterior distribution q(s_1:t | y_1:t, c_1:t) and q(z_1:t | y_1:t, c_1:t)
        # ----- 1. embedding network -----
        self.infer_network_emb = build_dense_network(
            self.node_dim * 2, [self.node_dim, self.node_dim], [nn.LeakyReLU(0.2), None]
        )

        # ----- 2. variational posterior distribution q(s_1:t | y_1:t, c_1:t) = q(s_1:t | emb_1:t) -----
        time_step = int(self.args.max_step * self.args.train_time_ratio)
        self.infer_network_posterior_s = InferenceNet(
            self.node_dim, self.dim_s, self.num_category, time_step
        )

        # ----- 3. variational posterior distribution q(z_1:t | y_1:t, c_1:t) -----
        self.infer_network_posterior_z = nn.LSTM(
            input_size=self.node_dim,
            hidden_size=self.node_dim * 2,
            bidirectional=False,
            batch_first=True,
        )
        self.infer_network_posterior_mean_var_z = VAEEncoder(
            self.node_dim * 2, self.node_dim, self.num_node
        )

    def _initialize_gaussian_mean_log_var(
        self, dim: int, use_trainable_cov: bool, num_sample: int = 1
    ) -> Tuple[nn.Parameter, nn.Parameter]:
        """
        Construct the initial mean and covariance matrix for the multivariate Gaussian distribution.

        Args:
            dim: an integer representing the dimension of the Gaussian distribution
            use_trainable_cov: a boolean indicating whether to use a trainable covariance matrix
            num_sample: an integer representing the number of samples to generate

        Returns:
            x0_mean: a PyTorch parameter representing the initial mean of the Gaussian distribution
            x0_scale: a PyTorch parameter representing the initial covariance matrix of the Gaussian distribution
        """
        x0_mean = self._init_params((num_sample, dim))

        x0_log_var = torch.ones((num_sample, dim)) * torch.log(torch.tensor(COV_MIN))
        x0_log_var = nn.Parameter(x0_log_var, requires_grad=use_trainable_cov)

        return x0_mean, x0_log_var

    def st_transition_gen(
        self,
        qs_dist: MultivariateNormal,
        eval: bool = False,
    ) -> MultivariateNormal:
        """
        Generate state transitions based on a given multivariate normal distribution.

        This function computes state transitions for a sequence of states based on the
        provided multivariate normal distribution `qs_dist`. The resulting state
        transitions are returned as a new multivariate normal distribution.

        Args:
            qs_dist (MultivariateNormal): A multivariate normal distribution representing
                the initial states. It contains mean and covariance information.
            eval (bool): A flag indicating whether to perform evaluation mode (default: False).

        Returns:
            MultivariateNormal: A multivariate normal distribution representing the state
            transitions. It contains mean and covariance information for the generated states.
        """
        qs_mean = qs_dist.mean  # [bs, 1, time, dim_s]
        qs_cov_mat = qs_dist.covariance_matrix  # [bs, 1, time, dim_s, dim_s]
        device = qs_mean.device
        bs = qs_mean.shape[0]

        # retreive initalized variables for p(s0)
        ps0_cov_mat = torch.diag_embed(torch.exp(self.gen_s0_log_var.to(device)) + EPS)
        ps0_mean = self.gen_s0_mean.to(device)

        # -- 2. prior of single step of H, R --
        pst_mean = qs_mean[:, :, :-1] @ self.gen_st_h  # [bs, 1, time-1, dim_s]
        pst_transition_var = torch.exp(self.gen_st_log_r)
        pst_transition_cov_mat = torch.diag_embed(pst_transition_var + EPS)
        pst_cov_mat = (
            self.gen_st_h @ qs_cov_mat[:, :, :-1] @ self.gen_st_h.transpose(-1, -2)
            + pst_transition_cov_mat
        )  # [bs, 1, time-1, dim_s, dim_s]

        # concatenate sequential st and initial s0
        ps0_mean_bs = ps0_mean.reshape(1, 1, 1, self.dim_s).repeat(bs, 1, 1, 1)
        ps_mean = torch.cat([ps0_mean_bs, pst_mean], dim=-2)  # [bs, 1, time, dim_s]
        ps0_cov_mat_mc = ps0_cov_mat.reshape(1, 1, 1, self.dim_s, self.dim_s).repeat(
            bs, 1, 1, 1, 1
        )
        ps_cov_mat = torch.cat(
            [ps0_cov_mat_mc, pst_cov_mat], dim=-3
        )  # [bs, 1, time, dim_s, dim_s]

        ps_dist = MultivariateNormal(loc=ps_mean, scale_tril=torch.tril(ps_cov_mat))

        if not eval:  # For debugging
            self.register_buffer("ps_mean", ps_mean.clone().detach())
            self.register_buffer("ps_cov_mat", ps_cov_mat.clone().detach())

        return ps_dist

    def zt_transition_gen(
        self,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = 0,
        qs_dist: distributions.MultivariateNormal = None,
        qz_dist: distributions.MultivariateNormal = None,
        eval: bool = False,
        qs_sampled: torch.Tensor = None,
    ):
        """
        Generate state transitions in a dynamic system based on OU process.

        This function computes state transitions for a dynamic system based on the
        provided input data and latent variables. It models the transition process
        and returns a distribution representing the generated states.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input data tensors.
            idx (int, optional): An index parameter used to limit the number of time steps (default: 0).
            qs_dist (MultivariateNormal, optional): A multivariate normal distribution representing the latent variables qs (default: None).
            qz_dist (MultivariateNormal, optional): A multivariate normal distribution representing the latent variables qz (default: None).
            eval (bool, optional): A flag indicating whether to perform evaluation mode (default: False).
            qs_sampled (torch.Tensor, optional): A tensor representing sampled qs values (default: None).

        Returns:
            MultivariateNormal: A multivariate normal distribution representing the state transitions.
                It contains mean and covariance information for the generated states.
        """
        if idx:
            input_t = feed_dict["time_seq"][:, :idx]
        else:
            input_t = feed_dict["time_seq"]
        bs, num_steps = input_t.shape
        device = input_t.device

        # -- p(z0) --
        pz0_mean = self.gen_z0_mean.to(device)
        pz0_var = torch.exp(self.gen_z0_log_var.to(device))

        # -- p(z_n | z_{n-1}, s_n) --
        # calculate time difference dt
        dt = (
            torch.diff(input_t, dim=-1).unsqueeze(-1) / T_SCALE + EPS
        )  # [bs, num_steps-1, 1]

        # retreive variables from qs_dist.mean
        # z_{n-1}
        qz_mean = qz_dist.mean[:, :-1]  # [bs, time-1, num_node]
        # s_n and its disentangled elements
        qs_mean = qs_dist.mean[:, 0, 1:]  # [bs, time-1, dim_s]
        q_alpha = torch.relu(qs_mean[..., 0:1]) + EPS
        q_mu = qs_mean[..., 1:2]  # torch.tanh(qs_mean[..., 1:2]) # 
        q_sigma = qs_mean[
            ..., 2:3
        ]  # TODO  q_sigma = torch.minimum(qs_mean[..., 2:3], self.var_log_max.to(device))
        q_gamma = torch.sigmoid(qs_mean[..., 3:4])  # torch.zeros_like(q_sigma)

        # calculate useful variables
        # exp(-alpha * dt)
        pz_ou_decay = torch.exp(-q_alpha * dt)  # [bs, num_steps-1, 1]
        # empower^{\ell,k}_n = gamma^\ell_n * \sum_{i=1}^K (a^{ik} * (z^{\ell,k}_{n-1})) * (1/num_node)
        pz_graph_adj = (
            self.node_dist.sample_A(self.num_sample)[-1][:, 0].mean(0).to(device)
        )  # adj_ij means i has influence on j
        pz_empower = q_gamma * (qz_mean @ pz_graph_adj) / self.num_node
        # mu^{\ell,k}_n = q_mu^\ell_n + empower^{\ell,k}_n
        pz_empowered_mu = q_mu + pz_empower  # [bs, time-1, num_node]

        # OU process
        # mean m^{\ell,k}_n = mu^{\ell,k}_n * (1 - exp(-alpha * dt)) + m^{\ell,k}_{n-1} * exp(-alpha * dt)
        #                   = pz_empowered_mu * (1 - pz_ou_decay) + qz_mean * pz_ou_decay
        pz_ou_mean = (
            pz_ou_decay * qz_mean + (1 - pz_ou_decay) * pz_empowered_mu
        )  # [bs, time-1, num_node]
        # var v^{\ell,k}_n = (sigma^{\ell,k}_n)^2 * (1 - exp(-2 * alpha * dt)) / (2 * alpha)
        pz_ou_var = (
            q_sigma * q_sigma * (1 - pz_ou_decay * pz_ou_decay) / (2 * q_alpha + EPS)
            + EPS
        )  # [bs, num_steps-1, 1]

        # pz_dist
        pz0_mean_mc = pz0_mean.reshape(1, 1, 1).repeat(bs, 1, self.num_node)
        pz_mean = torch.cat([pz0_mean_mc, pz_ou_mean], dim=1)  # [bs, time, num_node]
        pz0_var_mc = pz0_var.reshape(1, 1, 1).repeat(bs, 1, 1)
        pz_var = torch.cat([pz0_var_mc, pz_ou_var], dim=1).repeat(
            1, 1, self.num_node
        )  # [bs, time, num_node]

        pz_dist = MultivariateNormal(
            loc=pz_mean, scale_tril=torch.tril(torch.diag_embed(pz_var + EPS))
        )

        # if qs_sampled is None:
        #     samples = qs_dist.rsample((self.num_sample,)) # [n, bs, time, dim_s]
        #     qs_sampled = samples.transpose(1,0).reshape(bsn, 1, num_steps, self.dim_s)

        if not eval:
            self.register_buffer("pz_decay", pz_ou_decay.clone().detach())
            self.register_buffer("pz_empower", pz_empower.clone().detach())
            self.register_buffer("pz_empowered_mu", pz_empowered_mu.clone().detach())

            self.register_buffer(name="pz_mean", tensor=pz_mean.clone().detach())
            self.register_buffer(name="pz_var", tensor=pz_var.clone().detach())

        return pz_dist

    def st_transition_infer(
        self,
        emb_inputs: torch.Tensor,
        num_sample: int = 0,
        eval: bool = False,
    ) -> torch.distributions.MultivariateNormal:
        """
        Perform state transition inference.

        Args:
            emb_inputs (torch.Tensor): Embedding inputs with dtype torch.Tensor.
            num_sample (int, optional): Number of samples (default: 0).
            eval (bool, optional): Flag to indicate evaluation mode (default: False).

        Returns:
            torch.distributions.MultivariateNormal: Multivariate Gaussian distribution.
        """

        num_sample = self.num_sample if num_sample == 0 else num_sample

        qs_out_inf = self.infer_network_posterior_s(
            emb_inputs,
            self.qs_temperature,
            self.qs_hard,
        )

        s_category = qs_out_inf["categorical"]  # [bs, 1, num_cat]
        s_mean = qs_out_inf["s_mu_infer"]  # [bs, 1, time, dim_s]
        s_var = qs_out_inf["s_var_infer"]  # [bs, 1, time, dim_s]

        s_var_mat = torch.diag_embed(s_var + EPS)  # [bs, 1, time, dim_s, dim_s]
        qs_dist = MultivariateNormal(loc=s_mean, scale_tril=torch.tril(s_var_mat))

        # NOTE: For debug use
        if not eval:
            self.register_buffer(
                "qs_category_logits", qs_out_inf["logits"].clone().detach()
            )
            self.register_buffer(name="qs_mean", tensor=s_mean.clone().detach())
            self.register_buffer(name="qs_var", tensor=s_var.clone().detach())
            self.logits = qs_out_inf["logits"]
            self.probs = qs_out_inf["prob_cat"]
            self.s_category = s_category
        self.register_buffer("qs_category", s_category.clone().detach())

        return qs_dist

    def zt_transition_infer(
        self,
        feed_dict: Tuple[torch.Tensor, torch.Tensor],
        emb_inputs: Optional[torch.Tensor] = None,
        eval: bool = False,
    ) -> MultivariateNormal:
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
        self.infer_network_posterior_z.flatten_parameters()  # useful when using DistributedDataParallel (DDP)
        qz_emb_out, _ = self.infer_network_posterior_z(
            emb_inputs, None
        )  # [bs, times, dim*2]

        # Compute the mean and covariance matrix of the posterior distribution of `z_t`
        qz_mean, qz_log_var = self.infer_network_posterior_mean_var_z(qz_emb_out)

        qz_log_var = torch.minimum(qz_log_var, self.var_log_max.to(qz_log_var.device))
        qz_cov_mat = torch.diag_embed(
            torch.exp(qz_log_var) + EPS
        )  # [bs, times, num_node, num_node]
        qz_dist = MultivariateNormal(
            loc=qz_mean, scale_tril=torch.tril(qz_cov_mat)
        )  # [bs, times, num_node]; [bs, times, num_node, num_node]

        if not eval:
            self.register_buffer(name="qz_mean", tensor=qz_mean.clone().detach())
            self.register_buffer(
                name="qz_var", tensor=torch.exp(qz_log_var).clone().detach()
            )

        return qz_dist

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
            t_pe = self.get_time_embedding(time, "absolute")  # [bs, times, dim]
            y_pe = torch.tile(label.unsqueeze(-1), (1, 1, self.node_dim))  # + t_pe
            node_pe = self.node_dist._get_node_embedding()[item]  # [bs, times, dim]
            emb_input = torch.cat([node_pe, y_pe], dim=-1)  # [bs, times, dim*2]
            self.infer_network_emb.flatten_parameters()
            emb_history, _ = self.infer_network_emb(emb_input)  # [bs, trian_t, dim]
            emb_history = emb_history + t_pe

        else:
            t_emb = self.get_time_embedding(time, "absolute")  # [bs, times, dim]
            y_emb = torch.tile(label.unsqueeze(-1), (1, 1, self.node_dim)) + t_emb
            node_emb = self.node_dist._get_node_embedding()[item]  # [bs, times, dim]
            emb_input = torch.cat([node_emb, y_emb], dim=-1)  # [bs, times, dim*2]
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
        qz_dist = self.zt_transition_infer(
            feed_dict=feed_dict, emb_inputs=emb_history, eval=eval
        )

        return qs_dist, qz_dist

    def generative_process(
        self,
        qs_dist: distributions.MultivariateNormal,
        qz_dist: distributions.MultivariateNormal,
        feed_dict: Dict[str, torch.Tensor] = None,
        eval: bool = False,
    ) -> Tuple[distributions.MultivariateNormal, distributions.MultivariateNormal]:
        """
        Perform generative process.

        Args:
            qs_dist (distributions.MultivariateNormal): Multivariate Gaussian distribution for s.
            qz_dist (distributions.MultivariateNormal): Multivariate Gaussian distribution for z.
            feed_dict (Dict[str, torch.Tensor], optional): Dictionary of feed-forward tensors (default: None).
            eval (bool, optional): Flag to indicate evaluation mode (default: False).

        Returns:
            Tuple[distributions.MultivariateNormal, distributions.MultivariateNormal]: Tuple of generative
            distributions for s and z.
        """
        # generative model for s (Karman filter)
        ps_dist = self.st_transition_gen(qs_dist, eval=eval)

        # generative model for z (OU process)
        pz_dist = self.zt_transition_gen(
            qs_dist=qs_dist, qz_dist=qz_dist, feed_dict=feed_dict, eval=eval
        )

        return ps_dist, pz_dist

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
        t_train = feed_dict["time_seq"]
        y_train = feed_dict["label_seq"]
        item_train = feed_dict["skill_seq"]
        emb_history = self.embedding_process(
            time=t_train, label=y_train, item=item_train
        )

        # Compute the posterior distribution of `s_t` and `z_t`
        qs_dist, qz_dist = self.inference_process(emb_history, feed_dict)

        # Compute the prior distribution of `s_t` and `z_t`
        ps_dist, pz_dist = self.generative_process(qs_dist, qz_dist, feed_dict)

        return_dict = self.get_objective_values(
            [qs_dist, qz_dist],
            [ps_dist, pz_dist],
            feed_dict,
        )

        self.register_buffer(
            name="output_emb_input", tensor=emb_history.clone().detach()
        )

        return return_dict

    def predictive_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform predictive modeling.

        Args:
            feed_dict (Dict[str, torch.Tensor]): Dictionary containing input data with dtype torch.Tensor.
        """
        t_all = feed_dict["time_seq"]
        y_all = feed_dict["label_seq"]
        item_all = feed_dict["skill_seq"]
        bs, time_step = t_all.shape
        bsn = bs * self.num_sample

        max_step = self.args.max_step
        train_step = int(max_step * self.args.train_time_ratio)
        test_step = int(max_step * self.args.test_time_ratio)

        emb_history = self.embedding_process(
            time=t_all[:, :train_step],
            label=y_all[:, :train_step],
            item=item_all[:, :train_step],
        )

        qs_dist, qz_dist = self.inference_process(emb_history, eval=True)

        # analytical solution
        pred_s_mean, pred_s_var, pred_z_mean, pred_z_var = [], [], [], []
        s_last_mean = qs_dist.mean[:, 0, -1:]  # [bs, 1, dim_s]
        s_last_cov_mat = qs_dist.covariance_matrix[:, 0, -1:]  # [bs, 1, dim_s, dim_s]
        st_tran_r = torch.diag_embed(torch.exp(self.gen_st_log_r) + EPS)
        z_last_mean = qz_dist.mean[:, -1:]  # [bs, 1, num_node]
        pz_graph_adj = (
            self.node_dist.sample_A(self.num_sample)[-1][:, 0]
            .mean(0)
            .to(z_last_mean.device)
        )  # # adj =  torch.exp(self.node_dist.edge_log_probs()[0]).to(sampled_s.device) # adj_ij means i has influence on j
        dt = (
            torch.diff(t_all[:, -test_step - 1 :], dim=-1).unsqueeze(-1) / T_SCALE + EPS
        )  # [bs, num_steps-1, 1]
        for i in range(test_step):
            # p(st-1) = N(m, P), p(st|st-1) = N(st|H*st-1 + b, R)
            # p(st) = N(st|H*m + b, H*P*H' + R)
            s_next_mean = s_last_mean @ self.gen_st_h  # [bs, 1, dim_s]
            s_next_cov_mat = (
                self.gen_st_h @ s_last_cov_mat @ self.gen_st_h.transpose(-1, -2)
                + st_tran_r
            )  # [bs, 1, dim_s, dim_s]
            pred_s_mean.append(s_next_mean)
            pred_s_var.append(s_next_cov_mat)
            s_last_mean = s_next_mean
            s_last_cov_mat = s_next_cov_mat

            # p(zt) = N(zt|zt-1, st)
            q_alpha = torch.relu(s_next_mean[..., 0:1]) + EPS
            q_mu = s_next_mean[..., 1:2]  # torch.tanh(s_next_mean[..., 1:2])  # 
            q_sigma = s_next_mean[..., 2:3]  # [bs, 1, 1]
            q_gamma = torch.sigmoid(
                s_next_mean[..., 3:4]
            )  # torch.zeros_like(q_sigma) #
            # calculate useful variables
            pz_ou_decay = torch.exp(-q_alpha * dt[:, i : i + 1])  # [bs, 1, 1]
            pz_ou_var = (
                q_sigma
                * q_sigma
                * (1 - pz_ou_decay * pz_ou_decay)
                / (2 * q_alpha + EPS)
            )  # [bs, num_steps-1, 1]
            pz_empower = (z_last_mean @ pz_graph_adj) / self.num_node * q_gamma
            pz_empowered_mu = q_mu + pz_empower  # [bs, time-1, num_node]
            pz_ou_mean = (
                pz_ou_decay * z_last_mean + (1 - pz_ou_decay) * pz_empowered_mu
            )  # [bs, 1, num_node]
            z_last_mean = pz_ou_mean
            pred_z_mean.append(pz_ou_mean)
            pred_z_var.append(pz_ou_var)

        pred_s_mean = torch.cat(pred_s_mean, dim=1)  # [bs, time, dim_s]
        pred_s_cov_mat = torch.cat(pred_s_var, dim=1)  # [bs, time, dim_s, dim_s]
        pred_z_mean = torch.cat(pred_z_mean, dim=1)  # [bs, time, num_node]
        pred_z_var = torch.cat(pred_z_var, dim=1).repeat(
            1, 1, self.num_node
        )  # [bs, time, num_node]

        pred_z_dist = MultivariateNormal(
            loc=pred_z_mean, scale_tril=torch.tril(torch.diag_embed(pred_z_var + EPS))
        )
        pred_z_sampled = pred_z_dist.sample(
            (self.num_sample,)
        )  # [n, bs, time, num_node]
        pred_z_sampled = pred_z_sampled.transpose(1, 0).reshape(
            bsn, test_step, self.num_node
        )  # [bsn, time, num_node]
        pred_z_sampled = pred_z_sampled.transpose(
            -1, -2
        ).contiguous()  # [bsn, num_node, time]

        item_test = item_all[:, -test_step:]
        item_test_mc = (
            item_test.unsqueeze(1)
            .repeat(1, self.num_sample, 1)
            .reshape(bsn, 1, test_step)
        )  # [bsn, 1, time]
        pred_z_sampled_item = (
            torch.gather(pred_z_sampled, 1, item_test_mc).transpose(-1, -2).contiguous()
        )  # [bsn, time, 1]

        # # here are Karman filter
        # ps_sampled_future = []
        # qs_sampled = qs_dist.rsample((self.num_sample,)) # [n, bs, 1, time, dim_s]
        # qs_sampled = qs_sampled.transpose(1,0).reshape(-1, 1, train_step, self.dim_s)
        # ps_prev = qs_sampled[:,:,-1:]
        # for i in range(10): # TODO
        #     ps_next = ps_prev @ self.gen_st_h + self.gen_st_b
        #     ps_sampled_future.append(ps_next)
        #     ps_prev = ps_next
        # ps_sampled_future = torch.cat(ps_sampled_future, dim=-2)
        # s_sampled = torch.cat([qs_sampled, ps_sampled_future], dim=-2) # [bsn, 1, time, dim_s]
        # z_prior_dist = self.zt_transition_gen(qs_sampled = s_sampled, feed_dict=feed_dict, eval=True)
        # sampled_scalar_z = z_prior_dist.rsample()[:,:,-test_step:]
        # bsn = sampled_scalar_z.shape[0]
        # items = item_all.unsqueeze(1).repeat(1, self.num_sample, 1).reshape(bsn, 1, -1)[:,:,-test_step:] # [bsn, 1, time]
        # sampled_scalar_z_item = torch.gather(sampled_scalar_z[..., 0], 1, items).transpose(-1,-2).contiguous() # [bsn, time, 1]

        pred_y_test = self.y_emit(pred_z_sampled_item)
        pred = pred_y_test.reshape(bs, self.num_sample, test_step)
        mc_label = y_all[:, -test_step:].unsqueeze(1).repeat(1, self.num_sample, 1)

        return {
            "prediction": pred,
            "label": mc_label,
        }

    def get_objective_values(
        self,
        q_dists: Tuple[
            distributions.MultivariateNormal, distributions.MultivariateNormal
        ],
        p_dists: Tuple[
            distributions.MultivariateNormal, distributions.MultivariateNormal
        ],
        feed_dict: Dict[str, torch.Tensor],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate objective values.

        Args:
            q_dists (Tuple[distributions.MultivariateNormal, distributions.MultivariateNormal]):
                Tuple of Multivariate Gaussian distributions for q (qs_dist, qz_dist).
            p_dists (Tuple[distributions.MultivariateNormal, distributions.MultivariateNormal]):
                Tuple of Multivariate Gaussian distributions for p (ps_dist, pz_dist).
            feed_dict (Dict[str, torch.Tensor]): Dictionary containing input data with dtype torch.Tensor.
            temperature (float, optional): Temperature parameter (default: 1.0).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of st_log_prob, zt_log_prob, yt_log_prob.
        """

        qs_dist, qz_dist = q_dists
        ps_dist, pz_dist = p_dists

        item = feed_dict["skill_seq"]
        label = feed_dict["label_seq"]
        bs, time_step = item.shape
        bsn = bs * self.num_sample

        # st_log_prob
        qs_sampled = qs_dist.rsample(
            (self.num_sample,)
        )  # [num_sample, bs, 1, time, dim_s]
        qs_log_prob = ps_dist.log_prob(qs_sampled)  # [num_sample, bs, 1, time]
        qs_log_prob = qs_log_prob.reshape(-1, time_step).squeeze(1)  # [bs, time]

        # zt_log_prob
        qz_sampled = qz_dist.rsample(
            (self.num_sample,)
        )  # [num_sample, bs, time, num_node]
        qz_log_prob = pz_dist.log_prob(qz_sampled)  # [num_sample, bs, time]
        qz_log_prob = qz_log_prob.reshape(-1, time_step).squeeze(1)  # [bs, time]

        # yt_log_prob
        items = (
            item.unsqueeze(1).repeat(1, self.num_sample, 1).reshape(bsn, 1, -1)
        )  # [bsn, 1, time]
        qz_sampled = (
            qz_sampled.permute(1, 0, 3, 2).contiguous().reshape(bsn, self.num_node, -1)
        )  # [bsn, num_node, time]
        qz_sampled_item = (
            torch.gather(qz_sampled, 1, items).transpose(-1, -2).contiguous()
        )  # [bsn, time, 1]

        y_prob_train = self.y_emit(qz_sampled_item)
        y_dist_train = torch.distributions.bernoulli.Bernoulli(probs=y_prob_train)
        y_train_mc = (
            label.unsqueeze(1).repeat(1, self.num_sample, 1).reshape(bsn, -1, 1).float()
        )
        yt_log_prob = y_dist_train.log_prob(y_train_mc)  # [bsn, time, 1]
        yt_log_prob = yt_log_prob.squeeze(-1)

        recon_inputs = self.y_emit(qz_sampled)  # [bsn, num_node, time]
        recon_inputs = recon_inputs.reshape(
            bs, self.num_sample, self.num_node, time_step, -1
        )
        recon_inputs_items = y_prob_train.reshape(bs, self.num_sample, 1, time_step, -1)

        if not eval:
            self.register_buffer(
                name="pred_y_all_sampled", tensor=recon_inputs.clone().detach()
            )
            self.register_buffer(
                name="pred_y_sampled", tensor=recon_inputs_items.clone().detach()
            )
            self.register_buffer(name="output_items", tensor=items.clone().detach())
            self.register_buffer(
                name="pred_z_sampled", tensor=qz_sampled_item.clone().detach()
            )

        temp_s, temp_z = self.args.s_entropy_weight, self.args.z_entropy_weight
        w_s, w_z, w_y = (
            self.args.s_log_weight,
            self.args.z_log_weight,
            self.args.y_log_weight,
        )
        sequence_likelihood = (
            w_s * qs_log_prob[:, 1:]
            + w_z * qz_log_prob[:, 1:] 
            + w_y * yt_log_prob[:, 1:]
        ) / 3  # [bs,]
        initial_likelihood = (
            w_s * qs_log_prob[:, 0]  + w_z * qz_log_prob[:, 0] + w_y * yt_log_prob[:, 0]
        ) / 3

        t1_mean = torch.mean(sequence_likelihood)
        t2_mean = torch.mean(initial_likelihood) * 1e-4

        t3_mean = torch.mean(qs_dist.entropy())
        t4_mean = torch.mean(qz_dist.entropy())  # TODO

        elbo = (
            t1_mean + t2_mean + temp_s * t3_mean + temp_z * t4_mean
        )  # /self.num_node*t4_mean

        return dict(
            elbo=elbo,
            sequence_likelihood=t1_mean,
            initial_likelihood=t2_mean,
            st_log_prob=qs_log_prob.mean(),
            zt_log_prob=qz_log_prob.mean(),
            yt_log_prob=yt_log_prob.mean(),
            st_entropy=t3_mean,
            zt_entropy=t4_mean,
            prediction=recon_inputs_items.reshape(
                bsn, 1, time_step, 1
            ),  # [bsn, 1, time, dim_y]
            label=label[:, None, :, None],  # [bs, 1, time, 1]
            sampled_s=qs_sampled,
        )

    def loss(
        self,
        feed_dict: Dict[str, torch.Tensor],
        outdict: Dict[str, torch.Tensor],
        metrics: List[str] = None,
    ):
        """ """
        
        losses = defaultdict(lambda: torch.zeros(()))  # , device=self.device))

        # Calculate binary cross-entropy loss -> not used for optimization only for visualization
        gt = outdict["label"].repeat(
            1, self.num_sample, 1, 1
        )  # .repeat(self.num_sample, 1)
        pred = outdict["prediction"]

        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred.flatten(), gt.float().flatten())
        losses["loss_bce"] = bceloss

        for key in [
            "elbo",
            "initial_likelihood",
            "sequence_likelihood",
            "st_entropy",
            "zt_entropy",
            "yt_log_prob",
            "zt_log_prob",
            "st_log_prob",
        ]:
            losses[key] = outdict[key].mean()

        # Still NOT for optimization
        edge_log_probs = self.node_dist.edge_log_probs().to(pred.device)
        pred_att = torch.exp(edge_log_probs[0]).to(pred.device)
        pred_adj = torch.nn.functional.gumbel_softmax(edge_log_probs, hard=True, dim=0)[
            0
        ].sum()

        losses["sparsity"] = (pred_att >= 0.5).sum()
        losses["loss_sparsity"] = pred_adj * self.args.sparsity_loss_weight

        if "junyi15" in self.args.dataset:
            gt_adj = self.adj.to(pred.device)
            losses["adj_0_att_1"] = (1 * (pred_att >= 0.5) * (1 - gt_adj)).sum()
            losses["adj_1_att_0"] = (1 * (pred_att < 0.5) * gt_adj).sum()
            self.register_buffer(
                name="output_gt_graph_weights", tensor=gt_adj.clone().detach()
            )

        gmvae_loss = LossFunctions()
        loss_cat = -gmvae_loss.entropy(self.logits, self.probs) - numpy.log(0.1)
        losses["loss_cat"] = loss_cat * self.args.cat_weight

        # loss_cat_in_entropy = gmvae_loss.prior_entropy(self.num_category, self.gen_network_transition_s, self.device)
        # losses['loss_cat_in_entropy'] = loss_cat_in_entropy * self.args.cat_in_entropy_weight
        # loss_f1 = gmvae_loss.f1_regularization(pred, gt)
        # losses['loss_f1'] = loss_f1 * 0. # 1e-2

        losses["loss_total"] = (
            -outdict["elbo"].mean() + losses["loss_sparsity"] + losses["loss_cat"]
        )

        # Register output predictions
        self.register_buffer(name="output_predictions", tensor=pred.clone().detach())
        self.register_buffer(name="output_gt", tensor=gt.clone().detach())
        self.register_buffer(
            name="output_attention_weights", tensor=pred_att.clone().detach()
        )

        # Evaluate metrics
        if metrics != None:
            pred = pred.detach().cpu().data.numpy()
            gt = gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
            for key in evaluations.keys():
                losses[key] = evaluations[key]

        # Calculate mean and variance of the Ornstein-Uhlenbeck process
        losses["ou_speed"] = outdict["sampled_s"][..., 0].mean()
        losses["ou_mean"] = outdict["sampled_s"][..., 1].mean()
        losses["ou_vola"] = outdict["sampled_s"][..., 2].mean()
        if self.dim_s == 4:
            losses["ou_gamma"] = outdict["sampled_s"][..., 3].mean()
        # print(losses)

        return losses


class ContinualPSIKT(AmortizedPSIKT):
    def __init__(
        self,
        mode="train",
        num_node=1,
        num_seq=1,
        args=None,
        device="cpu",
        logs=None,
        nx_graph=None,
    ) -> None:
        super().__init__(mode, num_node, args, device, logs, nx_graph)

        time_step_save = args.max_step
        self.num_seq_save = num_seq

        s_shape = (num_seq, 1, time_step_save, self.dim_s)
        z_shape = (num_seq, 1, time_step_save, self.num_node)
        z_shape_pred = (num_seq, 1, time_step_save, self.num_node)

        self.pred_s_means = nn.Parameter(
            torch.zeros(s_shape, device=self.device), requires_grad=False
        )
        self.pred_s_vars = nn.Parameter(
            torch.zeros(s_shape, device=self.device), requires_grad=False
        )
        self.infer_s_means = nn.Parameter(
            torch.zeros(s_shape, device=self.device), requires_grad=False
        )
        self.infer_s_vars = nn.Parameter(
            torch.zeros(s_shape, device=self.device), requires_grad=False
        )

        self.pred_s_means_update = nn.Parameter(
            torch.zeros(s_shape, device=self.device), requires_grad=False
        )
        self.pred_s_vars_update = nn.Parameter(
            torch.zeros(s_shape, device=self.device), requires_grad=False
        )
        self.infer_s_means_update = nn.Parameter(
            torch.zeros(s_shape, device=self.device), requires_grad=False
        )
        self.infer_s_vars_update = nn.Parameter(
            torch.zeros(s_shape, device=self.device), requires_grad=False
        )

        self.pred_z_means = nn.Parameter(
            torch.zeros(z_shape_pred, device=self.device), requires_grad=False
        )
        self.pred_z_vars = nn.Parameter(
            torch.zeros(z_shape_pred, device=self.device), requires_grad=False
        )
        self.infer_z_means = nn.Parameter(
            torch.zeros(z_shape, device=self.device), requires_grad=False
        )
        self.infer_z_vars = nn.Parameter(
            torch.zeros(z_shape, device=self.device), requires_grad=False
        )

        self.pred_z_means_update = nn.Parameter(
            torch.zeros(z_shape_pred, device=self.device), requires_grad=False
        )
        self.pred_z_vars_update = nn.Parameter(
            torch.zeros(z_shape_pred, device=self.device), requires_grad=False
        )
        self.infer_z_means_update = nn.Parameter(
            torch.zeros(z_shape, device=self.device), requires_grad=False
        )
        self.infer_z_vars_update = nn.Parameter(
            torch.zeros(z_shape, device=self.device), requires_grad=False
        )

        self.var_minimum = torch.log(torch.tensor(1).to(self.device))

        self.infer_network_emb = build_dense_network(
            self.node_dim * 2, [self.node_dim, self.node_dim], [nn.LeakyReLU(0.2), None]
        )

        self.infer_network_posterior_s = nn.LSTM(
            input_size=self.node_dim,
            hidden_size=self.node_dim * 2,
            bidirectional=False,
            batch_first=True,
        )
        self.infer_network_posterior_z = nn.LSTM(
            input_size=self.node_dim,
            hidden_size=self.node_dim * 2,
            bidirectional=False,
            batch_first=True,
        )
        self.infer_network_posterior_mean_var_s = VAEEncoder(
            self.node_dim * 2, self.node_dim, self.dim_s
        )
        # self.infer_network_posterior_mean_var_s = VAEEncoder(
        #     self.node_dim, self.node_dim, self.dim_s
        # )
        # self.infer_network_posterior_mean_var_z = VAEEncoder(
        #     self.node_dim, self.node_dim, self.num_node
        # )

    def st_transition_infer(
        self,
        feed_dict: Dict[str, torch.Tensor] = None,
        emb_inputs: torch.Tensor = None,
        idx: int = None,
    ) -> torch.distributions.MultivariateNormal:
        """
        Compute the posterior distribution of the latent variable `s_t` given the input and output sequences.
        """
        output, _ = self.infer_network_posterior_s(emb_inputs)
        output = output[:, -1:]
        mean, log_var = self.infer_network_posterior_mean_var_s(
            output
        )  # [bs, time_step, dim_s]

        # mean, log_var = self.infer_network_posterior_mean_var_s(emb_inputs)
        log_var = torch.minimum(log_var, self.var_minimum)
        cov_mat = torch.diag_embed(torch.exp(log_var) + EPS)
        dist_s = distributions.multivariate_normal.MultivariateNormal(
            loc=mean, scale_tril=torch.tril(cov_mat)
        )

        return dist_s

    def zt_transition_infer(
        self,
        feed_dict: Dict[str, torch.Tensor] = None,
        emb_inputs: torch.Tensor = None,
        idx: int = None,
    ):
        """
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
        """

        output, _ = self.infer_network_posterior_z(emb_inputs)
        output = output[:, -1:]
        mean, log_var = self.infer_network_posterior_mean_var_z(output)

        # mean, log_var = self.infer_network_posterior_mean_var_z(emb_inputs)  # [batch_size, time_step, dim_s]
        log_var = torch.minimum(log_var, self.var_minimum)
        cov_mat = torch.diag_embed(torch.exp(log_var) + EPS)

        dist_z = distributions.multivariate_normal.MultivariateNormal(
            loc=mean, scale_tril=torch.tril(cov_mat)
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
    ) -> Tuple[
        torch.distributions.MultivariateNormal, torch.distributions.MultivariateNormal
    ]:
        """"""
        user = feed_dict["user_id"]
        y_idx = feed_dict["label_seq"][:, idx : idx + 1]  # [bs, 1]
        device = y_idx.device
        bs = y_idx.shape[0]

        if idx == 0:

            t_idx = feed_dict["time_seq"][:, : idx + 1]  # [bs, 2]
            dt = t_idx / T_SCALE + EPS
            s_tilde_dist = distributions.MultivariateNormal(
                self.gen_s0_mean.unsqueeze(0).repeat(bs, 1, 1),
                scale_tril=torch.tril(
                    torch.diag_embed(torch.exp(self.gen_s0_log_var) + EPS)
                ),
            )
            
            s_tilde_dist_mean = s_tilde_dist.mean  # [num_seq, 1, dim_s]
            s_tilde_dist_var = torch.diagonal(s_tilde_dist.scale_tril, dim1=-2, dim2=-1)

            # p_theta(z_0)
            z_tilde_dist = distributions.MultivariateNormal(
                self.gen_z0_mean.unsqueeze(0).repeat(
                    bs, 1, self.num_node
                ),
                scale_tril=torch.tril(
                    torch.diag_embed(
                        torch.exp(self.gen_z0_log_var.repeat(1, self.num_node)) + EPS
                    )
                ).unsqueeze(0),
            )
            z_tilde_dist_mean = z_tilde_dist.mean # [1, 1, dim_z]
            z_tilde_dist_var = torch.diagonal(z_tilde_dist.scale_tril, dim1=-2, dim2=-1)

        else:

            t_idx = feed_dict["time_seq"][:, idx - 1 : idx + 1]  # [bs, 2]
            dt = torch.diff(t_idx, dim=-1) / T_SCALE + EPS

            if s_prior != None:
                s_prior_mean = s_prior.mean
                s_prior_cov = torch.diagonal(s_prior.scale_tril, dim1=1, dim2=2)

            else:
                s_prior_mean = self.infer_s_means_update[user, :, idx - 1]
                s_prior_cov = self.infer_s_vars_update[user, :, idx - 1]

            s_tilde_dist_mean = s_prior_mean @ self.gen_st_h  # [bs, 1, dim_s]
            s_prior_cov_mat = torch.diag_embed(s_prior_cov)  # [bs, 1, dim_s, dim_s]
            pst_transition_var = torch.exp(self.gen_st_log_r)
            pst_transition_cov_mat = torch.diag_embed(
                pst_transition_var + EPS
            )  # [1, dim_s, dim_s]
            s_tilde_dist_var_mat = (
                self.gen_st_h @ s_prior_cov_mat @ self.gen_st_h.transpose(-1, -2)
                + pst_transition_cov_mat
            )  # [bs, 1, dim_s, dim_s]
            s_tilde_dist_var = torch.diagonal(
                s_tilde_dist_var_mat, dim1=-2, dim2=-1
            )  # [bs, 1, dim_s]
            s_tilde_dist = distributions.multivariate_normal.MultivariateNormal(
                loc=s_tilde_dist_mean,
                scale_tril=torch.tril(torch.diag_embed(s_tilde_dist_var)),
            )

            # q_phi(z_t-1) the posterior of last time step is the prior of this time step
            if z_prior != None:
                z_prior_mean = z_prior.mean
                z_prior_cov = torch.diagonal(z_prior.scale_tril, dim1=1, dim2=2)  # TODO
            else:
                z_prior_mean = self.infer_z_means_update[
                    user, :, idx - 1
                ]  # [bs, 1, dim_z]
                z_prior_cov = self.infer_z_vars_update[user, :, idx - 1]
                
            s_next_sample = s_tilde_dist_mean # [bs, 1, dim_s]
            z_last_sample = z_prior_mean # [bs, 1, num_node]
            
            sampled_alpha = (
                torch.relu(s_next_sample[..., 0:1]) + EPS
            )  # TODO change # [bs, 1, 1]
            sampled_mu = s_next_sample[..., 1:2]
            sampled_sigma = s_next_sample[..., 2:3]
            sampled_gamma = torch.sigmoid(s_next_sample[..., 3:4])

            ou_decay = torch.exp(-sampled_alpha * dt.reshape(bs, 1, 1))  # [bs, 1, 1]
            graph_adj = (
                self.node_dist.sample_A(self.num_sample)[-1][:, 0].mean(0).to(device)
            )
            empower = (
                sampled_gamma * (z_last_sample @ graph_adj) / self.num_node
            )  # [bs, 1, num_node]
            empowered_mu = sampled_mu + empower

            z_tilde_dist_mean = ou_decay * z_prior_mean + (1 - ou_decay) * empowered_mu
            z_tilde_dist_var = (
                sampled_sigma ** 2 * (1 - ou_decay ** 2) / (2 * sampled_alpha + EPS)
                + EPS
            )
            z_tilde_dist = distributions.multivariate_normal.MultivariateNormal(
                loc=z_tilde_dist_mean,
                scale_tril=torch.tril(
                    torch.diag_embed(z_tilde_dist_var.repeat(1, 1, self.num_node))
                ),
            )

        if not eval:
            if not update:
                self.pred_s_means[user, :, idx] = s_tilde_dist_mean.detach().clone()
                self.pred_s_vars[user, :, idx] = s_tilde_dist_var.detach().clone()
                self.pred_z_means[user, :, idx] = z_tilde_dist_mean.detach().clone()
                self.pred_z_vars[user, :, idx] = z_tilde_dist_var.detach().clone()
            else:
                self.pred_s_means_update[
                    user, :, idx
                ] = s_tilde_dist_mean.detach().clone()
                self.pred_s_vars_update[
                    user, :, idx
                ] = s_tilde_dist_var.detach().clone()
                self.pred_z_means_update[
                    user, :, idx
                ] = z_tilde_dist_mean.detach().clone()
                self.pred_z_vars_update[
                    user, :, idx
                ] = z_tilde_dist_var.detach().clone()

        return s_tilde_dist, z_tilde_dist

    def inference_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
        eval: bool = False,
        update: bool = False,
    ):
        """
        Args:
            eval: if True, it will not update the parameters.
                    Usually this happens during evaluation or comparison when there is no gradient.
            update: if True, it means the network is optimized by the update loss.
                    We should save the interested parameters in the updated list.
        """
        # y_idx = feed_dict['label_seq'][:, :idx+1] # [bs, times]
        # t_idx = feed_dict['time_seq'][:, :idx+1]
        # item = feed_dict['skill_seq'][:, :idx+1] # [bs, times]
        y_idx = feed_dict["label_seq"][:, idx : idx + 1]  # [bs, times]
        t_idx = feed_dict["time_seq"][:, idx : idx + 1]
        item = feed_dict["skill_seq"][:, idx : idx + 1]  # [bs, times]
        
        # ----- embedding -----
        t_emb = self.get_time_embedding(t_idx, "absolute")  # [bs, times, node_dim]
        y_emb = torch.tile(
            y_idx.unsqueeze(-1), (1, 1, self.node_dim)
        )  # [bs, times, node_dim]
        node_emb = self.node_dist._get_node_embedding()[item]  # [bs, times, dim]
        emb_input = torch.cat([node_emb, y_emb], dim=-1)  # [bs, times, dim*2]
        emb_history = self.infer_network_emb(emb_input)

        emb_rnn_inputs = emb_history + t_emb

        # -----  `q(s[t] | y[1:t])' -----
        s_dist = self.st_transition_infer(feed_dict, emb_inputs=emb_rnn_inputs, idx=idx)

        # ----- `q(z[t] | y[1:t])' -----
        z_dist = self.zt_transition_infer(
            feed_dict,
            emb_inputs=emb_rnn_inputs,
            idx=idx,
        )

        if not eval:
            users = feed_dict["user_id"]
            if not update:
                self.infer_s_means[users, :, idx] = s_dist.mean.detach().clone()
                self.infer_s_vars[users, :, idx] = (
                    torch.diagonal(s_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()
                )
                self.infer_z_means[users, :, idx] = z_dist.mean.detach().clone()
                self.infer_z_vars[users, :, idx] = (
                    torch.diagonal(z_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()
                )
            else:
                self.infer_s_means_update[users, :, idx] = s_dist.mean.detach().clone()
                self.infer_s_vars_update[users, :, idx] = (
                    torch.diagonal(s_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()
                )
                self.infer_z_means_update[users, :, idx] = z_dist.mean.detach().clone()
                self.infer_z_vars_update[users, :, idx] = (
                    torch.diagonal(z_dist.scale_tril, dim1=-2, dim2=-1).detach().clone()
                )

        return s_dist, z_dist

    def objective_function(
        self,
        feed_dict_idx: Dict[str, torch.Tensor],
        idx: int,
        pred_dist=None,
        post_dist=None,
    ):

        y_idx = feed_dict_idx["label_seq"][:, idx : idx + 1]
        item_idx = feed_dict_idx["skill_seq"][:, idx : idx + 1]

        # p_tilde_theta(s_t) = \int p_theta(s_t | s_t-1) q_phi(s_t-1 | y_1:t-1) ds_t
        # p_tilde_theta(z_t) = \int p_theta(z_t | s_t, z_t-1) q_phi(z_t-1 | y_1:t-1) dz_t
        s_tilde_dist, z_tilde_dist = pred_dist
        # q_phi(s_t | y_1:t), q_phi(z_t | y_1:t)
        s_infer_dist, z_infer_dist = post_dist

        # log tilde_p_theta(s_t)
        s_vp_sample = s_infer_dist.rsample(
            (self.num_sample,)
        )  # [num_sample, bs, 1, dim_s]
        log_prob_st = s_tilde_dist.log_prob(s_vp_sample)  # [num_sample, bs, 1, dim_s]
        log_prob_st = log_prob_st.mean() / self.dim_s

        z_vp_sample = z_infer_dist.rsample(
            (self.num_sample,)
        )  # [num_sample, bs, 1, dim_z]
        log_prob_zt = z_tilde_dist.log_prob(z_vp_sample)  # [num_sample, bs, 1, dim_z]
        log_prob_zt = log_prob_zt.mean() / self.dim_z

        item_idx_mc = item_idx.unsqueeze(0).repeat(self.num_sample, 1, 1)  # [n, bs, 1]
        sampled_scalar_z = torch.gather(z_vp_sample[:, :, 0], dim=2, index=item_idx_mc)

        emission_prob = self.y_emit(sampled_scalar_z)  # [n, bs, 1]
        emission_dist = torch.distributions.bernoulli.Bernoulli(probs=emission_prob)
        prediction = emission_dist.sample()  # NOTE: only for evaluation; no gradient
        label = y_idx.unsqueeze(0).repeat(self.num_sample, 1, 1).float()
        log_prob_yt = emission_dist.log_prob(label).mean()

        st_entropy = s_infer_dist.entropy().mean()
        zt_entropy = z_infer_dist.entropy().mean()

        elbo = (
            log_prob_st
            + log_prob_zt
            + self.args.y_log_weight * log_prob_yt
            + self.args.s_entropy_weight * st_entropy
            + self.args.z_entropy_weight * zt_entropy
        )

        return dict(
            elbo=elbo,
            sequence_likelihood=log_prob_yt + log_prob_zt + log_prob_st,
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

        comparison = defaultdict(lambda: torch.zeros(()))  # , device=self.device))
        users = feed_dict["user_id"]
        labels = feed_dict["label_seq"][:, idx].float()  # [bs, times]
        items = feed_dict["skill_seq"][:, idx]  # [bs, times]

        # ------ comparison 1: check if optimization works ------
        old_z_tilde_dist = torch.distributions.MultivariateNormal(
            loc=self.pred_z_means[users, :, idx],
            scale_tril=torch.diag_embed(self.pred_z_vars[users, :, idx]),
        )
        new_z_tilde_dist = torch.distributions.MultivariateNormal(
            loc=self.pred_z_means_update[users, :, idx],
            scale_tril=torch.diag_embed(self.pred_z_vars_update[users, :, idx]),
        )
        old_y = self.y_emit(old_z_tilde_dist.sample((self.num_sample,))).mean(0)
        new_y = self.y_emit(new_z_tilde_dist.sample((self.num_sample,))).mean(0)
        old_y = torch.gather(old_y, dim=2, index=items.reshape(-1, 1, 1))
        new_y = torch.gather(new_y, dim=2, index=items.reshape(-1, 1, 1))

        comparison["comp_1_tilde"] = loss_fn(
            new_y.flatten(), labels.flatten()
        ) - loss_fn(old_y.flatten(), labels.flatten())

        old_z_infer_dist = torch.distributions.MultivariateNormal(
            loc=self.infer_z_means[users, :, idx],
            scale_tril=torch.diag_embed(self.infer_z_vars[users, :, idx]),
        )
        new_z_infer_dist = torch.distributions.MultivariateNormal(
            loc=self.infer_z_means_update[users, :, idx],
            scale_tril=torch.diag_embed(self.infer_z_vars_update[users, :, idx]),
        )
        old_y = self.y_emit(old_z_infer_dist.sample((self.num_sample,))).mean(0)
        new_y = self.y_emit(new_z_infer_dist.sample((self.num_sample,))).mean(0)
        old_y = torch.gather(old_y, dim=2, index=items.reshape(-1, 1, 1))
        new_y = torch.gather(new_y, dim=2, index=items.reshape(-1, 1, 1))

        comparison["comp_1_infer"] = loss_fn(
            new_y.flatten(), labels.flatten()
        ) - loss_fn(old_y.flatten(), labels.flatten())

        return comparison



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
    
            
            
    def eval_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform predictive modeling.

        Args:
            feed_dict (Dict[str, torch.Tensor]): Dictionary containing input data with dtype torch.Tensor.
        """
        
        test_step = 10
        user = feed_dict['user_id']
        s_tilde_dist = torch.distributions.MultivariateNormal(
                loc=self.infer_s_means_update[user, :, idx],
                scale_tril=torch.diag_embed(self.infer_s_vars_update[user, :, idx]),
            )
        z_tilde_dist = torch.distributions.MultivariateNormal(
                loc=self.infer_z_means_update[user, :, idx],
                scale_tril=torch.diag_embed(self.infer_z_vars_update[user, :, idx]),
            )

        y_all = feed_dict['label_seq'][:, idx+1:idx+test_step+1] # [bs, times]
        item_all = feed_dict["skill_seq"][:, idx+1:idx+test_step+1]
        t_all = feed_dict['time_seq'][:, idx:idx+test_step+1] # [bs, 10]
        bs, _ = t_all.shape
        bsn = bs * self.num_sample
    
        qs_dist, qz_dist = s_tilde_dist, z_tilde_dist

        # analytical solution
        pred_s_mean, pred_s_var, pred_z_mean, pred_z_var = [], [], [], []
        s_last_mean = qs_dist.mean  # [bs, 1, dim_s]
        s_last_cov_mat = qs_dist.covariance_matrix  # [bs, 1, dim_s, dim_s]
        st_tran_r = torch.diag_embed(torch.exp(self.gen_st_log_r) + EPS)
        z_last_mean = qz_dist.mean  # [bs, 1, num_node]
        pz_graph_adj = (
            self.node_dist.sample_A(self.num_sample)[-1][:, 0]
            .mean(0)
            .to(z_last_mean.device)
        )  
        dt = (
            torch.diff(t_all, dim=-1).unsqueeze(-1) / T_SCALE + EPS
        )  # [bs, num_steps, 1]
        for i in range(test_step):
            # p(st-1) = N(m, P), p(st|st-1) = N(st|H*st-1 + b, R)
            # p(st) = N(st|H*m + b, H*P*H' + R)
            s_next_mean = s_last_mean @ self.gen_st_h  # [bs, 1, dim_s]
            s_next_cov_mat = (
                self.gen_st_h @ s_last_cov_mat @ self.gen_st_h.transpose(-1, -2)
                + st_tran_r
            )  # [bs, 1, dim_s, dim_s]
            pred_s_mean.append(s_next_mean)
            pred_s_var.append(s_next_cov_mat)
            s_last_mean = s_next_mean
            s_last_cov_mat = s_next_cov_mat

            # p(zt) = N(zt|zt-1, st)
            q_alpha = torch.relu(s_next_mean[..., 0:1]) + EPS
            q_mu = s_next_mean[..., 1:2]  # torch.tanh(s_next_mean[..., 1:2])  #
            q_sigma = s_next_mean[..., 2:3]  # [bs, 1, 1]
            q_gamma = torch.sigmoid(
                s_next_mean[..., 3:4]
            )  # torch.zeros_like(q_sigma) #
            # calculate useful variables
            pz_ou_decay = torch.exp(-q_alpha * dt[:, i : i + 1])  # [bs, 1, 1]
            pz_ou_var = (
                q_sigma
                * q_sigma
                * (1 - pz_ou_decay * pz_ou_decay)
                / (2 * q_alpha + EPS)
            )  # [bs, num_steps-1, 1]
            pz_empower = (z_last_mean @ pz_graph_adj) / self.num_node * q_gamma
            pz_empowered_mu = q_mu + pz_empower  # [bs, time-1, num_node]
            pz_ou_mean = (
                pz_ou_decay * z_last_mean + (1 - pz_ou_decay) * pz_empowered_mu
            )  # [bs, 1, num_node]
            z_last_mean = pz_ou_mean
            pred_z_mean.append(pz_ou_mean)
            pred_z_var.append(pz_ou_var)

        pred_s_mean = torch.cat(pred_s_mean, dim=1)  # [bs, time, dim_s]
        pred_s_cov_mat = torch.cat(pred_s_var, dim=1)  # [bs, time, dim_s, dim_s]
        pred_z_mean = torch.cat(pred_z_mean, dim=1)  # [bs, time, num_node]
        pred_z_var = torch.cat(pred_z_var, dim=1).repeat(
            1, 1, self.num_node
        )  # [bs, time, num_node]

        pred_z_dist = torch.distributions.MultivariateNormal(
            loc=pred_z_mean, scale_tril=torch.tril(torch.diag_embed(pred_z_var + EPS))
        )
        pred_z_sampled = pred_z_dist.sample(
            (self.num_sample,)
        )  # [n, bs, time, num_node]
        pred_z_sampled = pred_z_sampled.transpose(1, 0).reshape(
            bsn, test_step, self.num_node
        )  # [bsn, time, num_node]
        pred_z_sampled = pred_z_sampled.transpose(
            -1, -2
        ).contiguous()  # [bsn, num_node, time]

        item_test = item_all
        item_test_mc = (
            item_test.unsqueeze(1)
            .repeat(1, self.num_sample, 1)
            .reshape(bsn, 1, test_step)
        )  # [bsn, 1, time]
        pred_z_sampled_item = (
            torch.gather(pred_z_sampled, 1, item_test_mc).transpose(-1, -2).contiguous()
        )  # [bsn, time, 1]
        
        pred_y_test = self.y_emit(pred_z_sampled_item)
        pred = pred_y_test.reshape(bs, self.num_sample, test_step)
        mc_label = y_all.unsqueeze(1).repeat(1, self.num_sample, 1)

        return self.pred_evaluate_method(pred.cpu(), mc_label.cpu(), self.metrics)
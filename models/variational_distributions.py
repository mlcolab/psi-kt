
from typing import List, Optional

import numpy as np

import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn

import ipdb
EPS = 1e-6
class VarGT(nn.Module):
    def __init__(self, device, num_nodes, gt_adj_path=None):
        super(VarGT, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.gt_adj_path = gt_adj_path
    def sample_A(self, num_graph=1):
        adj = np.load(self.gt_adj_path)
        adj = torch.from_numpy(adj).to(self.device).unsqueeze(0)
        return None, None, adj

class VarDistribution(nn.Module):
    def __init__(self, device, num_nodes, tau_gumbel=None):
        super(VarDistribution, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.tau_gumbel = tau_gumbel

    def edge_log_probs(self, latents=None):
        pass

    def sample_A(self, num_graph=1):
        logits = self.edge_log_probs()
        off_diag_mask = 1 - torch.eye(self.num_nodes, device=self.device)
        # ipdb.set_trace()

        probs = [F.gumbel_softmax(logits, tau=self.tau_gumbel, hard=False, dim=0)[1:] for _ in range(num_graph)]
        probs = torch.stack(probs)
        probs = probs # * off_diag_mask # [num_graphs, 2, num_nodes, num_nodes]

        sample = [F.gumbel_softmax(logits, tau=self.tau_gumbel, hard=True, dim=0)[1:] for _ in range(num_graph)] 
        sample = torch.stack(sample)
        adj = sample  # * off_diag_mask  # Force zero diagonals
        return logits, probs, adj


class VarConstant(VarDistribution):
    def __init__(self, device, num_nodes, tau_gumbel=None):
        super().__init__(device, num_nodes, tau_gumbel)

    def sample_A(self, num_graph=1):
        probs = torch.ones((1, 1, self.num_nodes, self.num_nodes)).to(self.device)
        adj = probs
        logits = torch.log(probs)
        return logits, probs, adj


class VarBasic(VarDistribution):
    def __init__(self, device, num_nodes, tau_gumbel=None, dense_init=False):
        super().__init__(device, num_nodes, tau_gumbel)
        self.dense_init = dense_init
        self.latents = self._initial_latents()

    def _initial_latents(self):        
        if not self.dense_init:
            # Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
            latents = torch.rand(size=(self.num_nodes, self.num_nodes), device=self.device)
        else:
            latents = torch.ones(size=(self.num_nodes, self.num_nodes), device=self.device)
        latents = nn.Parameter(latents, requires_grad=True)
        return latents

    def edge_log_probs(self):
        scores = self.latents
        log_probs, log_probs_neg = F.logsigmoid(scores), F.logsigmoid(-scores) 
        return torch.stack([log_probs, log_probs_neg])


class VarENCO(VarDistribution):
    """
    Variational distribution for the binary adjacency matrix, following the parameterization from
    the ENCO paper (https://arxiv.org/pdf/2107.10483.pdf). For each edge, parameterizes the existence
    and orientation separately. Main benefit is that it avoids length 2 cycles automatically.
    Orientation is somewhat over-parameterized.
    """

    def __init__(self, device, num_nodes, tau_gumbel, dense_init = False):
        super().__init__(device, num_nodes, tau_gumbel)
        self.dense_init = dense_init
        self.logits_edges = self._initialize_edge_logits()
        self.params_orient = self._initialize_orient_params()

    def _initialize_edge_logits(self):
        logits = torch.zeros(2, self.num_nodes, self.num_nodes, device=self.device)  # Shape (2, n, n)
        if self.dense_init:
            logits[1, :, :] += 3
        else:
            logits[1, :, :] -= 1
        return nn.Parameter(logits, requires_grad=True)

    def _initialize_orient_params(self):
        if self.dense_init:
            params = torch.ones(self.num_nodes, self.num_nodes, device=self.device)  # (n, n)
        else:
            params = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)  # (n, n)
        return nn.Parameter(params, requires_grad=True)

    def _build_logits_orient(self) -> torch.Tensor:
        """
        Auxiliary function that computes the (softmax) logits to sample orientation for the edges given the parameters.
        """
        logits_0 = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)  # Shape (n, n)
        # Get logits_1 strictly upper triangular
        logits_1 = torch.triu(self.params_orient)
        logits_1 = logits_1 * (1.0 - torch.eye(self.num_nodes, self.num_nodes, device=self.device)) # remove the 1 in the diagonal
        logits_1 = logits_1 - torch.transpose(logits_1, 0, 1)  # Make logit_ij = -logit_ji
        return torch.stack([logits_0, logits_1])

    def edge_log_probs(self) -> torch.Tensor:
        """
        Auxiliary function to compute the (softmax) logits from both edge logits and orientation logits. Notice
        the logits for the softmax are computed differently than those for Bernoulli (latter uses sigmoid, equivalent
        if the logits for zero filled with zeros).

        Simply put, to sample an edge i->j you need to both sample the precense of that edge, and sample its orientation.
        """
        logits_edges = self.logits_edges  # Shape (2, n, n)
        logits_orient = self._build_logits_orient()  # Shape (2, n, n)
        logits_1 = logits_edges[1, :, :] + logits_orient[1, :, :]  # Shape (n, n)
        aux = torch.stack(
            [
                logits_edges[1, :, :] + logits_orient[0, :, :],
                logits_edges[0, :, :] + logits_orient[1, :, :],
                logits_edges[0, :, :] + logits_orient[0, :, :],
            ]
        )  # Shape (3, num_nodes, num_nodes)
        logits_0 = torch.logsumexp(aux, dim=0)  # Shape (num_nodes, num_nodes)
        logits = torch.stack([logits_0, logits_1])  # Shape (2, num_nodes, num_nodes)
        return logits

    def log_prob_A(self, A):
        """
        Evaluates the variational distribution q(A) at a sampled adjacency A.
        Args:
            A: A binary adjacency matrix, size (num_nodes, num_nodes).
        Returns:
            The log probability of the sample A. A number if A has size (num_nodes, num_nodes).
        """
        return self._build_bernoulli().log_prob(A)

    def get_adj_matrix(self, do_round):
        """
        Returns the adjacency matrix.
        """
        probs_edges = F.softmax(self.logits_edges, dim=0)[1, :, :]  # Shape (num_nodes, num_nodes)
        logits_orient = self._build_logits_orient()
        probs_orient = F.softmax(logits_orient, dim=0)[1, :, :]  # Shape (num_nodes, num_nodes)
        probs_1 = probs_edges * probs_orient
        probs_1 = probs_1 * (1.0 - torch.eye(self.num_nodes, device=self.device))
        if do_round:
            return probs_1.round()
        return probs_1

    def _build_bernoulli(self) -> td.Distribution:
        """
        Builds and returns the bernoulli distributions obtained using the (softmax) logits.
        """
        logits = self._get_logits_softmax()  # (2, n, n)
        logits_bernoulli_1 = logits[1, :, :] - logits[0, :, :]  # (n, n)
        # Diagonal elements are set to 0
        logits_bernoulli_1 -= 1e10 * torch.eye(self.num_nodes, device=self.device)
        dist = td.Independent(td.Bernoulli(logits=logits_bernoulli_1), 2)
        return dist

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q.
        """
        return self._build_bernoulli().entropy()



class VarDIBS(VarDistribution):
    def __init__(self, device, num_nodes, tau_gumbel, dense_init = False, 
                            latent_prior_std = None, latent_dim = 128):
        """
        Args:
            device: Device used.
            num_nodes: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
            alpha_linear (float): slope of of linear schedule for inverse temperature :math:`\\alpha`
                                    of sigmoid in latent graph model :math:`p(G | Z)`
            dense_init: whether the initialization of latent variables is from a uniform distribution (False) or torch.ones(True)
        """
        super().__init__(device, num_nodes, tau_gumbel)
        self.dense_init = dense_init # start from all 1?
        self.latent_prior_std = latent_prior_std
        self.latent_dim = latent_dim

        alpha_linear = 0.05
        self.alpha = lambda t: (alpha_linear * t)

        self.u, self.v = self._initial_random_particles()

    def _initial_random_particles(self):
        """
        Args:
            n_particles (int): number of particles inferred
            n_dim (int): size of latent dimension :math:`k`. Defaults to ``n_vars``, s.t. :math:`k = d`

        Returns:
            batch of latent tensors ``[n_particles, d, k, 2]``
        """
        dim = torch.tensor(self.latent_dim)
        
        if not self.dense_init:
            std = self.latent_prior_std or (1.0 / torch.sqrt(dim))
            u = torch.randn(size=(self.num_nodes, self.latent_dim), device=self.device) * std
            v = torch.randn(size=(self.num_nodes, self.latent_dim), device=self.device) * std
            # z = torch.randn(size=(self.num_nodes, self.latent_dim, 2), requires_grad=True) * std # TODO check what distribution this results in
            # torch.randn returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 
            # is it actually sparse?

        else:
            u = torch.ones(size=(self.num_nodes, self.latent_dim), device=self.device) * (1.0 / torch.sqrt(dim))
            v = torch.ones(size=(self.num_nodes, self.latent_dim), device=self.device) * (1.0 / torch.sqrt(dim)) # u*v_T = 1, sigmoid(u*v_T) = 0.7311
            # z = torch.ones(size=(self.num_nodes, self.latent_dim, 2), requires_grad=True) * (1.0 / torch.sqrt(dim))
        u = nn.Parameter(u, requires_grad=True)
        v = nn.Parameter(v, requires_grad=True)
        return u, v


    def edge_log_probs(self):
        """
        Edge log probabilities encoded by latent representation
        Args:
            z (ndarray): latent tensors :math:`Z` ``[..., d, k, 2]``
        Returns:
            tuple of tensors ``[..., d, d], [..., d, d]`` corresponding to ``log(p)`` and ``log(1-p)``
        """
        u, v = self.u, self.v # u, z [num_nodes, latent_dim]
        scores = torch.matmul(u, v.transpose(0,1))
        # ipdb.set_trace()
        log_probs, log_probs_neg = F.logsigmoid(scores), F.logsigmoid(-scores) # TODO in dibs paper, there is an alpha control the temperature here

        # scores = jnp.einsum('...ik,...jk->...ij', u, v)
        # log_probs, log_probs_neg = log_sigmoid(self.alpha(t) * scores), log_sigmoid(self.alpha(t) * -scores)

        # # mask diagonal since it is explicitly not modeled
        # # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
        # log_probs = log_probs.at[..., jnp.arange(log_probs.shape[-1]), jnp.arange(log_probs.shape[-1])].set(0.0)
        # log_probs_neg = log_probs_neg.at[..., jnp.arange(log_probs_neg.shape[-1]), jnp.arange(log_probs_neg.shape[-1])].set(0.0)
        return torch.stack([log_probs, log_probs_neg])


class VarAttention(VarDistribution):
    def __init__(self, device, num_nodes, tau_gumbel, dense_init = False, 
                            latent_prior_std = None, latent_dim = 128):
        """
        Args:
            device: Device used.
            num_nodes: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
            alpha_linear (float): slope of of linear schedule for inverse temperature :math:`\\alpha`
                                    of sigmoid in latent graph model :math:`p(G | Z)`
            dense_init: whether the initialization of latent variables is from a uniform distribution (False) or torch.ones(True)
        """
        super().__init__(device, num_nodes, tau_gumbel)
        self.dense_init = dense_init # start from all 1?
        self.latent_prior_std = latent_prior_std
        self.latent_dim = latent_dim

        alpha_linear = 0.05
        self.alpha = lambda t: (alpha_linear * t)

        self.u = self._initial_random_particles()
        
        self.query_layer = nn.Linear(self.latent_dim, self.latent_dim)#, device=self.device)
        self.key_layer = nn.Linear(self.latent_dim, self.latent_dim)#, device=self.device)
        self.value_layer = nn.Linear(self.latent_dim, self.latent_dim)#, device=self.device)
        self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=4, dropout=0.0, bias=True, batch_first=True)#, device=self.device)
        # the attention in pytorch has softmax across all nodes, which is not what we want
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        # https://discuss.pytorch.org/t/what-does-increasing-number-of-heads-do-in-the-multi-head-attention/101294/7
        # num_heads – Number of parallel attention heads. Note that embed_dim will be split across num_heads (i.e. each head will have dimension embed_dim // num_heads).


    def _initial_random_particles(self):
        """
        Args:
            n_particles (int): number of particles inferred
            n_dim (int): size of latent dimension :math:`k`. Defaults to ``n_vars``, s.t. :math:`k = d`

        Returns:
            batch of latent tensors ``[n_particles, d, k, 2]``
        """
        # sample points uniformly from a sphere surface 
        if not self.dense_init:
            u = torch.randn(size=(self.num_nodes, self.latent_dim))#, device=self.device)
            u = u / (torch.norm(u, dim=1, keepdim=True) + EPS)
        else:
            u = torch.rand(size=(self.num_nodes, self.latent_dim))#, device=self.device)
            u = u / (torch.norm(u, dim=1, keepdim=True) + EPS)
        u = nn.Parameter(u, requires_grad=True)
        return u


    def _get_atten_weights(self):
        """
        Edge log probabilities encoded by latent representation
        Args:
            z (ndarray): latent tensors :math:`Z` ``[..., d, k, 2]``
        Returns:
            tuple of tensors ``[..., d, d], [..., d, d]`` corresponding to ``log(p)`` and ``log(1-p)``
        """
        u = self.u
        u = u / (torch.norm(u, dim=1, keepdim=True) + EPS)
       
        query_u = self.query_layer(u)
        key_u = self.key_layer(u)
        value_u = self.value_layer(u) 
        
        attn_output, attn_output_weights = self.attention_layer(query_u, key_u, value_u) 
        
        return query_u, key_u, value_u, attn_output, attn_output_weights


    def _get_node_embedding(self):
        u = self.u
        u = u / (torch.norm(u, dim=1, keepdim=True) + EPS)
        return u
        
    def edge_log_probs(self):
        """
        Edge log probabilities encoded by latent representation
        Args:
            z (ndarray): latent tensors :math:`Z` ``[..., d, k, 2]``
        Returns:
            tuple of tensors ``[..., d, d], [..., d, d]`` corresponding to ``log(p)`` and ``log(1-p)``
        """
        query_u, key_u, value_u, attn_output, attn_output_weights = self._get_atten_weights()
        # TODO need to mask acyclic edges
        # TODO constrain the sparsity of edges: https://aclanthology.org/2021.acl-short.17.pdf; https://arxiv.org/pdf/2110.11299.pdf; https://arxiv.org/abs/1705.07704
        # https://github.com/datnnt1997/multi-head_self-attention/blob/master/SelfAttention.ipynb
        
        log_probs, log_probs_neg = torch.log(attn_output_weights), torch.log(1-attn_output_weights) # TODO in dibs paper, there is an alpha control the temperature here
        
        return torch.stack([log_probs, log_probs_neg])


class VarTransformation(VarDistribution):
    def __init__(self, device, num_nodes, tau_gumbel, dense_init = False, 
                            latent_prior_std = None, latent_dim = 128):
        """
        Args:
            device: Device used.
            num_nodes: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
            alpha_linear (float): slope of of linear schedule for inverse temperature :math:`\\alpha`
                                    of sigmoid in latent graph model :math:`p(G | Z)`
            dense_init: whether the initialization of latent variables is from a uniform distribution (False) or torch.ones(True)
        """
        super().__init__(device, num_nodes, tau_gumbel)
        self.dense_init = dense_init # start from all 1?
        self.latent_prior_std = latent_prior_std
        self.latent_dim = latent_dim

        alpha_linear = 0.05
        self.alpha = lambda t: (alpha_linear * t)

        self.u = self._initial_random_particles()
        
        transformation_layer = torch.randn(size=(self.latent_dim, self.latent_dim))
        self.transformation_layer = nn.Parameter(transformation_layer, requires_grad=True)
        

    def _initial_random_particles(self):
        """
        Args:
            n_particles (int): number of particles inferred
            n_dim (int): size of latent dimension :math:`k`. Defaults to ``n_vars``, s.t. :math:`k = d`

        Returns:
            batch of latent tensors ``[n_particles, d, k, 2]``
        """
        # sample points uniformly from a sphere surface 
        if not self.dense_init:
            u = torch.randn(size=(self.num_nodes, self.latent_dim))#, device=self.device)
            u = u / (torch.norm(u, dim=1, keepdim=True) + EPS)
        else:
            u = torch.rand(size=(self.num_nodes, self.latent_dim))#, device=self.device)
            u = u / (torch.norm(u, dim=1, keepdim=True) + EPS)
        u = nn.Parameter(u, requires_grad=True)
        return u

 
    def _get_node_embedding(self):
        u = self.u
        u = u / (torch.norm(u, dim=1, keepdim=True) + EPS)
        return u
        
        
    def edge_log_probs(self):
        """
        Edge log probabilities encoded by latent representation
        Args:
            z (ndarray): latent tensors :math:`Z` ``[..., d, k, 2]``
        Returns:
            tuple of tensors ``[..., d, d], [..., d, d]`` corresponding to ``log(p)`` and ``log(1-p)``
        """
        u = self._get_node_embedding()
        
        prob_edge_existing = torch.sigmoid(u @ u.transpose(-1,-2))
        prob_edge_directed_ab = torch.sigmoid(u @ (self.transformation_layer-self.transformation_layer.transpose(-1,-2)) @ u.transpose(-1,-2))
        # prob_edge_directed_ba = 1 - prob_edge_directed_ab
        
        probs = prob_edge_existing * prob_edge_directed_ab
        log_probs, log_probs_neg = torch.log(probs), torch.log(1-probs) 
        
        return torch.stack([log_probs, log_probs_neg])

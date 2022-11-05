
from typing import List, Optional

import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn

import ipdb

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
            latents = torch.rand(size=(self.num_nodes, self.num_nodes))
        else:
            latents = torch.ones(size=(self.num_nodes, self.num_nodes))
        latents = nn.Parameter(latents, requires_grad=True).to(self.device)
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

    # def sample_A(self, num_graph=1): # TODO num_graph
    #     """
    #     Sample an adjacency matrix from the variational distribution. It uses the gumbel_softmax trick,
    #     and returns hard samples (straight through gradient estimator). Adjacency returned always has
    #     zeros in its diagonal (no self loops).

    #     V1: Returns one sample to be used for the whole batch.
    #     """
    #     logits = self.edge_log_probs()

    #     off_diag_mask = 1 - torch.eye(self.num_nodes, device=self.device)

    #     probs = F.gumbel_softmax(logits, tau=self.tau_gumbel, hard=False, dim=0)
    #     probs = probs * off_diag_mask.unsqueeze(0)

    #     sample = F.gumbel_softmax(logits, tau=self.tau_gumbel, hard=True, dim=0)  # (2, n, n) binary
    #     sample = sample[1]  # (n, n)
    #     adj = sample # * off_diag_mask  # Force zero diagonals
    #     return logits, probs, adj

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

        self.latents = self._initial_random_particles()

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
            u = torch.randn(size=(self.num_nodes, self.latent_dim)) * std
            v = torch.randn(size=(self.num_nodes, self.latent_dim)) * std
            # z = torch.randn(size=(self.num_nodes, self.latent_dim, 2), requires_grad=True) * std # TODO check what distribution this results in
            # torch.randn returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 
            # is it actually sparse?

        else:
            u = torch.ones(size=(self.num_nodes, self.latent_dim)) * (1.0 / torch.sqrt(dim))
            v = torch.ones(size=(self.num_nodes, self.latent_dim)) * (1.0 / torch.sqrt(dim)) # u*v_T = 1, sigmoid(u*v_T) = 0.7311
            # z = torch.ones(size=(self.num_nodes, self.latent_dim, 2), requires_grad=True) * (1.0 / torch.sqrt(dim))

        u = nn.Parameter(u, requires_grad=True).to(self.device)
        v = nn.Parameter(v, requires_grad=True).to(self.device)
        return [u, v]


    def edge_log_probs(self):
        """
        Edge log probabilities encoded by latent representation
        Args:
            z (ndarray): latent tensors :math:`Z` ``[..., d, k, 2]``
        Returns:
            tuple of tensors ``[..., d, d], [..., d, d]`` corresponding to ``log(p)`` and ``log(1-p)``
        """
        z = self.latents
        u, v = z[0], z[1] # u, z [num_nodes, latent_dim]
        scores = torch.matmul(u, v.transpose(0,1))
        log_probs, log_probs_neg = F.logsigmoid(scores), F.logsigmoid(-scores) # TODO in dibs paper, there is an alpha control the temperature here

        # scores = jnp.einsum('...ik,...jk->...ij', u, v)
        # log_probs, log_probs_neg = log_sigmoid(self.alpha(t) * scores), log_sigmoid(self.alpha(t) * -scores)

        # # mask diagonal since it is explicitly not modeled
        # # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
        # log_probs = log_probs.at[..., jnp.arange(log_probs.shape[-1]), jnp.arange(log_probs.shape[-1])].set(0.0)
        # log_probs_neg = log_probs_neg.at[..., jnp.arange(log_probs_neg.shape[-1]), jnp.arange(log_probs_neg.shape[-1])].set(0.0)
        return torch.stack([log_probs, log_probs_neg])

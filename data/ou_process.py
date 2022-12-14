'''
https://github.com/felix-clark/ornstein-uhlenbeck/blob/master/ornstein_uhlenbeck.py
and
https://github.com/jwergieluk/ou_noise/blob/master/ou_noise/ou.py
'''

"""
A module for evaluation and analysis with an Ornstein-Uhlenbeck process:
dX_t = - eta * X_t + sigma * dW_t
 where W_t is a Weiner process.
The log-likelihood function is:
l = - (n/2) log(sigma^2/(2*eta)) - 1/2 sum log [1 - exp(-2*eta*dt)] - (n*eta/sigma^2) * D
where D = (1/n) * sum{ [(x_t-mu) - (x_{t-1}-mu)*exp(-eta*dt)]^2 /
(1-exp(-2*eta*dt)) } is a modified sum of squares.
Setting dl/d(sigma^2) = 0 yields sigma^2/2*eta = D so for the purposes of
minimization the log-likelihood can be simplified:
l = - (n/2) [1 + log(D)] - 1/2 sum log [1 - exp(-2*eta*dt)]
Keep in mind that this will not be valid when evaluating variations from the
minimum such as when computing multi-parameter uncertainties.
The mean mu is also a function of eta, but for our purposes it should be close
to zero so we will treat it as a constant and update iteratively if needed.
The MLE estimators are biased, but we should be in the large-N limit where any
correction is small.
"""

from typing import List, Tuple, Optional
import logging
import numpy as np
import scipy.optimize as opt

import math

import scipy.constants
import scipy.stats
import scipy.optimize
# from . import quadratic_variation

import networkx as nx 
import scipy
from scipy import stats

import ipdb


class VanillaOU():
    def __init__(self, mean_rev_speed, mean_rev_level, vola, num_seq=1):
        self.mean_rev_speed = mean_rev_speed
        self.mean_rev_level = mean_rev_level
        self.vola = vola
        self.num_seq = num_seq
        assert self.mean_rev_speed >= 0
        # assert self.mean_rev_level >= 0
        assert self.vola >= 0


    def variance(self, t):
        return self.vola * self.vola * (1.0 - np.exp(- 2.0 * self.mean_rev_speed * t)) / (2 * self.mean_rev_speed)

    def std(self, t):
        return np.sqrt(self.variance(t))

    def mean(self, x0, t, mean_speed=None, mean_level=None):
        speed = mean_speed if mean_speed is not None else self.mean_rev_speed
        level = mean_level if mean_level is not None else self.mean_rev_level
        return x0 * np.exp(-speed * t) + (1.0 - np.exp(- speed * t)) * level

    def simulate_path(self, x0, t):
        """ 
        Simulates a sample path
        dX = A(alpha-X)dt + v dB
        """
        assert len(t) > 1
        time_step = t.shape[1]
        num_node = len(x0)

        dt = np.diff(t).reshape(self.num_seq, -1, 1, 1)
        dt = np.repeat(dt, num_node, -2)

        x = np.zeros((self.num_seq, time_step, num_node, 1))
        x[:, 0] += x0

        noise = scipy.stats.norm.rvs(size=x.shape)
        scale = self.std(dt)
        x[:, 1:] += noise[:, 1:] * scale

        for i in range(1, time_step):
            x[:, i] += self.mean(x[:, i-1], dt[:, i-1])
        return x


class VanillaGraphOU(VanillaOU):
    def __init__(self, mean_rev_speed, mean_rev_level, vola, num_seq, nx_graph):
        self.graph = nx_graph
        super().__init__(mean_rev_speed, mean_rev_level, vola)
    
    def simulate_path(self, x0, t):
        assert len(t) > 1
        dt = np.diff(t).reshape(-1, 1, 1)
        dt = np.repeat(dt, len(x0), -2)

        x = np.zeros((len(t), len(x0), 1))
        x[0] = x0
        
        noise = scipy.stats.norm.rvs(size=(len(t), len(x0), 1)) # shape [times, num_node]
        scale = self.std(dt)
        x[1:] = noise[1:] * scale

        adj = nx.adjacency_matrix(self.graph).toarray()
        adj_t = np.transpose(adj, (-1,-2))
        in_degree = adj_t.sum(axis=1).reshape(-1,1)
        ind = np.where(in_degree == 0)[0] # find degree 0

        for i in range(1, len(x)):
            tmp_mean_level = (-self.mean_rev_speed/self.mean_rev_level) 
            s = (1/(in_degree+1e-7)) * adj_t@x[i-1]
            s[ind] = 1 
            tmp_mean_level = tmp_mean_level * s
            x[i] += self.mean(x[i - 1], dt[i - 1], mean_level=tmp_mean_level)

        return x



class RewriteGraphOU(VanillaOU):
    def __init__(self, mean_rev_speed, mean_rev_level, vola, num_seq, nx_graph):
        self.graph = nx_graph
        super().__init__(mean_rev_speed, mean_rev_level, vola, num_seq)
    
    def simulate_path(self, x0, t):
        assert len(t) > 1
        num_node = len(x0)
        time_step = t.shape[1]

        dt = np.diff(t).reshape(self.num_seq, -1, 1, 1)
        dt = np.repeat(dt, num_node, -2)

        x = np.zeros((self.num_seq, time_step, num_node, 1))
        x[:, 0] += x0
        # ipdb.set_trace()
        noise = scipy.stats.norm.rvs(size=x.shape) 
        scale = self.std(dt)
        x[:, 1:] += noise[:, 1:] * scale

        adj = nx.adjacency_matrix(self.graph).toarray()
        adj_t = np.transpose(adj, (-1,-2))
        in_degree = adj_t.sum(axis=1).reshape(1,-1,1)
    
        # find degree 0
        ind = np.where(in_degree == 0)[0]

        for i in range(1, time_step):
            # ipdb.set_trace()
            s = (1/(in_degree+1e-7)) * adj_t@x[:, i-1] # [num_seq, num_node, 1]
            s[ind] = 1 
            tmp_mean_level = self.mean_rev_level * s
            x[:, i] += self.mean(x[:, i-1], dt[:, i-1], mean_level=tmp_mean_level)
        # ipdb.set_trace()
        return x



class ExtendGraphOU(VanillaOU):
    def __init__(self, mean_rev_speed, mean_rev_level, vola, num_seq, nx_graph):
        self.graph = nx_graph
        super().__init__(mean_rev_speed, mean_rev_level, vola)
    
    def simulate_path(self, x0, t):
        assert len(t) > 1
        dt = np.diff(t).reshape(-1, 1, 1)
        dt = np.repeat(dt, len(x0), -2)

        x = np.zeros((len(t), len(x0), 1))
        x[0] = x0
        
        noise = scipy.stats.norm.rvs(size=(len(t), len(x0), 1)) # shape [times, num_node]
        scale = self.std(dt)
        x[1:] = noise[1:] * scale

        adj = nx.adjacency_matrix(self.graph).toarray()
        adj_t = np.transpose(adj, (-1,-2))
        in_degree = adj_t.sum(axis=1).reshape(-1,1)
    
        # find degree 0
        ind = np.where(in_degree == 0)[0]

        for i in range(1, len(x)):
            s = (1/(in_degree+1e-7)) * adj_t@x[i-1]
            s[ind] = 1 
            tmp_mean_level = self.mean_rev_level * s
            x[i] += self.mean(x[i - 1], dt[i - 1], mean_level=tmp_mean_level)
        # ipdb.set_trace()
        return x

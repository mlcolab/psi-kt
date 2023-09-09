"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
import ipdb
DIM = 64
# Inference Network
class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
      super(InferenceNet, self).__init__()

      # q(y|x)
      self.inference_qyx = torch.nn.ModuleList([
          nn.Linear(x_dim, DIM),
          nn.ReLU(),
          nn.Linear(DIM, DIM),
          nn.ReLU(),
          GumbelSoftmax(DIM, y_dim)
      ])

      # q(z|y,x)
      self.inference_qzyx = torch.nn.ModuleList([
          nn.Linear(x_dim // 10 + y_dim, DIM),
          nn.ReLU(),
          nn.Linear(DIM, DIM),
          nn.ReLU(),
          Gaussian(DIM, z_dim)
      ])

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                #last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)  
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat
  
    def forward(self, x, temperature=1.0, hard=0):
        x_faltten = x.reshape(x.size(0), -1)
        
        # q(y|x) 
        logits, prob, y = self.qyx(x_faltten, temperature, hard) # [bs, num_categories]
        
        # q(z|x,y)
        # ipdb.set_trace() # TODO we can do it at every time step
        mu, var, z = self.qzxy(x.mean(1), y) # [bs, z_dim] # TODO would it be a little imbalanced? x, y are not the same dimension

        output = {'mean': mu, 'var': var, 'gaussian': z, 
                  'logits': logits, 'prob_cat': prob, 'categorical': y}
        
        return output


# Generative Network
class GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GenerativeNet, self).__init__()

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(z_dim, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, x_dim),
            torch.nn.Sigmoid()
        ])

    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var
  
    # p(x|z)
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, y, z=None):
        # p(z|y)
        y_mu, y_var = self.pzy(y)
        
        # p(x|z)
        # x_rec = self.pxz(z)
        x_rec = None

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output


# GMVAE Network
class GMVAENet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GMVAENet, self).__init__()

    self.inference = InferenceNet(x_dim, z_dim, y_dim)
    self.generative = GenerativeNet(x_dim, z_dim, y_dim)

    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias.data is not None:
          init.constant_(m.bias, 0) 

  def forward(self, x, temperature=1.0, hard=0):
    x = x.view(x.size(0), -1)
    out_inf = self.inference(x, temperature, hard)
    z, y = out_inf['gaussian'], out_inf['categorical']
    out_gen = self.generative(z, y)
    
    # merge output
    output = out_inf
    for key, value in out_gen.items():
      output[key] = value
    return output
  







"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Custom Layers

"""
import torch
from torch import nn
from torch.nn import functional as F

# Flatten layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Reshape layer    
class Reshape(nn.Module):
  def __init__(self, outer_shape):
    super(Reshape, self).__init__()
    self.outer_shape = outer_shape
  def forward(self, x):
    return x.view(x.size(0), *self.outer_shape)

# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim
     
    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        #categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard 
  
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y

# Sample from a Gaussian distribution
class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z      

    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z 
  









# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Loss functions used for training our model

"""
import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class LossFunctions:
    eps = 1e-8

    def mean_squared_error(self, real, predictions):
        """Mean Squared Error between the true and predicted outputs
            loss = (1/n)*Σ(real - predicted)^2

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels
    
        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        loss = (real - predictions).pow(2)
        return loss.sum(-1).mean()


    def reconstruction_loss(self, real, predicted, rec_type='mse' ):
        """Reconstruction loss between the true and predicted outputs
            mse = (1/n)*Σ(real - predicted)^2
            bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels
    
        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        if rec_type == 'mse':
            loss = (real - predicted).pow(2)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none')
        else:
            raise "invalid loss function... try bce or mse..."
        return loss.sum(-1).mean()


    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
            log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
            x: (array) corresponding array containing the input
            mu: (array) corresponding array containing the mean 
            var: (array) corresponding array containing the variance

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)


    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        """Variational loss when using labeled data without considering reconstruction loss
            loss = log q(z|x,y) - log p(z) - log p(y)

        Args:
            z: (array) array containing the gaussian latent variable
            z_mu: (array) array containing the mean of the inference model
            z_var: (array) array containing the variance of the inference model
            z_mu_prior: (array) array containing the prior mean of the generative model
            z_var_prior: (array) array containing the prior variance of the generative mode
            
        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()


    def entropy(self, logits, targets):
        """Entropy loss
            loss = (1/n) * -Σ targets*log(predicted)

        Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels
    
        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))
    
    def prior_entropy(self, num_classes, generative_network, device):

        category = F.one_hot(torch.arange(num_classes), num_classes=num_classes).to(device)
        out = generative_network(category.float())
        mu = out['y_mean']
        
        return -mu.std(0).sum()

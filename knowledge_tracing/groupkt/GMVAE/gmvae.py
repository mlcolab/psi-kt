"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
-- Modified by: Hanqi Zhou
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F

import math
import numpy as np

from knowledge_tracing.groupkt import *

DIM = 64

class InferenceNet(nn.Module):
    def __init__(self, in_dim, latent_dim, cate_dim, time_step):
        super(InferenceNet, self).__init__()

        # q(class|input)
        self.inference_qyx = torch.nn.ModuleList([
            nn.LSTM(
                input_size=in_dim,  
                hidden_size=in_dim * 2,
                bidirectional=False,
                batch_first=True,
            ),
            nn.Linear(in_dim*2, in_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_dim, DIM),
            nn.LeakyReLU(0.2),
            GumbelSoftmax(DIM, cate_dim)
        ])

        # q(latents|class, input)
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(in_dim + cate_dim, DIM),
            nn.ReLU(),
            nn.Linear(DIM, in_dim),
            nn.ReLU(),
            Gaussian(in_dim, latent_dim)
        ])

    # q(y|x) -> q(category|emb_history)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                #last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y) -> q(s|emb_history, category)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=-1)  
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat
  
    def forward(self, inputs, temperature=1.0, hard=0, time_dependent_s=False):
        input_faltten = inputs.reshape(inputs.size(0), -1) 

        w_logits, w_prob, w_sample = self.qyx(input_faltten, temperature, hard) # [bs, num_categories]

        if time_dependent_s:
            w_sample_mc = w_sample.unsqueeze(1).repeat(1,inputs.shape[1],1)
            s_mu_infer, s_var_infer, _ = self.qzxy(inputs, w_sample_mc) # [bs, seq_len, latent_dim]
        else:
            s_mu_infer, s_var_infer, _ = self.qzxy(inputs.mean(1), w_sample) # [bs, latent_dim] # TODO would it be a little imbalanced? x, y are not the same dimension

        output = {'s_mu_infer': s_mu_infer, 's_var_infer': s_var_infer,
                  'logits': w_logits, 'prob_cat': w_prob, 'categorical': w_sample}
        
        for key, value in output.items():
            output[key] = value.unsqueeze(1)
        
        return output


class GenerativeNet(nn.Module):
    """
    Initializes a GenerativeNet instance.

    Args:
        x_dim (int): Dimensionality of the input data x.
        z_dim (int): Dimensionality of the latent variable z.
        y_dim (int): Dimensionality of the latent categorical variable y.
    """

    def __init__(self, x_dim: int, z_dim: int, y_dim: int) -> None:
        super(GenerativeNet, self).__init__()

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList(
            [
                nn.Linear(z_dim, DIM),
                nn.ReLU(),
                nn.Linear(DIM, DIM),
                nn.ReLU(),
                nn.Linear(DIM, x_dim),
                torch.nn.Sigmoid(),
            ]
        )

    def pzy(self, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Calculates the parameters of the distribution p(z|y).

        Args:
            y (torch.Tensor): Latent categorical variable y.

        Returns:
            torch.Tensor: Mean of the distribution p(z|y).
            torch.Tensor: Variance of the distribution p(z|y).
        """
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    def pxz(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generates samples from the distribution p(x|z).

        Args:
            z (torch.Tensor): Latent variable z.

        Returns:
            torch.Tensor: Generated samples from the distribution p(x|z).
        """
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, y: torch.Tensor, z: torch.Tensor = None) -> dict:
        """
        Forward pass of the GenerativeNet.

        Args:
            y (torch.Tensor): Latent categorical variable y.
            z (torch.Tensor, optional): Latent variable z. Defaults to None.

        Returns:
            dict: A dictionary containing output tensors:
                  - 'y_mean': Mean of the distribution p(z|y).
                  - 'y_var': Variance of the distribution p(z|y).
                  - 'x_rec': Generated samples from the distribution p(x|z).
        """
        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # p(x|z)
        # x_rec = self.pxz(z)
        x_rec = None

        output = {"y_mean": y_mu, "y_var": y_var, "x_rec": x_rec}
        return output


class GMVAENet(nn.Module):
    """
    Gaussian Mixture Variational Autoencoder Network.

    This class represents a GMVAE network that combines an inference network and a generative network
    to perform variational autoencoding with Gaussian mixture latent variables.

    Args:
        x_dim (int): Dimensionality of the input data x.
        z_dim (int): Dimensionality of the latent variable z.
        y_dim (int): Dimensionality of the latent categorical variable y.
    """

    def __init__(self, x_dim: int, z_dim: int, y_dim: int):
        super(GMVAENet, self).__init__()

        self.inference = InferenceNet(x_dim, z_dim, y_dim)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim)

        # weight initialization
        for m in self.modules():
            if (
                type(m) == nn.Linear
                or type(m) == nn.Conv2d
                or type(m) == nn.ConvTranspose2d
            ):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, temperature: float = 1.0, hard: int = 0) -> dict:
        """
        Forward pass of the GMVAENet.

        Args:
            x (torch.Tensor): Input data x.
                A tensor containing the input data of shape (batch_size, channels, height, width).
            temperature (float, optional): Temperature parameter for Gumbel-Softmax relaxation. Defaults to 1.0.
            hard (int, optional): Flag for hard Gumbel-Softmax sampling. Defaults to 0.

        Returns:
            dict: A dictionary containing output tensors:
                  - 'gaussian': Sampled latent variable z from the inference network.
                  - 'categorical': Sampled latent categorical variable y from the inference network.
                  - Other keys from the generative network's output.
        """
        x = x.view(x.size(0), -1)
        out_inf = self.inference(x, temperature, hard)
        z, y = out_inf["gaussian"], out_inf["categorical"]
        out_gen = self.generative(z, y)

        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output


"""

Custom Layers

"""


class Flatten(nn.Module):
    """
    A custom Flatten layer.

    This layer flattens the input tensor by reshaping it into a 2D tensor.

    Usage:
    >>> flatten_layer = Flatten()
    >>> flattened_tensor = flatten_layer(input_tensor)

    Args:
        None

    Returns:
        torch.Tensor: Flattened tensor with shape (batch_size, -1).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Flatten layer.

        Args:
            x (torch.Tensor): Input tensor to be flattened.

        Returns:
            torch.Tensor: Flattened tensor with shape (batch_size, -1).
        """
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    """
    A custom Reshape layer.

    This layer reshapes the input tensor into the specified shape.

    Usage:
    >>> reshape_layer = Reshape(outer_shape)
    >>> reshaped_tensor = reshape_layer(input_tensor)

    Args:
        outer_shape (tuple): Desired shape for the output tensor, excluding the batch dimension.

    Returns:
        torch.Tensor: Reshaped tensor with the specified outer shape.
    """

    def __init__(self, outer_shape: tuple):
        """
        Initializes a Reshape instance.

        Args:
            outer_shape (tuple): Desired shape for the output tensor, excluding the batch dimension.
        """
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Reshape layer.

        Args:
            x (torch.Tensor): Input tensor to be reshaped.

        Returns:
            torch.Tensor: Reshaped tensor with the specified outer shape.
        """
        return x.view(x.size(0), *self.outer_shape)


class GumbelSoftmax(nn.Module):
    """
    Gumbel-Softmax distribution sampler.

    This class provides methods for sampling from the Gumbel-Softmax distribution and optionally
    discretizing the samples.

    Args:
        f_dim (int): Dimensionality of the input features.
        c_dim (int): Dimensionality of the categorical variable.

    Usage:
    >>> gs_sampler = GumbelSoftmax(f_dim, c_dim)
    >>> logits, prob, sample = gs_sampler(input_features, temperature, hard)
    """

    def __init__(self, f_dim: int, c_dim: int):
        """
        Initializes a GumbelSoftmax instance.

        Args:
            f_dim (int): Dimensionality of the input features.
            c_dim (int): Dimensionality of the categorical variable.
        """
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(
        self, shape: tuple, is_cuda: bool = False, eps: float = 1e-20
    ) -> torch.Tensor:
        """
        Sample from the Gumbel distribution.

        Args:
            shape (tuple): Shape of the sample.
            is_cuda (bool, optional): Flag indicating whether to use GPU. Defaults to False.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-20.

        Returns:
            torch.Tensor: Sampled values from the Gumbel distribution.
        """
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(
        self, logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """
        Sample from the Gumbel-Softmax distribution.

        Args:
            logits (torch.Tensor): Logits representing the categorical distribution.
            temperature (float): Temperature parameter for the Gumbel-Softmax distribution.

        Returns:
            torch.Tensor: Sampled values from the Gumbel-Softmax distribution.
        """
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(
        self, logits: torch.Tensor, temperature: float, hard: bool = False
    ) -> torch.Tensor:
        """
        Gumbel-Softmax distribution with optional hard sampling.

        Args:
            logits (torch.Tensor): Logits representing the categorical distribution.
            temperature (float): Temperature parameter for the Gumbel-Softmax distribution.
            hard (bool, optional): Flag for hard Gumbel-Softmax sampling. Defaults to False.

        Returns:
            torch.Tensor: Sampled values from the Gumbel-Softmax distribution.
        """
        # categorical_dim = 10
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

    def forward(
        self, x: torch.Tensor, temperature: float = 1.0, hard: bool = False
    ) -> tuple:
        """
        Forward pass of the GumbelSoftmax layer.

        Args:
            x (torch.Tensor): Input tensor.
                A tensor containing the input features of shape (batch_size, f_dim).
            temperature (float, optional): Temperature parameter for Gumbel-Softmax sampling. Defaults to 1.0.
            hard (bool, optional): Flag for hard Gumbel-Softmax sampling. Defaults to False.

        Returns:
            tuple: A tuple containing:
                  - torch.Tensor: Logits representing the categorical distribution.
                  - torch.Tensor: Probabilities computed from the softmax of logits.
                  - torch.Tensor: Sampled values from the Gumbel-Softmax distribution.
        """
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


class Gaussian(nn.Module):
    """
    Gaussian distribution sampler.

    This class provides methods for sampling from a Gaussian distribution and reparameterizing.

    Args:
        in_dim (int): Dimensionality of the input features.
        z_dim (int): Dimensionality of the latent variable z.

    Usage:
    >>> gaussian_sampler = Gaussian(in_dim, z_dim)
    >>> mu, var, z = gaussian_sampler(input_features)
    """

    def __init__(self, in_dim: int, z_dim: int):
        """
        Initializes a Gaussian instance.

        Args:
            in_dim (int): Dimensionality of the input features.
            z_dim (int): Dimensionality of the latent variable z.
        """
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterize the Gaussian distribution.

        Args:
            mu (torch.Tensor): Mean of the Gaussian distribution.
            var (torch.Tensor): Variance of the Gaussian distribution.

        Returns:
            torch.Tensor: Sampled values from the Gaussian distribution using reparameterization.
        """
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward pass of the Gaussian layer.

        Args:
            x (torch.Tensor): Input tensor.
                A tensor containing the input features of shape (batch_size, in_dim).

        Returns:
            tuple: A tuple containing:
                  - torch.Tensor: Mean of the Gaussian distribution.
                  - torch.Tensor: Variance of the Gaussian distribution.
                  - torch.Tensor: Sampled values from the Gaussian distribution using reparameterization.
        """
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


"""

Loss functions used for training our model

"""


class LossFunctions:
    """
    Collection of loss functions for various tasks.

    This class provides methods for calculating different types of loss functions.

    Usage:
    >>> loss_calculator = LossFunctions()
    >>> mse_loss = loss_calculator.mean_squared_error(real, predictions)
    """

    def mean_squared_error(
        self, real: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the Mean Squared Error between the true and predicted outputs.
            loss = (1/n)*Σ(real - predicted)^2

        Args:
            real (torch.Tensor): Corresponding tensor containing the true labels.
            predictions (torch.Tensor): Corresponding tensor containing the predicted labels.

        Returns:
            torch.Tensor: Mean squared error loss.
        """
        loss = (real - predictions).pow(2)
        return loss.sum(-1).mean()

    def reconstruction_loss(
        self, real: torch.Tensor, predicted: torch.Tensor, rec_type="mse"
    ) -> torch.Tensor:
        """
        Calculates the reconstruction loss between the true and predicted outputs.

            mse = (1/n)*Σ(real - predicted)^2
            bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

        Args:
            real (torch.Tensor): Corresponding tensor containing the true labels.
            predicted (torch.Tensor): Corresponding tensor containing the predicted labels.
            rec_type (str, optional): Type of reconstruction loss ('mse' or 'bce'). Defaults to 'mse'.

        Returns:
            torch.Tensor: Reconstruction loss.
        """
        if rec_type == "mse":
            loss = (real - predicted).pow(2)
        elif rec_type == "bce":
            loss = F.binary_cross_entropy(predicted, real, reduction="none")
        else:
            raise "invalid loss function... try bce or mse..."
        return loss.sum(-1).mean()

    def log_normal(
        self, x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the logarithm of the normal distribution with mean=mu and variance=var.

            log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
            x (torch.Tensor): Corresponding tensor containing the input.
            mu (torch.Tensor): Corresponding tensor containing the mean.
            var (torch.Tensor): Corresponding tensor containing the variance.

        Returns:
            torch.Tensor: Logarithm of the normal distribution.
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1
        )

    def gaussian_loss(
        self,
        z: torch.Tensor,
        z_mu: torch.Tensor,
        z_var: torch.Tensor,
        z_mu_prior: torch.Tensor,
        z_var_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the variational loss when using labeled data without considering reconstruction loss.

            loss = log q(z|x,y) - log p(z) - log p(y)
        Args:
            z (torch.Tensor): Tensor containing the Gaussian latent variable.
            z_mu (torch.Tensor): Tensor containing the mean of the inference model.
            z_var (torch.Tensor): Tensor containing the variance of the inference model.
            z_mu_prior (torch.Tensor): Tensor containing the prior mean of the generative model.
            z_var_prior (torch.Tensor): Tensor containing the prior variance of the generative model.

        Returns:
            torch.Tensor: Gaussian loss.
        """
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(
            z, z_mu_prior, z_var_prior
        )
        return loss.mean()

    def entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the entropy loss.
            loss = (1/n) * -Σ targets*log(predicted)
        Args:
            logits (torch.Tensor): Corresponding tensor containing the logits of the categorical variable.
            targets (torch.Tensor): Corresponding tensor containing the true labels.

        Returns:
            torch.Tensor: Entropy loss.
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

    def prior_entropy(
        self, num_classes: int, generative_network, device
    ) -> torch.Tensor:
        """
        Calculates the entropy of the prior distribution.

        Args:
            num_classes (int): Number of classes/categories.
            generative_network: Generative network for which the entropy is computed.
            device: Device to which tensors are moved.

        Returns:
            torch.Tensor: Prior entropy loss.
        """
        category = F.one_hot(torch.arange(num_classes), num_classes=num_classes).to(
            device
        )
        out = generative_network(category.float())
        mu = out["y_mean"]

        return -mu.std(0).sum()

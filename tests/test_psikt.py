import pytest

# I haven't figured out how to import the module from the root directory yet
# I currently use this as the hacky way to import the module...
import sys

sys.path.append("..")

import math
import numpy as np

import torch

from knowledge_tracing.groupkt import EPS
from knowledge_tracing.groupkt.groupkt_graph_representation import VarTransformation
from knowledge_tracing.groupkt.groupkt import AmortizedGroupKT
from knowledge_tracing.groupkt.GMVAE.gmvae import InferenceNet
from knowledge_tracing.groupkt.modules import VAEEncoder
from knowledge_tracing.runner.runner import KTRunner
from knowledge_tracing.utils.logger import Logger

DIM_S = 4

@pytest.fixture
def groupkt():
    class args_example(object):
        def __init__(
            self,
        ):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.learned_graph = "w_gt"
            self.var_log_max = 1.0
            self.node_dim = 16
            self.num_category = 5
            self.time_dependent_s = 1
            self.num_sample = 1
            
            self.max_step = 50
            self.train_time_ratio = 0.04
            self.test_time_ratio = 0.04
            self.val_time_ratio = 0.04
            
            self.overfit = 1
            self.epoch = 10
            self.batch_size_multiGPU = 32
            self.eval_batch_size = 32
            self.metric = "Accuracy, F1, Recall, Precision, AUC"
            self.early_stop = 1
            self.create_logs = 0
            self.log_path = 'logs'

    nx_graph = torch.randn(10, 10)
    num_node = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    args = args_example()
    logs = Logger(args)
    model = AmortizedGroupKT(num_node=num_node, args=args, device=device, logs=logs, nx_graph=nx_graph)
    
    return model

@pytest.fixture
def ps_dist():  
    ps_mean = torch.randn(2, 1, 10, DIM_S)
    ps_cov_mat = torch.randn(2, 1, 10, DIM_S, DIM_S)
    ps_dist = torch.distributions.MultivariateNormal(ps_mean, ps_cov_mat)
    return ps_dist

@pytest.fixture
def qs_dist():
    # Create a sample MultivariateNormal distribution for qs_dist (replace with actual parameters)
    qs_mean = torch.randn(1, 1, 2, DIM_S)
    qs_cov_mat = torch.ones((1, 1, 2, DIM_S)) * 1e-4 # torch.pow(torch.randn(1, 1, 2, DIM_S), 2)
    qs_dist = torch.distributions.MultivariateNormal(qs_mean, torch.diag_embed(qs_cov_mat))
    return qs_dist

# Test the _init_weights method
def test_init_weights(groupkt):
    # Call the _init_weights method
    groupkt._init_weights()

    # Check that the model attributes are initialized correctly
    assert isinstance(groupkt.node_dist, VarTransformation)
    assert isinstance(groupkt.gen_s0_mean, torch.Tensor)
    assert isinstance(groupkt.gen_s0_log_var, torch.Tensor)
    assert isinstance(groupkt.gen_z0_mean, torch.Tensor)
    assert isinstance(groupkt.gen_z0_log_var, torch.Tensor)
    assert isinstance(groupkt.gen_st_h, torch.Tensor)
    assert isinstance(groupkt.gen_st_log_r, torch.Tensor)
    assert isinstance(groupkt.y_emit, torch.nn.Sigmoid)
    assert isinstance(groupkt.infer_network_emb, torch.nn.Module)
    assert isinstance(groupkt.infer_network_posterior_s, InferenceNet)  # Replace with the actual class name
    assert isinstance(groupkt.infer_network_posterior_z, torch.nn.LSTM)
    assert isinstance(groupkt.infer_network_posterior_mean_var_z, VAEEncoder)  # Replace with the actual class name


def test_construct_univariate_normal(groupkt):
    mean = torch.tensor([2.0])
    std = torch.tensor([[1.0]])
    dist = groupkt._construct_normal_from_mean_std(mean, std)
    
    from torch.distributions import MultivariateNormal
    # Check if the constructed distribution is a univariate normal
    assert isinstance(dist, MultivariateNormal)
    
    # Check if the mean and covariance are correct
    assert torch.allclose(dist.mean, mean, atol=EPS)
    assert torch.allclose(dist.covariance_matrix, std.pow(2), atol=EPS)
    

def test_initialize_gaussian_mean_log_var(groupkt):
    dim = 3
    use_trainable_cov = True
    num_sample = 2
    cov_min = 0.05
    
    x0_mean, x0_log_var = groupkt._initialize_gaussian_mean_log_var(dim, use_trainable_cov, num_sample)
    
    # Check if the shapes of mean and log variance match the expected shapes
    assert x0_mean.shape == (num_sample, dim)
    assert x0_log_var.shape == (num_sample, dim)
    
    # Check if mean is initialized with Xavier uniform initialization
    for i in range(num_sample):
        assert torch.allclose(torch.mean(x0_mean[i]), torch.zeros(1), atol=np.sqrt(6/(dim+num_sample)))
    
    # Check if log variance is initialized to log(COV_MIN)
    assert torch.allclose(x0_log_var, torch.log(torch.tensor(cov_min)), atol=1e-6)


def test_positional_encoding1d(groupkt):
    d_model = 16
    length = 4
    
    actual_time = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    
    # Call the method with actual_time
    pe = groupkt._positional_encoding1d(d_model, length, actual_time)
    assert pe.shape == (1, length, d_model)
    assert torch.allclose(pe, torch.zeros_like(pe), atol=1)
    
    # Calculate the expected positional encoding values
    expected_pe = torch.zeros(1, length, d_model)
    for i in range(length):
        for j in range(d_model // 2):
            angle = i / math.pow(10000, 2 * j / d_model)
            expected_pe[0, i, 2 * j] = math.sin(angle)
            expected_pe[0, i, 2 * j + 1] = math.cos(angle)
    
    # Check if the positional encoding matrix values are close to the expected values
    assert torch.allclose(pe, expected_pe, atol=1)
    

def test_st_transition_gen(groupkt, qs_dist):
    bs, _, time, _ = qs_dist.mean.shape
    
    transition_st_h = groupkt.gen_st_h
    assert isinstance(transition_st_h, torch.Tensor)
    assert transition_st_h.shape == (DIM_S, DIM_S)
    assert transition_st_h.requires_grad
    
    # Call the st_transition_gen method
    ps_dist = groupkt.st_transition_gen(qs_dist, eval=False)
    assert isinstance(ps_dist, torch.distributions.MultivariateNormal)
    ps_dist_mean = ps_dist.mean
    ps_dist_cov_mat = ps_dist.covariance_matrix
    assert ps_dist_mean.shape == (bs, 1, time, DIM_S)
    assert ps_dist_cov_mat.shape == (bs, 1, time, DIM_S, DIM_S)
    
    num_samples = int(1e6)
    qs_sample = qs_dist.sample((num_samples,))  # [num_samples, bs, 1, time, dim_s]
    qs_sample_transition = (
        qs_sample[:, :, :, :-1] @ groupkt.gen_st_h 
    )  # [num_samples, bs, 1, time-1, dim_s]
    qs_new_dist = torch.distributions.MultivariateNormal(qs_sample_transition, torch.diag_embed(torch.exp(groupkt.gen_st_log_r)+EPS))
    # qs_sample_transition_mean = qs_sample_transition.mean(dim=0)  # [bs, 1, time-1, dim_s]
    # qs_sample_transition_cov_mat = qs_sample_transition.var(dim=0)  # [bs, 1, time-1, dim_s]

    # assert torch.allclose(qs_sample_transition_mean, ps_dist_mean[:,:,1:], atol=1e-1)
    # assert torch.allclose(qs_sample_transition_cov_mat, ps_dist_cov_mat[:,:,1:], atol=1e-1)
    
    # new_dist = torch.distributions.MultivariateNormal(ps_dist_mean[:,:,1:], ps_dist_cov_mat[:,:,1:])
    # assert new_dist.log_prob(qs_sample_transition_mean).mean() > -1e-1
    
    
def test_zt_transition_gen(groupkt):
    # Prepare test data
    feed_dict = {"time_seq": torch.tensor([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])}  # Replace with appropriate data
    # Prepare other required arguments if necessary

    # Call the function
    pz_dist = groupkt.zt_transition_gen(feed_dict)

    # Add assertions to check if the output is as expected
    assert isinstance(pz_dist, torch.distributions.MultivariateNormal)
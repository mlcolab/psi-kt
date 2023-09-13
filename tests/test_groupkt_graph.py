import pytest

# I haven't figured out how to import the module from the root directory yet
# I'll first use this as the hacky way to import the module...
import sys

sys.path.append("..")

import torch

from knowledge_tracing.runner.runner import KTRunner
from knowledge_tracing.utils.logger import Logger
from knowledge_tracing.groupkt import groupkt_graph_representation as ktgraph


@pytest.fixture
def kt_runner():
    class args_example(object):
        def __init__(
            self,
        ):
            self.overfit = 1
            self.epoch = 10
            self.batch_size_multiGPU = 32
            self.eval_batch_size = 32
            self.metric = "Accuracy, F1, Recall, Precision, AUC"
            self.early_stop = 1
            self.device = "cpu"
            self.create_logs = 0

    args = args_example()
    logs = Logger(args)
    kt_runner = KTRunner(args, logs)
    return kt_runner


@pytest.fixture
def var_transformation_instance():
    device = "cpu"
    num_nodes = 10
    tau_gumbel = 0.1
    return ktgraph.VarTransformation(
        device, num_nodes, tau_gumbel, dense_init=False, latent_dim=16
    )


def test_var_transformation_init(var_transformation_instance):
    assert (
        var_transformation_instance.device == "cpu"
    )  # Check if the device is set correctly
    assert (
        var_transformation_instance.num_nodes == 10
    )  # Check if the number of nodes is set correctly
    assert (
        var_transformation_instance.tau_gumbel == 0.1
    )  # Check if tau_gumbel is set correctly
    assert (
        not var_transformation_instance.dense_init
    )  # Check if dense_init is set to False by default
    assert (
        var_transformation_instance.latent_prior_std is None
    )  # Check if latent_prior_std is set to None by default
    assert (
        var_transformation_instance.latent_dim == 16
    )  # Check if latent_dim is set correctly
    assert var_transformation_instance.alpha(1) == 0.05  # Check the alpha function


def test_sample_A(var_transformation_instance):
    num_graph = 2  # Number of graphs to sample
    logits, probs, adj = var_transformation_instance.sample_A(num_graph)

    # Check the shapes of the output tensors
    assert logits.shape == (2, 10, 10)
    assert probs.shape == (2, 1, 10, 10)
    assert adj.shape == (2, 1, 10, 10)

    # Check that the adjacency matrix has zeros on the diagonal
    diagonal_elements = torch.diagonal(
        adj.view(
            -1,
            var_transformation_instance.num_nodes,
            var_transformation_instance.num_nodes,
        ),
        dim1=-1,
        dim2=-2,
    )
    assert torch.all(diagonal_elements == 0)


def test_initial_random_particles(var_transformation_instance):
    # Test when dense_init is False
    var_transformation_instance.dense_init = False
    u, transformation_layer = var_transformation_instance._initial_random_particles()

    # Check the shapes of the output tensors
    assert u.shape == (10, 16)  # Check the shape of u
    assert transformation_layer.shape == (
        16,
        16,
    )  # Check the shape of transformation_layer

    # Check that u is a parameter with requires_grad set to True
    assert isinstance(u, torch.nn.Parameter)
    assert u.requires_grad
    _, _, adj = var_transformation_instance.sample_A(
        1000
    )  # Call the sample_A method to update the transformation_layer


# Test the _get_node_embedding method
def test_get_node_embedding(var_transformation_instance):
    u = var_transformation_instance._get_node_embedding()
    assert u.shape == (10, 16)  # Check the shape of the embedding tensor


# Test the edge_log_probs method
def test_edge_log_probs(var_transformation_instance):
    log_probs = var_transformation_instance.edge_log_probs()
    assert log_probs.shape == (2, 10, 10)  # Check the shape of the log_probs tensor
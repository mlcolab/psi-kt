import pytest

import sys

sys.path.append("..")

import numpy as np

import torch
from torch import distributions
from torch.testing import assert_allclose

from knowledge_tracing.baseline.basemodel import BaseModel, BaseLearnerModel
from knowledge_tracing.utils.logger import Logger


@pytest.fixture
def pred_and_true():
    # Generate random predictions and true labels
    y_pred = np.random.rand(100)
    y_true = np.random.randint(0, 2, size=100)
    return y_pred, y_true


# @pytest.fixture
# def base_learner_model():
#     class args_example(object):
#         def __init__(
#             self,
#         ):
#             self.overfit = 1
#             self.epoch = 10
#             self.batch_size_multiGPU = 32
#             self.eval_batch_size = 32
#             self.metric = "Accuracy, F1, Recall, Precision, AUC"
#             self.early_stop = 1
#             self.device = "cpu"
#             self.create_logs = 0
#             self.mode = 'ls_split_train'

#     args = args_example()
#     logs = Logger(args)
#     base_learner_model = BaseLearnerModel(args.mode, args.device, logs)

#     return base_learner_model


def test_find_whole_stats():
    # Prepare input tensors for testing
    all_feature = torch.tensor([[[[3, 1, 2], [4, 2, 2], [5, 2, 3]]]], dtype=torch.int64)
    t = torch.tensor([[26, 48, 76]], dtype=torch.int64)
    items = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    num_node = 1
    # Call the _find_whole_stats method
    whole_stats, whole_last_time = BaseLearnerModel._find_whole_stats(
        all_feature, t, items, num_node
    )

    # Perform assertions to check if the output tensors have the expected shapes
    assert whole_stats.shape == (1, 1, 3, 3)
    assert torch.equal(
        whole_stats,
        torch.tensor([[[[0, 0, 0], [3, 1, 2], [4, 2, 2]]]], dtype=torch.int64),
    )
    assert whole_last_time.shape == (1, 1, 4)
    assert torch.equal(
        whole_last_time, torch.tensor([[[0, 26, 48, 76]]], dtype=torch.int64)
    )


def test_pred_evaluate_method(pred_and_true):
    # Generate random predictions and true labels
    y_pred, y_true = pred_and_true

    # Define the metrics to evaluate
    metrics = ["mse", "mae", "f1", "accuracy", "precision", "recall"]

    def calculate_f1_score(y_true, y_pred):
        # Calculate true positives, false positives, and false negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate F1 score
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return precision, recall, f1_score

    precision, recall, f1_score = calculate_f1_score(y_true, y_pred > 0.5)
    # Calculate the expected results
    expected_results = {
        "mse": np.mean((y_pred - y_true) ** 2),
        "mae": np.mean(np.absolute(y_true - y_pred)),
        "f1": f1_score,
        "accuracy": np.mean(y_true == (y_pred > 0.5)),
        "precision": precision,
        "recall": recall,
    }

    # Evaluate the predictions using the optimized function
    evaluations = BaseModel.pred_evaluate_method(y_pred, y_true, metrics)

    # Compare the actual and expected results
    for metric in metrics:
        assert np.isclose(
            evaluations[metric], expected_results[metric]
        ), f"Failed for {metric}: expected {expected_results[metric]}, but got {evaluations[metric]}"

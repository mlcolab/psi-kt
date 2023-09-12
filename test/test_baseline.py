import pytest

from knowledge_tracing.runner.runner import KTRunner
from knowledge_tracing.utils.logger import Logger

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


# test KTRunner._eva_termination
def test_eva_termination_multiple_metrics(kt_runner):
    # Set up the necessary attributes
    kt_runner.metrics = ["accuracy", "auc"]  # replace with the desired metrics
    kt_runner.logs.val_results = {
        "accuracy": [0.9, 0.92, 0.91, 0.93, 0.92, 0.91, 0.92, 0.9, 0.89, 0.88, 0.87],
        "auc": [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    }
    # Call the _eva_termination function
    result = kt_runner._eva_termination(
        None, kt_runner.metrics, kt_runner.logs.val_results
    )
    # unittest.TestCase.assertFalse(result)
    assert result == False


def test_eva_termination_multiple_metrics_all_decreasing(kt_runner):
    # Set up the necessary attributes
    kt_runner.metrics = ["accuracy", "auc"]  # replace with the desired metrics
    kt_runner.logs.val_results = {
        "accuracy": [0.95, 0.95, 0.94, 0.93, 0.92, 0.92, 0.92, 0.90, 0.89, 0.88, 0.87],
        "auc": [0.9, 0.9, 0.88, 0.85, 0.81, 0.77, 0.73, 0.70, 0.65, 0.65, 0.57],
    }
    # Call the _eva_termination function
    result = kt_runner._eva_termination(
        None, kt_runner.metrics, kt_runner.logs.val_results
    )

    # unittest.TestCase.assertFalse(result)
    assert result == True

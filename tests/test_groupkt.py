import pytest

# I haven't figured out how to import the module from the root directory yet
# I currently use this as the hacky way to import the module...
import sys

sys.path.append("..")

import torch

from knowledge_tracing.groupkt import EPS
from knowledge_tracing.groupkt.groupkt import AmortizedGroupKT
from knowledge_tracing.runner.runner import KTRunner
from knowledge_tracing.utils.logger import Logger


@pytest.fixture
def groupkt():
    class args_example(object):
        def __init__(
            self,
        ):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.num_seq = 10
            self.learned_graph = "w_gt"
            self.var_log_max = torch.tensor(1.0)
            
            self.overfit = 1
            self.epoch = 10
            self.batch_size_multiGPU = 32
            self.eval_batch_size = 32
            self.metric = "Accuracy, F1, Recall, Precision, AUC"
            self.early_stop = 1
            self.create_logs = 0

    args = args_example()
    logs = Logger(args)
    model = AmortizedGroupKT(args, logs)
    return model

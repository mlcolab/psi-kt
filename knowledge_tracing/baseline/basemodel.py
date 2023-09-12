import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm
from sklearn.metrics import *

import torch

from knowledge_tracing.utils import utils, logger


class BaseModel(torch.nn.Module):
    """
    Base class for neural network models.
    Args:
        model_path (str, optional): The path to save/load the model. Defaults to '../model/Model/Model_{}_{}.pt'.

    Attributes:
        runner (str): The type of runner for the model.
        extra_log_args (list): Additional arguments for logging.
    """

    runner = "KTRunner"
    extra_log_args = []

    def __init__(
        self,
        model_path: str = "../model/Model/Model_{}_{}.pt",
    ):
        super(BaseModel, self).__init__()
        self.model_path = model_path
        self._init_weights()
        self.optimizer = None

    @staticmethod
    def parse_model_args(
        parser: argparse.ArgumentParser,
        model_name: str = "BaseModel",
    ):
        parser.add_argument(
            "--model_path", type=str, default="", help="Model save path."
        )
        return parser

    @staticmethod
    def pred_evaluate_method(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        metrics: List[str],
    ) -> dict:
        """
        Compute evaluation metrics for a set of predictions.

        Args:
            y_pred: The predicted values as a NumPy array.
            y_true: The ground truth values as a NumPy array.
            metrics: A list of evaluation metrics to compute.

        Returns:
            A dictionary containing the evaluation metrics and their values.
        """
        # Flatten the arrays to one dimension
        y_pred = np.ravel(y_pred)
        y_true = np.ravel(y_true)

        # Convert the predictions to binary values based on a threshold of 0.5
        y_pred_binary = (y_pred > 0.5).astype(int)
        evaluation_funcs = {
            "rmse": mean_squared_error,
            "mae": mean_absolute_error,
            "auc": roc_auc_score,
            "f1": f1_score,
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
        }

        # Define the evaluation functions for each metric
        evaluations = {}
        for metric in metrics:
            if metric in evaluation_funcs:
                evaluations[metric] = evaluation_funcs[metric](
                    y_true,
                    y_pred_binary
                    if metric in ["f1", "accuracy", "precision", "recall"]
                    else y_pred,
                )

        return evaluations

    @staticmethod
    def init_weights(m: torch.nn.Module) -> None:
        """
        Initialize weights and biases of the neural network module.

        Args:
            m (torch.nn.Module): The neural network module to initialize.

        Returns:
            None: The method modifies the weights and biases of the input module in-place.
        """

        # TODO: add more initialization methods
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif (
            isinstance(m, torch.nn.RNNCell)
            or isinstance(m, torch.nn.GRUCell)
            or isinstance(m, torch.nn.RNN)
        ):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
        elif isinstance(m, torch.nn.LSTMCell) or isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name or "weight_ch" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)

    @staticmethod
    def batch_to_gpu(
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Move the tensors in a batch to the specified GPU device.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing tensors as values.
            device (torch.device): The target GPU device to move the tensors to.

        Returns:
            Dict[str, torch.Tensor]: The batch with tensors moved to the specified device.
        """

        if torch.cuda.device_count() > 0:
            for key in batch:
                batch[key] = batch[key].to(device)
        return batch

    def _init_weights(self):
        """
        Initialize the model's weights. Subclasses should override this method.
        """
        raise NotImplementedError(
            "Subclasses of BaseModel must implement _init_weights() method."
        )

    def forward(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform the forward pass of the model.

        Args:
            feed_dict (dict): A dictionary containing input tensors.
        """
        raise NotImplementedError(
            "Subclasses of BaseModel must implement forward() method."
        )

    def get_feed_dict(
        self,
        corpus,
        data,
        batch_start: int,
        batch_size: int,
        phase: str,
    ):
        """
        Create a feed dictionary containing input tensors for the model's forward pass.

        Args:
            corpus (DataReader): An instance of the DataReader class containing corpus data.
            data (pd.DataFrame): The DataFrame containing the batch data.
            batch_start (int): The starting index of the batch.
            batch_size (int): The size of the batch.
            phase (str): The phase of the data, e.g., 'train', 'valid', or 'test'.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing input tensors for the model.
                - 'skill_seq' (torch.Tensor): Padded skill sequence tensor of shape [batch_size, real_max_step].
                - 'quest_seq' (torch.Tensor): Padded question sequence tensor of shape [batch_size, real_max_step].
                - 'label_seq' (torch.Tensor): Padded label sequence tensor of shape [batch_size, real_max_step].
        """

        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start

        feed_dict_keys = {
            "skill_seq": "skill_seq",
            "label_seq": "correct_seq",
            "time_seq": "time_seq",
            "problem_seq": "problem_seq",
            "num_history": "num_history",
            "num_success": "num_success",
            "num_failure": "num_failure",
            "user_id": "user_id",
        }

        feed_dict = utils.get_feed_general(
            keys=feed_dict_keys,
            data=data,
            start=batch_start,
            batch_size=real_batch_size,
        )  # [batch_size, seq_len]

        return feed_dict

    def prepare_batches(
        self,
        corpus,
        data: List[Tuple],
        batch_size: int,
        phase: str,
    ) -> List:
        """
        Prepare the data into batches for training/validation/test.

        Args:
            corpus: the corpus object
            data: the training/validation/test data which needs to be batched
            batch_size: the batch size
            phase: the current training phase ('train', 'valid', or 'test')

        Returns:
            A list of batches of the input data
        """

        num_examples = len(data)
        total_batches = (num_examples + batch_size - 1) // batch_size
        assert num_examples > 0

        # Prepare the batches using a list comprehension
        batches = []
        for batch in tqdm(
            range(total_batches),
            leave=False,
            ncols=100,
            mininterval=1,
            desc="Prepare Batches",
        ):
            batches.append(
                self.get_feed_dict(corpus, data, batch * batch_size, batch_size, phase)
            )

        return batches

    def count_variables(
        self,
    ) -> int:
        """
        Counts the number of trainable parameters in a PyTorch model.

        Args:
            model: A PyTorch model.

        Returns:
            The total number of trainable parameters in the model.
        """

        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return total_parameters

    def save_model(
        self, epoch: int, mini_epoch: int = 0, model_path: str = None
    ) -> None:
        """
        Save the model to a file.

        Args:
            epoch: the current epoch number
            model_path: the path to save the model to
        """
        if model_path is None:
            model_path = self.model_path
        model_path = model_path.format(epoch, mini_epoch)

        Path(model_path).parents[0].touch()
        torch.save(self.state_dict(), model_path)
        self.logs.write_to_log_file("Save model to " + model_path)

    def load_model(
        self,
        model_path: str = None,
    ) -> None:
        """
        Load the model from a file.

        Args:
            model_path: the path to load the model from
        """

        if model_path is None:
            model_path = self.model_path

        self.load_state_dict(torch.load(model_path))
        self.eval()
        self.logs.write_to_log_file("Load model from " + model_path)

    def customize_parameters(
        self,
    ) -> List[Dict]:
        """
        Customize the optimizer settings for different parameters.

        Returns:
            A list of dictionaries specifying the optimization settings for each parameter group
        """

        weight_p, bias_p = [], []
        # Find parameters that require gradient
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if "bias" in name:
                bias_p.append(p)
            else:
                weight_p.append(p)

        optimize_dict = [{"params": weight_p}, {"params": bias_p, "weight_decay": 0}]

        return optimize_dict

    def actions_before_train(self) -> None:
        """
        Perform actions before starting the training process.

        1. Compute the total number of trainable parameters in the model.
        2. Write the total number of parameters to a log file.

        Returns:
            None
        """
        # Step 1: Compute the total number of trainable parameters
        total_parameters = self.count_variables()

        # Step 2: Write the total number of parameters to a log file
        self.logs.write_to_log_file("#params: %d" % total_parameters)

    def actions_after_train(self) -> None:
        """
        Perform actions after completing the training process.

        1. Compute the total training time.
        2. Get the final training loss.
        3. Write the training time and final training loss to a log file.

        Returns:
            None
        """
        # Step 1: Compute the total training time
        end_time = time.time()
        train_time = end_time - self.start_time

        # Step 2: Get the final training loss
        final_loss = self.logs.get_last_loss()

        # Step 3: Write the training time and final training loss to a log file
        self.logs.write_to_log_file("Training time: {:.2f} seconds".format(train_time))
        self.logs.write_to_log_file("Final training loss: {:.4f}".format(final_loss))

        # TODO: Add more actions if needed


##########################################################################################
# Learner Model
# It is previously used to simulate learning trajectories based on PPE/HLR/OU process.
##########################################################################################


class BaseLearnerModel(BaseModel):
    """
    Base class for learner models.

    Args:
        mode (str): The mode of the learner model (e.g., 'train', 'val', 'test').
        device (str, optional): The device to run the model on (e.g., 'cpu', 'cuda:0'). Defaults to 'cpu'.
        logs (LogWriter, optional): An instance of LogWriter class for logging. Defaults to None.
    """

    def __init__(
        self,
        mode: str,
        device: torch.device,
        logs: logger.Logger,
    ) -> None:
        super(BaseLearnerModel, self).__init__()
        # Store the mode, device, and logs
        self.mode = mode
        self.device = device
        self.logs = logs

        # Initialize optimizer (set to None by default)
        self.optimizer = None

        # Set the model_path for saving the trained model
        if logs is not None:
            self.model_path = Path(logs.args.log_path, "Model/Model_{}_{}.pt")
        else:
            self.model_path = None

    @staticmethod
    def _find_whole_stats(
        all_feature: torch.Tensor,
        t: torch.Tensor,
        items: torch.Tensor,
        num_node: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute learning history statistics for the given input. Including
        - Number of total interactions for specific KC, number of success, and number of failure.
        - The time stampes of last interactions.

        Args:
            all_feature (torch.Tensor): Tensor of shape [bs, 1, num_step, 3].
            t (torch.Tensor): Tensor of shape [bs, num_step].
            items (torch.Tensor): Tensor of shape [bs, num_step].
            num_node (int): Number of nodes (items).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the computed whole_stats and whole_last_time tensors.
        """

        all_feature = all_feature.long()
        device = all_feature.device
        num_seq, num_step = t.shape

        # Allocate memory without initializing tensors
        whole_stats = torch.zeros(
            (num_seq, num_node, num_step, 3), device=device, dtype=torch.int64
        )
        whole_last_time = torch.zeros(
            (num_seq, num_node, num_step + 1), device=device, dtype=torch.int64
        )

        # Precompute index tensor
        seq_indices = torch.arange(num_seq, device=device)

        # Set initial values for whole_last_time
        whole_last_time[seq_indices, items[:, 0], 1] = t[:, 0]

        # Loop over time steps
        for i in range(1, num_step):
            cur_item = items[:, i]  # [num_seq, ]
            cur_feat = all_feature[:, 0, i]  # [bs, 1, 3]

            # Accumulate whole_stats
            whole_stats[:, :, i] = whole_stats[:, :, i - 1]  # whole_stats[:,:,i-1] #
            whole_stats[seq_indices, cur_item, i] = cur_feat

            whole_last_time[:, :, i + 1] = whole_last_time[
                :, :, i
            ]  # + whole_last_time[seq_indices,:,i]
            whole_last_time[seq_indices, cur_item, i + 1] = t[:, i]

        return whole_stats, whole_last_time

    @staticmethod
    def _compute_all_features(
        num_seq: int,
        num_node: int,
        time_step: int,
        device: torch.device,
        stats_cal_on_fly: bool = False,
        items: torch.Tensor = None,
        stats: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the tensor 'all_feature' based on the provided input arguments.

        Args:
            num_seq (int): The number of sequences (batch size).
            num_node (int): Number of nodes (items).
            time_step (int): The number of time steps.
            device (torch.device): The device to run the computation on (e.g., 'cpu', 'cuda:0').
            stats_cal_on_fly (bool, optional): Whether to compute 'all_feature' on the fly. Defaults to False.
            items (torch.Tensor, optional): Tensor of shape [num_seq, num_time_step] containing item indices. Defaults to None.
            stats (torch.Tensor, optional): Tensor of shape [num_seq, num_node, num_time_step, 3] containing precomputed stats. Defaults to None.

        Returns:
            torch.Tensor: The computed tensor 'all_feature'.
        """

        if stats_cal_on_fly or items is None:
            item_start = items[:, 0]
            all_feature = torch.zeros((num_seq, num_node, 3), device=device)
            all_feature[torch.arange(0, num_seq), item_start, 0] += 1
            all_feature[torch.arange(0, num_seq), item_start, 2] += 1
            all_feature = all_feature.unsqueeze(-2).tile((1, 1, time_step, 1))
        else:
            all_feature = stats.float()  # [num_seq/bs, num_node, num_time_step, 3]

        return all_feature

    @staticmethod
    def _initialize_parameter(
        shape: Tuple,
        device: torch.device,
    ) -> torch.nn.Parameter:
        """
        A static method to initialize a PyTorch parameter tensor with Xavier initialization.

        Args:
            shape (Tuple): A tuple specifying the shape of the parameter tensor.
            device (torch.device): A PyTorch device object specifying the device where the parameter tensor will be created.

        Returns:
            param (nn.Parameter): A PyTorch parameter tensor with the specified shape, initialized using Xavier initialization.

        """

        # create a parameter tensor with the specified shape on the specified device
        param = torch.nn.Parameter(torch.empty(shape, device=device))

        # apply Xavier initialization to the parameter tensor
        torch.nn.init.xavier_uniform_(param)

        return param

    def forward(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input data as tensors.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing output tensors, including predictions and labels.
        """

        # Extract input data from the feed_dict
        skills = feed_dict["skill_seq"]  # [batch_size, seq_len]
        times = feed_dict["time_seq"]  # [batch_size, seq_len]
        labels = feed_dict["label_seq"]  # [batch_size, seq_len]

        bs, _ = labels.shape
        self.num_seq = bs

        # Set initial state x0 for simulation
        x0 = torch.zeros((bs, self.num_node), requires_grad=True).to(labels.device)
        if self.num_node > 1:
            x0[torch.arange(bs), skills[:, 0]] += labels[:, 0]
            items = skills
        else:
            x0[:, 0] += labels[:, 0]
            items = None

        # Prepare stats for simulation
        stats = torch.stack(
            [
                feed_dict["num_history"],
                feed_dict["num_success"],
                feed_dict["num_failure"],
            ],
            dim=-1,
        )
        stats = stats.unsqueeze(1)

        # Perform simulation using the 'simulate_path' method
        out_dict = self.simulate_path(
            x0=x0,
            t=times,
            items=items,
            user_id=feed_dict["user_id"],
            stats=stats,
        )

        # Update the output dictionary with predictions and labels
        out_dict.update(
            {
                "prediction": out_dict["x_item_pred"],  # Add the prediction tensor
                "label": labels.unsqueeze(1),  # Add the label tensor [bs, 1, time]
            }
        )

        return out_dict

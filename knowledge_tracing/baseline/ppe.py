from collections import defaultdict
from typing import List, Dict

import numpy as np

import torch
import torch.nn as nn

from knowledge_tracing.baseline import *
from knowledge_tracing.baseline.basemodel import BaseModel, BaseLearnerModel
from knowledge_tracing.utils import utils, logger
from knowledge_tracing.data.data_loader import DataReader
from knowledge_tracing.baseline import EPS, T_SCALE


class PPE(BaseLearnerModel):
    """
    An implementation of the PPE model, extending the BaseLearnerModel,
    original paper: https://pubmed.ncbi.nlm.nih.gov/29498437/

    Args:
        args (object): An object containing arguments related to the model.
        corpus (DataReader): KT data loader.
        logs (Logger): The clasee containing all of the logging methods. Defaults to None.
        lr (float, optional): Learning rate for the model. Defaults to 0.1.
        variable_x (float, optional): Steepness of weighting for weight each repetition decreases exponentially with time. Defaults to 0.6.
        variable_b (float, optional): Base level of the decay. Defaults to 0.04.
        variable_m (float, optional): Slope of the decay. Defaults to 0.08.
        variable_tau (float, optional): Exponential . Defaults to 0.9.
        variable_s (float, optional): Exponential temperature. Defaults to 0.04.
        num_seq (int, optional): Number of sequences. Defaults to 1.
        num_node (int, optional): Number of nodes. Defaults to 1.
        mode (str, optional): Mode of operation. Defaults to "ls_split_time".
        device (str, optional): Device to use (e.g., 'cpu', 'cuda'). Defaults to "cpu".

    """

    def __init__(
        self,
        args,
        corpus: DataReader,
        logs: logger.Logger = None,
        lr: float = 0.1,
        variable_x: float = 0.6,
        variable_b: float = 0.04,
        variable_m: float = 0.08,
        variable_tau: float = 0.9,
        variable_s: float = 0.04,
        num_seq: int = 1,
        num_node: int = 1,
        mode: str = "ls_split_time",
        device: torch.device = torch.device("cpu"),
    ):
        self.args = args
        self.logs = logs
        self.num_node = num_node
        self.mode = mode
        self.device = device

        if args.multi_node:
            self.num_node = int(corpus.n_skills)
        else:
            self.num_node = 1

        if "ls_" in args.train_mode:
            self.num_seq = int(corpus.n_users)
        else:
            self.num_seq = num_seq

        self.lr = torch.tensor(lr, device=device).float()
        self.variable_x = torch.tensor(variable_x, device=device).float()
        self.variable_b = torch.tensor(variable_b, device=device).float()
        self.variable_m = torch.tensor(variable_m, device=device).float()
        self.variable_tau = torch.tensor(variable_tau, device=device).float()
        self.variable_s = torch.tensor(variable_s, device=device).float()

        BaseLearnerModel.__init__(
            self, mode=args.train_mode, device=args.device, logs=logs
        )

    def _init_weights(self) -> None:
        """
        Initialize the weights based on the chosen training mode.

        This method initializes the model's weights based on the selected training mode. It assigns
        appropriate values to the learning rate, variables related to the mode, and other parameters.

        Note:
            This method is intended for internal use within the PPE class.

        Raises:
            ValueError: If an invalid mode is specified.

        """

        if self.mode != "synthetic":
            try:
                shape = utils.get_theta_shape(self.num_seq, self.num_node, 1)[
                    self.mode.lower()
                ]
            except KeyError:
                raise ValueError(f"Invalid mode: {self.mode}")
            self.lr = self._initialize_parameter(shape, self.device)
            self.variable_x = self._initialize_parameter(shape, self.device)
            self.variable_b = self._initialize_parameter(shape, self.device)
            self.variable_m = self._initialize_parameter(shape, self.device)

            tau = torch.ones_like(self.lr, device=self.device) * 0.9
            s = torch.ones_like(self.lr, device=self.device) * 0.04
            self.variable_tau = nn.Parameter(tau, requires_grad=False)
            self.variable_s = nn.Parameter(s, requires_grad=False)

    def simulate_path(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        items: torch.Tensor = None,
        stats_cal_on_fly: bool = False,
        stats: torch.Tensor = None,
        user_id: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate the learner model's path over time.

        Args:
            x0: shape[num_seq/batch_size, num_node]; the initial state of the learner model
            t: shape[num_seq/batch_size, num_time_step]
            items: [num_seq/batch_size, num_time_step];
                ** it cannot be None when mode=synthetic
            stats_cal_on_fly: whether calculate the stats of history based on the prediction
                ** TODO test. it causes gradient error now
            stats: [num_seq/batch_size, num_node, num_time_step, 3]; it contains [N_total, N_success, N_failure]
        """
        assert t.numel() > 0  # check if tensor is not empty
        num_node = x0.shape[-1]
        num_seq, time_step = t.shape

        dt = torch.diff(t).unsqueeze(1)
        dt = torch.tile(dt, (1, num_node, 1)) / T_SCALE + EPS  # [batch_size, num_node, time-1]

        # ----- compute the stats of history -----
        if items == None or num_node == 1:
            items = torch.zeros_like(t, device=self.device, dtype=torch.long)
        all_feature = self._compute_all_features(
            num_seq, num_node, time_step, self.device, stats_cal_on_fly, items, stats
        )
        whole_stats, whole_last_time = self._find_whole_stats(
            all_feature, t, items, num_node
        )

        # ----- adapt to different modes -----
        if "simple" in self.mode:
            user_id = Ellipsis
        elif "ls_" in self.mode or "ln_" in self.mode:
            user_id = user_id

        batch_lr = torch.relu(self.lr[user_id]) + EPS
        batch_x = torch.relu(self.variable_x[user_id]) + EPS
        batch_b = torch.relu(self.variable_b[user_id]) + EPS
        batch_m = torch.relu(self.variable_m[user_id]) + EPS
        batch_tau = self.variable_tau[user_id]
        batch_s = self.variable_s[user_id]

        # ----- simulate the path -----
        t = torch.tile(t.unsqueeze(1), (1, num_node, 1))
        x_pred = [x0.unsqueeze(-1)]
        x_item_pred = [x0[torch.arange(0, num_seq), items[:, 0]][:, None]]

        for i in range(1, time_step):
            cur_item = items[:, i]  # [num_seq, ]

            # for PPE part
            # - small d (decay)
            cur_repeat = whole_stats[:, :, i, 0]
            cur_history_last_time = whole_last_time[
                :, :, : i + 1
            ]  # [batch_size, num_node, i+1]
            lags = (
                torch.diff(cur_history_last_time) / T_SCALE + EPS
            )  # [batch_size, num_node, i]
            lag_mask = lags > 0
            dn = ((1 / torch.log(abs(lags + EPS) + np.e)) * lag_mask).sum(dim=-1) / (
                cur_repeat - 1 + EPS
            )  # [batch_size, num_node]
            dn = batch_b + batch_m * dn.unsqueeze(-1)  # [batch_size, num_node, 1]

            # - big T
            small_t = (
                t[..., i : i + 1] - whole_last_time[..., : i + 1]
            ) / T_SCALE + EPS
            small_t = torch.minimum(small_t, torch.tensor(1e2))
            big_t = torch.nan_to_num(torch.pow(small_t + EPS, -batch_x)) / (
                torch.sum(
                    torch.nan_to_num(torch.pow(small_t + EPS, -batch_x)),
                    1,
                    keepdims=True,
                )
                + EPS
            )
            big_t = torch.sum(big_t * small_t, -1)[..., None]  # [batch_size, num_node]
            big_t = torch.nan_to_num(big_t)

            big_t_mask = big_t != 0
            mn = (
                torch.nan_to_num(
                    torch.pow((whole_stats[:, :, i : i + 1, 0] + 1), batch_lr)
                )
                * torch.nan_to_num(torch.pow((big_t + EPS), -dn))
                * big_t_mask
            )
            mn = torch.nan_to_num(mn)

            pn = 1 / (
                1
                + torch.nan_to_num(
                    torch.exp((batch_tau - mn) / (batch_s + EPS) + EPS) + EPS
                )
            )

            # ----- update the stats -----
            if stats_cal_on_fly or self.mode == "synthetic":
                success = (pn >= 0.5) * 1

                all_feature[torch.arange(num_seq), cur_item, i:, 0] += 1
                all_feature[torch.arange(num_seq), :, i:, 1] += success
                all_feature[torch.arange(num_seq), :, i:, 2] += 1 - success

            x_pred.append(pn)
            x_item_pred.append(pn[torch.arange(num_seq), cur_item])

        x_pred = torch.cat(x_pred, -1)  # [num_seq, num_node, time_step]
        x_item_pred = torch.stack(x_item_pred, -1)
        params = {"x_item_pred": x_item_pred}

        return params

    def forward_cl(
        self,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a forward pass of the learner model with continual learning.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input data tensors.
            idx (int, optional): Index for clipping the input sequences. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing output tensors from the forward pass.
        """
        # Extract input tensors from feed_dict
        cur_feed_dict = utils.get_feed_continual(
            keys=[
                "skill_seq",
                "label_seq",
                "time_seq",
                "num_history",
                "num_success",
                "num_failure",
                "user_id",
            ],
            data=feed_dict,
            idx=idx,
        )

        out_dict = self.forward(cur_feed_dict)

        return out_dict

    def evaluate_cl(
        self,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
        metrics=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the learner model's performance.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input data tensors.
            idx (int, optional): Index of the evaluation batch. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing evaluation results.
        """
        test_step = 10

        skills = feed_dict["skill_seq"][:, idx : idx + test_step + 1]
        labels = feed_dict["label_seq"][:, idx : idx + test_step + 1]
        times = feed_dict["time_seq"][:, idx : idx + test_step + 1]

        batch_size, _ = labels.shape
        self.num_seq = batch_size

        x0 = torch.zeros((batch_size, self.num_node)).to(labels.device)
        if self.num_node > 1:
            x0[torch.arange(batch_size), skills[:, 0]] += labels[:, 0]
            items = skills
        else:
            x0[:, 0] += labels[:, 0]
            items = None

        stats = torch.stack(
            [
                feed_dict["num_history"],
                feed_dict["num_success"],
                feed_dict["num_failure"],
            ],
            dim=-1,
        )
        stats = stats.unsqueeze(1)

        out_dict = self.simulate_path(
            x0=x0,
            t=times,
            items=items,
            user_id=feed_dict["user_id"],
            stats=stats,
            stats_cal_on_fly=True,
        )

        labels = labels.unsqueeze(1)[..., 1:].float()
        prediction = out_dict["x_item_pred"][..., 1:]

        return self.pred_evaluate_method(
            prediction.flatten().cpu(), labels.flatten().cpu(), metrics
        )

    def predictive_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
        single_step: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform predictive modeling using the learner model.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input data tensors.
            single_step (bool, optional): Whether to perform single-step prediction. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing predicted outputs and related information.
        """

        train_step = int(self.args.max_step * self.args.train_time_ratio)

        skills = feed_dict["skill_seq"][:, train_step - 1 :]  # [batch_size, seq_len]
        times = feed_dict["time_seq"][:, train_step - 1 :]  # [batch_size, seq_len]
        labels = feed_dict["label_seq"][:, train_step - 1 :]  # [batch_size, seq_len]

        batch_size, _ = labels.shape
        self.num_seq = batch_size

        x0 = torch.zeros((batch_size, self.num_node)).to(labels.device)
        if self.num_node > 1:
            x0[torch.arange(batch_size), skills[:, 0]] += labels[:, 0]
            items = skills
        else:
            x0[:, 0] += labels[:, 0]
            items = None

        stats = torch.stack(
            [
                feed_dict["num_history"],
                feed_dict["num_success"],
                feed_dict["num_failure"],
            ],
            dim=-1,
        )
        stats = stats.unsqueeze(1)

        out_dict = self.simulate_path(
            x0=x0,
            t=times,
            items=items,
            user_id=feed_dict["user_id"],
            stats=stats,
            stats_cal_on_fly=True,
        )

        out_dict.update(
            {
                "prediction": out_dict["x_item_pred"][..., 1:],
                "label": labels.unsqueeze(1)[..., 1:],  # [batch_size, 1, time]
            }
        )

        return out_dict

    def loss(
        self,
        feed_dict: Dict[str, torch.Tensor],
        out_dict: Dict[str, torch.Tensor],
        metrics: List[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss and evaluation metrics for the learner model.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input data tensors.
            out_dict (Dict[str, torch.Tensor]): A dictionary containing output tensors from the model.
            metrics (List[str], optional): List of evaluation metric names. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing computed loss values and evaluation metrics.
        """
        losses = defaultdict(lambda: torch.zeros((), device=self.device))

        pred = out_dict["x_item_pred"]
        label = out_dict["label"]

        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred, label.float())
        losses["loss_total"] = bceloss

        if metrics != None:
            pred = pred.detach().cpu().data.numpy()
            label = label.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, label, metrics)
            losses.update(evaluations)

        return losses

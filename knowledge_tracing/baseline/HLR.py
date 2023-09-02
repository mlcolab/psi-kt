import numpy as np
import argparse

import torch
import torch.nn as nn

from typing import List, Dict

from knowledge_tracing.baseline.BaseModel import BaseModel, BaseLearnerModel
from knowledge_tracing.utils import logger
from knowledge_tracing.data.data_loader import DataReader

from enum import Enum

from knowledge_tracing.baseline import *


class HLR(BaseLearnerModel):
    """
    An implementation of the HLR model, extending the BaseLearnerModel.

    This class defines the HLR (Half-life regression) model,
    which extends the BaseLearnerModel class. It includes methods for parsing model
    arguments and initializing the instance.

    Modified from:
        https://github.com/duolingo/halflife-regression/blob/0041df0dcd436bf1b4aa7a17a020d9c670db70d8/experiment.py

    Args:
        theta: [bs/num_seq, num_node, 3]; should be 3D vector indicates the parameters of the model;
            the näive version is to compute the dot product of theta and [N_total, N_success, N_failure]
        base: the base of HLR model
        num_seq: when mode==synthetic, it is the number of sequences to generate;
            is mode==train, it is the number of batch size
        items: [bs/num_seq, time_step]
        mode: [synthetic, train]; synthetic is to generate new sequences based on given theta; train is to
            train the parameters theta given observed data.
        device: cpu or cuda to put all variables and train the model

    """

    def __init__(
        self,
        args: argparse.Namespace,
        corpus: DataReader,
        logs: logger.Logger,
        theta: torch.Tensor = None,
        base: float = 2.0,
        num_seq: int = 1,
        num_node: int = 1,
        nx_graph: np.array = None,
    ):
        if args.multi_node:
            self.num_node = int(corpus.n_skills)
        else:
            self.num_node = 1

        if "ls_" in args.train_mode:
            self.num_seq = int(corpus.n_users)
        else:
            self.num_seq = num_seq

        self.base = base
        self.args = args
        self.theta = theta
        self.mode = args.train_mode
        self.device = args.device

        if num_node > 1 and self.mode == "synthetic":
            self.adj = torch.tensor(nx_graph, device=self.device)
            assert self.adj.shape[-1] == num_node
        else:
            self.adj = None

        super().__init__(mode=args.train_mode, device=args.device, logs=logs)

    def _init_weights(self) -> None:
        # Training mode choosing
        class ThetaShape(Enum):
            SIMPLE_SPLIT_TIME = (1, 1, 3)
            SIMPLE_SPLIT_LEARNER = (1, 1, 3)
            LS_SPLIT_TIME = (self.num_seq, 1, 3)
            NS_SPLIT_TIME = (1, self.num_node, 3)
            NS_SPLIT_LEARNER = (1, self.num_node, 3)
            LN_SPLIT_TIME = (self.num_seq, self.num_node, 3)

        if self.mode == "synthetic":
            self.theta = torch.tensor(self.theta, device=self.device).float()
        else:
            try:
                shape = ThetaShape[self.mode.upper()].value
            except KeyError:
                raise ValueError(f"Invalid mode: {self.mode}")
            self.theta = self._initialize_parameter(shape, self.device)

    @staticmethod
    def hclip(
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clip the input tensor of half-life values to a specific range.

        Args:
            h (torch.Tensor): Input tensor containing half-life values.

        Returns:
            torch.Tensor: Clipped tensor with half-life values bounded within a specific range.
        """
        MIN_HALF_LIFE = torch.tensor(15.0 / (24 * 60), device=h.device)  # 15 minutes
        MAX_HALF_LIFE = torch.tensor(274.0, device=h.device)  # 9 months
        return torch.min(torch.max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)

    @staticmethod
    def pclip(
        p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clip the model predictions to ensure they fall within a specific range.

        This function helps with loss optimization by bounding the model predictions between
        a minimum and maximum value.

        Args:
            p (torch.Tensor): Input tensor containing model predictions.

        Returns:
            torch.Tensor: Clipped tensor with model predictions bounded within a specific range.
        """
        MIN_P = torch.tensor(0.0001, device=p.device)
        MAX_P = torch.tensor(0.9999, device=p.device)
        return torch.min(torch.max(p, MIN_P), MAX_P)

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
            x0 (torch.Tensor): Initial state of the learner model.
            t (torch.Tensor): Time steps for simulation.
            items (torch.Tensor, optional): Items associated with each time step.
                                            Defaults to None.
            stats_cal_on_fly (bool, optional): Whether to calculate history statistics based on predictions.
                                            Defaults to False.
            stats (torch.Tensor, optional): History statistics. Defaults to None.
            user_id (int, optional): User identifier. Defaults to None.

        Returns:
            dict: A dictionary containing simulated path parameters, including half-life, predictions,
                history statistics, and more.
        """
        assert t.numel() > 0  # check if tensor is not empty
        num_seq, time_step = t.shape

        dt = torch.diff(t).unsqueeze(1)
        dt = (
            torch.tile(dt, (1, self.num_node, 1)) / T_SCALE + EPS
        )  # [bs, num_node, time-1]

        # ----- compute the stats of history -----
        if items == None or self.num_node == 1:
            items = torch.zeros_like(t, device=self.device, dtype=torch.long)
        all_feature = self._compute_all_features(
            num_seq,
            self.num_node,
            time_step,
            self.device,
            stats_cal_on_fly,
            items,
            stats,
        )  #
        whole_stats, whole_last_time = self._find_whole_stats(
            all_feature, t, items, self.num_node
        )

        # ----- adapt to different modes -----
        theta_map = {
            "simple": self.theta,
            "ls_split_time": self.theta[user_id],
            "ns_": torch.tile(self.theta, (num_seq, 1, 1)),
            "ln_": self.theta[user_id],
        }
        batch_theta = None
        for mode, value in theta_map.items():
            if mode in self.mode:
                batch_theta = value
                break

        # ----- simulate the path -----
        x_pred = [x0]
        x_item_pred = [x0[torch.arange(num_seq), items[:, 0]]]
        half_lifes = [torch.zeros_like(x0, device=self.device)]

        for i in range(1, time_step):
            cur_item = items[:, i]  # [num_seq, ]
            cur_dt = (
                t[:, None, i] - whole_last_time[..., i]
            ) / T_SCALE + eps  # [bs, num_node]
            cur_feat = whole_stats[:, :, i]

            feat = torch.mul(cur_feat, batch_theta).sum(-1)
            feat = torch.minimum(feat, torch.tensor(1e2).to(self.device))
            # half_life = self.hclip(self.base ** feat)
            half_life = self.base**feat
            p_all = self.pclip(self.base ** (-cur_dt / half_life))  # [bs, num_node]
            p_item = p_all[torch.arange(num_seq), cur_item]  # [bs, ]

            if stats_cal_on_fly or self.mode == "synthetic":
                success = nn.functional.gumbel_softmax(
                    torch.log(p_item), hard=True
                )  # TODO
                success = success.unsqueeze(-1)
                all_feature[torch.arange(num_seq), cur_item, i:, 0] += 1
                all_feature[torch.arange(num_seq), cur_item, i:, 1] += success
                all_feature[torch.arange(num_seq), cur_item, i:, 2] += 1 - success

            half_lifes.append(half_life)
            x_item_pred.append(p_item)
            x_pred.append(p_all)

        half_lifes = torch.stack(half_lifes, -1)
        x_pred = torch.stack(x_pred, -1)
        x_item_pred = torch.stack(x_item_pred, -1).unsqueeze(1)

        params = {
            # NOTE: the first element of the following values in out_dict is not predicted
            "half_life": half_lifes,  # [bs, num_node, times]
            "x_item_pred": x_item_pred,  # [bs, 1, times]
            "x_all_pred": x_pred,  # [bs, num_node, times]
            "num_history": all_feature[..., 0],  # [bs, num_node, times]
            "num_success": all_feature[..., 1],
            "num_failure": all_feature[..., 2],
        }

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
        cur_feed_dict = {
            "skill_seq": feed_dict["skill_seq"][:, : idx + 1],
            "label_seq": feed_dict["label_seq"][:, : idx + 1],
            "time_seq": feed_dict["time_seq"][:, : idx + 1],
            "num_history": feed_dict["num_history"][:, : idx + 1],
            "num_success": feed_dict["num_success"][:, : idx + 1],
            "num_failure": feed_dict["num_failure"][:, : idx + 1],
            "user_id": feed_dict["user_id"],
        }

        out_dict = self.forward(cur_feed_dict)

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

        losses = {}

        pred = out_dict["prediction"]
        label = out_dict["label"]

        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred, label.float())
        losses["loss_total"] = bceloss

        if metrics:
            pred = pred.detach().cpu().data.numpy()
            label = label.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, label, metrics)
            losses.update(evaluations)

        # For debugging results
        if "simple" in self.mode:
            losses["theta_0"] = self.theta.clone()[0, 0, 0]
            losses["theta_1"] = self.theta.clone()[0, 0, 1]
            losses["theta_2"] = self.theta.clone()[0, 0, 2]

        return losses

    def predictive_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
        single_step: bool = True,
    ) -> Dict[str, torch.Tensor]:
        train_step = int(self.args.max_step * self.args.train_time_ratio)

        skills_test = feed_dict["skill_seq"][:, train_step - 1 :]
        labels_test = feed_dict["label_seq"][:, train_step - 1 :]
        times_test = feed_dict["time_seq"][:, train_step - 1 :]

        bs, _ = labels_test.shape
        self.num_seq = bs

        x0 = torch.zeros((bs, self.num_node)).to(labels_test.device)
        if self.num_node > 1:
            x0[torch.arange(bs), skills_test[:, 0]] += labels_test[:, 0]
            items = skills_test
        else:
            x0[:, 0] += labels_test[:, 0]
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
            t=times_test,
            items=items,
            user_id=feed_dict["user_id"],
            stats=stats,
            stats_cal_on_fly=True,
        )

        out_dict.update(
            {
                "prediction": out_dict["x_item_pred"][..., 1:],
                "label": labels_test.unsqueeze(1)[..., 1:],  # [bs, 1, time]
            }
        )

        return out_dict

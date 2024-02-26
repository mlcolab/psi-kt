from collections import defaultdict
from typing import Dict, Any

import numpy as np

import torch
import torch.nn as nn

from knowledge_tracing.baseline import *
from knowledge_tracing.baseline.basemodel import BaseLearnerModel, BaseModel


##########################################################################################
# OU Process
# This is not used in current experiments. Only for simulation purpose.
##########################################################################################


class VanillaOU(BaseLearnerModel):
    """
    Modified from
        https://github.com/jwergieluk/ou_noise/tree/master/ou_noise
        https://github.com/felix-clark/ornstein-uhlenbeck/blob/master/ornstein_uhlenbeck.py
        https://github.com/369geofreeman/Ornstein_Uhlenbeck_Model/blob/main/research.ipynb
    Args:
        mean_rev_speed: [bs, num_node, 1] the speed of mean reversion in OU process (alpha).
        mean_rev_level: [bs, num_node, 1] the mean level of OU process (mu).
        vola: [bs, num_node, 1] the volatility of OU process (sigma).
        num_seq: when training mode, the num_seq will be automatically the number of batch size;
                    when synthetic mode, the num_seq will be the number of sequences to generate
        mode: can be 'training' or 'synthetic'
        device: cpu or cuda
        logs: the logger

    """

    def __init__(
        self,
        mean_rev_speed: torch.Tensor = None,
        mean_rev_level: torch.Tensor = None,
        vola: torch.Tensor = None,
        num_seq: int = 1,
        num_node: int = 1,
        mode: str = "train",
        nx_graph: np.ndarray = None,
        device: torch.device = "cpu",
        logs: Any = None,
    ):
        super().__init__(mode=mode, device=device, logs=logs)
        self.num_node = num_node
        self.num_seq = num_seq

        if num_node > 1:
            self.adj = torch.tensor(nx_graph, device=self.device)
            assert self.adj.shape[-1] == num_node
        else:
            self.adj = None

        self.mean_rev_speed = nn.Parameter(
            torch.empty(1, 1, 1, device=device), requires_grad=True
        )
        self.mean_rev_level = nn.Parameter(
            torch.empty(1, 1, 1, device=device), requires_grad=True
        )
        self.vola = nn.Parameter(
            torch.empty(1, 1, 1, device=device), requires_grad=True
        )

        if "simple_" in mode:
            self.initialize_parameters(1, 1, 1, device)
        elif "ls_" in mode:
            self.initialize_parameters(num_seq, 1, 1, device)
        elif "ns_" in mode:
            self.initialize_parameters(1, num_node, 1, device)
        elif "ln_" in mode:
            self.initialize_parameters(num_seq, num_node, 1, device)
        elif mode == "synthetic":
            assert mean_rev_speed is not None
            self.mean_rev_speed = mean_rev_speed
            self.mean_rev_level = mean_rev_level
            self.vola = vola
        else:
            raise Exception("It is not a compatible mode")

        assert torch.min(self.mean_rev_speed) >= 0
        assert torch.min(self.vola) >= 0

    def _initialize_parameters(
        self, num_seq: int, num_node: int, num_values: int, device: torch.device
    ) -> None:
        """
        Initialize the parameters of OU process
        Args:
            num_seq: the number of sequences
            num_node: the number of nodes
            num_values: the number of values for each node
        """

        speed = torch.rand((num_seq, num_node, num_values), device=device)
        level = torch.rand((num_seq, num_node, num_values), device=device)
        vola = torch.rand((num_seq, num_node, num_values), device=device)
        self.mean_rev_speed = nn.Parameter(speed, requires_grad=True)
        self.mean_rev_level = nn.Parameter(level, requires_grad=True)
        self.vola.data = nn.Parameter(vola, requires_grad=True)

    def variance(
        self,
        t: torch.Tensor,
        speed: torch.Tensor = None,
        vola: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        The variances introduced by the parameter vola, time difference and Wiener process (Gaussian noise)
        Args:
            t: [bs/num_seq, num_node, times-1]; the time difference
            speed: [bs/num_seq, num_node/1]
            vola: [bs/num_seq, num_node/1]
        """
        speed = speed if speed is not None else self.mean_rev_speed
        vola = vola if vola is not None else self.vola
        speed = speed.unsqueeze(-1)
        vola = vola.unsqueeze(-1)

        return vola * vola * (1.0 - torch.exp(-2.0 * speed * t)) / (2 * speed + EPS)

    def std(self, t, speed=None, vola=None) -> torch.Tensor:
        """
        Args:
            t: [num_seq/bs, num_node, times] usually is the time difference of a sequence
        """
        return torch.sqrt(self.variance(t, speed, vola) + EPS)

    def mean(self, x0, t, speed=None, level=None) -> torch.Tensor:
        """
        Args:
            x0:
            t:
        """
        speed = speed if speed is not None else self.mean_rev_speed
        level = level if level is not None else self.mean_rev_level

        return x0 * torch.exp(-speed * t) + (1.0 - torch.exp(-speed * t)) * level

    def logll(self, x, t, speed=None, level=None, vola=None) -> torch.Tensor:
        """
        Calculates log likelihood of a path
        Args:
            t: [num_seq/bs, time_step]
            x: [num_seq/bs, time_step] it should be the same size as t
        Return:
            log_pdf: [num_seq/bs, 1]
        """
        speed = speed if speed is not None else self.mean_rev_speed
        level = level if level is not None else self.mean_rev_level
        vola = vola if vola is not None else self.vola

        dt = torch.diff(t)
        dt = torch.log(dt)
        mu = self.mean(x, dt, speed, level)
        sigma = self.std(dt, speed, vola)
        var = self.variance(dt, speed, vola)

        dist = torch.distributions.normal.Normal(loc=mu, scale=var)
        log_pdf = dist.log_prob(x).sum(-1)

        return log_pdf

    def simulate_path(
        self, x0, t, items=None, user_id=None, stats_cal_on_fly=None, stats=None
    ) -> Dict[str, torch.Tensor]:
        """
        Simulates a sample path or forward based on the parameters (speed, level, vola)
        dX = speed*(level-X)dt + vola*dB
        ** the num_node here can be considered as multivariate case of OU process
            while the correlations between nodes wdo not matter
        Args:
            x0: [num_seq/bs, num_node] the initial states for each node in each sequences
            t: [num_seq/bs, time_step] the time points to sample (or interact);
                It should be the same for all nodes
            items:
        Return:
            x_pred: [num_seq/bs, num_node, time_step]
        """
        assert len(t) > 0
        num_node = x0.shape[-1]
        num_seq, time_step = t.shape

        dt_normalize = T_SCALE
        dt = torch.diff(t).unsqueeze(1) / dt_normalize + EPS
        dt = torch.tile(dt, (1, num_node, 1))  # [bs, num_node, time-1]

        if items == None or num_node == 1:
            items = torch.zeros_like(t, device=self.device, dtype=torch.long)

        if "simple" in self.mode:
            batch_speed = torch.tile(
                torch.relu(self.mean_rev_speed)[..., 0] + EPS, (num_seq, num_node)
            )
            batch_level = torch.tile(self.mean_rev_level[..., 0], (num_seq, num_node))
            batch_vola = torch.tile(
                torch.relu(self.vola)[..., 0] + EPS, (num_seq, num_node)
            )
        elif ("ls_" in self.mode) or ("ln" in self.mode):
            batch_speed = torch.relu(self.mean_rev_speed[user_id])[..., 0] + EPS
            batch_level = self.mean_rev_level[user_id][..., 0]
            batch_vola = torch.relu(self.vola[user_id])[..., 0] + EPS
        elif "ns_" in self.mode:
            batch_speed = torch.tile(
                torch.relu(self.mean_rev_speed[user_id])[..., 0] + EPS, (num_seq, 1)
            )
            batch_level = torch.tile(self.mean_rev_level[user_id][..., 0], (num_seq, 1))
            batch_vola = torch.tile(
                torch.relu(self.vola[user_id])[..., 0] + EPS, (num_seq, 1)
            )
        else:
            batch_speed = None
            batch_level = None
            batch_vola = None

        scale = self.std(dt, speed=batch_speed, vola=batch_vola)  # [bs, num_node, t-1]
        noise = torch.randn(size=scale.shape, device=self.device)

        x_last = x0
        x_pred = []
        x_pred.append(x_last)
        x_item_pred = []
        x_item_pred.append(x0[torch.arange(0, num_seq), items[:, 0]])

        if stats_cal_on_fly or self.mode == "synthetic":
            item_start = items[:, 0]
            all_feature = torch.zeros((num_seq, num_node, 3), device=self.device)
            all_feature[torch.arange(0, num_seq), item_start, 0] += 1
            all_feature[torch.arange(0, num_seq), item_start, 2] += 1
            all_feature = all_feature.unsqueeze(-2).tile((1, 1, time_step, 1))
        else:
            all_feature = stats.float()  # [num_seq/bs, num_node, num_time_step, 3]

        _, whole_last_time = self._find_whole_stats(all_feature, t, items, num_node)

        for i in range(1, time_step):
            cur_item = items[:, i]

            cur_dt = (
                t[:, None, i] - whole_last_time[..., i]
            ) / dt_normalize + EPS  # [bs, num_node]

            x_next = self.mean(
                x_last, cur_dt, speed=batch_speed, level=batch_level
            )  # [bs, num_node]
            x_next = x_next + noise[..., i - 1] * scale[..., i - 1]
            x_pred.append(x_next)

            x_pred_item = x_next[torch.arange(0, num_seq), cur_item]  # [bs, ]
            x_item_pred.append(x_pred_item)

            x_last = x_next

        x_pred = torch.stack(x_pred, -1)
        x_item_pred = torch.stack(x_item_pred, -1).unsqueeze(1)

        params = {
            "x_original_item_pred": x_item_pred,  # [bs, 1, times]
            "x_original_all_pred": x_pred,  # [bs, num_node, times]
            "x_all_pred": torch.sigmoid(x_pred),
            "x_item_pred": torch.sigmoid(x_item_pred),
            "std": noise * scale,
            "user_id": user_id,
            "times": t,
            "items": items,
        }

        return params

    def loss(self, feed_dict, out_dict, metrics=None) -> Dict[str, torch.Tensor]:
        losses = defaultdict(lambda: torch.zeros((), device=self.device))

        pred = out_dict["prediction"]
        label = out_dict["label"]

        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred, label.float())
        losses["loss_total"] = bceloss

        if metrics != None:
            pred = pred.detach().cpu().data.numpy()
            label = label.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, label, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]

        return losses


class GraphOU(VanillaOU):
    """
    Extend the VanillaOU to graph case by adding interactions for each random variable.
    Args:
        mean_rev_speed: [bs, num_node, 1] the speed of mean reversion in OU process (alpha).
        mean_rev_level: [bs, num_node, 1] the mean level of OU process (mu).
        vola: [bs, num_node, 1] the volatility of OU process (sigma).
        gamma: [bs, num_node, 1] the interaction strength of OU process (gamma).
        num_seq: when training mode, the num_seq will be automatically the number of batch size;
        num_node: the number of nodes in the graph
        mode: can be 'training' or 'synthetic'
        nx_graph: the adjacency matrix of the graph
        device: cpu or cuda
        logs: the logger
    """

    def __init__(
        self,
        mean_rev_speed=None,
        mean_rev_level=None,
        vola=None,
        gamma=None,
        num_seq=1,
        num_node=1,
        mode="train",
        nx_graph=None,
        device="cpu",
        logs=None,
    ):
        super().__init__(
            mean_rev_speed,
            mean_rev_level,
            vola,
            num_seq,
            num_node,
            mode,
            nx_graph,
            device,
            logs,
        )

        if "simple_" in mode:
            gamma = torch.rand((1, 1, 1), device=device)
            self.gamma = nn.Parameter(gamma, requires_grad=True)
        elif "ls_" in mode:
            gamma = torch.rand((num_seq, 1, 1), device=device)
            self.gamma = nn.Parameter(gamma, requires_grad=True)
        elif "ns_" in mode:
            gamma = torch.rand((1, num_node, 1), device=device)
            self.gamma = nn.Parameter(gamma, requires_grad=True)
        elif "ln_" in mode:
            gamma = torch.rand((num_seq, num_node, 1), device=device)
            self.gamma = nn.Parameter(gamma, requires_grad=True)

    def simulate_path(self, x0, t, items=None, user_id=None, stats=None) -> Dict:
        """
        Simulates a sample path or forward based on the parameters (speed, level, vola)
        dX = speed*(level-X)dt + vola*dB
        ** the num_node here can be considered as multivariate case of OU process
            while the correlations between nodes wdo not matter
        Args:
            x0: [num_seq/bs, num_node] the initial states for each node in each sequences
            t: [num_seq/bs, time_step] the time points to sample (or interact);
                It should be the same for all nodes
            items: [num_seq/bs, time_step]
        Return:
            x_pred: [num_seq/bs, num_node, time_step]
        """
        assert len(t) > 0
        omega = 0.5
        rho = 50
        self.rho = torch.tensor(rho, device=self.device)
        num_node = x0.shape[-1]
        num_seq, time_step = t.shape

        t = t
        dt = torch.diff(t).unsqueeze(1) / T_SCALE + EPS
        dt = torch.tile(dt, (1, num_node, 1)) + EPS  # [bs, num_node, time-1]

        if items == None or num_node == 1:
            items = torch.zeros_like(t, device=self.device, dtype=torch.long)

        if "simple" in self.mode:
            batch_speed = torch.tile(
                torch.relu(self.mean_rev_speed)[..., 0] + EPS, (num_seq, num_node)
            )
            batch_level = torch.tile(self.mean_rev_level[..., 0], (num_seq, num_node))
            batch_vola = torch.tile(
                torch.relu(self.vola)[..., 0] + EPS, (num_seq, num_node)
            )
            batch_gamma = torch.tile(self.gamma[..., 0], (num_seq, num_node))
        elif ("ls_" in self.mode) or ("ln_" in self.mode):
            batch_speed = torch.relu(self.mean_rev_speed[user_id])[..., 0] + EPS
            batch_level = self.mean_rev_level[user_id][..., 0]
            batch_vola = torch.relu(self.vola[user_id])[..., 0] + EPS
            batch_gamma = self.gamma[user_id][..., 0]
        elif "ns_" in self.mode:
            batch_speed = torch.tile(
                torch.relu(self.mean_rev_speed)[..., 0] + EPS, (num_seq, 1)
            )
            batch_level = torch.tile(self.mean_rev_level[..., 0], (num_seq, 1))
            batch_vola = torch.tile(torch.relu(self.vola)[..., 0] + EPS, (num_seq, 1))
            batch_gamma = torch.tile(self.gamma[..., 0], (num_seq, 1))

        # graph
        # TODO future work could test with multiple power of adj -> multi-step transition
        adj = self.adj.float()
        adj_t = torch.transpose(adj, -1, -2)
        assert num_node == adj.shape[-1]

        scale = self.std(dt, speed=batch_speed, vola=batch_vola)  # [bs, num_node, t-1]
        noise = torch.randn(size=scale.shape, device=self.device)

        x_pred = []
        x_item_pred = []
        x_last = x0  # [bs, num_node]
        x_pred.append(x_last)
        x_item_pred.append(x0[torch.arange(0, num_seq), items[:, 0]])

        # find degree 0
        in_degree = adj_t.sum(dim=-1)
        ind = torch.where(in_degree == 0)[0]

        for i in range(1, time_step):
            cur_item = items[:, i]  # [num_seq, ]

            empower = torch.einsum("ij, ai->aj", adj_t, x_last)
            empower = (1 / (in_degree[None, :] + EPS)) * batch_gamma * empower
            empower[:, ind] = 0

            stable = batch_speed
            tmp_mean_level = batch_level
            tmp_batch_speed = torch.relu(omega * empower + (1 - omega) * stable) + EPS

            x_next = self.mean(
                x_last, dt[..., i - 1], speed=tmp_batch_speed, level=tmp_mean_level
            )  # [num_seq/bs, num_node]
            x_next = x_next + noise[..., i - 1] * scale[..., i - 1]
            x_pred.append(x_next)

            x_pred_item = x_next[torch.arange(0, num_seq), cur_item]  # [bs, ]
            x_item_pred.append(x_pred_item)

            x_last = x_next

        x_pred = torch.stack(x_pred, -1)
        x_item_pred = torch.stack(x_item_pred, -1).unsqueeze(1)

        params = {
            "x_original_item_pred": x_item_pred,  # [bs, times]
            "x_original_all_pred": x_pred,  # [bs, num_node, times]
            "x_all_pred": torch.sigmoid(x_pred),
            "x_item_pred": torch.sigmoid(x_item_pred),
            "std": noise * scale,
            "user_id": user_id,
            "items": items,
            "times": t,
        }

        return params

    def loss(self, feed_dict, out_dict, metrics=None) -> Dict[str, torch.Tensor]:
        losses = defaultdict(lambda: torch.zeros((), device=self.device))

        pred = out_dict["prediction"]
        label = out_dict["label"]

        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred, label.float())
        losses["loss_total"] = bceloss

        if metrics != None:
            pred = pred.detach().cpu().data.numpy()
            label = label.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, label, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]

        return losses

from typing import List, Optional, Type
import argparse

import torch
from torch import nn
from torch.nn import LayerNorm, Sequential
import torch.nn.functional as F

from knowledge_tracing.groupkt import *
from knowledge_tracing.groupkt.modules import generate_fully_connected
from knowledge_tracing.utils.utils import create_rel_rec_send


class Graph(nn.Module):
    """
    A class representing a graph for a our GroupKT task.

    This class defines the structure and properties of the graph for the task.

    Args:
        device (torch.device): The device to run the computations on.
        num_nodes (int): The total number of nodes in the graph.
        args (argparse.Namespace): Command-line arguments containing various configuration settings.
    """

    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        args: argparse.Namespace,
    ) -> None:
        super(Graph, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.args = args

        self.norm_layer = LayerNorm
        self.dropout = args.dropout
        self.res_connection = True
        self.encoder_layer_sizes = None
        self.decoder_layer_sizes = None  # TODO

        self.time_log = args.time_log
        self.time_lag = args.time_lag

        self.emb_history = args.emb_history
        self.skill_base = args.skill_base
        self.problem_base = args.problem_base

        self.rec_ind, self.send_ind = create_rel_rec_send(self.num_nodes, self.device)

    def _init_weights(self) -> None:
        """
        Initialize the weights of the graph model.
        """
        pass

    def forward(
        self,
        feed_dict: dict,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the graph model.

        Args:
            feed_dict (dict): A dictionary containing input data.

        Returns:
            torch.Tensor: The output tensor from the graph model.
        """
        pass


class GraphTargetMessage(Graph):
    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        args: argparse.Namespace,
        emb_size: Optional[int] = None,
    ) -> None:
        super().__init__(device, num_nodes, args)
        self.emb_size = emb_size or args.emb_size
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize the weights of the graph model.

        This method initializes the weights for the graph model's layers based on the configuration settings.
        """

        a = max(self.emb_size, 64)
        layers_f = self.decoder_layer_sizes or [a, a]
        in_dim_f = self.emb_size
        layers_g = self.encoder_layer_sizes or [a, a]
        in_dim_g = self.emb_size * 2

        if self.emb_history == 1:
            self.emb_history_layer = generate_fully_connected(
                input_dim=in_dim_g,
                output_dim=self.emb_size,
                hidden_dims=layers_g,
                p_dropout=self.dropout,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=self.device,
                normalization=self.norm_layer,
                res_connection=self.res_connection,
            )

        self.interv_edges = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=self.emb_size,
            hidden_dims=layers_g,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )

        self.final_output_layer = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=2,
            hidden_dims=layers_f,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )

    def forward(
        self,
        feed_dict: dict,
        adj: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_base: Optional[torch.Tensor] = None,
        node_bias_base: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the graph model.

        Args:
            feed_dict (dict): A dictionary containing input data.
            adj (torch.Tensor): Adjacency matrix of the graph.
            node_embeddings (torch.Tensor): Node embeddings.
            node_base (Optional[torch.Tensor]): Node base embeddings.
            node_bias_base (Optional[torch.Tensor]): Node bias base embeddings.

        Returns:
            torch.Tensor: The output tensor from the forward pass.
        """

        skills = feed_dict["skill_seq"]  # [bs, seq_len]
        problems = feed_dict["problem_seq"]  # [bs, seq_len]
        times = feed_dict["time_seq"]  # [bs, seq_len]
        labels = feed_dict["label_seq"]  # [bs, seq_len]

        bs, time_steps = labels.shape
        time_lag = self.args.time_lag

        bs_skill_emb = torch.tile(node_embeddings, (bs, 1, 1)).to(self.device)

        if self.emb_history == 0:
            labels_pro = torch.where(labels == 0, -2, labels)
            labels_pro = torch.where(labels_pro == -1, 0, labels_pro)
            labels_pro = torch.where(labels_pro == -2, -1, labels_pro)
        else:
            labels_pro = labels

        delta_t = (
            (times[:, :, None] - times[:, None, :]).abs().double()
        )  # [bs, seq_len, seq_len] # symmetry ranging from 13.47e8-13.74e8
        delta_t = torch.log(delta_t + EPS) / torch.log(
            torch.tensor(self.time_log)
        )  # ??? time_log # ranging from 0 to 17

        preds = []

        # TODO
        for t in range(time_lag, time_steps):
            label_history = labels_pro[:, t - time_lag : t]  # [bs, time_lag]
            skill_history = skills[:, t - time_lag : t]  # [bs, time_lag]
            skill_target = skills[:, t : t + 1]  # [bs, 1]

            delta_t_weight = delta_t.transpose(-1, -2)[
                :, t, t - time_lag : t
            ]  # [bs, time_lag]  range from 0 to around 10
            emb_history = [
                bs_skill_emb[torch.arange(0, bs), skill_history[:, i]]
                for i in range(time_lag)
            ]
            emb_history = torch.stack(emb_history, dim=1)  # [bs, time_lag, emd_size]

            if self.emb_history == 0:
                emb_history *= label_history.unsqueeze(-1)
            elif self.emb_history == 1:
                label_history = torch.tile(
                    label_history.unsqueeze(-1), (1, 1, self.embedding_size)
                )
                emb_history = torch.cat([emb_history, label_history], dim=-1).double()
                emb_history = self.emb_history_layer(emb_history)

            # w_ij means the effect from i to j
            # column j has all of the effects coming to j
            # so index in j-th row of w_transpose
            graph_weight = adj.transpose(-1, -2)[
                :, torch.arange(0, bs), skill_target[:, 0]
            ]  # [num_graph, bs, num_nodes]
            cross_weight = [
                graph_weight[:, torch.arange(0, bs), skill_history[:, i]]
                for i in range(time_lag)
            ]
            cross_weight = torch.stack(cross_weight, -1)  # [bs, time_lag]

            emb_target = torch.tile(
                bs_skill_emb[torch.arange(0, bs), skill_target[:, 0]].unsqueeze(1),
                (1, time_lag, 1),
            )
            edge_feat = torch.cat(
                [emb_target, emb_history], dim=-1
            ).double()  # [bs, time_lag, emb_size*2]

            msg = self.interv_edges(edge_feat)  # [bs, time_lag, emb_size]

            # use a softmax here? TODO
            cross_effect = (cross_weight * torch.exp(-delta_t_weight)).unsqueeze(
                -1
            ) * msg
            cross_effect = cross_effect.sum(-2)  # [num_graph, bs, emb_size]

            pred = F.softmax(self.final_output_layer(cross_effect), dim=-1)[
                ..., 1:
            ]  # problem_base_target + skill_base_target
            preds.append(pred)


class GraphRecurrent(Graph):
    """
    A class representing a recurrent graph for a specific task.

    This class inherits from the Graph class and introduces recurrent computations.

    Args:
        device (torch.device): The device to run the computations on.
        num_nodes (int): The total number of nodes in the graph.
        args (argparse.Namespace): Command-line arguments containing various configuration settings.
        emb_size (int, optional): The size of the node embeddings.
        dev0 (Any, optional): Additional device-specific configuration.
        dev1 (Any, optional): Additional device-specific configuration.
    """

    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        args: argparse.Namespace,
        emb_size: Optional[int] = None,
        dev0: Optional[Type[torch.device]] = None,
        dev1: Optional[Type[torch.device]] = None,
    ) -> None:
        super().__init__(device, num_nodes, args)
        self.emb_size = emb_size or args.emb_size
        self.dev0 = dev0
        self.dev1 = dev1
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize the weights of the graph model.

        This method initializes the weights for the graph model's layers based on the configuration settings.
        """

        a = max(self.emb_size, 64)
        n_hid = self.emb_size

        layers_g = self.encoder_layer_sizes or [a, a]
        in_dim_g = self.emb_size * 2

        if self.emb_history == 1:
            self.emb_history_layer = generate_fully_connected(
                input_dim=in_dim_g,
                output_dim=self.emb_size,
                hidden_dims=layers_g,
                p_dropout=self.dropout,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=self.device,
                normalization=self.norm_layer,
                res_connection=self.res_connection,
            )

        self.msg_fc1 = nn.Linear(2 * n_hid, n_hid, device=self.dev0)
        self.msg_fc2 = nn.Linear(n_hid, n_hid, device=self.dev1)

        self.msg_out_shape = n_hid
        self.skip_first_edge_type = True

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False).to(self.dev1)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False).to(self.dev1)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False).to(self.dev1)

        # n_in_node = self.num_nodes
        n_in_node = n_hid
        self.input_r = nn.Linear(n_in_node, n_hid, bias=True).to(self.dev1)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True).to(self.dev1)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True).to(self.dev1)

        self.out_fc1 = nn.Linear(n_hid, n_hid).to(self.dev1)
        self.out_fc2 = nn.Linear(n_hid, n_hid).to(self.dev1)
        self.out_fc3 = nn.Linear(n_hid, n_in_node).to(self.dev1)

    def forward(
        self,
        feed_dict: dict,
        adj: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the recurrent graph model.

        Args:
            feed_dict (dict): A dictionary containing input data.
            adj (torch.Tensor): Adjacency matrix of the graph.
            node_embeddings (torch.Tensor): Node embeddings.

        Returns:
            torch.Tensor: The output tensor from the forward pass.
        """
        skills = feed_dict["skill_seq"]  # [bs, seq_len]
        problems = feed_dict["problem_seq"]  # [bs, seq_len]
        times = feed_dict["time_seq"]  # [bs, seq_len]
        labels = feed_dict["label_seq"]  # [bs, seq_len]

        bs, time_steps = labels.shape
        time_lag = self.args.time_lag
        num_graph = adj.shape[0]

        node_embeddings = node_embeddings.double().to(self.dev1)
        adj = adj.to(self.dev1)
        hidden = torch.zeros(bs, self.num_nodes, self.msg_out_shape).to(self.dev1)

        pred_all = []

        for t in range(0, time_steps - 1):
            rel_type = adj.reshape(num_graph, bs, -1)  # [num_graph, bs, num_edge]
            ins = skills[:, t : t + 1]

            # node2edge
            receivers = torch.index_select(
                hidden, dim=1, index=self.rec_ind
            )  # [bs, num_edges, dim]
            senders = torch.index_select(hidden, dim=1, index=self.send_ind)
            pre_msg = torch.cat([senders, receivers], dim=-1).double().to(self.dev0)

            # Run separate MLP for every edge type
            # NOTE: To exclude one edge type, simply offset range by 1
            msg = torch.tanh(self.msg_fc1(pre_msg))
            msg = F.dropout(msg, p=self.dropout)

            msg = msg.to(self.dev1)
            msg = torch.tanh(self.msg_fc2(msg))  # [bs, num_edge, dim]

            msg = msg * rel_type.unsqueeze(-1)  # [num_graph, bs, num_edge, dim]
            agg_msgs = msg.reshape(num_graph, bs, self.num_nodes, self.num_nodes, -1)
            agg_msgs = agg_msgs.sum(-2)
            agg_msgs = agg_msgs.contiguous() / self.num_nodes  # Average

            # GRU-style gated aggregation
            bs_emb = node_embeddings.repeat(bs, 1, 1)
            inputs = bs_emb[torch.arange(bs), ins[:, 0]]
            inputs = inputs.reshape(1, bs, 1, self.emb_size)
            r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
            i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
            n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
            hidden = (1 - i) * n + i * hidden  # [num_graph, bs, num_nodes, emb_size]

            # Output MLP
            pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout)
            pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout)
            pred = self.out_fc3(pred)

            # # node2edge
            # receivers = torch.matmul(rel_rec, hidden) # hidden states of receiver nodes [bs, num_nodes, hid_dim]
            # senders = torch.matmul(rel_send, hidden)
            # pre_msg = torch.cat([senders, receivers], dim=-1)

            # agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
            # agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

            # # GRU-style gated aggregation
            # r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
            # i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
            # n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
            # hidden = (1 - i) * n + i * hidden

            # # Output MLP
            # pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout)
            # pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout)
            # pred = self.out_fc3(pred)

            # # Predict position/velocity difference
            # pred = inputs + pred
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()


class GraphWholeMessage(Graph):
    """
    A class representing a graph with whole message aggregation for a specific task.

    This class inherits from the Graph class and introduces whole message aggregation.

    Args:
        device (torch.device): The device to run the computations on.
        num_nodes (int): The total number of nodes in the graph.
        args (argparse.Namespace): Command-line arguments containing various configuration settings.
        emb_size (int, optional): The size of the node embeddings.
    """

    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        args: argparse.Namespace,
        emb_size: Optional[int] = None,
    ) -> None:
        super().__init__(device, num_nodes, args)
        self.emb_size = emb_size or args.emb_size
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize the weights of the whole message aggregation graph model.
        """
        a = max(self.emb_size, 64)
        layers_f = self.decoder_layer_sizes or [a, a]
        in_dim_f = self.emb_size
        layers_g = self.encoder_layer_sizes or [a, a]
        in_dim_g = self.emb_size * 2

        if self.emb_history == 1:
            self.emb_history_layer = generate_fully_connected(
                input_dim=in_dim_g,
                output_dim=self.emb_size,
                hidden_dims=layers_g,
                p_dropout=self.dropout,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=self.device,
                normalization=self.norm_layer,
                res_connection=self.res_connection,
            )

        self.final_output_layer = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=2,
            hidden_dims=layers_f,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )

        n_hid = self.emb_size
        self.msg_fc1 = nn.Linear(2 * n_hid, n_hid, device=self.device)
        self.msg_fc2 = nn.Linear(n_hid, n_hid, device=self.device)

    def forward(
        self,
        feed_dict: dict,
        adj: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_base: Optional[torch.Tensor] = None,
        node_bias_base: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the whole message aggregation graph model.

        Args:
            feed_dict (dict): A dictionary containing input data.
            adj (torch.Tensor): Adjacency matrix of the graph.
            node_embeddings (torch.Tensor): Node embeddings.
            node_base (Optional[torch.Tensor]): Node base embeddings.
            node_bias_base (Optional[torch.Tensor]): Node bias base embeddings.

        Returns:
            torch.Tensor: The output tensor from the forward pass.
        """
        skills = feed_dict["skill_seq"].to(self.device)  # [bs, seq_len]
        problems = feed_dict["problem_seq"].to(self.device)  # [bs, seq_len]
        times = feed_dict["time_seq"].to(self.device)  # [bs, seq_len]
        labels = feed_dict["label_seq"].to(self.device)  # [bs, seq_len]

        bs, time_steps = labels.shape
        time_lag = self.args.time_lag

        bs_skill_emb = torch.tile(node_embeddings, (bs, 1, 1)).to(self.device)
        if self.emb_history == 0:
            labels_pro = torch.where(labels == 0, -2, labels)
            labels_pro = torch.where(labels_pro == -1, 0, labels_pro)
            labels_pro = torch.where(labels_pro == -2, -1, labels_pro)
        else:
            labels_pro = labels

        # -- calculate the time difference
        delta_t = (
            (times[:, :, None] - times[:, None, :]).abs().double()
        )  # [bs, seq_len, seq_len]
        delta_t = torch.log(delta_t + EPS) / torch.log(torch.tensor(self.time_log))

        # -- initialize hidden features
        hidden = torch.zeros(bs, self.num_nodes, self.emb_size).to(self.device)
        # ---- the first #time_lag features
        skill_init = skills[:, :time_lag]  # [bs, time_lag]
        label_init = labels_pro[:, :time_lag]
        emb_init = [
            bs_skill_emb[torch.arange(0, bs), skill_init[:, i]] for i in range(time_lag)
        ]
        emb_init = torch.stack(emb_init, dim=1)  # [bs, time_lag, emd_size]
        if self.emb_history == 0:
            emb_init = emb_init * label_init.unsqueeze(-1)
        elif self.emb_history == 1:
            label_init = torch.tile(label_init.unsqueeze(-1), (1, 1, self.emb_size))
            emb_init = torch.cat([emb_init, label_init], dim=-1).double()
            emb_init = self.emb_history_layer(emb_init)
        # ---- brutal searching TODO
        unique_skill = [skill_init[i].unique().tolist() for i in range(bs)]
        for i in range(bs):
            skill = unique_skill[i]
            for j in range(len(skill)):
                cur_skill = skill[j]
                last_time_ind = (skill_init[i] == cur_skill).nonzero()[-1]
                hidden[i][cur_skill] += emb_init[i][last_time_ind][0]

        preds = []

        for t in range(time_lag, time_steps):
            skill_history = skills[:, t - time_lag : t]  # [bs, time_lag]
            skill_target = skills[:, t : t + 1]  # [bs, 1]

            hidden_history = [
                hidden[torch.arange(0, bs), skill_history[:, i]]
                for i in range(time_lag)
            ]  # [bs, time_lag, emb_size]
            hidden_history = torch.stack(hidden_history, 1)

            delta_t_weight = delta_t.transpose(-1, -2)[
                :, t, t - time_lag : t
            ]  # [bs, time_lag]  range from 0 to around 10
            emb_history = [
                bs_skill_emb[torch.arange(0, bs), skill_history[:, i]]
                for i in range(time_lag)
            ]
            emb_history = torch.stack(emb_history, dim=1)  # [bs, time_lag, emd_size]

            # -- w_ij means the effect from i to j
            cross_weight = [
                adj[:, torch.arange(0, bs), skill_history[:, i]]
                for i in range(time_lag)
            ]
            cross_weight = torch.stack(
                cross_weight, 2
            )  # [num_graph, bs, time_lag, num_nodes]
            # step 2
            adj_2 = torch.matrix_power(adj, 2) / self.num_nodes
            # adj_3 = torch.matrix_power(adj,3)/self.num_nodes/self.num_nodes
            adj_spatial = torch.stack(
                [adj, adj_2], dim=2
            )  # [num_graph, bs, power, num_nodes, num_nodes]

            # adj_spatial = adj_spatial.swapaxes(0, 1).reshape(bs, -1, self.num_nodes, self.num_nodes) # [bs, num_graph*power, num_nodes, num_nodes]
            cross_weight = [
                adj_spatial[:, torch.arange(0, bs), :, skill_history[:, i]]
                for i in range(time_lag)
            ]
            cross_weight = torch.stack(
                cross_weight, -2
            )  # [num_graph, bs, power, time_lag, num_nodes]

            # -- node2edge
            pre_msg = torch.cat(
                [
                    hidden_history.unsqueeze(-2).repeat(1, 1, self.num_nodes, 1),
                    hidden.unsqueeze(1).repeat(1, time_lag, 1, 1),
                ],
                dim=-1,
            ).double()
            msg = torch.tanh(self.msg_fc1(pre_msg))
            msg = F.dropout(msg, p=self.dropout)
            msg = torch.tanh(self.msg_fc2(msg))  # [bs, time_lag, num_nodes, dim]
            msg = msg.unsqueeze(1).unsqueeze(1) * cross_weight.unsqueeze(
                -1
            )  # [bs, num_graph, power, time_lag, num_nodes, emb_dim] # ~e-05 e-04

            # --edge2node
            agg_msg = torch.exp(-delta_t_weight.reshape(bs, 1, 1, time_lag, 1, 1)) * msg
            agg_msg = agg_msg.sum(3).sum(2)
            hidden = agg_msg.mean(1)
            pred = F.softmax(self.final_output_layer(agg_msg), dim=-1)[
                torch.arange(bs), :, skill_target[:, 0], 1:
            ]  # problem_base_target + skill_base_target
            preds.append(pred)

        return preds


class GraphPerformanceTrace(Graph):
    """
    A class for performance tracing using graph data.

    This class extends the base Graph class and provides additional functionality
    for tracing performance using graph data.

    Args:
        device (str): The device to use for computations (e.g., 'cpu', 'cuda').
        num_nodes (int): The number of nodes in the graph.
        args (argparse.Namespace): Command-line arguments.
        emb_size (int, optional): The embedding size. If not provided, it uses args.emb_size.

    """

    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        args: argparse.Namespace,
        emb_size: Optional[int] = None,
    ) -> None:
        super().__init__(device, num_nodes, args)
        self.emb_size = emb_size or args.emb_size
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes the weights and layers of the performance tracing model.
        """
        self.update_user = Sequential(
            nn.Conv1d(32, 32, 5, device=self.device),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1, device=self.device),
            # nn.ReLU(),
        )
        self.update_f = nn.Linear(32, 32, device=self.device)
        self.update_b = nn.Linear(32, 32, device=self.device)

    def forward(
        self,
        feed_dict: dict,
        adj: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_base: Optional[torch.Tensor] = None,
        node_bias_base: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Perform a forward pass through the performance tracing graph model.

        Args:
            feed_dict (dict): A dictionary containing input data.
            adj (torch.Tensor): Adjacency matrix of the graph.
            node_embeddings (torch.Tensor): Node embeddings.
            node_base (Optional[torch.Tensor]): Node base embeddings.
            node_bias_base (Optional[torch.Tensor]): Node bias base embeddings.

        Returns:
            List[torch.Tensor]: A list of tensors containing the predicted performance traces.
        """

        skills = feed_dict["skill_seq"].to(self.device)  # [bs, seq_len]
        problems = feed_dict["problem_seq"].to(self.device)  # [bs, seq_len]
        times = feed_dict["time_seq"].to(self.device)  # [bs, seq_len]
        labels = feed_dict["label_seq"].to(self.device)  # [bs, seq_len]
        bs, time_steps = labels.shape

        # -- multi-step spatial on graph
        power = 3
        adj_list = [
            torch.matrix_power(adj, i)
            / torch.float_power(torch.tensor(self.num_nodes), i)
            for i in range(1, power + 1)
        ]
        adj_list = (
            torch.stack(adj_list, 2).permute(1, 0, 2, 3, 4).double()
        )  # [bs, num_graph, power, num_nodes, num_nodes]

        latents_dim = node_embeddings.shape[-1] // 2
        node_embeddings = node_embeddings.double()
        node_send, node_rec = torch.split(
            node_embeddings, latents_dim, dim=-1
        )  # [1, num_node, latent_dim]

        # -- initialize hidden features
        user_b = torch.rand(
            (bs, latents_dim), device=self.device
        ).double()  # TODO other initialization
        user_f = torch.rand((bs, latents_dim), device=self.device).double()

        alpha = 1e-1  # TODO
        last_perf = torch.matmul(user_b, node_rec.permute(1, 2, 0)).permute(
            1, 2, 0
        )  # [bs, 1, num_nodes]
        node_last_forget = torch.matmul(user_f, node_send.permute(1, 2, 0)).permute(
            1, 2, 0
        )  # [bs, 1, num_nodes]

        preds = []
        for t in range(1, time_steps):  # from t=1
            node_cur_base = torch.matmul(user_b, node_rec.permute(1, 2, 0)).permute(
                1, 2, 0
            )  # [bs, 1, num_nodes]

            cur_delta_t = (
                torch.log((times[:, t] - times[:, t - 1]).abs().double() + EPS)
                / torch.log(torch.tensor(self.time_log))
            ).reshape(-1, 1, 1)
            tmporal_effect = (
                torch.exp(-cur_delta_t * alpha * node_last_forget) * last_perf
            )  # [bs, 1, num_nodes] # cur_delta_t has some extreme large values!

            spatial_effect = torch.einsum("bij, banjk->banik", last_perf, adj_list)[
                :, :, :, 0
            ]  # [bs, num_graph, power, num_nodes]
            spatial_effect = spatial_effect.sum(2) / power / self.num_nodes

            cur_perf = torch.sigmoid(node_cur_base + tmporal_effect + spatial_effect)
            if (
                torch.isnan(node_cur_base).sum()
                + torch.isnan(cur_delta_t).sum()
                + torch.isnan(spatial_effect).sum()
                + torch.isnan(cur_perf).sum()
                > 0
            ):
                ipdb.set_trace()
            node_last_forget = torch.matmul(user_f, node_send.permute(1, 2, 0)).permute(
                1, 2, 0
            )
            last_perf = cur_perf

            # -- update user_b and user_f
            cur_label = labels[:, t].unsqueeze(-1).repeat(1, latents_dim)
            cur_skill = skills[:, t]
            user = torch.stack(
                [
                    user_b,
                    user_f,
                    cur_label,
                    node_rec.repeat(bs, 1, 1)[torch.arange(bs), cur_skill],
                    node_send.repeat(bs, 1, 1)[torch.arange(bs), cur_skill],
                ],
                dim=-1,
            ).double()

            user_update = self.update_user(user)[..., 0]  # nan in update user
            user_f = torch.tanh(self.update_f(user_update))
            user_b = torch.tanh(self.update_b(user_update))

            if (
                torch.isnan(user).sum()
                + torch.isnan(user_f).sum()
                + torch.isnan(user_b).sum()
                > 0
            ):
                ipdb.set_trace()
            preds.append(cur_perf[torch.arange(bs), :, cur_skill])
        return preds


class GraphOUProcess(Graph):
    """
    A class for simulating an Ornstein-Uhlenbeck process using graph data.

    This class extends the base Graph class and provides functionality
    for simulating an Ornstein-Uhlenbeck process using graph data.

    Args:
        device (str): The device to use for computations (e.g., 'cpu', 'cuda').
        num_nodes (int): The number of nodes in the graph.
        args (argparse.Namespace): Command-line arguments.
        emb_size (int, optional): The embedding size. If not provided, it uses args.emb_size.

    """

    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        args: argparse.Namespace,
        emb_size: Optional[int] = None,
    ) -> None:
        super().__init__(device, num_nodes, args)
        self.emb_size = emb_size or args.emb_size
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes the weights and layers for the Ornstein-Uhlenbeck process model.
        """

        self.mean_linear = generate_fully_connected(
            input_dim=3,
            output_dim=1,
            hidden_dims=32,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Sigmoid,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )
        self.speed_linear = generate_fully_connected(
            input_dim=3,
            output_dim=1,
            hidden_dims=32,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Sigmoid,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )
        self.beta_linear = generate_fully_connected(
            input_dim=3,
            output_dim=1,
            hidden_dims=32,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Sigmoid,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )

    def std(
        self,
        t: torch.Tensor,
        vola: float,
        mean_rev_speed: float,
    ) -> torch.Tensor:
        """
        Calculates the standard deviation of the Ornstein-Uhlenbeck process.

        Args:
            t (torch.Tensor): Time values.
            vola (float): Volatility parameter.
            mean_rev_speed (float): Mean reversion speed parameter.

        Returns:
            torch.Tensor: Standard deviation of the process.
        """

        variance = (
            vola
            * vola
            * (1.0 - torch.exp(-2.0 * mean_rev_speed * t))
            / (2 * mean_rev_speed)
        )
        return torch.sqrt(variance)

    def mean(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        mean_speed: float,
        mean_level: float,
    ) -> torch.Tensor:
        """
        Calculates the mean of the Ornstein-Uhlenbeck process.

        Args:
            x0 (float): Initial value.
            t (torch.Tensor): Time values.
            mean_speed (float): Mean speed parameter.
            mean_level (float): Mean level parameter.

        Returns:
            torch.Tensor: Mean of the process.
        """

        return (
            x0 * torch.exp(-mean_speed * t)
            + (1.0 - torch.exp(-mean_speed * t)) * mean_level
        )

    def forward(
        self,
        feed_dict: dict,
        adj: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_base: Optional[torch.Tensor] = None,
        node_bias_base: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Performs a forward pass through the Ornstein-Uhlenbeck process model.

        Args:
            feed_dict (dict): Input data dictionary containing skill_seq, problem_seq, time_seq, and label_seq.
            adj (torch.Tensor): Adjacency matrix.
            node_embeddings (torch.Tensor): Node embeddings.
            node_base (torch.Tensor, optional): Node base embeddings.
            node_bias_base (torch.Tensor, optional): Node bias base embeddings.
        """
        skills = feed_dict["skill_seq"].to(self.device)  # [bs, seq_len]
        problems = feed_dict["problem_seq"].to(self.device)  # [bs, seq_len]
        times = feed_dict["time_seq"].to(self.device)  # [bs, seq_len]
        labels = feed_dict["label_seq"].to(self.device)  # [bs, seq_len]
        bs, time_steps = labels.shape
        num_graph = adj.shape[0]

        # -- multi-step spatial on graph
        power = 3
        adj_list = [
            torch.matrix_power(adj, i)
            / torch.float_power(torch.tensor(self.num_nodes), i)
            for i in range(1, power + 1)
        ]
        adj_list = (
            torch.stack(adj_list, 2).permute(1, 0, 2, 3, 4).double()
        )  # [bs, num_graph, power, num_nodes, num_nodes]

        # -- initialize alpha, mu, beta
        mean_rev_speed = torch.rand((bs, 1), device=self.device).double()
        mean_rev_level = torch.rand((bs, 1), device=self.device).double()
        vola = torch.rand((bs, 1), device=self.device).double()
        x_last = torch.zeros(
            (bs, num_graph, power, self.num_nodes, 1), device=self.device
        ).double()

        preds = []
        for t in range(1, time_steps):  # from t=1
            dt = times[:, t : t + 1] - times[:, t - 1 : t]
            dt = dt.reshape(bs, -1, 1, 1, 1)
            dt = dt.tile(1, num_graph, 1, self.num_nodes, 1)

            noise = torch.randn(size=x_last.shape, device=self.device)
            scale = self.std(dt, vola, mean_rev_speed)

            xt = noise * scale  # [bs, num_graph, power, n, n]

            adj = adj_list  # [bs, num_graph, power, n, n]
            adj_t = torch.transpose(adj, -1, -2)
            in_degree = adj_t[0].sum(dim=-1)  # .reshape(1,-1,1)

            for j in range(num_graph):
                in_j = in_degree[j, 0]  # TODO consider power
                # find degree 0
                ind = torch.where(in_j == 0)[0]

                s = (
                    (1 / (in_degree[j : j + 1] + 1e-7)) * adj_t[:, j] @ x_last[:, j]
                )  # [bs, power, n, 1]
                s[:, :, ind] = 1
                tmp_mean_level = mean_rev_level * s  # [bs, power, n, 1]
                xt[:, j] += self.mean(
                    x_last[:, j], dt[:, j], mean_rev_speed, tmp_mean_level
                )
            xt = torch.sigmoid(xt)

            cur_skill = skills[:, t]
            cur_label = labels[:, t]
            pred = xt[torch.arange(bs), :, :, cur_skill]
            preds.append(pred)  # [bs, num_graph, power, 1]
            x_last = xt

            # update
            gt = labels[:, t : t + 1].reshape(-1, 1, 1, 1)
            prev_speed = mean_rev_speed.reshape(-1, 1, 1, 1).tile(bs, 1, 1, 1)
            mean_rev_speed = self.speed_linear(torch.cat([prev_speed, gt - pred], 1))
            mean_rev_level = mean_rev_level, labels[:, t], xt

        return preds

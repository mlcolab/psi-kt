import math
import argparse
from collections import defaultdict
from typing import List, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from knowledge_tracing.baseline.basemodel import BaseModel
from knowledge_tracing.utils import logger
from knowledge_tracing.data.data_loader import DataReader


class GKT(BaseModel):
    """Graph-based Knowledge Tracing Modeling Student Proficiency Using Graph Neural Network

    Args:
        num_c (int): total num of unique questions
        hidden_dim (int): hidden dimension for MLP
        emb_size (int): embedding dimension for question embedding layer
        graph_type (str, optional): graph type, dense or transition. Defaults to "dense".
        graph (_type_, optional): graph. Defaults to None.
        bias (bool, optional): add bias for DNN. Defaults to True.
    """

    @staticmethod
    def parse_model_args(
        parser: argparse.ArgumentParser,
        model_name: str = "GKT",
    ) -> argparse.Namespace:
        """
        Parse GKT-specific model arguments from the command line.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
            model_name (str, optional): Name of the model. Defaults to "GKT".

        Returns:
            argparse.Namespace: Parsed command-line arguments.

        """
        parser.add_argument(
            "--emb_size", type=int, default=16, help="Size of embedding vectors."
        )
        parser.add_argument("--graph_type", type=str, default="dense", help="")
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(
        self,
        args: argparse.Namespace,
        corpus: DataReader,
        logs: logger.Logger,
        graph=None,
    ) -> None:
        self.model_name = "gkt"

        self.num_c = int(corpus.n_skills)
        self.hidden_dim = args.emb_size
        self.emb_size = args.emb_size
        self.res_len = 2
        self.graph_type = args.graph_type

        self.bias = True
        self.dropout = args.dropout

        self.args = args
        self.logs = logs
        self.device = args.device
        BaseModel.__init__(self, model_path=Path(args.log_path, "Model"))

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model components.

        This method initializes various components such as embeddings and loss function.
        """
        self.graph = torch.nn.Parameter(
            torch.empty(self.num_c, self.num_c, device=self.device)
        )
        self.graph.requires_grad = True  # fix parameter

        # one-hot feature and question
        self.one_hot_feat = torch.eye(self.res_len * self.num_c).to(self.device)
        self.one_hot_q = torch.eye(self.num_c).to(self.device)
        zero_padding = torch.zeros(1, self.num_c).to(self.device)
        self.one_hot_q = torch.cat((self.one_hot_q, zero_padding), dim=0)

        # concept and concept & response embeddings
        self.interaction_emb = nn.Embedding(self.res_len * self.num_c, self.emb_size)
        # last embedding is used for padding, so dim + 1
        self.emb_c = nn.Embedding(self.num_c + 1, self.emb_size, padding_idx=-1)

        # f_self function
        mlp_input_dim = self.hidden_dim + self.emb_size
        self.f_self = MLP(
            mlp_input_dim,
            self.hidden_dim,
            self.hidden_dim,
            dropout=self.dropout,
            bias=self.bias,
        )

        # f_neighbor functions
        self.f_neighbor_list = nn.ModuleList()

        # f_in functions
        self.f_neighbor_list.append(
            MLP(
                2 * mlp_input_dim,
                self.hidden_dim,
                self.hidden_dim,
                dropout=self.dropout,
                bias=self.bias,
            )
        )
        # f_out functions
        self.f_neighbor_list.append(
            MLP(
                2 * mlp_input_dim,
                self.hidden_dim,
                self.hidden_dim,
                dropout=self.dropout,
                bias=self.bias,
            )
        )

        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(self.hidden_dim, self.num_c)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim, bias=self.bias)
        # prediction layer
        self.predict = nn.Linear(self.hidden_dim, 1, bias=self.bias)
        # Binary Cross Entropy Loss
        self.loss_function = nn.BCELoss()

    # Aggregate step, as shown in Section 3.2.1 of the paper
    def _aggregate(
        self, xt: torch.Tensor, qt: torch.Tensor, ht: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Args:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state and concept(& response) embedding
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        x_idx_mat = torch.arange(self.res_len * self.num_c, device=self.device)
        x_embedding = self.interaction_emb(x_idx_mat)  # [res_len * num_c, emb_size]

        masked_feat = F.embedding(
            xt[qt_mask], self.one_hot_feat
        )  # [batch_size, res_len * num_c] A simple lookup table that looks up embeddings in a fixed dictionary and size.

        res_embedding = masked_feat.mm(x_embedding)  # [batch_size, emb_size]
        mask_num = res_embedding.shape[0]

        concept_idx_mat = (
            self.num_c * torch.ones((batch_size, self.num_c), device=self.device).long()
        )
        concept_idx_mat[qt_mask, :] = torch.arange(
            self.num_c, device=self.device
        )  # [batch_size, num_c]
        concept_embedding = self.emb_c(concept_idx_mat)  # [batch_size, num_c, emb_size]

        index_tuple = (torch.arange(mask_num, device=self.device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(
            index_tuple, res_embedding
        )
        tmp_ht = torch.cat(
            (ht, concept_embedding), dim=-1
        )  # [batch_size, num_c, hidden_dim + emb_size]
        return tmp_ht

    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            qt: question indices for all students in a batch at the current timestamp
        Return:
            m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        masked_qt = qt[qt_mask]  # [mask_num, ]
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, num_c, hidden_dim + emb_size]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + emb_size]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(
            1, self.num_c, 1
        )  # [mask_num, num_c, hidden_dim + emb_size]
        neigh_ht = torch.cat(
            (expanded_self_ht, masked_tmp_ht), dim=-1
        )  # [mask_num, num_c, 2 * (hidden_dim + emb_size)]
        concept_embedding, rec_embedding, z_prob = None, None, None

        adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)  # [mask_num, num_c, 1]
        reverse_adj = (
            self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)
        )  # [mask_num, num_c, 1]
        neigh_features = adj * self.f_neighbor_list[0](
            neigh_ht
        ) + reverse_adj * self.f_neighbor_list[1](neigh_ht)

        # neigh_features: [mask_num, num_c, hidden_dim]
        m_next = tmp_ht[:, :, : self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next, concept_embedding, rec_embedding, z_prob

    # Update step, as shown in Section 3.3.2 of the paper
    def _update(
        self, tmp_ht: torch.Tensor, ht: torch.Tensor, qt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Return:
            h_next: hidden representations of all concepts at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        mask_num = qt_mask.nonzero().shape[0]
        # GNN Aggregation
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(
            tmp_ht, qt
        )  # [batch_size, num_c, hidden_dim]
        # Erase & Add Gate
        m_next[qt_mask] = self.erase_add_gate(
            m_next[qt_mask]
        )  # [mask_num, num_c, hidden_dim]
        # GRU
        h_next = m_next
        res = self.gru(
            m_next[qt_mask].reshape(-1, self.hidden_dim),
            ht[qt_mask].reshape(-1, self.hidden_dim),
        )  # [mask_num * num_c, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device),)
        h_next[qt_mask] = h_next[qt_mask].index_put(
            index_tuple, res.reshape(-1, self.num_c, self.hidden_dim)
        )
        return h_next, concept_embedding, rec_embedding, z_prob

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, num_c]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, num_c]
        return y

    def _get_next_pred(self, yt: torch.Tensor, q_next: torch.Tensor) -> torch.Tensor:
        """
        Args:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        next_qt = q_next
        next_qt = torch.where(
            next_qt != -1,
            next_qt,
            self.num_c * torch.ones_like(next_qt, device=yt.device),
        )
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)  # [batch_size, num_c]
        # dot product between yt and one_hot_qt
        pred = (yt * one_hot_qt).sum(dim=1)  # [batch_size, ]
        return pred

    def forward(self, feed_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the model for a given input feed dictionary.

        Args:
            feed_dict: A dictionary containing the input tensors for the model.

        Returns:
            A dictionary containing the output tensors for the model.
        """
        c = feed_dict["skill_seq"]
        r = feed_dict["label_seq"]

        features = c * 2 + r
        questions = c

        batch_size, seq_len = features.shape
        ht = Variable(
            torch.zeros((batch_size, self.num_c, self.hidden_dim), device=self.device)
        )

        pred_list = []
        for i in range(seq_len):
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]
            qt_mask = torch.ne(qt, -1)  # [batch_size], next_qt != -1
            tmp_ht = self._aggregate(
                xt, qt, ht, batch_size
            )  # [batch_size, num_c, hidden_dim + emb_size]
            h_next, _, _, _ = self._update(
                tmp_ht, ht, qt
            )  # [batch_size, num_c, hidden_dim]
            ht[qt_mask] = h_next[qt_mask]  # update new ht
            yt = self._predict(h_next, qt)  # [batch_size, num_c]
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]

        out_dict = {
            "prediction": pred_res,
            "label": r[:, 1:].double() if seq_len > 1 else r.double(),
        }

        return out_dict

    def loss(
        self,
        feed_dict: Dict[str, torch.Tensor],
        out_dict: Dict[str, torch.Tensor],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the model for a given input feed dictionary.

        Args:
            feed_dict: A dictionary containing the input tensors for the model.

        Returns:
            A dictionary containing the output tensors for the model.
        """

        losses = defaultdict(lambda: torch.zeros((), device=self.device))

        predictions = out_dict["prediction"].flatten().float()
        labels = out_dict["label"].flatten().float()
        mask = labels > -1
        loss = self.loss_function(predictions[mask], labels[mask])
        losses["loss_total"] = loss

        # Compute the evaluation metrics for the main prediction task
        if metrics is not None:
            pred = predictions.detach().cpu().data.numpy()
            gt = labels.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
            for key in evaluations.keys():
                losses[key] = evaluations[key]

        if "cl_prediction" in out_dict.keys():
            cl_predictions, cl_labels = out_dict["cl_prediction"], out_dict["cl_label"]
            cl_loss = self.loss_function(cl_predictions, cl_labels.float())
            losses["cl_loss"] = cl_loss
            pred = cl_predictions.detach().cpu().data.numpy()
            gt = cl_labels.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
            for key in evaluations.keys():
                losses["cl_" + key] = evaluations[key]

        return losses

    def predictive_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
        single_step: bool = True,
    ) -> Dict[str, torch.Tensor]:
        train_step = int(self.args.max_step * self.args.train_time_ratio)

        cur_feed_dict = {
            "skill_seq": feed_dict["skill_seq"][:, train_step - 1 :],
            "label_seq": feed_dict["label_seq"][:, train_step - 1 :],
            "problem_seq": feed_dict["problem_seq"][:, train_step - 1 :],
            "time_seq": feed_dict["time_seq"][:, train_step - 1 :],
        }

        out_dict = self.forward(cur_feed_dict)
        return out_dict

    def forward_cl(
        self,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the classification task, given the input feed_dict and the current time index.

        Args:
            feed_dict: A dictionary containing the input tensors for the model.
            idx: The current time index. Only items and labels up to this index are used.

        Returns:
            A dictionary containing the output tensors for the classification task.
        """

        # Extract input tensors from feed_dict
        cur_feed_dict = {
            "skill_seq": feed_dict["skill_seq"][:, : idx + 1],
            "label_seq": feed_dict["label_seq"][:, : idx + 1],
            "problem_seq": feed_dict["problem_seq"][:, : idx + 1],
        }

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
        cur_feed_dict = {
            "skill_seq": feed_dict["skill_seq"][:, idx : idx + test_step + 1],
            "label_seq": feed_dict["label_seq"][:, idx : idx + test_step + 1],
            "problem_seq": feed_dict["problem_seq"][:, idx : idx + test_step + 1],
        }

        out_dict = self.forward(cur_feed_dict)

        labels = out_dict["label"]
        prediction = out_dict["prediction"]

        return self.pred_evaluate_method(
            prediction.flatten().cpu(), labels.flatten().cpu(), metrics
        )


# Multi-Layer Perceptron(MLP) layer
class MLP(nn.Module):
    """Two-layer fully-connected ReLU net with batch norm."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        # the paper said they added Batch Normalization for the output of MLPs, as shown in Section 4.2
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 or 0 will cause BatchNorm error, so return the input directly
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:  # len(input_size()) == 2
            return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(
            x, self.dropout, training=self.training
        )  # pay attention to add training=self.training
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    """Erase & Add Gate module
    NOTE: this erase & add gate is a bit different from that in DKVMN.
    For more information about Erase & Add gate, please refer to the paper "Dynamic Key-Value Memory Networks for Knowledge Tracing"
    The paper can be found in https://arxiv.org/abs/1611.08108

    Args:
        nn (_type_): _description_
    """

    def __init__(self, feature_dim, num_c, bias=True):
        super(EraseAddGate, self).__init__()
        # weight
        self.weight = nn.Parameter(torch.rand(num_c))
        self.reset_parameters()
        # erase gate
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        # add gate
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Params:
            x: input feature matrix

        Shape:
            x: [batch_size, num_c, feature_dim]
            res: [batch_size, num_c, feature_dim]

        Return:
            res: returned feature matrix with old information erased and new information added
            The GKT paper didn't provide detailed explanation about this erase-add gate. As the erase-add gate in the GKT only has one input parameter,
            this gate is different with that of the DKVMN. We used the input matrix to build the erase and add gates, rather than $\mathbf{v}_{t}$ vector in the DKVMN.

        """
        erase_gate = torch.sigmoid(self.erase(x))  # [batch_size, num_c, feature_dim]
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))  # [batch_size, num_c, feature_dim]
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res

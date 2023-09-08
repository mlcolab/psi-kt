# @Date: 2023/07/29

import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict

from knowledge_tracing.baseline.basemodel import BaseModel
from knowledge_tracing.utils import utils, logger
from knowledge_tracing.data.data_loader import DataReader


class AKT(BaseModel):
<<<<<<< HEAD:knowledge_tracing/baseline/AKT.py
=======
    """
    An implementation of the AKT model, extending the BaseModel.

    This class defines the AKT (Attention-based Knowledge Tracing) model,
    which extends the BaseModel class. It includes methods for parsing model
    arguments and initializing the instance.

    Args:
        args (argparse.Namespace):
            Namespace containing parsed command-line arguments.
        corpus (DataReader):
            An instance of the DataReader class containing corpus data.
        logs (Logger):
            An instance of the Logger class for logging purposes.

    Attributes:
        extra_log_args (List[str]): List of additional arguments to include in logs.
            These are specific to the AKT model.

    Methods:
        parse_model_args(parser, model_name="AKT"):
            Parse AKT-specific model arguments from the command line.

        __init__(args, corpus, logs):
            Initialize an instance of the AKT class.

    """
>>>>>>> b4df127d6c418aa808d749d4ec8151f719b98d17:knowledge_tracing/baseline/akt.py

    extra_log_args = ["num_layer", "num_head"]

    @staticmethod
    def parse_model_args(
        parser: argparse.ArgumentParser,
        model_name: str = "AKT",
    ) -> argparse.Namespace:
        """
        Parse AKT-specific model arguments from the command line.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
            model_name (str, optional): Name of the model. Defaults to "AKT".

        Returns:
            argparse.Namespace: Parsed command-line arguments.

        """

        parser.add_argument(
            "--emb_size", type=int, default=16, help="Size of embedding vectors."
        )
        parser.add_argument(
            "--num_layer", type=int, default=1, help="Self-attention layers."
        )
        parser.add_argument(
            "--num_head", type=int, default=4, help="Self-attention heads."
        )
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(
        self,
        args: argparse.Namespace,
        corpus: DataReader,
        logs: logger.Logger,
    ) -> None:
        self.skill_num = int(corpus.n_skills)
        self.question_num = int(corpus.n_problems)
        self.emb_size = args.emb_size
        self.num_head = args.num_head
        self.dropout = args.dropout

        # Set the device to use for computations
        self.device = args.device

        # Store the arguments and logs for later use
        self.args = args
        self.logs = logs
        super().__init__(model_path=os.path.join(args.log_path, "Model/Model_{}_{}.pt"))

    def _init_weights(
        self,
    ) -> None:
        """
        Initialize the weights of the model's embeddings and Transformer layers.

        This method initializes the embedding matrices and Transformer layers' weights
        with appropriate dimensions and configurations.

        Returns:
            None

        """

        self.skill_embeddings = nn.Embedding(self.skill_num, self.emb_size)
        self.inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size)
        self.difficult_param = nn.Embedding(self.question_num, 1)
        self.skill_diff = nn.Embedding(self.skill_num, self.emb_size)
        self.inter_diff = nn.Embedding(self.skill_num * 2, self.emb_size)

        self.blocks_1 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=self.emb_size,
                    d_feature=self.emb_size // self.num_head,
                    d_ff=self.emb_size,
                    dropout=self.dropout,
                    n_heads=self.num_head,
                    kq_same=False,
                    gpu=self.args.gpu,
                )
                for _ in range(self.args.num_layer)
            ]
        )
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=self.emb_size,
                    d_feature=self.emb_size // self.num_head,
                    d_ff=self.emb_size,
                    dropout=self.dropout,
                    n_heads=self.num_head,
                    kq_same=False,
                    gpu=self.args.gpu,
                )
                for _ in range(self.args.num_layer * 2)
            ]
        )

        self.out = nn.Sequential(
            nn.Linear(self.emb_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1),
        )

        self.loss_function = nn.BCELoss(reduction="sum")

    def forward_cl(
        self,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the continual learning task, given the input feed_dict and the current time index.

        Args:
            feed_dict: A dictionary containing the input tensors for the model.
            idx: The current time index. Only items and labels up to this index are used.

        Returns:
            A dictionary containing the output tensors for the continual learning task.
        """

        # Extract input tensors from feed_dict
        cur_feed_dict = {
            "skill_seq": feed_dict["skill_seq"][:, : idx + 1],
            "label_seq": feed_dict["label_seq"][:, : idx + 1],
            "quest_seq": feed_dict["quest_seq"][:, : idx + 1],
        }

        out_dict = self.forward(cur_feed_dict)

        return out_dict

    def forward(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input tensors.
                - 'skill_seq' (torch.Tensor): Skill sequence tensor of shape [batch_size, real_max_step].
                - 'quest_seq' (torch.Tensor): Question sequence tensor of shape [batch_size, real_max_step].
                - 'label_seq' (torch.Tensor): Label sequence tensor of shape [batch_size, real_max_step].

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the model's output tensors.
                - 'prediction' (torch.Tensor): Predicted probabilities of shape [batch_size, time_step-1].
                - 'label' (torch.Tensor): Ground truth labels of shape [batch_size, time_step-1].
                - 'emb' (torch.Tensor): Encoded skill representations of shape [batch_size, emb_size].
        """

        skills = feed_dict["skill_seq"]
        questions = feed_dict["quest_seq"]
        labels = feed_dict["label_seq"]
        time_step = labels.shape[-1]

        mask_labels = labels * (labels > -1).long()
        inters = skills + mask_labels * self.skill_num
        skill_data = self.skill_embeddings(skills)
        inter_data = self.inter_embeddings(inters)

        skill_diff_data = self.skill_diff(skills)
        inter_diff_data = self.inter_diff(inters)
        q_diff = self.difficult_param(questions)
        skill_data = skill_data + q_diff * skill_diff_data
        inter_data = inter_data + q_diff * inter_diff_data

        x, y = skill_data, inter_data
        for block in self.blocks_1:  # encode
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:  # don't peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True

        concat_q = torch.cat([x, skill_data], dim=-1)
        prediction = self.out(concat_q).squeeze(-1).sigmoid()

        # Remove the first time step if it's a padding step
        prediction = prediction[:, 1:] if time_step > 1 else prediction
        label = labels[:, 1:] if time_step > 1 else labels

        out_dict = {
            "prediction": prediction,
            "label": label.float(),
            "emb": x,
        }

        return out_dict

    def predictive_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform prediction using the trained model.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input tensors.
                - 'skill_seq' (torch.Tensor): Skill sequence tensor of shape [batch_size, real_max_step].
                - 'quest_seq' (torch.Tensor): Question sequence tensor of shape [batch_size, real_max_step].
                - 'label_seq' (torch.Tensor): Label sequence tensor of shape [batch_size, real_max_step].

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the model's output tensors.
                - 'predictions' (List[torch.Tensor]): List of predicted probabilities for each time step.
        """

        skills = feed_dict["skill_seq"]  # [batch_size, real_max_step]
        questions = feed_dict["quest_seq"]  # [batch_size, real_max_step]
        labels = feed_dict["label_seq"]  # [batch_size, real_max_step]

        all_step = skills.shape[-1]
        train_step = int(self.args.max_step * self.args.train_time_ratio)
        test_time = int(self.args.max_step * self.args.test_time_ratio)

        predictions = []

        for i in range(0, test_time):
            if i == 0:
                inters = (
                    skills[:, train_step - 1 : train_step + 1]
                    + labels[:, train_step - 1 : train_step + 1] * self.skill_num
                )

            else:
                pred_labels = torch.cat(
                    [
                        labels[:, train_step - 1 : train_step],
                        (torch.cat(predictions, -1) >= 0.5) * 1,
                        labels[:, train_step + i : train_step + i + 1],
                    ],
                    dim=-1,
                )

                inters = (
                    skills[:, train_step - 1 : train_step + i + 1]
                    + pred_labels * self.skill_num
                )

            skill_data = self.skill_embeddings(
                skills[:, train_step - 1 : train_step + i + 1]
            )
            inter_data = self.inter_embeddings(inters)

            skill_diff_data = self.skill_diff(
                skills[:, train_step - 1 : train_step + i + 1]
            )
            inter_diff_data = self.inter_diff(inters)

            q_diff = self.difficult_param(
                questions[:, train_step - 1 : train_step + i + 1]
            )
            skill_data = skill_data + q_diff * skill_diff_data
            inter_data = inter_data + q_diff * inter_diff_data

            x, y = skill_data, inter_data
            for block in self.blocks_1:  # encode
                y = block(mask=1, query=y, key=y, values=y)
            flag_first = True
            for block in self.blocks_2:
                if flag_first:  # peek current question
                    x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                    flag_first = False
                else:  # don't peek current response
                    x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                    flag_first = True

            concat_q = torch.cat([x, skill_data], dim=-1)
            prediction = self.out(concat_q).squeeze(-1).sigmoid()[:, -1:]

            predictions.append(prediction)

        prediction = torch.cat(predictions, dim=-1)
        out_dict = {
            "prediction": prediction,
            "label": labels[:, train_step:].float() if all_step > 1 else labels.float(),
            "emb": x,
        }

        return out_dict

    def loss(
        self,
        feed_dict: Dict[str, torch.Tensor],
        out_dict: Dict[str, torch.Tensor],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss and evaluation metrics for the model's predictions.

        Args:
            feed_dict (Dict[str, torch.Tensor]): A dictionary containing input tensors.
            out_dict (Dict[str, torch.Tensor]): A dictionary containing the model's output tensors.
            metrics (Optional[List[str]]): A list of evaluation metrics to compute (default: None).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing computed losses and evaluation metrics.
        """

        losses = defaultdict(lambda: torch.zeros((), device=self.device))

        predictions = out_dict["prediction"].flatten()
        labels = out_dict["label"].flatten()
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

        return losses

    def get_feed_dict(
        self,
        corpus: DataReader,
        data: pd.DataFrame,
        batch_start: int,
        batch_size: int,
        phase: str,
    ) -> Dict[str, torch.Tensor]:
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
        skill_seqs = data["skill_seq"][
            batch_start : batch_start + real_batch_size
        ].values
        quest_seqs = data["problem_seq"][
            batch_start : batch_start + real_batch_size
        ].values
        label_seqs = data["correct_seq"][
            batch_start : batch_start + real_batch_size
        ].values
        user_seqs = data["user_id"][batch_start : batch_start + real_batch_size].values

        feed_dict = {
            "skill_seq": torch.from_numpy(
                utils.pad_lst(skill_seqs)
            ),  # [batch_size, real_max_step]
            "quest_seq": torch.from_numpy(
                utils.pad_lst(quest_seqs)
            ),  # [batch_size, real_max_step]
            "label_seq": torch.from_numpy(
                utils.pad_lst(label_seqs, value=-1)
            ),  # [batch_size, real_max_step]
            "user_seq": torch.from_numpy(user_seqs),  # [batch_size, real_max_step]
        }

        return feed_dict


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        gpu: str = "",
    ) -> None:
        """
        This is a Basic Block of the Transformer paper. It contains one Multi-head attention object.
        Followed by layer norm and position-wise feedforward net and dropout layer.

        Args:
            d_model (int): The input dimension of the model.
            d_feature (int): The dimension of the feature in the attention mechanism.
            d_ff (int): The dimension of the feedforward network.
            n_heads (int): The number of attention heads.
            dropout (float): Dropout rate for regularization.
            kq_same (bool): Whether to use the same linear transformation for keys and queries.
            gpu (str): The GPU device identifier if used (default: "").

        """
        super().__init__()
        self.gpu = gpu
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, gpu=gpu
        )

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        mask: int,
        query: torch.Tensor,
        key: torch.Tensor,
        values: torch.Tensor,
        apply_pos: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer layer.

        Args:
            mask (int): Mask value indicating whether to use padding.
            query (torch.Tensor): Query tensor of shape [batch_size, seqlen, d_model].
            key (torch.Tensor): Key tensor of shape [batch_size, seqlen, d_model].
            values (torch.Tensor): Value tensor of shape [batch_size, seqlen, d_model].
            apply_pos (bool): Whether to apply position-wise feedforward operations (default: True).

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seqlen, d_model].
        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = torch.from_numpy(nopeek_mask) == 0
        src_mask = src_mask.cuda() if self.gpu != "" else src_mask
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True
            )
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False
            )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        bias: bool = True,
        gpu: str = "",
    ) -> None:
        """
        Multi-Head Attention module for Transformer architecture.

        Args:
            d_model (int): The dimensionality of the input and output vectors.
            d_feature (int): The dimensionality of each head in multi-head attention.
            n_heads (int): The number of attention heads.
            dropout (float): The dropout rate applied to the attention scores.
            kq_same (bool): Whether to use the same weights for key and query projections.
            bias (bool, optional): Whether to include bias terms (default: True).
            gpu (str, optional): GPU device identifier to use (default: "").
        """
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.gpu = gpu

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        zero_pad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the Multi-Head Attention layer.

        Args:
            q (torch.Tensor): Queries tensor with shape (batch_size, query_length, d_model).
            k (torch.Tensor): Keys tensor with shape (batch_size, key_length, d_model).
            v (torch.Tensor): Values tensor with shape (batch_size, value_length, d_model).
            mask (torch.Tensor): Mask tensor to mask out padded positions during attention calculation.
            zero_pad (torch.Tensor): Zero padding tensor to mark padded positions.

        Returns:
            torch.Tensor: Output tensor after applying Multi-Head Attention.
        """

        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(
            q, k, v, self.d_k, mask, self.dropout, zero_pad, self.gammas
        )

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).reshape(bs, -1, self.d_model)
        output = self.out_proj(concat)

        return output

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        d_k: int,
        mask: torch.Tensor,
        dropout: nn.Module,
        zero_pad: torch.Tensor,
        gamma: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate the attention scores and apply attention mechanism.

        Args:
            q (torch.Tensor): Queries tensor with shape (batch_size, n_heads, query_length, d_k).
            k (torch.Tensor): Keys tensor with shape (batch_size, n_heads, key_length, d_k).
            v (torch.Tensor): Values tensor with shape (batch_size, n_heads, value_length, d_k).
            d_k (int): The dimensionality of keys and queries in each head.
            mask (torch.Tensor): Mask tensor to mask out padded positions during attention calculation.
            dropout (nn.Module): Dropout layer for applying dropout to attention scores.
            zero_pad (torch.Tensor): Zero padding tensor to mark padded positions.
            gamma (torch.Tensor, optional): Gamma parameter for position effect (default: None).

        Returns:
            torch.Tensor: Output tensor after applying attention mechanism.
        """
        scores = (
            torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5
        )  # BS, head, seqlen, seqlen
        bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

        x1 = torch.arange(seqlen).expand(seqlen, -1)
        x1 = x1.cuda() if self.gpu != "" else x1
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
            scores_ = scores_ * mask.float()
            scores_ = scores_.cuda() if self.gpu != "" else scores_
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
            position_effect = torch.abs(x1 - x2)[
                None, None, :, :
            ].float()  # 1, 1, seqlen, seqlen
            position_effect = (
                position_effect.cuda() if self.gpu != "" else position_effect
            )
            # bs, 8, sl, sl positive distance
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()
        m = nn.Softplus()
        gamma = -1.0 * m(gamma).unsqueeze(0)  # 1,8,1,1
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
        )
        scores = scores * total_effect

        maxim = torch.tensor(-1e20).to(scores.device)
        scores = scores.masked_fill(mask == 0, maxim)  # float('-inf'))
        scores = F.softmax(scores, dim=-1)  # BS, head, seqlen, seqlen
        if zero_pad:
            pad_zero = torch.zeros(bs, head, 1, seqlen).float()
            pad_zero = pad_zero.cuda() if self.gpu != "" else pad_zero
            scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

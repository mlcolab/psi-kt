from pathlib import Path
from typing import List, Dict, Optional
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from knowledge_tracing.baseline.basemodel import BaseModel
from knowledge_tracing.utils import utils, logger
from knowledge_tracing.data.data_loader import DataReader


class HKT(BaseModel):
    """
    An implementation of the HKT model, extending the BaseModel.

    This class defines the HKT (Hawkes Knowledge Tracing) model,
    original paper: https://dl.acm.org/doi/10.1145/3437963.3441802

    Args:
        args (argparse.Namespace):
            Namespace containing parsed command-line arguments.
        corpus (DataReader):
            An instance of the DataReader class containing corpus data.
        logs (Logger):
            An instance of the Logger class for logging purposes.

    Attributes:
        extra_log_args (List[str]): List of additional arguments to include in logs.
            These are specific to the HKT model.

    Methods:
        parse_model_args(parser, model_name="HKT"):
            Parse HKT-specific model arguments from the command line.

        __init__(args, corpus, logs):
            Initialize an instance of the HKT class.

    """

    extra_log_args = ["time_log"]

    @staticmethod
    def parse_model_args(
        parser: argparse.ArgumentParser,
        model_name: str = "HKT",
    ) -> argparse.Namespace:
        """
        Parse HKT-specific model arguments from the command line.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
            model_name (str, optional): Name of the model. Defaults to "HKT".

        Returns:
            argparse.Namespace: Parsed command-line arguments.

        """

        parser.add_argument(
            "--emb_size", type=int, default=16, help="Size of embedding vectors."
        )
        parser.add_argument(
            "--time_log", type=float, default=np.e, help="Log base of time intervals."
        )
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(
        self,
        args: argparse.Namespace,
        corpus: DataReader,
        logs: logger.Logger,
    ) -> None:
        self.dataset = args.dataset
        self.problem_num = int(corpus.n_problems)
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.time_log = args.time_log

        # Set the device to use for computations
        self.device = args.device

        # Store the arguments and logs for later use
        self.args = args
        self.logs = logs

        BaseModel.__init__(self, model_path=Path(args.log_path, "Model"))

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model components.

        This method initializes various components such as embeddings and loss function.
        """

        # Problem and skill embeddings with size 1 for bias terms
        self.problem_base = nn.Embedding(self.problem_num, 1)
        self.skill_base = nn.Embedding(self.skill_num, 1)

        # Embeddings for alpha and beta for interaction and skill terms
        self.alpha_inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size)
        self.alpha_skill_embeddings = nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = nn.Embedding(self.skill_num, self.emb_size)

    def forward(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the model given the input feed_dict.

        Args:
            feed_dict: A dictionary containing the input tensors for the model.

        Returns:
            A dictionary containing the output tensors for the model.
        """

        items = feed_dict["skill_seq"]  # [batch_size, seq_len]
        problems = feed_dict["problem_seq"]  # [batch_size, seq_len]
        times = feed_dict["time_seq"]  # [batch_size, seq_len]
        labels = feed_dict["label_seq"]  # [batch_size, seq_len]

        mask_labels = labels * (labels > -1).long()
        inters = items + mask_labels * self.skill_num  # (batch_size, seq_len)

        # alpha: for each learner, how much influence from previous skill and performance on other items
        # although it is for each learner, but the skill embedding is universal
        alpha_src_emb = self.alpha_inter_embeddings(
            inters
        )  # [batch_size, seq_len, emb]
        alpha_target_emb = self.alpha_skill_embeddings(
            items
        )  # [batch_size, seq_len, emb]
        alphas = torch.matmul(
            alpha_src_emb, alpha_target_emb.transpose(-2, -1)
        )  # [batch_size, seq_len, seq_len]

        beta_src_emb = self.beta_inter_embeddings(inters)  # [batch_size, seq_len, emb]
        beta_target_emb = self.beta_skill_embeddings(items)
        betas = torch.matmul(
            beta_src_emb, beta_target_emb.transpose(-2, -1)
        )  # [batch_size, seq_len, seq_len]
        betas = torch.clamp(betas + 1, min=0, max=10)

        delta_t = (
            (times[:, :, None] - times[:, None, :]).abs().double()
        )  # [batch_size, seq_len, seq_len]
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        cross_effects = alphas * torch.exp(
            -betas * delta_t
        )  # [batch_size, seq_len, seq_len]

        seq_len = items.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0).to(cross_effects.device)
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2)  # [batch_size, seq_len]

        problem_bias = self.problem_base(problems).squeeze(dim=-1)
        skill_bias = self.skill_base(items).squeeze(dim=-1)

        prediction = (problem_bias + skill_bias + sum_t).sigmoid()

        # Return predictions and labels from the second position in the sequence
        time_step = items.shape[1]
        out_dict = {
            "prediction": prediction[:, 1:] if time_step > 1 else prediction,
            "label": labels[:, 1:].double() if time_step > 1 else labels.double(),
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

    def get_feed_dict(
        self,
        corpus: DataReader,
        data: pd.DataFrame,
        batch_start: int,
        batch_size: int,
        phase: str,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get the input feed dictionary for a batch of data.

        Args:
            corpus: The Corpus object containing the vocabulary and other dataset information.
            data: A DataFrame containing the input data for the batch.
            batch_start: The starting index of the batch.
            batch_size: The size of the batch.
            phase: The phase of the model (e.g. 'train', 'eval', 'test').
            device: The device to place the tensors on.

        Returns:
            A dictionary containing the input tensors for the model.
        """

        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        skill_seqs = data["skill_seq"][
            batch_start : batch_start + real_batch_size
        ].values
        label_seqs = data["correct_seq"][
            batch_start : batch_start + real_batch_size
        ].values
        time_seqs = data["time_seq"][batch_start : batch_start + real_batch_size].values
        problem_seqs = data["problem_seq"][
            batch_start : batch_start + real_batch_size
        ].values

        feed_dict = {
            "skill_seq": torch.from_numpy(utils.pad_lst(skill_seqs)),
            "label_seq": torch.from_numpy(
                utils.pad_lst(label_seqs, value=-1)
            ),  # [batch_size, seq_len]
            "problem_seq": torch.from_numpy(
                utils.pad_lst(problem_seqs)
            ),  # [batch_size, seq_len]
            "time_seq": torch.from_numpy(
                utils.pad_lst(time_seqs)
            ),  # [batch_size, seq_len]
        }
        return feed_dict

    def predictive_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the model for a given input feed dictionary.

        Args:
            feed_dict: A dictionary containing the input tensors for the model.

        Returns:
            A dictionary containing the output tensors for the model.
        """

        train_step = int(self.args.max_step * self.args.train_time_ratio)

        items = feed_dict["skill_seq"][:, train_step - 1 :]  # [batch_size, seq_len]
        problems = feed_dict["problem_seq"][
            :, train_step - 1 :
        ]  # [batch_size, seq_len]
        times = feed_dict["time_seq"][:, train_step - 1 :]  # [batch_size, seq_len]
        labels = feed_dict["label_seq"][:, train_step - 1 :]  # [batch_size, seq_len]

        test_time = items.shape[-1]
        predictions = []
        for i in range(0, test_time - 1):
            if i == 0:
                inters = items[:, 0:1] + labels[:, 0:1] * self.skill_num
                delta_t = times[:, :1].unsqueeze(dim=-1)
            else:
                pred_labels = torch.cat(
                    [labels[:, 0:1], (torch.cat(predictions, -1) >= 0.5) * 1], dim=-1
                )
                inters = items[:, : i + 1] + pred_labels * self.skill_num

                cur_time = times[:, : i + 1]
                delta_t = (
                    (cur_time[:, :, None] - cur_time[:, None, :]).abs().double()
                )  # [batch_size, seq_len, seq_len]
                delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

            alpha_src_emb = self.alpha_inter_embeddings(
                inters
            )  # [batch_size, seq_len, emb]
            alpha_target_emb = self.alpha_skill_embeddings(
                items[:, i + 1 : i + 2]
            )  # [batch_size, seq_len, emb]
            alphas = torch.matmul(
                alpha_src_emb, alpha_target_emb.transpose(-2, -1)
            )  # [batch_size, seq_len, seq_len]

            beta_src_emb = self.beta_inter_embeddings(
                inters
            )  # [batch_size, seq_len, emb]
            beta_target_emb = self.beta_skill_embeddings(items[:, i + 1 : i + 2])
            betas = torch.matmul(
                beta_src_emb, beta_target_emb.transpose(-2, -1)
            )  # [batch_size, seq_len, seq_len]
            betas = torch.clamp(betas + 1, min=0, max=10)

            cross_effects = alphas * torch.exp(
                -betas * delta_t
            )  # [batch_size, seq_len, seq_len]

            sum_t = cross_effects.sum(-2)  # [batch_size, seq_len]

            problem_bias = self.problem_base(problems[:, i + 1 : i + 2]).squeeze(dim=-1)
            skill_bias = self.skill_base(items[:, i + 1 : i + 2]).squeeze(dim=-1)

            prediction = (problem_bias + skill_bias + sum_t).sigmoid()[:, -1:]
            predictions.append(prediction)

        prediction = torch.cat(predictions, dim=-1)
        # Return predictions and labels from the second position in the sequence
        out_dict = {
            "prediction": prediction,
            "label": labels[:, 1:].double(),
        }

        return out_dict

    def forward_cl(
        self,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
    ):
        """
        Forward pass for the classification task, given the input feed_dict and the current time index.

        Args:
            feed_dict: A dictionary containing the input tensors for the model.
            idx: The current time index. Only items and labels up to this index are used.

        Returns:
            A dictionary containing the output tensors for the classification task.
        """

        # Extract input tensors from feed_dict
        cur_feed_dict = utils.get_feed_continual(
            keys=["skill_seq", "label_seq", "problem_seq", "time_seq"],
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

        items = feed_dict["skill_seq"][:, idx : idx + test_step + 1]
        problems = feed_dict["problem_seq"][:, idx : idx + test_step + 1]
        times = feed_dict["time_seq"][:, idx : idx + test_step + 1]
        labels = feed_dict["label_seq"][:, idx : idx + test_step + 1]

        predictions = []
        for i in range(0, test_step):
            if i == 0:
                inters = items[:, 0:1] + labels[:, 0:1] * self.skill_num
                delta_t = times[:, :1].unsqueeze(dim=-1)
            else:
                pred_labels = torch.cat(
                    [labels[:, 0:1], (torch.cat(predictions, -1) >= 0.5) * 1], dim=-1
                )
                inters = items[:, : i + 1] + pred_labels * self.skill_num

                cur_time = times[:, : i + 1]
                delta_t = (
                    (cur_time[:, :, None] - cur_time[:, None, :]).abs().double()
                )  # [batch_size, seq_len, seq_len]
                delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

            alpha_src_emb = self.alpha_inter_embeddings(
                inters
            )  # [batch_size, seq_len, emb]
            alpha_target_emb = self.alpha_skill_embeddings(
                items[:, i + 1 : i + 2]
            )  # [batch_size, seq_len, emb]
            alphas = torch.matmul(
                alpha_src_emb, alpha_target_emb.transpose(-2, -1)
            )  # [batch_size, seq_len, seq_len]

            beta_src_emb = self.beta_inter_embeddings(
                inters
            )  # [batch_size, seq_len, emb]
            beta_target_emb = self.beta_skill_embeddings(items[:, i + 1 : i + 2])
            betas = torch.matmul(
                beta_src_emb, beta_target_emb.transpose(-2, -1)
            )  # [batch_size, seq_len, seq_len]
            betas = torch.clamp(betas + 1, min=0, max=10)

            cross_effects = alphas * torch.exp(
                -betas * delta_t
            )  # [batch_size, seq_len, seq_len]

            sum_t = cross_effects.sum(-2)  # [batch_size, seq_len]

            problem_bias = self.problem_base(problems[:, i + 1 : i + 2]).squeeze(dim=-1)
            skill_bias = self.skill_base(items[:, i + 1 : i + 2]).squeeze(dim=-1)

            prediction = (problem_bias + skill_bias + sum_t).sigmoid()[:, -1:]
            predictions.append(prediction)

        labels = labels[:, 1:].float()
        prediction = torch.cat(predictions, -1)

        return self.pred_evaluate_method(
            prediction.flatten().cpu(), labels.flatten().cpu(), metrics
        )

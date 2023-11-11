import os
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from pykt.utils import debug_print
from sklearn import metrics
from torch.utils.data import DataLoader

# from .loss import Loss
from scipy.special import softmax

import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

import torch

from knowledge_tracing.baseline.pykt.que_base_model import QueBaseModel, QueEmb
from knowledge_tracing.baseline.basemodel import BaseModel
from knowledge_tracing.utils import utils, logger
from knowledge_tracing.data.data_loader import DataReader


class MLP(nn.Module):
    """
    classifier decoder implemented with mlp
    """

    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer)]
        )
        self.dropout = nn.Dropout(p=dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


def get_outputs(self, emb_qc_shift, h, data, add_name="", model_type="question"):
    outputs = {}
    # https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/datasets/que_data_loader.py
    if model_type == "question":
        h_next = torch.cat([emb_qc_shift, h], axis=-1)
        y_question_next = torch.sigmoid(self.out_question_next(h_next))
        y_question_all = torch.sigmoid(self.out_question_all(h))
        outputs["y_question_next" + add_name] = y_question_next.squeeze(
            -1
        )  # [bs, time-1]
        outputs["y_question_all" + add_name] = (
            y_question_all * F.one_hot(data.long(), self.num_q)
        ).sum(-1)
    else:
        h_next = torch.cat([emb_qc_shift, h], axis=-1)
        y_concept_next = torch.sigmoid(self.out_concept_next(h_next))
        y_concept_all = torch.sigmoid(self.out_concept_all(h))
        outputs["y_concept_next" + add_name] = self.get_avg_fusion_concepts(
            y_concept_next, data.long()
        )
        outputs["y_concept_all" + add_name] = self.get_avg_fusion_concepts(
            y_concept_all, data.long()
        )

    return outputs


class QIKT(BaseModel):
    @staticmethod
    def parse_model_args(
        parser: argparse.ArgumentParser,
        model_name: str = "QIKT",
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
        parser.add_argument("--mlp_layer_num", type=int, default=2, help="")
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(
        self,
        args: argparse.Namespace,
        corpus: DataReader,
        logs: logger.Logger,
        emb_type="qaid_qc",
        emb_path="",
        pretrain_dim=768,
        device="cpu",
        mlp_layer_num=1,
        other_config={},
    ):
        self.model_name = "qcaid"
        self.num_q = int(corpus.n_problems)
        self.num_c = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.hidden_size = args.emb_size
        self.mlp_layer_num = args.mlp_layer_num
        self.dropout = args.dropout
        self.device = args.device

        self.emb_path = emb_path
        self.pretrain_dim = pretrain_dim
        self.other_config = other_config
        self.output_mode = self.other_config.get("output_mode", "an")

        # Store the arguments and logs for later use
        self.args = args
        self.logs = logs

        self.emb_type = emb_type

        BaseModel.__init__(self, model_path=Path(args.log_path, "Model"))

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model components.

        This method initializes various components such as embeddings and loss function.
        """
        self.que_emb = QueEmb(
            num_q=self.num_q,
            num_c=self.num_c,
            emb_size=self.emb_size,
            emb_type=self.emb_type,
            model_name=self.model_name,
            device=self.device,
            emb_path=self.emb_path,
            pretrain_dim=self.pretrain_dim,
        )

        self.que_lstm_layer = nn.LSTM(
            self.emb_size * 4, self.hidden_size, batch_first=True
        )
        self.concept_lstm_layer = nn.LSTM(
            self.emb_size * 2, self.hidden_size, batch_first=True
        )

        self.dropout_layer = nn.Dropout(self.dropout)

        self.out_question_next = MLP(
            self.mlp_layer_num, self.hidden_size * 3, 1, self.dropout
        )
        self.out_question_all = MLP(
            self.mlp_layer_num, self.hidden_size, self.num_q, self.dropout
        )

        self.out_concept_next = MLP(
            self.mlp_layer_num, self.hidden_size * 3, self.num_c, self.dropout
        )
        self.out_concept_all = MLP(
            self.mlp_layer_num, self.hidden_size, self.num_c, self.dropout
        )

        self.que_disc = MLP(self.mlp_layer_num, self.hidden_size * 2, 1, self.dropout)

        # Binary Cross Entropy Loss
        self.loss_function = nn.BCELoss()

    def sigmoid_inverse(self, x, epsilon=1e-8):
        return torch.log(x / (1 - x + epsilon) + epsilon)

    def get_avg_fusion_concepts(self, y_concept, cshft):
        # import ipdb; ipdb.set_trace()
        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long() == -1, False, True)
        concept_index = F.one_hot(torch.where(cshft != -1, cshft, 0), self.num_c)
        # concept_sum = (
        #     y_concept.unsqueeze(2).repeat(1, 1, max_num_concept, 1) * concept_index
        # ).sum(-1)
        # concept_sum = concept_sum * concept_mask  # remove mask
        # y_concept = concept_sum.sum(-1) / torch.where(
        #     concept_mask.sum(-1) != 0, concept_mask.sum(-1), 1
        # )
        # return y_concept
        concept_sum = (y_concept * concept_index).sum(-1)
        return concept_sum

    def forward(self, feed_dict, data=None):
        q = feed_dict["problem_seq"]
        c = feed_dict["skill_seq"]
        r = feed_dict["label_seq"]

        # _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(
        #     q, c, r
        # )  # [batch_size,emb_size*4],[batch_size,emb_size*2],[batch_size,emb_size*1],[batch_size,emb_size*1]

        _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(q, c, r)
        emb_qc_shift = emb_qc[:, 1:, :]
        emb_qca_current = emb_qca[:, :-1, :]

        # question model
        que_h = self.dropout_layer(
            self.que_lstm_layer(emb_qca_current)[0]
        )  # [bs, time-1, emb]
        que_outputs = get_outputs(
            self, emb_qc_shift, que_h, q[:, 1:], add_name="", model_type="question"
        )  # ['y_question_next', 'y_question_all'] [bs, time-1] [bs, time-1]
        out_dict = que_outputs

        # concept model
        emb_ca = torch.cat(
            [
                emb_c.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
                emb_c.mul((r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
            ],
            dim=-1,
        )  # s_t for corectness and incorrectness; [bs, time, emb*2]

        emb_ca_current = emb_ca[:, :-1, :]
        concept_h = self.dropout_layer(
            self.concept_lstm_layer(emb_ca_current)[0]
        )  # [bs, time-1, emb]
        concept_outputs = get_outputs(
            self, emb_qc_shift, concept_h, c[:, 1:], add_name="", model_type="concept"
        )
        out_dict["y_concept_all"] = concept_outputs["y_concept_all"]
        out_dict["y_concept_next"] = concept_outputs["y_concept_next"]

        y = (
            self.sigmoid_inverse(out_dict["y_question_all"])
            + self.sigmoid_inverse(out_dict["y_concept_all"])
            + self.sigmoid_inverse(out_dict["y_concept_next"])
        )
        y = torch.sigmoid(y)

        out_dict["prediction"] = y
        out_dict["label"] = r[:, 1:].double()

        return out_dict

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

        c = feed_dict["skill_seq"][:, train_step - 1 :]  # [bs, seq_len]
        q = feed_dict["problem_seq"][:, train_step - 1 :]  # [bs, seq_len]
        r = feed_dict["label_seq"][:, train_step - 1 :]  # [bs, seq_len]

        test_time = c.shape[-1]
        predictions = []

        for i in range(0, test_time - 1):
            if i < 1:
                history_labels = r[:, 0:2]
                _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(
                    q[:, 0:2], c[:, 0:2], history_labels
                )
            else:
                history_labels = torch.cat(
                    [r[:, 0:1], (torch.cat(predictions, -1) >= 0.5) * 1, r[:, i+1:i+2]], dim=-1
                )
                # import ipdb; ipdb.set_trace()
                _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(
                    q[:, : i + 2], c[:, : i + 2], history_labels
                )
            emb_qc_shift = emb_qc[:, 1:, :]
            emb_qca_current = emb_qca[:, :-1, :]

            # question model
            que_h = self.dropout_layer(
                self.que_lstm_layer(emb_qca_current)[0]
            )  # [bs, time-1, emb]
            que_outputs = get_outputs(
                self, emb_qc_shift, que_h, q[:, 1:i+2], add_name="", model_type="question"
            )  # ['y_question_next', 'y_question_all'] [bs, time-1] [bs, time-1]
            out_dict = que_outputs

            # concept model
            emb_ca = torch.cat(
                [
                    emb_c.mul((1 - history_labels).unsqueeze(-1).repeat(1, 1, self.emb_size)),
                    emb_c.mul((history_labels).unsqueeze(-1).repeat(1, 1, self.emb_size)),
                ],
                dim=-1,
            )  # s_t for corectness and incorrectness; [bs, time, emb*2]

            emb_ca_current = emb_ca[:, :-1, :]
            concept_h = self.dropout_layer(
                self.concept_lstm_layer(emb_ca_current)[0]
            )  # [bs, time-1, emb]
            concept_outputs = get_outputs(
                self,
                emb_qc_shift,
                concept_h,
                c[:, 1:i+2],
                add_name="",
                model_type="concept",
            )
            out_dict["y_concept_all"] = concept_outputs["y_concept_all"]
            out_dict["y_concept_next"] = concept_outputs["y_concept_next"]

            y = (
                self.sigmoid_inverse(out_dict["y_question_all"])
                + self.sigmoid_inverse(out_dict["y_concept_all"])
                + self.sigmoid_inverse(out_dict["y_concept_next"])
            )
            y = torch.sigmoid(y)

            predictions.append(y[:, -1:])

        prediction = torch.cat(predictions, dim=-1)
        # Return predictions and labels from the second position in the sequence
        out_dict = {
            "prediction": prediction,
            "label": r[:, 1:].double(),
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

        predictions = out_dict["prediction"].flatten().double()
        y_question_all = out_dict["y_question_all"].flatten().double()
        y_concept_all = out_dict["y_concept_all"].flatten().double()
        y_question_next = out_dict["y_question_next"].flatten().double()
        y_concept_next = out_dict["y_concept_next"].flatten().double()

        labels = out_dict["label"].flatten()
        mask = labels > -1
        loss_all = self.loss_function(predictions[mask], labels[mask])
        loss_qall = self.loss_function(y_question_all[mask], labels[mask])
        loss_call = self.loss_function(y_concept_all[mask], labels[mask])
        loss_cnext = self.loss_function(y_concept_next[mask], labels[mask])
        losses["loss_total"] = loss_all + loss_qall + loss_call + loss_cnext

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
            ),  # [bs, seq_len]
            "problem_seq": torch.from_numpy(
                utils.pad_lst(problem_seqs)
            ),  # [bs, seq_len]
            "time_seq": torch.from_numpy(utils.pad_lst(time_seqs)),  # [bs, seq_len]
        }
        return feed_dict

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
        cur_feed_dict = {
            "skill_seq": feed_dict["skill_seq"][:, : idx + 1],
            "label_seq": feed_dict["label_seq"][:, : idx + 1],
            "problem_seq": feed_dict["problem_seq"][:, : idx + 1],
            "time_seq": feed_dict["time_seq"][:, : idx + 1],
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
        
        c = feed_dict["skill_seq"][:, idx : idx + test_step + 1]
        q = feed_dict["problem_seq"][
            :, idx : idx + test_step + 1]
        r = feed_dict["label_seq"][:, idx : idx + test_step + 1]

        predictions = []
        
        for i in range(0, test_step):
            # import ipdb; ipdb.set_trace()
            if i < 1:
                history_labels = r[:, 0:2]
                _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(
                    q[:, 0:2], c[:, 0:2], history_labels
                )
            else:
                history_labels = torch.cat(
                    [r[:, 0:1], (torch.cat(predictions, -1) >= 0.5) * 1, r[:, i+1:i+2]], dim=-1
                )
                # import ipdb; ipdb.set_trace()
                _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(
                    q[:, : i + 2], c[:, : i + 2], history_labels
                )
            emb_qc_shift = emb_qc[:, 1:, :]
            emb_qca_current = emb_qca[:, :-1, :]

            # question model
            que_h = self.dropout_layer(
                self.que_lstm_layer(emb_qca_current)[0]
            )  # [bs, time-1, emb]
            que_outputs = get_outputs(
                self, emb_qc_shift, que_h, q[:, 1:i+2], add_name="", model_type="question"
            )  # ['y_question_next', 'y_question_all'] [bs, time-1] [bs, time-1]
            out_dict = que_outputs

            # concept model
            emb_ca = torch.cat(
                [
                    emb_c.mul((1 - history_labels).unsqueeze(-1).repeat(1, 1, self.emb_size)),
                    emb_c.mul((history_labels).unsqueeze(-1).repeat(1, 1, self.emb_size)),
                ],
                dim=-1,
            )  # s_t for corectness and incorrectness; [bs, time, emb*2]

            emb_ca_current = emb_ca[:, :-1, :]
            concept_h = self.dropout_layer(
                self.concept_lstm_layer(emb_ca_current)[0]
            )  # [bs, time-1, emb]
            concept_outputs = get_outputs(
                self,
                emb_qc_shift,
                concept_h,
                c[:, 1:i+2],
                add_name="",
                model_type="concept",
            )
            out_dict["y_concept_all"] = concept_outputs["y_concept_all"]
            out_dict["y_concept_next"] = concept_outputs["y_concept_next"]

            y = (
                self.sigmoid_inverse(out_dict["y_question_all"])
                + self.sigmoid_inverse(out_dict["y_concept_all"])
                + self.sigmoid_inverse(out_dict["y_concept_next"])
            )
            y = torch.sigmoid(y)

            predictions.append(y[:, -1:])
            
        prediction = torch.cat(predictions, dim=-1)
        labels = r[:, 1:]

        return self.pred_evaluate_method(
                    prediction.flatten().cpu(), labels.flatten().cpu(), metrics
                )
        

#     def train_one_step(self, data, process=True, return_all=False):
#         outputs, data_new = self.predict_one_step(
#             data, return_details=True, process=process
#         )

#         def get_loss_lambda(x):
#             return self.model.other_config.get(
#                 f"loss_{x}", 0
#             ) * self.model.other_config.get(f"output_{x}", 0)

#         # loss weight
#         loss_c_all_lambda = get_loss_lambda("c_all_lambda")
#         loss_c_next_lambda = get_loss_lambda("c_next_lambda")
#         loss_q_all_lambda = get_loss_lambda("q_all_lambda")
#         loss_q_next_lambda = get_loss_lambda("q_next_lambda")

#         if self.model.output_mode == "an_irt":
#             loss = (
#                 loss_kt
#                 + loss_q_all_lambda * loss_q_all
#                 + loss_c_all_lambda * loss_c_all
#                 + loss_c_next_lambda * loss_c_next
#             )
#         else:
#             loss = (
#                 loss_kt
#                 + loss_q_all_lambda * loss_q_all
#                 + loss_c_all_lambda * loss_c_all
#                 + loss_c_next_lambda * loss_c_next
#                 + loss_q_next_lambda * loss_q_next
#             )
#         # print(f"loss={loss:.3f},loss_kt={loss_kt:.3f},loss_q_all={loss_q_all:.3f},loss_c_all={loss_c_all:.3f},loss_q_next={loss_q_next:.3f},loss_c_next={loss_c_next:.3f}")
#         return outputs["y"], loss  # y_question没用


#     def predict(self, dataset, batch_size, return_ts=False, process=True):
#         test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#         self.model.eval()
#         with torch.no_grad():
#             y_trues = []
#             y_pred_dict = {}
#             for data in test_loader:
#                 new_data = self.batch_to_device(data, process=process)
#                 outputs, data_new = self.predict_one_step(data, return_details=True)

#                 for key in outputs:
#                     if not key.startswith("y") or key in ["y_qc_predict"]:
#                         continue
#                     elif key not in y_pred_dict:
#                         y_pred_dict[key] = []
#                     y = (
#                         torch.masked_select(outputs[key], new_data["sm"]).detach().cpu()
#                     )  # get label
#                     y_pred_dict[key].append(y.numpy())

#                 t = (
#                     torch.masked_select(new_data["rshft"], new_data["sm"])
#                     .detach()
#                     .cpu()
#                 )
#                 y_trues.append(t.numpy())

#         results = y_pred_dict
#         for key in results:
#             results[key] = np.concatenate(results[key], axis=0)
#         ts = np.concatenate(y_trues, axis=0)
#         results["ts"] = ts
#         return results

#     def evaluate(self, dataset, batch_size, acc_threshold=0.5):
#         results = self.predict(dataset, batch_size=batch_size)
#         eval_result = {}
#         ts = results["ts"]
#         for key in results:
#             if not key.startswith("y") or key in ["y_qc_predict"]:
#                 pass
#             else:
#                 ps = results[key]
#                 kt_auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
#                 prelabels = [1 if p >= acc_threshold else 0 for p in ps]
#                 kt_acc = metrics.accuracy_score(ts, prelabels)
#                 if key != "y":
#                     eval_result["{}_kt_auc".format(key)] = kt_auc
#                     eval_result["{}_kt_acc".format(key)] = kt_acc
#                 else:
#                     eval_result["auc"] = kt_auc
#                     eval_result["acc"] = kt_acc

#         self.eval_result = eval_result
#         return eval_result

#     def predict_one_step(
#         self, data, return_details=False, process=True, return_raw=False
#     ):
#         data_new = self.batch_to_device(data, process=process)
#         outputs = self.model(
#             data_new["cq"].long(), data_new["cc"], data_new["cr"].long(), data=data_new
#         )
#         output_c_all_lambda = self.model.other_config.get("output_c_all_lambda", 1)
#         output_c_next_lambda = self.model.other_config.get("output_c_next_lambda", 1)
#         output_q_all_lambda = self.model.other_config.get("output_q_all_lambda", 1)
#         output_q_next_lambda = self.model.other_config.get(
#             "output_q_next_lambda", 0
#         )  # not use this

#         if self.model.output_mode == "an_irt":

#             def sigmoid_inverse(x, epsilon=1e-8):
#                 return torch.log(x / (1 - x + epsilon) + epsilon)

#             y = (
#                 sigmoid_inverse(outputs["y_question_all"]) * output_q_all_lambda
#                 + sigmoid_inverse(outputs["y_concept_all"]) * output_c_all_lambda
#                 + sigmoid_inverse(outputs["y_concept_next"]) * output_c_next_lambda
#             )
#             y = torch.sigmoid(y)
#         else:
#             # output weight
#             y = (
#                 outputs["y_question_all"] * output_q_all_lambda
#                 + outputs["y_concept_all"] * output_c_all_lambda
#                 + outputs["y_concept_next"] * output_c_next_lambda
#             )
#             y = y / (output_q_all_lambda + output_c_all_lambda + output_c_next_lambda)
#         outputs["y"] = y

#         if return_details:
#             return outputs, data_new
#         else:
#             return y

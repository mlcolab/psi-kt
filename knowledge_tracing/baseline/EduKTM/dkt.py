import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

import torch

from knowledge_tracing.baseline.basemodel import BaseModel
from knowledge_tracing.utils import utils, logger
from knowledge_tracing.data.data_loader import DataReader


class DKT(BaseModel):
    """
    An implementation of the DKT model, extending the BaseModel.

    This class defines the DKT (Deep Knowledge Tracing) model,
    originally proposed in [Piech et al., 2015](https://arxiv.org/abs/1506.05908).

    Args:
        args (argparse.Namespace):
            Namespace containing parsed command-line arguments.
        corpus (DataReader):
            An instance of the DataReader class containing corpus data.
        logs (Logger):
            An instance of the Logger class for logging purposes.

    Attributes:
        extra_log_args (List[str]): List of additional arguments to include in logs.
            These are specific to the DKT model.

    """

    extra_log_args = ["hidden_size"]

    @staticmethod
    def parse_model_args(
        parser: argparse.ArgumentParser,
        model_name: str = "DKT",
    ) -> argparse.ArgumentParser:
        """
        Parse DKT-specific model arguments from the command line.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
            model_name (str, optional): Name of the model. Defaults to "DKT".

        Returns:
            argparse.Namespace: Parsed command-line arguments.

        """

        parser.add_argument(
            "--emb_size", type=int, default=16, help="Size of embedding vectors."
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=16,
            help="Size of hidden vectors in LSTM.",
        )
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(
        self,
        args: argparse.Namespace,
        corpus: DataReader,
        logs: logger.Logger,
    ):
        # Set the size of the skill embedding, the hidden size of the LSTM layer, the number of LSTM layers, and the dropout rate
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout

        # Set the device to use for computations
        self.device = args.device

        # Store the arguments and logs for later use
        self.args = args
        self.logs = logs
        BaseModel.__init__(self, model_path=Path(args.log_path, "Model"))

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model.

        This function creates and initializes the layers and weights of the learner model,
        including skill embeddings, an LSTM layer, and an output linear layer.

        Returns:
            None
        """

        # Define the skill embeddings layer
        self.skill_embeddings = torch.nn.Embedding(
            self.skill_num * 2, self.emb_size, device=self.device
        )

        self.rnn = torch.nn.LSTM(
            input_size=self.emb_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )

        self.out = torch.nn.Linear(self.hidden_size, self.skill_num)

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
            A dictionary containing the output tensors for the ontinual learning task.
        """

        # Extract input tensors from feed_dict
        cur_feed_dict = cur_feed_dict = utils.get_feed_continual(
            keys=["skill_seq", "label_seq", "user_id", "inverse_indice", "length"],
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
        test_item = feed_dict["skill_seq"][:, idx + 1 : idx + test_step + 1]

        predictions, hiddens = [], []
        last_emb = self.skill_embeddings(
            feed_dict["skill_seq"][:, idx : idx + 1]
            + feed_dict["label_seq"][:, idx : idx + 1] * self.skill_num
        )
        latent_states = None
        for i in range(test_step):
            rnn_input, latent_states = self.rnn(last_emb, latent_states)
            pred_vector = self.out(rnn_input)  # [batch_size, 1, skill_num]
            target_item = test_item[:, i : i + 1]
            prediction_sorted = torch.gather(
                pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)
            ).squeeze(
                dim=-1
            )  # [batch_size, 1]
            prediction = torch.sigmoid(prediction_sorted)
            last_emb = self.skill_embeddings(
                test_item[:, i : i + 1] + (prediction >= 0.5) * self.skill_num
            )  # [batch_size, 1, emb_size]
            predictions.append(prediction)
            hiddens.append(rnn_input)

        # Label
        labels = feed_dict["label_seq"][:, idx + 1 : idx + test_step + 1].float()
        prediction = torch.cat(predictions, -1)

        return self.pred_evaluate_method(
            prediction.flatten().cpu(), labels.flatten().cpu(), metrics
        )

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

        # Extract input tensors from feed_dict
        items = feed_dict["skill_seq"]  # [batch_size, history_max]
        labels = feed_dict["label_seq"]  # [batch_size, history_max]
        lengths = feed_dict["length"]  # [batch_size]
        indices = feed_dict["inverse_indice"]

        time_step = items.shape[-1]

        if items.is_cuda:
            lengths = lengths.cpu().int()

        # Embed the history of items and labels
        embed_history_i = self.skill_embeddings(
            items + labels * self.skill_num
        )  # [batch_size, time, emb_size]

        # Pack the embedded history and run through the RNN
        # pack: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # issues with 'lengths must be on cpu': https://github.com/pytorch/pytorch/issues/43227
        embed_history_i_packed = embed_history_i  # torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths - 1, batch_first=True) # embed_history_i_packed.data [(time-1)*batch_size, emb_size]
        output, _ = self.rnn(
            embed_history_i_packed, None
        )  # [batch_size, time, emb_size]

        # Unpack the output of the RNN and run it through the output layer
        pred_vector = self.out(output)  # [batch_size, time, skill_num]

        # Extract the prediction for the next item and the corresponding label
        target_item = items[:, 1:] if time_step > 1 else items
        prediction_sorted = torch.gather(
            pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        prediction_sorted = torch.sigmoid(prediction_sorted)
        prediction = prediction_sorted[indices]

        # Label
        label = labels[:, 1:] if time_step > 1 else labels
        label = label[indices].double()

        out_dict = {
            "prediction": prediction,
            "label": label,
        }

        return out_dict

    def predictive_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
        single_step: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the model for a given input feed dictionary.

        Args:
            feed_dict: A dictionary containing the input tensors for the model.

        Returns:
            A dictionary containing the output tensors for the model.
        """

        # Extract input tensors from feed_dict
        items = feed_dict["skill_seq"]  # [batch_size, history_max]
        labels = feed_dict["label_seq"]  # [batch_size, history_max]
        lengths = feed_dict["length"]  # [batch_size]
        indices = feed_dict["inverse_indice"]

        all_step = items.shape[-1]
        train_step = int(self.args.max_step * self.args.train_time_ratio)
        test_step = int(self.args.max_step * self.args.test_time_ratio)
        test_item = items[:, train_step:]

        if items.is_cuda:
            lengths = lengths.cpu().int()

        predictions, hiddens = [], []
        last_emb = self.skill_embeddings(
            items[:, train_step - 1 : train_step]
            + labels[:, train_step - 1 : train_step] * self.skill_num
        )  # [batch_size, 1, emb_size]
        for i in range(test_step):
            if i == 0:
                rnn_input, latent_states = self.rnn(last_emb, None)
            else:
                rnn_input, latent_states = self.rnn(last_emb, latent_states)
            pred_vector = self.out(rnn_input)  # [batch_size, 1, skill_num]
            target_item = test_item[:, i : i + 1]
            prediction_sorted = torch.gather(
                pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)
            ).squeeze(
                dim=-1
            )  # [batch_size, 1]
            prediction_sorted = torch.sigmoid(prediction_sorted)
            prediction = prediction_sorted[indices]
            last_emb = self.skill_embeddings(
                test_item[:, i : i + 1] + (prediction >= 0.5) * 1 * self.skill_num
            )  # [batch_size, 1, emb_size]
            predictions.append(prediction)
            hiddens.append(rnn_input)

        # Label
        label = labels[:, train_step:] if all_step > 1 else labels
        label = label[indices].double()
        prediction = torch.cat(predictions, -1)
        hiddens = torch.cat(hiddens, 1)

        out_dict = {
            "prediction": prediction,
            "label": label,
        }

        return out_dict

    def loss(
        self,
        feed_dict: Dict[str, torch.Tensor],
        out_dict: Dict[str, torch.Tensor],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss and evaluation metrics for the model given the input feed_dict and the output out_dict.

        Args:
            feed_dict: A dictionary containing the input tensors for the model.
            out_dict: A dictionary containing the output tensors for the model.
            metrics: A list of evaluation metrics to compute.

        Returns:
            A dictionary containing the loss and evaluation metrics.
        """
        losses = defaultdict(lambda: torch.zeros((), device=self.device))

        # Extract indices and lengths from feed_dict
        indice = feed_dict["indice"]
        lengths = feed_dict["length"] - 1
        if lengths.is_cuda:
            lengths = lengths.cpu().int()

        # Compute the loss for the main prediction task
        predictions, labels = out_dict["prediction"][indice], out_dict["label"][indice]
        loss = self.loss_function(predictions, labels.float())
        losses["loss_total"] = loss

        # Compute the evaluation metrics for the main prediction task
        if metrics is not None:
            pred = predictions.detach().cpu().data.numpy()
            gt = labels.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
            for key in evaluations.keys():
                losses[key] = evaluations[key]

        if "cl_prediction" in out_dict.keys():
            cl_predictions, cl_labels = (
                out_dict["cl_prediction"][indice],
                out_dict["cl_label"][indice],
            )
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

        # Extract the user_ids, user_seqs, and label_seqs from the data
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        user_ids = data["user_id"][batch_start : batch_start + real_batch_size].values
        user_seqs = data["skill_seq"][
            batch_start : batch_start + real_batch_size
        ].values
        label_seqs = data["correct_seq"][
            batch_start : batch_start + real_batch_size
        ].values
        time_seqs = data["time_seq"][batch_start : batch_start + real_batch_size].values

        # Compute the lengths, indice, and inverse_indice arrays for sorting the batch by length
        lengths = np.array(list(map(lambda lst: len(lst), user_seqs)))
        indice = np.array(np.argsort(lengths, axis=-1)[::-1])
        inverse_indice = np.zeros_like(indice)
        for i, idx in enumerate(indice):
            inverse_indice[idx] = i

        # Initialize the feed_dict with the input tensors for the model
        if device is None:
            device = self.device

        feed_dict = {
            "user_id": torch.from_numpy(user_ids[indice]).to(device),
            "skill_seq": torch.from_numpy(utils.pad_lst(user_seqs[indice])).to(
                device
            ),  # [batch_size, num of items to predict]
            "label_seq": torch.from_numpy(utils.pad_lst(label_seqs[indice])).to(
                device
            ),  # [batch_size, num of items to predict]
            "time_seq": torch.from_numpy(utils.pad_lst(time_seqs[indice])).to(
                device
            ),  # [batch_size, num of items to predict]
            "length": torch.from_numpy(lengths[indice]).to(device),  # [batch_size]
            "inverse_indice": torch.from_numpy(inverse_indice).to(device),
            "indice": torch.from_numpy(indice).to(device),
        }

        return feed_dict

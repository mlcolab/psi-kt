# @Date: 2023/07/25

import sys

sys.path.append("..")

import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict

import torch

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from knowledge_tracing.baseline.BaseModel import BaseModel
from knowledge_tracing.utils import utils, logger
from knowledge_tracing.data.data_loader import DataReader


class DKT(BaseModel):
    extra_log_args = ["hidden_size", "num_layer"]

    @staticmethod
    def parse_model_args(parser, model_name="DKT"):
        parser.add_argument(
            "--emb_size", type=int, default=16, help="Size of embedding vectors."
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=16,
            help="Size of hidden vectors in LSTM.",
        )
        parser.add_argument(
            "--num_layer", type=int, default=1, help="Number of GRU layers."
        )
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(
        self,
        args: argparse.Namespace,
        corpus: DataReader,
        logs: logger.Logger,
    ):
        """
        Initialize the model.

        This function creates an instance of the model based on the provided arguments.

        Args:
            args: An object containing the arguments for the model.
            corpus: An object containing the corpus of data for the model.
            logs: An object containing the logs for the model.

        Returns:
            None
        """

        # Set the size of the skill embedding, the hidden size of the LSTM layer, the number of LSTM layers, and the dropout rate
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.dropout = args.dropout

        # Set the device to use for computations
        self.device = args.device

        # Store the arguments and logs for later use
        self.args = args
        self.logs = logs
        BaseModel.__init__(
            self, model_path=os.path.join(args.log_path, "Model/Model_{}_{}.pt")
        )

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model.

        This function creates the necessary layers of the model and initializes their weights.

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
            num_layers=self.num_layer,
        )

        self.out = torch.nn.Linear(self.hidden_size, self.skill_num)

        self.loss_function = torch.nn.BCELoss()

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
            "inverse_indice": feed_dict["inverse_indice"],
            "length": feed_dict["length"],
        }

        out_dict = self.forward(cur_feed_dict)
        # items = feed_dict['skill_seq'][:, :idx+1]     # [bs, time]
        # labels = feed_dict['label_seq'][:, :idx+1]  # [bs, time]
        # indices = feed_dict['inverse_indice']

        # # Embed the input items and labels
        # embed_history_i = self.skill_embeddings(items + labels * self.skill_num) # [bs, time, emb_size]

        # # Pass the embeddings through the RNN and the output layer
        # output, _ = self.rnn(embed_history_i, None) # [bs, time, emb_size]
        # pred_vector = self.out(output) # [bs, time, skill_num]

        # # Extract the prediction for the next item
        # target_item = feed_dict['skill_seq'][:, 1:idx+2]
        # prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1)
        # prediction_sorted = torch.sigmoid(prediction_sorted)
        # prediction = prediction_sorted[indices]

        # # Split the predictions into training and evaluation predictions
        # train_pred = prediction[:, :-1]
        # eval_pred = prediction[:, -1:]

        # # Extract the labels for the training and evaluation predictions
        # train_label = feed_dict['label_seq'][:, 1:idx+1]
        # train_label = train_label[indices].double()
        # eval_label = feed_dict['label_seq'][:, idx+1:idx+2]
        # eval_label = eval_label[indices].double()

        # out_dict = {
        #     'prediction': train_pred,
        #     'label': train_label,
        #     'cl_prediction': eval_pred,
        #     'cl_label': eval_label,
        # }
        return out_dict

    def evaluate_cl(
        self,
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

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
        )  # [bs, time, emb_size]

        # Pack the embedded history and run through the RNN
        # pack: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # issues with 'lengths must be on cpu': https://github.com/pytorch/pytorch/issues/43227
        embed_history_i_packed = embed_history_i  # torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths - 1, batch_first=True) # embed_history_i_packed.data [(time-1)*bs, emb_size]
        output, _ = self.rnn(embed_history_i_packed, None)  # [bs, time, emb_size]

        # Unpack the output of the RNN and run it through the output layer
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # output.data [bs, time-1, emb_size]
        pred_vector = self.out(output)  # [bs, time, skill_num]

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
            "emb": output,
            "learner_id": feed_dict["user_id"],
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
        )  # [bs, 1, emb_size]
        for i in range(test_step):
            if i == 0:
                rnn_input, latent_states = self.rnn(last_emb, None)
            else:
                rnn_input, latent_states = self.rnn(last_emb, latent_states)
            pred_vector = self.out(rnn_input)  # [bs, 1, skill_num]
            target_item = test_item[:, i : i + 1]
            prediction_sorted = torch.gather(
                pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)
            ).squeeze(
                dim=-1
            )  # [bs, 1]
            prediction_sorted = torch.sigmoid(prediction_sorted)
            prediction = prediction_sorted[indices]
            last_emb = self.skill_embeddings(
                test_item[:, i : i + 1] + (prediction >= 0.5) * 1 * self.skill_num
            )  # [bs, 1, emb_size]
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
            "emb": hiddens,
            "item": items,
            "time": feed_dict["time_seq"],
        }  # TODO

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
        # predictions = torch.nn.utils.rnn.pack_padded_sequence(predictions, lengths, batch_first=True).data
        # labels = torch.nn.utils.rnn.pack_padded_sequence(labels, lengths, batch_first=True).data
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

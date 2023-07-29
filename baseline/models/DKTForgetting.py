# @Date: 2023/07/29

import sys
sys.path.append('..')

import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict

import torch

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from baseline.BaseModel import BaseModel
from utils import utils
from utils import logger
from data.data_loader import DataReader


class DKTForgetting(BaseModel):
    extra_log_args = ['hidden_size', 'num_layer']

    @staticmethod
    def parse_model_args(parser, model_name='DKTForgetting'):
        parser.add_argument('--emb_size', type=int, default=16,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=16,
                            help='Size of hidden vectors in LSTM.')
        parser.add_argument('--num_layer', type=int, default=1,
                            help='Number of GRU layers.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(
        self, 
        args: argparse.Namespace,
        corpus: DataReader,
        logs: logger.Logger,
    ) -> None:
        """
        Initialize the instance of your class.

        Parameters:
            args: The arguments object containing various configurations.
            corpus: An instance of the DataReader class containing corpus data.
            logs: An instance of the Logger class for logging purposes.
        """
        
        self.problem_num = int(corpus.n_problems)
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.dropout = args.dropout

        # Set the device to use for computations
        self.device = args.device

        # Store the logs and arguments for later use
        self.logs = logs
        self.args = args

        # Call the constructor of the superclass (BaseModel) with the specified model path
        BaseModel.__init__(self, model_path=os.path.join(args.log_path, 'Model/Model_{}.pt'))


    @staticmethod
    def get_time_features(
        item_seqs: List[List[int]], 
        time_seqs: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts time-related features from a sequence of item and time sequences.

        Args:
            item_seqs: A list of lists, where each inner list contains the items seen by a student.
            time_seqs: A list of lists, where each inner list contains the timestamps of the items seen by a student.

        Returns:
            A tuple containing three arrays: sequence_time_gap_seq, repeated_time_gap_seq, and past_trial_counts_seq.
        """
    
        bs = len(item_seqs)
        skill_max = max([max(i) for i in item_seqs]) # int
        inner_max_len = max(map(len, item_seqs)) # time (50)
        
        # Initialize arrays to store time-related features
        repeated_time_gap_seq = np.zeros([bs, inner_max_len, 1], np.float32)
        sequence_time_gap_seq = np.zeros([bs, inner_max_len, 1], np.float32)
        past_trial_counts_seq = np.zeros([bs, inner_max_len, 1], np.float32)
        
        # Compute time-related features for each learner sequence
        for i in range(bs):
            last_time = None
            skill_last_time = [None] * skill_max
            skill_cnt = [0] * skill_max
            
            for j in range(len(item_seqs[i])):
                sk = item_seqs[i][j] - 1
                ti = time_seqs[i][j]

                # Compute repeated time gap feature
                if skill_last_time[sk] is None:
                    repeated_time_gap_seq[i][j][0] = 0
                else:
                    repeated_time_gap_seq[i][j][0] = ti - skill_last_time[sk]
                skill_last_time[sk] = ti

                # Compute sequence time gap feature
                if last_time is None:
                    sequence_time_gap_seq[i][j][0] = 0
                else:
                    sequence_time_gap_seq[i][j][0] = (ti - last_time)
                last_time = ti

                # Compute past trial count feature
                past_trial_counts_seq[i][j][0] = (skill_cnt[sk])
                skill_cnt[sk] += 1
                
        # Perform data preprocessing on time-related features
        repeated_time_gap_seq[repeated_time_gap_seq < 0] = 1
        sequence_time_gap_seq[sequence_time_gap_seq < 0] = 1
        repeated_time_gap_seq[repeated_time_gap_seq == 0] = 1e4
        sequence_time_gap_seq[sequence_time_gap_seq == 0] = 1e4
        past_trial_counts_seq += 1
        sequence_time_gap_seq *= 1.0 / 60
        repeated_time_gap_seq *= 1.0 / 60

        sequence_time_gap_seq = np.log(sequence_time_gap_seq)
        repeated_time_gap_seq = np.log(repeated_time_gap_seq)
        past_trial_counts_seq = np.log(past_trial_counts_seq) 
        
        return sequence_time_gap_seq, repeated_time_gap_seq, past_trial_counts_seq
    
    
    def _init_weights(
        self
    ) -> None:
        """
        Initialize the weights of the model.

        This function creates the necessary layers of the model and initializes their weights.

        Returns:
            None
        """
        self.skill_embeddings = torch.nn.Embedding(
            self.skill_num * 2, 
            self.emb_size
        )
        self.rnn = torch.nn.LSTM(
            input_size=self.emb_size + 3, 
            hidden_size=self.hidden_size, 
            batch_first=True,
            num_layers=self.num_layer
        )
        self.fin = torch.nn.Linear(3, self.emb_size)
        self.fout = torch.nn.Linear(3, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size + 3, self.skill_num)

        self.loss_function = torch.nn.BCELoss()


    def forward_cl(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the forward pass of the model.

        Args:
            feed_dict: A dictionary containing input data.

        Returns:
            A dictionary containing the model's predictions and corresponding labels.
        """
        
        items = feed_dict['skill_seq'][:, :idx+1]   # [bs, time]
        labels = feed_dict['label_seq'][:, :idx+1]  # [bs, time]
        indices = feed_dict['inverse_indice']
            
        repeated_time_gap_seq = feed_dict['repeated_time_gap_seq'][:, :idx+1]  # [bs, time, 1]
        sequence_time_gap_seq = feed_dict['sequence_time_gap_seq'][:, :idx+1]  # [bs, time, 1]
        past_trial_counts_seq = feed_dict['past_trial_counts_seq'][:, :idx+1]  # [bs, time, 1]

        # Compute item embeddings and feature interaction
        embed_history_i = self.skill_embeddings(items + labels * self.skill_num) # [bs, time, emb_size]
        fin = self.fin(torch.cat((repeated_time_gap_seq, sequence_time_gap_seq, past_trial_counts_seq), dim=-1)) # [bs, time, emb_size]
        embed_history_i = torch.cat(
            (embed_history_i.mul(fin), repeated_time_gap_seq, sequence_time_gap_seq, past_trial_counts_seq), dim=-1
        )
        
        # Pack padded sequence and run through RNN layer
        self.rnn.flatten_parameters()
        output, _ = self.rnn(embed_history_i, None) # [bs, time, emb_dim]
        fout = self.fout(torch.cat((repeated_time_gap_seq, sequence_time_gap_seq, past_trial_counts_seq), dim=-1)) # [bs, time, emb_size]
        output = torch.cat((
            output.mul(fout), repeated_time_gap_seq,
            sequence_time_gap_seq, past_trial_counts_seq
        ), dim=-1) # [bs, time, emb_size+3]
        
        pred_vector = self.out(output) # [bs, time, skill_num]
        
        target_item = feed_dict['skill_seq'][:, 1:idx+2] 
        prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1)
        prediction_sorted = torch.sigmoid(prediction_sorted)
        prediction = prediction_sorted[feed_dict['inverse_indice']]

        train_pred = prediction[:, :-1]
        eval_pred = prediction[:, -1:]

        # Extract the labels for the training and evaluation predictions
        train_label = feed_dict['label_seq'][:, 1:idx+1]
        train_label = train_label[indices].double()
        eval_label = feed_dict['label_seq'][:, idx+1:idx+2]
        eval_label = eval_label[indices].double()

        out_dict = {
            'prediction': train_pred, 
            'label': train_label,
            'cl_prediction': eval_pred,
            'cl_label': eval_label,
        }

        return out_dict
    
    
    def forward(
        self, 
        feed_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the forward pass of the model.

        Args:
            feed_dict: A dictionary containing input data.

        Returns:
            A dictionary containing the model's predictions and corresponding labels.
        """
        
        items = feed_dict['skill_seq']  # [bs, max_step]
        labels = feed_dict['label_seq']  # [bs, max_step]
        lengths = feed_dict['length']  # [bs]

        if lengths.is_cuda:
            lengths = lengths.cpu().int()
            
        repeated_time_gap_seq = feed_dict['repeated_time_gap_seq']  # [bs, max_step]
        sequence_time_gap_seq = feed_dict['sequence_time_gap_seq']  # [bs, max_step]
        past_trial_counts_seq = feed_dict['past_trial_counts_seq']  # [bs, max_step]

        # Compute item embeddings and feature interaction
        embed_history_i = self.skill_embeddings(items + labels * self.skill_num) # [bs, max_stpe, emb_size]
        fin = self.fin(torch.cat((repeated_time_gap_seq, sequence_time_gap_seq, past_trial_counts_seq), dim=-1)) # [bs, max_stpe, emb_size]
        embed_history_i = torch.cat(
            (embed_history_i.mul(fin), repeated_time_gap_seq, sequence_time_gap_seq, past_trial_counts_seq), dim=-1
        )
        
        # Pack padded sequence and run through RNN layer
        embed_history_i_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths - 1, batch_first=True)
        output, _ = self.rnn(embed_history_i_packed, None)

        # Unpack padded sequence and apply output feature transformation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        fout = self.fout(torch.cat((repeated_time_gap_seq, sequence_time_gap_seq, past_trial_counts_seq), dim=-1))
        output = torch.cat((
            output.mul(fout[:, 1:, :]), repeated_time_gap_seq[:, 1:, :],
            sequence_time_gap_seq[:, 1:, :], past_trial_counts_seq[:, 1:, :]
        ), dim=-1) # [bs, time-1, emb_size+3]
        
        pred_vector = self.out(output) # [bs, time-1, skill_num]
        
        target_item = items[:, 1:]
        prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1)
        prediction_sorted = torch.sigmoid(prediction_sorted)
        prediction = prediction_sorted[feed_dict['inverse_indice']]
        
        label = labels[:, 1:]
        label = label[feed_dict['inverse_indice']].double()

        out_dict = {
            'prediction': prediction, 
            'label': label
        }

        return out_dict


    def loss(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        out_dict: Dict[str, torch.Tensor],
        metrics: List[str] = None,
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
        indice = feed_dict['indice']
        lengths = feed_dict['length'] - 1
        if lengths.is_cuda:
            lengths = lengths.cpu().int()

        # Compute the loss for the main prediction task
        predictions, labels = out_dict['prediction'][indice], out_dict['label'][indice]
        losses['loss_total'] = self.loss_function(predictions, labels.float())

        # Compute the evaluation metrics for the main prediction task
        if metrics is not None:
            pred = predictions.detach().cpu().data.numpy()
            gt = labels.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
            for key in evaluations.keys():
                losses[key] = evaluations[key]

        if 'cl_prediction' in out_dict.keys():
            cl_predictions, cl_labels = out_dict['cl_prediction'][indice], out_dict['cl_label'][indice]
            cl_loss = self.loss_function(cl_predictions, cl_labels.float())
            losses['cl_loss'] = cl_loss
            pred = cl_predictions.detach().cpu().data.numpy()
            gt = cl_labels.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
            for key in evaluations.keys():
                losses['cl_'+key] = evaluations[key]

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
        user_ids = data['user_id'][batch_start: batch_start + real_batch_size].values

        # *_seqs all have the same shape: [bs, ]; while each element in the bs is a list of times
        item_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'][batch_start: batch_start + real_batch_size].values 
        
        sequence_time_gap_seq, repeated_time_gap_seq, past_trial_counts_seq = \
            self.get_time_features(item_seqs, time_seqs)
        
        lengths = np.array(list(map(lambda lst: len(lst), item_seqs)))
        indice = np.array(np.argsort(lengths, axis=-1)[::-1])
        inverse_indice = np.zeros_like(indice)
        for i, idx in enumerate(indice):
            inverse_indice[idx] = i
            
        # Initialize the feed_dict with the input tensors for the model
        if device is None:
            device = self.device
            
        feed_dict = {
            'user_id': torch.from_numpy(user_ids[indice]).to(device),
            'skill_seq': torch.from_numpy(utils.pad_lst(item_seqs[indice])).to(device),  # [batch_size, max_step]
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs[indice])).to(device),  # [batch_size, max_step]
            'repeated_time_gap_seq': torch.from_numpy(repeated_time_gap_seq[indice]).to(device),  # [batch_size, max_step]
            'sequence_time_gap_seq': torch.from_numpy(sequence_time_gap_seq[indice]).to(device),  # [batch_size, max_step]
            'past_trial_counts_seq': torch.from_numpy(past_trial_counts_seq[indice]).to(device),  # [batch_size, max_step]
            'length': torch.from_numpy(lengths[indice]).to(device),  # [batch_size]
            'inverse_indice': torch.from_numpy(inverse_indice).to(device),
            'indice': torch.from_numpy(indice).to(device),
        }
        return feed_dict


    def predictive_model(
        self, 
        feed_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the forward pass of the model.

        Args:
            feed_dict: A dictionary containing input data.

        Returns:
            A dictionary containing the model's predictions and corresponding labels.
        """
        
        train_step = int(self.args.max_step * self.args.train_time_ratio)
        test_step = int(self.args.max_step * self.args.test_time_ratio)
        
        items = feed_dict['skill_seq'][:, train_step-1:]  # [bs, max_step]
        labels = feed_dict['label_seq'][:, train_step-1:]  # [bs, max_step]
        time_step = items.shape[-1]
            
        repeated_time_gap_seq = feed_dict['repeated_time_gap_seq'][:, train_step-1:]  # [bs, max_step]
        sequence_time_gap_seq = feed_dict['sequence_time_gap_seq'][:, train_step-1:]  # [bs, max_step]
        past_trial_counts_seq = feed_dict['past_trial_counts_seq'][:, train_step-1:]  # [bs, max_step]

        # Compute item embeddings and feature interaction
        predictions = []
        last_emb = self.skill_embeddings(items[:, 0:1] + labels[:, 0:1] * self.skill_num) # [bs, 1, emb_size]
        last_fin = self.fin(torch.cat((repeated_time_gap_seq[:, 0:1], 
                                       sequence_time_gap_seq[:, 0:1], 
                                       past_trial_counts_seq[:, 0:1]), dim=-1)) # [bs, max_stpe, emb_size]
        last_emb = torch.cat(
            (last_emb.mul(last_fin), 
             repeated_time_gap_seq[:, 0:1], 
             sequence_time_gap_seq[:, 0:1], 
             past_trial_counts_seq[:, 0:1]), dim=-1
        )
        
        for i in range(time_step - 1):
            if i == 0: 
                output, latent_states = self.rnn(last_emb, None) # [bs, 1, emb_size]
            else:
                output, latent_states = self.rnn(last_emb, latent_states)

            fout = self.fout(torch.cat((repeated_time_gap_seq[:, i:i+1], 
                                        sequence_time_gap_seq[:, i:i+1], 
                                        past_trial_counts_seq[:, i:i+1]), dim=-1)) # [bs, 1, emb_size]
            
            output = torch.cat((
                output.mul(fout), repeated_time_gap_seq[:, i+1:i+2],
                sequence_time_gap_seq[:, i+1:i+2], past_trial_counts_seq[:, i+1:i+2]
            ), dim=-1) # [bs, time-1, emb_size+3]
        
            pred_vector = self.out(output) # [bs, 1, skill_num]

            target_item = items[:, i+1:i+2]
            prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1) # [bs, 1]
            prediction_sorted = torch.sigmoid(prediction_sorted)
            prediction = prediction_sorted[feed_dict['inverse_indice']]
            predictions.append(prediction)

            # last embedding
            last_emb = self.skill_embeddings(items[:, i+1:i+2] + (prediction>=0.5)*1 * self.skill_num) # [bs, 1, emb_size]
            last_fin = self.fin(torch.cat((repeated_time_gap_seq[:, i+1:i+2], 
                                        sequence_time_gap_seq[:, i+1:i+2], 
                                        past_trial_counts_seq[:, i+1:i+2]), dim=-1)) # [bs, 1, emb_size]
            last_emb = torch.cat(
                (last_emb.mul(last_fin), 
                repeated_time_gap_seq[:, i+1:i+2], 
                sequence_time_gap_seq[:, i+1:i+2], 
                past_trial_counts_seq[:, i+1:i+2]), dim=-1
            )

        label = labels[:, 1:]
        label = label[feed_dict['inverse_indice']].double()
        prediction = torch.cat(predictions, -1)

        out_dict = {
            'prediction': prediction, 
            'label': label
        }

        return out_dict
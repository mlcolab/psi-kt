# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from models.BaseModel import BaseModel
from utils import utils
import ipdb


class DKT(BaseModel):
    extra_log_args = ['hidden_size', 'num_layer']

    @staticmethod
    def parse_model_args(parser, model_name='DKT'):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=32,
                            help='Size of hidden vectors in LSTM.')
        parser.add_argument('--num_layer', type=int, default=1,
                            help='Number of GRU layers.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus, logs):
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.dropout = args.dropout

        self.device = args.device

        self.args = args
        self.logs = logs
        BaseModel.__init__(self, model_path=os.path.join(args.log_path, 'Model_{}.pt'))


    def _init_weights(self):
        self.skill_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size, device=self.device)
        self.rnn = torch.nn.LSTM(
            input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True,
            num_layers=self.num_layer
        )
        self.out = torch.nn.Linear(self.hidden_size, self.skill_num)

        self.loss_function = torch.nn.BCELoss()


    def predictive_model(self, feed_dict):
        return self.forward(feed_dict)
    
    
    def forward_cl(
        self, 
        feed_dict: Dict[str, torch.Tensor],
        idx: int = None,
    ):
        
        items = feed_dict['skill_seq'][:, idx+1]     # [batch_size, history_max]
        labels = feed_dict['label_seq'][:, idx+1]  # [batch_size, history_max]
        indices = feed_dict['inverse_indice'][:, idx+1]
        lengths = idx
        
        ipdb.set_trace()
        if items.is_cuda: 
            lengths = lengths.cpu().int()

        ipdb.set_trace()
        embed_history_i = self.skill_embeddings(items + labels * self.skill_num) # [bs, time, emb_size]
        embed_history_i_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths, batch_first=True) # embed_history_i_packed.data [(time-1)*bs, emb_size]
        output, _ = self.rnn(embed_history_i_packed, None) # [bs, time-1, emb_size]
        output = output[:, -1:]

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # output.data [bs, time-1, emb_size]
        pred_vector = self.out(output) # [bs, time-1, skill_num] 
        target_item = items[:, 1:]
        prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1)
        label = labels[:, 1:]

        prediction_sorted = torch.sigmoid(prediction_sorted)
        
        prediction = prediction_sorted[indices]
        label = label[indices].double()

        out_dict = {'prediction': prediction, 'label': label}
        return out_dict


    def forward(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ):
        
        items = feed_dict['skill_seq']     # [batch_size, history_max]
        labels = feed_dict['label_seq']  # [batch_size, history_max]
        lengths = feed_dict['length']           # [batch_size]
        indices = feed_dict['inverse_indice']
        
        if items.is_cuda: 
            lengths = lengths.cpu().int()

        # History embedding
        embed_history_i = self.skill_embeddings(items + labels * self.skill_num) # [bs, time, emb_size] # TODO this is not fair...
        
        # pack: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # issues with 'lengths must be on cpu': https://github.com/pytorch/pytorch/issues/43227
        embed_history_i_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths - 1, batch_first=True) # embed_history_i_packed.data [(time-1)*bs, emb_size]
        output, _ = self.rnn(embed_history_i_packed, None) # [bs, time-1, emb_size]
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # output.data [bs, time-1, emb_size]
        
        # Prediction
        pred_vector = self.out(output) # [bs, time-1, skill_num] 
        target_item = items[:, 1:]
        prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1)
        prediction_sorted = torch.sigmoid(prediction_sorted)
        prediction = prediction_sorted[indices]
        
        # Label
        label = labels[:, 1:]
        label = label[indices].double()

        out_dict = {'prediction': prediction, 'label': label}
        return out_dict


    def loss(
        self, 
        feed_dict, 
        outdict, 
        metrics=None
    ):
        losses = defaultdict(lambda: torch.zeros((), device=self.device))

        indice = feed_dict['indice']
        lengths = feed_dict['length'] - 1
        if lengths.is_cuda:
            lengths = lengths.cpu().int()

        predictions, labels = outdict['prediction'][indice], outdict['label'][indice]
        predictions = torch.nn.utils.rnn.pack_padded_sequence(predictions, lengths, batch_first=True).data
        labels = torch.nn.utils.rnn.pack_padded_sequence(labels, lengths, batch_first=True).data
        loss = self.loss_function(predictions, labels.float())
        losses['loss_total'] = loss

        if metrics != None:
            pred = predictions.detach().cpu().data.numpy()
            gt = labels.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]

        return losses


    def get_feed_dict(
        self, 
        corpus, 
        data,
        batch_start: int,
        batch_size: int,
        phase: str,
        device: torch.device = None,
    ):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        user_ids = data['user_id'][batch_start: batch_start + real_batch_size].values
        user_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values

        lengths = np.array(list(map(lambda lst: len(lst), user_seqs)))
        indice = np.array(np.argsort(lengths, axis=-1)[::-1])
        inverse_indice = np.zeros_like(indice)
        for i, idx in enumerate(indice):
            inverse_indice[idx] = i

        if device is None:
            device = self.device
            
        feed_dict = {
            'user_id': torch.from_numpy(user_ids[indice]).to(device),
            'skill_seq': torch.from_numpy(utils.pad_lst(user_seqs[indice])).to(device),    # [batch_size, num of items to predict]
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs[indice])).to(device),   # [batch_size, num of items to predict]
            'length': torch.from_numpy(lengths[indice]).to(device),                        # [batch_size]
            'inverse_indice': torch.from_numpy(inverse_indice).to(device),
            'indice': torch.from_numpy(indice).to(device),
        }
        
        return feed_dict

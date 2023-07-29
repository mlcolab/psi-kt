# -*- coding: UTF-8 -*-
import os
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from collections import defaultdict

from baseline.BaseModel import BaseModel
from utils import utils

import ipdb

class HKT(BaseModel):
    extra_log_args = ['time_log']

    @staticmethod
    def parse_model_args(parser, model_name='HKT'):
        parser.add_argument('--emb_size', type=int, default=16,
                            help='Size of embedding vectors.')
        parser.add_argument('--time_log', type=float, default=np.e,
                            help='Log base of time intervals.')
        return BaseModel.parse_model_args(parser, model_name)


    def __init__(
        self, 
        args, 
        corpus, 
        logs
    ):
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
        super().__init__(model_path=os.path.join(args.log_path, 'Model/Model_{}_{}.pt'))


    def _init_weights(
        self
    ):
        self.problem_base = torch.nn.Embedding(self.problem_num, 1)
        self.skill_base = torch.nn.Embedding(self.skill_num, 1)

        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

        self.loss_function = torch.nn.BCELoss()


    def forward(
        self, 
        feed_dict: Dict[str, torch.Tensor],
    ):
        
        items = feed_dict['skill_seq']      # [batch_size, seq_len]
        problems = feed_dict['problem_seq']  # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]

        mask_labels = labels * (labels > -1).long()
        inters = items + mask_labels * self.skill_num # (bs, seq_len)

        # alpha: for each student, how much influence from previous skill and performance on other items
        # although it is for each student, but the skill embedding is universal
        alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
        alpha_target_emb = self.alpha_skill_embeddings(items)  # [bs, seq_len, emb]
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]

        beta_src_emb = self.beta_inter_embeddings(inters)  # [bs, seq_len, emb]
        beta_target_emb = self.beta_skill_embeddings(items)
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        betas = torch.clamp(betas + 1, min=0, max=10)

        delta_t = (times[:, :, None] - times[:, None, :]).abs().double() # [bs, seq_len, seq_len]
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        cross_effects = alphas * torch.exp(-betas * delta_t) # [bs, seq_len, seq_len]
        
        seq_len = items.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0).to(cross_effects.device)
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2) # [bs, seq_len]

        problem_bias = self.problem_base(problems).squeeze(dim=-1)
        skill_bias = self.skill_base(items).squeeze(dim=-1)

        prediction = (problem_bias + skill_bias + sum_t).sigmoid()

        # Return predictions and labels from the second position in the sequence
        out_dict = {
            'prediction': prediction[:, 1:], 
            'label': labels[:, 1:].double()
        }
        
        return out_dict


    def loss(
        self, 
        feed_dict: Dict[str, torch.Tensor], 
        out_dict: Dict[str, torch.Tensor], 
        metrics: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        
        losses = defaultdict(lambda: torch.zeros((), device=self.device))
        
        predictions = out_dict['prediction'].flatten()
        labels = out_dict['label'].flatten()
        mask = labels > -1
        loss = self.loss_function(predictions[mask], labels[mask])
        losses['loss_total'] = loss
        
        # Compute the evaluation metrics for the main prediction task
        if metrics is not None:
            pred = predictions.detach().cpu().data.numpy()
            gt = labels.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
            for key in evaluations.keys():
                losses[key] = evaluations[key]

        if 'cl_prediction' in out_dict.keys():
            cl_predictions, cl_labels = out_dict['cl_prediction'], out_dict['cl_label']
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
        corpus, 
        data, 
        batch_start, 
        batch_size, 
        phase
    ):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        skill_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'][batch_start: batch_start + real_batch_size].values
        problem_seqs = data['problem_seq'][batch_start: batch_start + real_batch_size].values

        feed_dict = {
            'skill_seq': torch.from_numpy(utils.pad_lst(skill_seqs)),            # [batch_size, seq_len] # TODO isn't this -1?
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs, value=-1)),  # [batch_size, seq_len]
            'problem_seq': torch.from_numpy(utils.pad_lst(problem_seqs)),        # [batch_size, seq_len]
            'time_seq': torch.from_numpy(utils.pad_lst(time_seqs)),              # [batch_size, seq_len]
        }
        return feed_dict

    # def actions_after_train(self):
    #     joblib.dump(self.alpha_inter_embeddings.weight.data.cpu().numpy(),
    #                 '../data/{}/alpha_inter_embeddings.npy'.format(self.dataset))
    #     joblib.dump(self.alpha_skill_embeddings.weight.data.cpu().numpy(),
    #                 '../data/{}/alpha_skill_embeddings.npy'.format(self.dataset))
    #     joblib.dump(self.beta_inter_embeddings.weight.data.cpu().numpy(),
    #                 '../data/{}/beta_inter_embeddings.npy'.format(self.dataset))
    #     joblib.dump(self.beta_skill_embeddings.weight.data.cpu().numpy(),
    #                 '../data/{}/beta_skill_embeddings.npy'.format(self.dataset))
    #     joblib.dump(self.problem_base.weight.data.cpu().numpy(),
    #                 '../data/{}/problem_base.npy'.format(self.dataset))
    #     joblib.dump(self.skill_base.weight.data.cpu().numpy(),
    #                 '../data/{}/skill_base.npy'.format(self.dataset))



    def predictive_model(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ):
        train_step = int(self.args.max_step * self.args.train_time_ratio)
        test_step = int(self.args.max_step * self.args.test_time_ratio)
        
        items = feed_dict['skill_seq'][:, train_step-1:]      # [batch_size, seq_len]
        problems = feed_dict['problem_seq'][:, train_step-1:]  # [batch_size, seq_len]
        times = feed_dict['time_seq'][:, train_step-1:]        # [batch_size, seq_len]
        labels = feed_dict['label_seq'][:, train_step-1:]      # [batch_size, seq_len]

        test_time = items.shape[-1]
        predictions = []
        for i in range(0, test_time-1):
            if i == 0:
                inters = items[:, 0:1] + labels[:, 0:1] * self.skill_num
                delta_t = times[:, :1].unsqueeze(dim=-1)
            else:
                pred_labels = torch.cat([labels[:, 0:1], (torch.cat(predictions,-1)>=0.5)*1], dim=-1)
                inters = items[:, :i+1] + pred_labels * self.skill_num 
                
                cur_time = times[:, :i+1]
                delta_t = (cur_time[:, :, None] - cur_time[:, None, :]).abs().double() # [bs, seq_len, seq_len]
                delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

            alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
            alpha_target_emb = self.alpha_skill_embeddings(items[:,i+1:i+2])  # [bs, seq_len, emb]
            alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]

            beta_src_emb = self.beta_inter_embeddings(inters)  # [bs, seq_len, emb]
            beta_target_emb = self.beta_skill_embeddings(items[:, i+1:i+2])
            betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
            betas = torch.clamp(betas + 1, min=0, max=10)

            cross_effects = alphas * torch.exp(-betas * delta_t) # [bs, seq_len, seq_len]
            
            sum_t = cross_effects.sum(-2) # [bs, seq_len]

            problem_bias = self.problem_base(problems[:, i+1:i+2]).squeeze(dim=-1)
            skill_bias = self.skill_base(items[:, i+1:i+2]).squeeze(dim=-1)

            prediction = (problem_bias + skill_bias + sum_t).sigmoid()[:, -1:]
            predictions.append(prediction)

        prediction = torch.cat(predictions, dim=-1)
        # Return predictions and labels from the second position in the sequence
        out_dict = {
            'prediction': prediction, 
            'label': labels[:, 1:].double(),
        }

        return out_dict
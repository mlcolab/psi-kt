import math, random, sys, os
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from tqdm import tqdm
from enum import Enum

import networkx as nx 

import ipdb
import torch
import torch.nn as nn

from baseline.BaseModel import BaseModel, BaseLearnerModel


##########################################################################################
# PPE Model
##########################################################################################


class PPE(BaseLearnerModel):
    def __init__(self, 
                 lr=0.1, 
                 variable_x=0.6,
                 variable_b=0.04,
                 variable_m=0.08,
                 variable_tau=0.9,
                 variable_s=0.04,
                 num_seq=1, 
                 num_node=1,
                 mode='train', 
                 nx_graph=None,
                 device='cpu',
                 logs=None,
                 ):
        '''
        Args:
            lr:
        '''
        super().__init__(mode=mode, device=device, logs=logs)
        self.num_node = num_node
        self.num_seq = num_seq

        if num_node > 1:
            self.adj = torch.tensor(nx_graph, device=self.device) 
            assert(self.adj.shape[-1] == num_node)
        else: 
            self.adj = None

        # Training mode choosing
        class ThetaShape(Enum):
            SIMPLE_SPLIT_TIME = (1, 1, 1)
            SIMPLE_SPLIT_LEARNER = (1, 1, 1)
            LS_SPLIT_TIME = (num_seq, 1, 1)
            NS_SPLIT_TIME = (1, num_node, 1)
            NS_SPLIT_LEARNER = (1, num_node, 1)
            LN_SPLIT_TIME = (num_seq, num_node, 1)
            
        if mode == 'synthetic':
            self.variable_x = torch.tensor(variable_x, device=device).float()
            self.variable_b = torch.tensor(variable_b, device=device).float() 
            self.variable_m = torch.tensor(variable_m, device=device).float()
            self.variable_tau = torch.tensor(variable_tau, device=device).float()
            self.variable_s = torch.tensor(variable_s, device=device).float()
            self.lr = torch.tensor(lr, device=device).float()
        else:
            try:
                shape = ThetaShape[mode.upper()].value
            except KeyError:
                raise ValueError(f"Invalid mode: {mode}")
            self.lr = self._initialize_parameter(shape, device)
            self.variable_x = self._initialize_parameter(shape, device)
            self.variable_b = self._initialize_parameter(shape, device)
            self.variable_m = self._initialize_parameter(shape, device)
            
            tau = torch.ones_like(self.lr, device=device) * 0.9
            s = torch.ones_like(self.lr, device=device) * 0.04
            self.variable_tau = nn.Parameter(tau, requires_grad=False)
            self.variable_s = nn.Parameter(s, requires_grad=False)

        
    def simulate_path(self, x0, t, items=None, stats=None, user_id=None, stats_cal_on_fly=False): 
        '''
        Args:
            x0: shape[num_seq/bs, num_node]; the initial state of the learner model
            t: shape[num_seq/bs, num_time_step]
            items: [num_seq/bs, num_time_step]; 
                ** it cannot be None when mode=synthetic
            stats_cal_on_fly: whether calculate the stats of history based on the prediction 
                ** TODO test. it causes gradient error now
            stats: [num_seq/bs, num_node, num_time_step, 3]; it contains [N_total, N_success, N_failure]
        '''
        assert t.numel() > 0 # check if tensor is not empty
        eps = 1e-6
        num_node = x0.shape[-1]
        num_seq, time_step = t.shape
        
        scale_factor = 1 # 1000
        dt = torch.diff(t).unsqueeze(1) 
        dt = torch.tile(dt, (1, num_node, 1))/60/60/24/scale_factor + eps # [bs, num_node, time-1]
        
        # ----- compute the stats of history -----
        if items == None or num_node == 1:
            items = torch.zeros_like(t, device=self.device, dtype=torch.long)
        all_feature = self._compute_all_features(
            num_seq, num_node, time_step, self.device, stats_cal_on_fly, items, stats
        )
        whole_stats, whole_last_time = self._find_whole_stats(
            all_feature, t, items, num_node
        )

        # ----- adapt to different modes -----
        if 'simple' in self.mode:
            user_id = Ellipsis
        elif 'ls_' in self.mode or 'ln_' in self.mode:
            user_id = user_id

        batch_lr = torch.relu(self.lr[user_id]) + eps
        batch_x = torch.sigmoid(self.variable_x[user_id]) + eps
        batch_b = torch.relu(self.variable_b[user_id]) + eps
        batch_m = torch.relu(self.variable_m[user_id]) + eps
        batch_tau = self.variable_tau[user_id]
        batch_s = self.variable_s[user_id]
        
        # ----- simulate the path -----
        t = torch.tile(t.unsqueeze(1), (1,num_node,1))
        drs = []
        x_pred = [x0.unsqueeze(-1)]
        x_item_pred = [x0[torch.arange(0, num_seq), items[:,0]][:, None]]
        
        for i in range(1, time_step):
            cur_item = items[:, i] # [num_seq, ] 
            
            # for PPE part 
            # - small d (decay)
            cur_repeat = whole_stats[:, :, i, 0]
            cur_history_last_time = whole_last_time[:, :, :i+1] # [bs, num_node, i+1]
            lags = torch.diff(cur_history_last_time) / (60 * 60 * 24 * scale_factor) + eps # [bs, num_node, i]
            lag_mask = (lags > 0)
            dn = ((1 / torch.log(abs(lags + eps) + np.e)) * lag_mask).sum(dim=-1) / (cur_repeat + eps) # [bs, num_node]
            dn = batch_b + batch_m * dn.unsqueeze(-1) # [bs, num_node, 1]

            # - big T
            small_t = (t[..., i:i+1] - whole_last_time[..., :i+1])/60/60/24/scale_factor + eps
            # cur_t.unsqueeze(-1) - cur_item_times # [bs, times]
            # mask1 = (cur_item_times!=0)
            # small_t *= mask1
            
            # small_t = torch.minimum(small_t, torch.tensor(1e2))
            big_t = torch.pow(small_t+eps, batch_x)/(torch.sum(torch.pow(small_t+eps, batch_x), 1, keepdims=True) + eps)
            big_t = torch.sum(big_t * small_t, -1)[..., None] # [bs, num_node]
            
            big_t_mask = (big_t!=0)
            mn = torch.pow((whole_stats[:,:,i:i+1,0]+1), batch_lr) * \
                        torch.pow((big_t + eps), -dn) * big_t_mask
            
            pn = 1/(1 + torch.exp((batch_tau - mn)/(batch_s + eps) + eps) + eps)
            
            # ----- update the stats -----    
            if stats_cal_on_fly or self.mode=='synthetic':
                success = (pn>=0.5)*1
                fail = (pn<0.5)*1

                all_feature[torch.arange(num_seq), cur_item, i:, 0] += 1
                all_feature[torch.arange(num_seq), cur_item, i:, 1] += success
                all_feature[torch.arange(num_seq), cur_item, i:, 2] += 1-success
            
            drs.append(dn)
            x_pred.append(pn)
            x_item_pred.append(pn[torch.arange(num_seq), cur_item])
            
        drs = torch.cat(drs, -1) # # [num_seq, num_node, time_step-1]
        x_pred = torch.cat(x_pred, -1) # [num_seq, num_node, time_step]
        x_item_pred = torch.stack(x_item_pred, -1)
        params = {
            'decay_rate': drs,
            'x_item_pred': x_item_pred
        }
        
        return params

        
    def loss(self, feed_dict, out_dict, metrics=None):
        losses = defaultdict(lambda: torch.zeros((), device=self.device))
        
        pred = out_dict['x_item_pred']
        label = out_dict['label']
        
        loss_fn = torch.nn.BCELoss()
        bceloss = loss_fn(pred, label.float())
        losses['loss_total'] = bceloss
        
        if metrics != None:
            pred = pred.detach().cpu().data.numpy()
            label = label.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, label, metrics)
            losses.update(evaluations)
        
        losses['learning_rate'] = self.lr.clone()[0,0]
        losses['variable_x'] = self.variable_x.clone()[0,0]
        losses['variable_b'] = self.variable_b.clone()[0,0]
        losses['variable_m'] = self.variable_m.clone()[0,0]
        losses['variable_tau'] = self.variable_tau.clone()[0,0]
        losses['variable_s'] = self.variable_s.clone()[0,0]
        losses['variable_b'] = self.variable_b.clone()[0,0]
            
        return losses
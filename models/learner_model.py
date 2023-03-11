import collections, math, random, sys, os
import numpy as np
import pandas as pd
from collections import defaultdict

from tqdm import tqdm

import scipy
import scipy.constants
import scipy.stats
import scipy.optimize
# from . import quadratic_variation

import networkx as nx 

import ipdb
import torch
import torch.nn as nn

from utils import utils
from models.BaseModel import BaseModel

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class BaseLearnerModel(nn.Module):
    def __init__(self, 
                 mode, 
                 device='cpu', 
                 logs=None):
        super(BaseLearnerModel, self).__init__()
        self.mode = mode
        self.device = device
        self.logs = logs
        self.optimizer = None
        
        self.pred_evaluate_method = BaseModel.pred_evaluate_method
        self.batch_to_gpu = BaseModel.batch_to_gpu
        self.model_path = os.path.join(logs.args.log_path, 'Model/Model_{}.pt')
        
    ##### the following functions aim for consistency with KTRunner
    def load_model(self, model_path=None):
        if not os.path.exists(model_path):
            raise Exception('Pre-trained model does not exist')
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        self.logs.write_to_log_file('Load model from ' + model_path)
    def actions_before_train(self):
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logs.write_to_log_file('#params: %d' % total_parameters)
    def customize_parameters(self):
        weight_p, bias_p = [], []
        # find parameters which require gradient
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()): 
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        return optimize_dict
    def prepare_batches(self, corpus, data, batch_size, phase):
        num_example = len(data)
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert(num_example > 0)
        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self.get_feed_dict(corpus, data, batch * batch_size, batch_size, phase))
        else: return batches
    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        skill_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'][batch_start: batch_start + real_batch_size].values
        problem_seqs = data['problem_seq'][batch_start: batch_start + real_batch_size].values
        num_history = data['num_history'][batch_start: batch_start + real_batch_size].values
        num_success = data['num_success'][batch_start: batch_start + real_batch_size].values
        num_failure = data['num_failure'][batch_start: batch_start + real_batch_size].values
        user_id = data['user_id'][batch_start: batch_start + real_batch_size].values
        feed_dict = {
            'skill_seq': torch.from_numpy(utils.pad_lst(skill_seqs)),            # [batch_size, seq_len] # TODO isn't this -1?
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs, value=-1)),  # [batch_size, seq_len]
            'problem_seq': torch.from_numpy(utils.pad_lst(problem_seqs)),        # [batch_size, seq_len]
            'time_seq': torch.from_numpy(utils.pad_lst(time_seqs)),              # [batch_size, seq_len]
            'num_history': torch.from_numpy(utils.pad_lst(num_history)), 
            'num_success': torch.from_numpy(utils.pad_lst(num_success)), 
            'num_failure': torch.from_numpy(utils.pad_lst(num_failure)), 
            'user_id': torch.from_numpy(user_id),
        }
        return feed_dict
    def save_model(self, epoch, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model_path = model_path.format(epoch)
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        self.logs.write_to_log_file('Save model to ' + model_path)
        
        
##########################################################################################
# OU Process
##########################################################################################

class VanillaOU(BaseLearnerModel):
    def __init__(self, 
                 mean_rev_speed=None, 
                 mean_rev_level=None, 
                 vola=None, 
                 num_seq=1, 
                 num_node=1,
                 mode='train',
                 device='cpu', 
                 logs=None):
        '''
        Modified from 
            https://github.com/jwergieluk/ou_noise/tree/master/ou_noise
            https://github.com/felix-clark/ornstein-uhlenbeck/blob/master/ornstein_uhlenbeck.py
            https://github.com/369geofreeman/Ornstein_Uhlenbeck_Model/blob/main/research.ipynb
        Args:
            mean_rev_speed: [bs/1, 1] or scalar
            mean_rev_level: [bs/1, 1] or scalar
            vola: [bs/1, 1] or scalar
            num_seq: when training mode, the num_seq will be automatically the number of batch size; 
                     when synthetic mode, the num_seq will be the number of sequences to generate
            mode: can be 'training' or 'synthetic'
            device:  
        '''
        super().__init__(mode=mode, device=device, logs=logs)
        if mode == 'train_split_learner': # TODO: for now it only considers parameters to optimize
            speed = torch.rand((num_seq, 1), device=device)
            level = torch.rand((num_seq, 1), device=device)
            vola = torch.rand((num_seq, 1), device=device)
            
            self.mean_rev_speed = nn.Parameter(speed, requires_grad=True)
            self.mean_rev_level = nn.Parameter(level, requires_grad=True)
            self.vola = nn.Parameter(vola, requires_grad=True)
        
        elif mode == 'train_split_time':
            speed = torch.rand((num_seq, 1), device=device)
            level = torch.rand((num_seq, 1), device=device)
            vola = torch.rand((num_seq, 1), device=device)
            
            self.mean_rev_speed = nn.Parameter(speed, requires_grad=True)
            self.mean_rev_level = nn.Parameter(level, requires_grad=True)
            self.vola = nn.Parameter(vola, requires_grad=True)

        elif mode == 'synthetic':
            assert mean_rev_speed is not None
            self.mean_rev_speed = mean_rev_speed
            self.mean_rev_level = mean_rev_level
            self.vola = vola
            
        else:
            raise Exception('It is not a compatible mode')
            
        self.num_seq = num_seq
        assert torch.min(self.mean_rev_speed) >= 0
        assert torch.min(self.vola) >= 0

    def variance(self, t, speed=None, vola=None):
        '''
        The variances introduced by the parameter vola, time difference and Wiener process (Gaussian noise)
        Args:
            t: [bs/num_seq, num_node, times-1]; the time difference
            speed: [bs/num_seq, 1]
            vola: [bs/num_seq, 1]
        '''
        speed = speed if speed is not None else self.mean_rev_speed
        vola = vola if vola is not None else self.vola
        speed = speed.unsqueeze(-1)
        vola = vola.unsqueeze(-1)
        
        return vola * vola * (1.0 - torch.exp(- 2.0 * speed * t)) / (2 * speed + 1e-6)

    def std(self, t, speed=None, vola=None):
        '''
        Args:
            t: [num_seq/bs, num_node, times] usually is the time difference of a sequence
        '''
        return torch.sqrt(self.variance(t, speed, vola))

    def mean(self, x0, t, speed=None, level=None):
        '''
        Args:
            x0: 
            t: 
        '''
        speed = speed if speed is not None else self.mean_rev_speed
        level = level if level is not None else self.mean_rev_level

        return x0 * torch.exp(-speed * t) + (1.0 - torch.exp(- speed * t)) * level

    def logll(self, x, t, speed=None, level=None, vola=None):
        """
        Calculates log likelihood of a path
        Args:
            t: [num_seq/bs, time_step]
            x: [num_seq/bs, time_step] it should be the same size as t
        Return:
            log_pdf: [num_seq/bs, 1]
        """
        speed = speed if speed is not None else self.mean_rev_speed
        level = level if level is not None else self.mean_rev_level
        vola = vola if vola is not None else self.vola
        
        dt = torch.diff(t)
        dt = torch.log(dt) # TODO TODO
        mu = self.mean(x, dt, speed, level)
        sigma = self.std(dt, speed, vola)
        var = self.variance(dt, speed, vola)

        dist = torch.distributions.normal.Normal(loc=mu, scale=var)
        log_pdf = dist.log_prob(x).sum(-1)
        # log_scale = torch.log(sigma) / 2
        # log_pdf = -((x - mu) ** 2) / (2 * var) - log_scale - torch.log(torch.sqrt(2 * torch.tensor(math.pi,device=device)))

        return log_pdf

    def simulate_path(self, x0, t, items=None, user_id=None):
        """ 
        Simulates a sample path or forward based on the parameters (speed, level, vola)
        dX = speed*(level-X)dt + vola*dB
        ** the num_node here can be considered as multivariate case of OU process 
            while the correlations between nodes wdo not matter
        Args:
            x0: [num_seq/bs, num_node] the initial states for each node in each sequences
            t: [num_seq/bs, time_step] the time points to sample (or interact);
                It should be the same for all nodes
            items: 
        Return: 
            x_pred: [num_seq/bs, num_node, time_step]
        """
        assert len(t) > 0
        num_seq, time_step = t.shape
        num_node = x0.shape[1]
        eps = 1e-6
        
        if items == None:
            items = torch.zeros_like(t, device=self.device)
        if self.mode == 'train_split_time':
            batch_speed = torch.relu(self.mean_rev_speed[user_id]) + eps # TODO 
            batch_level = self.mean_rev_level[user_id]
            batch_vola = torch.relu(self.vola[user_id]) + eps
        else: 
            batch_speed = None
            batch_level = None
            batch_vola = None
            
        dt = torch.diff(t).unsqueeze(1) 
        dt = torch.tile(dt, (1, num_node, 1))/60/60/24 + eps # [bs, num_node, time-1]
        # dt = torch.log(dt) # TODO to find the right temperature of time difference in different real-world datasets

        scale = self.std(dt, speed=batch_speed, vola=batch_vola) # [bs, num_node, t-1]
        noise = torch.randn(size=scale.shape, device=self.device)
        
        x_last = x0 
        x_pred = []
        x_pred.append(x_last)
        x_item_pred = []
        x_item_pred.append(x0[torch.arange(0,num_seq), items[:,0]])
        # ipdb.set_trace()
        for i in range(1, time_step):
            cur_item = items[:, i]
            
            x_next = self.mean(x_last, dt[..., i-1], speed=batch_speed, level=batch_level)  # [bs, num_node]
            x_next = x_next + noise[..., i-1] * scale[..., i-1]
            x_pred.append(x_next)
            
            x_pred_item = x_next[torch.arange(0,num_seq), cur_item] # [bs, ]
            x_item_pred.append(x_pred_item)
            
            x_last = x_next
        x_pred = torch.stack(x_pred, -1)
        x_item_pred = torch.stack(x_item_pred, -1)
        # ipdb.set_trace()
        params = {
            'x_original_item_pred': x_item_pred,        # [bs, times]
            'x_original_all_pred': x_pred,              # [bs, num_node, times]
            'x_item_pred': torch.sigmoid(x_item_pred),
            'x_all_pred': torch.sigmoid(x_pred),
        }
        
        return params
            
    def forward(self, feed_dict):
        # skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        # problems = feed_dict['problem_seq']  # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]

        bs, _ = labels.shape
        self.num_seq = bs
        
        x0 = labels[:, :1]
        outdict = self.simulate_path(x0=x0, t=times, user_id=feed_dict['user_id'])
        
        outdict['prediction'] = outdict['x_item_pred']
        outdict['label'] = labels
        
        # for p in model.parameters():
        #     p.data.clamp_(0)
        return outdict
        
    def loss(self, feed_dict, outdict, metrics=None):
        losses = defaultdict(lambda: torch.zeros((), device=self.device))
        loss_fn = torch.nn.BCELoss()
        pred = outdict['prediction']
        x_gt = outdict['label']
        # ipdb.set_trace()
        bceloss = loss_fn(pred, x_gt.float())
        losses['loss_total'] = bceloss
        # ipdb.set_trace()
        if metrics != None:
            pred = pred.detach().cpu().data.numpy()
            gt = x_gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
            
        # TODO better visualization
        losses['mean_rev_speed'] = self.mean_rev_speed[0].clone()
        losses['mean_rev_level'] = self.mean_rev_level[0].clone()
        losses['vola'] = self.vola[0].clone()
        
        return losses
    
    
##########################################################################################
# HLR Model
##########################################################################################


class HLR(BaseLearnerModel):
    def __init__(self, 
                 theta=None, 
                 base=2., 
                 num_seq=1, 
                 num_node=1, 
                 mode='train', 
                 device='cpu',
                 logs=None):
        '''
        TODO:
            multiple nodes have bugs 
        Modified from:
            https://github.com/duolingo/halflife-regression/blob/0041df0dcd436bf1b4aa7a17a020d9c670db70d8/experiment.py
        Args:
            theta: [bs/num_seq, 3]; should be 3D vector indicates the parameters of the model; 
                the näive version is to compute the dot product of theta and [N_total, N_success, N_failure]
            base: the base of HLR model
            num_seq: when mode==synthetic, it is the number of sequences to generate;
                is mode==train, it is the number of batch size
            items: [bs/num_seq, time_step]
            mode: [synthetic, train]; synthetic is to generate new sequences based on given theta; train is to 
                train the parameters theta given observed data.
            device: cpu or cuda to put all variables and train the model
        '''
        super().__init__(mode=mode, device=device, logs=logs)
        if mode == 'train_split_learner':
            theta = torch.empty(num_seq, 3, device=device)
            theta = torch.nn.init.xavier_uniform_(theta)
            self.theta = nn.Parameter(theta, requires_grad=True)
        elif mode == 'train_split_time':
            theta = torch.empty(num_seq, 3, device=device)
            theta = torch.nn.init.xavier_uniform_(theta)
            self.theta = nn.Parameter(theta, requires_grad=True)
        elif mode == 'synthetic':
            self.theta = torch.tensor(theta, device=device).float()
        else:
            raise Exception('It is not a compatible mode')
        
        self.num_seq = num_seq
        self.base = base
        self.num_node = num_node
        
    @staticmethod
    def hclip(h):
        '''
        bound min/max half-life
        '''
        MIN_HALF_LIFE = torch.tensor(15.0 / (24 * 60), device=h.device)    # 15 minutes
        MAX_HALF_LIFE = torch.tensor(274., device=h.device)                # 9 months
        return torch.min(torch.max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)
    @staticmethod
    def pclip(p):
        '''
        bound min/max model predictions (helps with loss optimization)
        '''
        MIN_P = torch.tensor(0.0001, device=p.device)
        MAX_P = torch.tensor(0.9999, device=p.device)
        return torch.min(torch.max(p, MIN_P), MAX_P)
    
    def simulate_path(self, x0, t, items=None, stats_cal_on_fly=False, stats=None, user_id=None):
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
        torch.autograd.set_detect_anomaly(True) 
        num_node = x0.shape[-1]
        num_seq, time_step = t.shape

        dt = torch.diff(t).unsqueeze(1) 
        dt = torch.tile(dt, (1, num_node, 1))/60/60/24 # [bs, num_node, time-1]
        # dt = torch.log(dt) # [bs, num_node, t-1] TODO
        
        if items == None:
            items = torch.zeros_like(t, device=self.device)
        if stats_cal_on_fly or self.mode=='synthetic':
            item_start = items[:, 0]
            all_feature = torch.zeros((num_seq, num_node, 3), device=self.device)
            all_feature[torch.arange(0, num_seq), item_start, 0] += 1
            all_feature[torch.arange(0, num_seq), item_start, 2] += 1
            all_feature = all_feature.unsqueeze(-2).tile((1,1,time_step,1))
        else: 
            all_feature = stats.float() # [num_seq/bs, num_node, num_time_step, 3]
        if self.mode == 'train_split_time':
            batch_theta = self.theta[user_id]
            
        x_pred = []
        x_pred.append(x0)
        x_item_pred = []
        
        x_item_pred.append(x0[torch.arange(0, num_seq), items[:,0]])
        half_lifes = []
        half_lifes.append(torch.zeros_like(x0, device=self.device))
        
        for i in range(1, time_step):
            cur_item = items[:, i] # [num_seq, ] 
            cur_dt = dt[..., i-1] # [bs, num_node]
            cur_feat = all_feature[:,:,i] # [bs, num_node, 3] # TODO it is changed Mar 10th

            feat = torch.einsum('bij,bj->bi', cur_feat, batch_theta)
            half_life = self.hclip(self.base ** (feat))
            p_all = torch.sigmoid(self.base ** (-cur_dt/half_life)) # [bs, num_node]
            p_item = p_all[torch.arange(0,num_seq), cur_item] # [bs, ]
            
            if stats_cal_on_fly or self.mode=='synthetic':
                success = nn.functional.gumbel_softmax(torch.log(p_item), hard=True) # TODO
                success = success.unsqueeze(-1)
                all_feature[torch.arange(num_seq), cur_item, i:, 0] += 1
                all_feature[torch.arange(num_seq), cur_item, i:, 1] += success
                all_feature[torch.arange(num_seq), cur_item, i:, 2] += 1-success

            half_lifes.append(half_life)
            x_item_pred.append(p_item)
            x_pred.append(p_all)
            
        half_lifes = torch.stack(half_lifes, -1)
        x_pred = torch.stack(x_pred, -1)
        x_item_pred = torch.stack(x_item_pred, -1)

        params = {
            # NOTE: the first element of the following values in outdict is not predicted
            'half_life': half_lifes,               # [bs, num_node, times]
            'x_item_pred': x_item_pred,            # [bs, times]
            'x_all_pred': x_pred,                  # [bs, num_node, times]
            'num_history': all_feature[..., 0:1],  # [bs, num_node, times, 1]
            'num_success': all_feature[..., 1:2],
            'num_failure': all_feature[..., 2:3],
        }
        
        return params
        
    def iterate_update(self, inputs, n_iter, learning_rate=1e-3): 
        t_data, x_data, stats = inputs
        x0 = x_data[:, :1]
        x_gt = x_data[:, 1:]
        
        LN2 = torch.tensor(math.log(2.), device=t_data.device)
        l2wt = torch.tensor(0.1, device=t_data.device)
        sigma = torch.tensor(1.0, device=t_data.device)
        
        loss_fn = torch.nn.BCELoss()
        
        for _ in range(n_iter):
            x_pred, params = self.simulate_path(x0=x0, t=t_data, stats=stats)
            p = params['x_item_pred']
            h = params['half_life']

            # dlp_dw = 2.*(p-x_gt)*(LN2**2)*p*(torch.log(torch.diff(t_data))/h)
            dlp_dw = 2.*(p-x_gt)*(LN2**2)*p*((torch.diff(t_data))/60/60/24/h)
            
            bceloss = loss_fn(p, x_gt.float())
            
            fcounts = defaultdict(int)                  
            for k in range(3):
                x_k = stats[..., 1:, k]                                                           
                rate = (1./(1+x_gt)) * learning_rate / torch.sqrt(torch.tensor(1 + fcounts[k], device=self.device))
                
                self.theta[k] -= (rate * dlp_dw * x_k).sum()/rate.shape[-1]

                # L2 regularization update
                self.theta[k] -= (rate * l2wt * self.theta[k]).sum() / sigma**2 /rate.shape[-1]

                # increment feature count for learning rate
                fcounts[k] += 1
            # self.theta = torch.nn.functional.normalize(self.theta, p=1.0, dim = 0)
            # print(self.theta)
            
    def forward(self, feed_dict):
        # skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        # problems = feed_dict['problem_seq']  # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]

        bs, _ = labels.shape
        self.num_seq = bs
        
        stats = torch.stack([feed_dict['num_history'], feed_dict['num_success'], feed_dict['num_failure']], dim=-1)
        stats = stats.unsqueeze(1)
        
        x0 = labels[:, :1]
        outdict = self.simulate_path(x0=x0, t=times, stats=stats, user_id=feed_dict['user_id'])
        
        outdict['prediction'] = outdict['x_item_pred']
        outdict['label'] = labels
        
        return outdict
        
    def loss(self, feed_dict, outdict, metrics=None):
        losses = defaultdict(lambda: torch.zeros((), device=self.device))
        loss_fn = torch.nn.BCELoss()
        
        p = outdict['x_item_pred']
        x_gt = outdict['label']

        bceloss = loss_fn(p, x_gt.float())
        losses['loss_total'] = bceloss
        
        if metrics != None:
            pred = outdict['x_item_pred'].detach().cpu().data.numpy()
            gt = x_gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
            
        losses['theta_0'] = self.theta.clone()[0, 0]
        losses['theta_1'] = self.theta.clone()[0, 1]
        losses['theta_2'] = self.theta.clone()[0, 2]
        
        return losses


##########################################################################################
# PPE Model
##########################################################################################


class PPE(BaseLearnerModel):
    def __init__(self, 
                 lr, 
                 variable_x=0.6,
                 variable_b=0.04,
                 variable_m=0.08,
                 variable_tau=0.9,
                 variable_s=0.04,
                 num_seq=1, 
                 num_node=1,
                 mode='train',
                 device='cpu',
                 logs=None,
                 ):
        '''
        Args:
            lr:
        '''
        super().__init__(mode=mode, device=device, logs=logs)
        if mode == 'train_split_learner':
            speed = torch.rand((num_seq, 1), device=device)
            level = torch.rand((num_seq, 1), device=device)
            vola = torch.rand((num_seq, 1), device=device)
            
            self.mean_rev_speed = nn.Parameter(speed, requires_grad=True)
            self.mean_rev_level = nn.Parameter(level, requires_grad=True)
            self.vola = nn.Parameter(vola, requires_grad=True)
        elif mode == 'train_split_time':
            lr = torch.rand((num_seq, 1), device=device)
            x = torch.rand((num_seq, 1), device=device)
            b = torch.rand((num_seq, 1), device=device)
            m = torch.rand((num_seq, 1), device=device)

            self.lr = nn.Parameter(lr, requires_grad=True)
            self.variable_x = nn.Parameter(x, requires_grad=True)
            self.variable_b = nn.Parameter(b, requires_grad=True)
            self.variable_m = nn.Parameter(m, requires_grad=True)
            
            tau = torch.ones((num_seq, 1), device=device) * 0.9
            s = torch.ones((num_seq, 1), device=device) * 0.04
            self.variable_tau = nn.Parameter(tau, requires_grad=False)
            self.variable_s = nn.Parameter(s, requires_grad=False)
        elif mode == 'synthetic':
            self.variable_x = variable_x
            self.variable_b = variable_b 
            self.variable_m = variable_m
            self.variable_tau = variable_tau
            self.variable_s = variable_s
            self.lr = lr
        
        self.num_seq = num_seq
        self.num_node = num_node
        

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
        
        eps = 1e-6
        num_node = x0.shape[-1]
        num_seq, time_step = t.shape
        
        scale_factor = 1000
        t = t/60/60/24 + eps # TODO
        t = t/scale_factor + eps
        # dt = torch.diff(t).reshape(num_seq, -1)/60/60/24 + eps
        
        if items == None:
            items = torch.zeros_like(t, device=self.device)
        items = items.long()
        item_start = items[:, 0]
        
        if stats_cal_on_fly or self.mode=='synthetic':
            all_feature = torch.zeros((num_seq, num_node, 3), device=self.device)
            all_feature[torch.arange(0, num_seq), item_start, 0] += 1
            all_feature[torch.arange(0, num_seq), item_start, 2] += 1
            all_feature = all_feature.unsqueeze(-2).tile((1,1,time_step,1))
        else: 
            all_feature = stats.float() # [num_seq/bs, num_node, num_time_step, 3]

        x_pred = []
        x_pred.append(x0)

        # ipdb.set_trace()
        max_items_repeat = [torch.max(torch.bincount(items[i])) for i in range(num_seq)]
        max_item_repeat = max(max_items_repeat)
        actuall_max_item_repeat = torch.max(all_feature)
        drift_repeat = int(actuall_max_item_repeat - max_item_repeat + 1)

        items_time_seq = torch.zeros((self.num_seq, num_node, max_item_repeat), device=self.device)
        items_time_seq[torch.arange(0, num_seq), item_start, 0] += t[:, 0]

        # all_features = []
        drs = []
        pns = []
        pns.append(x0)
        # ipdb.set_trace()
        for i in range(1, time_step):
            # print(i)
            # if i == 80: ipdb.set_trace()
            cur_item = items[:, i] # [num_seq, ] 
            cur_item_repeat = all_feature[torch.arange(num_seq), cur_item, i, 0].long()
            
            # put current time t in items_time_seq
            # TODO not sure about multiple nodes
            cur_t = t[:, i]
            
            items_time_seq[torch.arange(num_seq), cur_item, cur_item_repeat-drift_repeat] += cur_t # [bs, num_node, times]

            # for PPE part 
            # - small d (decay)
            cur_item_times = items_time_seq[torch.arange(0, num_seq), cur_item] # [bs, times]
            lags = torch.diff(cur_item_times)
            lag_mask = (lags>0) # TODO is log here ? yes - in paper
            dn = torch.sum(1/torch.log(abs(lags + eps) + np.e) * lag_mask, -1) * (1/(cur_item_repeat+eps)) # [bs, ]
            dn = self.variable_b[user_id] + self.variable_m[user_id] * dn[:, None] # [bs, 1]
            
            # - big T
            small_t = cur_t.unsqueeze(-1) - cur_item_times # [bs, times]
            mask1 = (cur_item_times!=0)
            small_t *= mask1
            
            small_t = torch.minimum(small_t, (1e+3)*torch.ones_like(small_t))
            big_t = torch.pow(small_t+eps, self.variable_x[user_id])/(torch.sum(torch.pow(small_t+eps, self.variable_x[user_id]), 1, keepdims=True) + eps)
            big_t = torch.sum(big_t * small_t, 1) # [bs,]

            # ipdb.set_trace()
            test2 = torch.pow((big_t + eps)[:, None], -dn)
            test = torch.pow(small_t+eps, self.variable_x[user_id])
            test1 = torch.pow((cur_item_repeat+1)[:, None], self.lr[user_id])
            
            # ipdb.set_trace()
            if test.isinf().sum() + test1.isinf().sum() + test2.isinf().sum() > 0:
                ipdb.set_trace()
            
            big_t_mask = (big_t!=0)
            mn = torch.pow((cur_item_repeat+1)[:, None], self.lr[user_id]) * \
                        torch.pow((big_t + eps)[:, None], -dn) * big_t_mask[:, None]
            
            pn = 1/(1 + torch.exp((self.variable_tau[user_id] - mn)/(self.variable_s[user_id] + eps) + eps) + eps)
            
            test = torch.exp((self.variable_tau[user_id] - mn)/(self.variable_s[user_id] + eps))
            if test.isinf().sum() > 0: 
                ipdb.set_trace()
            # success = (pn>=0.5)*1
            # fail = (pn<0.5)*1

            # all_feature[np.arange(num_seq), cur_item, 0] += 1
            # all_feature[np.arange(num_seq), cur_item, 1] += success
            # all_feature[np.arange(num_seq), cur_item, 2] += fail
            
            # tmp_feature = np.copy(all_feature)
            # all_features.append(tmp_feature)
            drs.append(dn)
            pns.append(pn)
            
            
        # ipdb.set_trace()
        # all_features = np.repeat(np.stack(all_features, 1).astype(int), num_node, -1) # [num_seq, time_step-1, num_node, 3]
        drs = torch.tile(torch.stack(drs, 1).reshape(num_seq, -1, 1), (1,1,num_node))
        pns = torch.tile(torch.stack(pns, 1).reshape(num_seq, -1, 1), (1,1,num_node))
        params = {
            'decay_rate': drs,
            'x_item_pred': pns[..., 0],
            'num_history': stats[..., 0:1],
            'num_success': stats[..., 1:2],
            'num_failure': stats[..., 2:3],
        }
        
        return params


    def forward(self, feed_dict):
        # skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        # problems = feed_dict['problem_seq']  # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]

        bs, _ = labels.shape
        self.num_seq = bs
        
        stats = torch.stack([feed_dict['num_history'], feed_dict['num_success'], feed_dict['num_failure']], dim=-1)
        stats = stats.unsqueeze(1)
        
        x0 = labels[:, :1]
        outdict = self.simulate_path(x0=x0, t=times, stats=stats, user_id=feed_dict['user_id'])
        # ipdb.set_trace()
        outdict['prediction'] = outdict['x_item_pred']
        outdict['label'] = labels
        
        return outdict
        
        
    def loss(self, feed_dict, outdict, metrics=None):
        losses = defaultdict(lambda: torch.zeros((), device=self.device))
        loss_fn = torch.nn.BCELoss()
        
        p = outdict['x_item_pred']
        x_gt = outdict['label']

        bceloss = loss_fn(p, x_gt.float())
        losses['loss_total'] = bceloss
        
        if metrics != None:
            pred = outdict['x_item_pred'].detach().cpu().data.numpy()
            gt = x_gt.detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
            
        losses['lr'] = self.lr.clone()[0]
        losses['variable_x'] = self.variable_x.clone()[0]
        losses['variable_b'] = self.variable_b.clone()[0]
        losses['variable_m'] = self.variable_m.clone()[0]
        losses['variable_tau'] = self.variable_tau.clone()[0]
        losses['variable_s'] = self.variable_s.clone()[0]
        losses['variable_b'] = self.variable_b.clone()[0]
            
        return losses
    
    
    
    
    
    


class GraphOU(VanillaOU):
    def __init__(self, mean_rev_speed, mean_rev_level, vola, num_seq, nx_graph, gamma=None, rho=None, omega=None):
        self.graph = nx_graph
        self.gamma = gamma
        self.rho = rho
        self.omega = omega
        super().__init__(mean_rev_speed, mean_rev_level, vola, num_seq)
    
    def simulate_path(self, x0, t, items=None):
        eps = 1e-6
        assert len(t) > 1
        num_node = len(x0)
        num_seq, time_step = t.shape
        
        dt = np.diff(t).reshape(self.num_seq, -1, 1, 1) 
        dt = np.repeat(dt, num_node, -2)

        x = np.zeros((self.num_seq, time_step, num_node, 1))
        x[:, 0] += x0
        
        noise = scipy.stats.norm.rvs(size=x.shape) 
        scale = self.std(dt)
        x[:, 1:] += noise[:, 1:] * scale

        adj = nx.adjacency_matrix(self.graph).toarray()
        adj_t = np.transpose(adj, (-1,-2))
        in_degree = adj_t.sum(axis=1).reshape(1,-1,1)
    
        item_start = items[:, 0]
        all_feature = np.zeros((self.num_seq, num_node, 3))
        all_feature[np.arange(num_seq), item_start, 0] += 1
        all_feature[np.arange(num_seq), item_start, 2] += 1
        
        # find degree 0
        ind = np.where(in_degree[0,:,0] == 0)[0]

        # TODO for debugging visualization
        empowers = []
        stables = []
        tmp_mean_levels = []
        all_features = []
        
        for i in range(1, time_step):
            empower = (1/(in_degree+1e-7)) * adj_t@x[:, i-1] * self.gamma # [num_seq, num_node, 1]
            empower[:, ind] = 0.0
            stable = np.power((all_feature[..., 1:2]/(all_feature[..., 0:1]+eps)), self.rho)
            tmp_mean_level = self.omega * empower + (1-self.omega) * stable
            perf = self.mean(x[:, i-1], dt[:, i-1], mean_level=tmp_mean_level)
            # x[:, i] += sigmoid(perf)
            x[:, i] += perf # TODO
            
            cur_item = items[:, i]
            cur_perf = sigmoid(x[np.arange(num_seq), i, cur_item])
            # cur_perf = x[np.arange(num_seq), i, cur_item]
            cur_success = (cur_perf>=0.5) * 1
            # ipdb.set_trace()
            all_feature[np.arange(num_seq), cur_item, 0:1] += 1
            all_feature[np.arange(num_seq), cur_item, 1:2] += cur_success
            tmp_feature = np.copy(all_feature)
            
            # DEBUG
            empowers.append(empower)
            stables.append(stable)
            tmp_mean_levels.append(tmp_mean_level)
            all_features.append(tmp_feature)
        
        # DEBUG
        # ipdb.set_trace()
        empowers = np.stack(empowers, 1) # [num_seq, time_step-1, num_node, 1]
        stables = np.stack(stables, 1)
        tmp_mean_levels = np.stack(tmp_mean_levels, 1)
        all_features = np.stack(all_features, 1).astype(int) # [num_seq, time_step-1, num_node, 3]
        params = {
            'empowers': empowers,
            'stables': stables,
            'tmp_mean_levels': tmp_mean_levels,
            # 'num_history': all_features[..., 0:1],
            # 'num_success': all_features[..., 1:2],
            # 'num_failure': all_features[..., 2:3],
        }
        
        return x, params


class ExtendGraphOU(VanillaOU):
    '''
    Args:
        mean_rev_speed: 
        mean_rev_level:
        vola:
    '''
    def __init__(self, mean_rev_speed, mean_rev_level, vola, num_seq, nx_graph=None):
        self.graph = nx_graph
        self.num_seq = num_seq
        super().__init__(mean_rev_speed, mean_rev_level, vola, num_seq)
    
    def find_most_recent_interaction(self, t, x, skill, num_node):
        num_seq, time_lag = t.shape
        
        x_last = torch.zeros((num_seq, num_node), device=x.device, dtype=x.dtype)
        t_last = torch.zeros((num_seq, num_node), device=t.device, dtype=t.dtype)
        num_last = torch.zeros((num_seq, num_node), device=t.device, dtype=t.dtype)
        success_last = torch.zeros((num_seq, num_node), device=t.device, dtype=t.dtype)
        for i in range(time_lag):
            cur_skill = skill[:, i]
            x_last[torch.arange(num_seq), cur_skill] = x[:, i]
            t_last[torch.arange(num_seq), cur_skill] = t[:, i]
            num_last[torch.arange(num_seq), cur_skill] += 1
            success_last[torch.arange(num_seq), cur_skill] += x[:, i]
            
        return x_last, t_last, num_last, success_last
    
    def variance(self, t):
        speed = self.mean_rev_speed.unsqueeze(-1)
        vola = self.vola.unsqueeze(-1)
        eps = 1e-6
        return vola * vola * (1.0 - torch.exp(- 2.0 * speed * t)) / (2 * speed + eps) # TODO log(dt) may be <0

    def std(self, t):
        return torch.sqrt(self.variance(t))

    def mean(self, x0, t, mean_speed=None, mean_level=None):
        speed = mean_speed if mean_speed is not None else self.mean_rev_speed
        level = mean_level if mean_level is not None else self.mean_rev_level
        
        return x0 * torch.exp(-speed * t) + (1.0 - torch.exp(- speed * t)) * level
    
    def simulate_path(self, feed_dict, time_lag, speed=None, gamma=None, vola=None, rho=None, adj=None):
        '''
        Args:
            speed: [num_seq/bs, 1]
        '''
        eps = 1e-6
        omega = 0.5
        rho = 50
        t_future = feed_dict['time_seq'][:, time_lag:]
        num_seq, time_step = t_future.shape # TODO??? num_seq # num_seq = self.num_seq # 
        assert time_step > 1
        device = t_future.device
        
        self.vola = vola if vola is not None else self.vola
        self.mean_rev_speed = speed if speed is not None else self.mean_rev_speed
        self.mean_rev_level = gamma if gamma is not None else self.mean_rev_level
        self.rho = torch.tensor(rho, device=device)
    
        if adj==None:
            adj = torch.from_numpy(nx.adjacency_matrix(self.graph).toarray()).to(self.device) # TODO test with dimension
        else: adj = adj
        num_node = adj.shape[-1]
        adj_t = torch.transpose(adj, -1, -2) # TODO test with multiple power of adj
        in_degree = adj_t.sum(dim=-1)
        # find degree 0
        ind = torch.where(in_degree[0,0,0] == 0)[0]
        
        # -- history statistics
        x_history = feed_dict['label_seq'][:, :time_lag]
        skill_history = feed_dict['skill_seq'][:, :time_lag]
        t_history = feed_dict['time_seq'][:, :time_lag]
        x_last, t_last, num_last, success_last = self.find_most_recent_interaction(t_history, x_history, skill_history, num_node)
        perf_last = x_last.float()
        # ipdb.set_trace()
        
        # -- calculate time difference
        t_future = torch.tile(t_future.unsqueeze(-1), (1,1,num_node))
        tmp = torch.cat([t_last.unsqueeze(1), t_future], dim=1)
        dt = torch.log(torch.diff(tmp, dim=1)+eps) # [num_seq/bs, pred_step, num_nodes] # TODO log time
        
        
        # ----- OU process -----
        perf_future = torch.zeros((num_seq, time_step, num_node), device=device)
        
        # -- calculate the noise/vola
        noise = torch.randn(size=perf_future.shape, device=device) 
        scale = self.std(dt)
        perf_future += noise * scale # [num_seq/bs, pred_steps, num_nodes]
        
        skill_future = feed_dict['skill_seq'][:, time_lag:]
        pred = []
        for i in range(time_step):
            
            empower = torch.einsum('abcin, an->abci', adj_t.double(), perf_last.double())
            empower = (1/(in_degree[:,0,0]+1e-7)) * gamma * empower[:,0,0]
            empower[:,ind] = 0
            stable = torch.pow((success_last/(num_last+eps)), self.rho)
            tmp_mean_level = omega * empower + (1-omega) * stable
            perf = self.mean(perf_last, dt[:, i], mean_level=tmp_mean_level) # [num_seq/bs, num_node]
            perf =  torch.sigmoid(perf)
            
            cur_skill = skill_future[:, i]
            cur_perf =perf[torch.arange(num_seq), cur_skill] # TODO sigmoid?
            cur_x = (cur_perf>=0.5) * 1
            ipdb.set_trace()
            num_last[torch.arange(num_seq), cur_skill] += 1
            success_last[torch.arange(num_seq), cur_skill] += cur_x
            
            perf_future[:, i] += perf
            pred.append(cur_perf)
            perf_last = perf
        
        # ipdb.set_trace()
        
        return pred
        
        
        
        
        
        



















# class HLR(object):
#     def __init__(self, theta, base=2, num_seq=1):
#         self.theta = theta
#         self.num_seq = num_seq
#         self.base = base

#     def simulate_path(self, x0, t, items=None): # TODO x0
#         num_node = len(x0)
#         num_seq, time_step = t.shape
#         dt = np.diff(t).reshape(num_seq, -1)

#         x = np.zeros((self.num_seq, time_step, num_node, 1))
#         x[:, 0] += x0

#         item_start = items[:, 0]
#         all_feature = np.zeros((self.num_seq, num_node, 3))
#         all_feature[np.arange(num_seq), item_start, 0] += 1
#         all_feature[np.arange(num_seq), item_start, 2] += 1

#         # DEBUG
#         all_features = []
#         half_lifes = []
#         for i in range(1, time_step):
#             cur_item = items[:, i] # [num_seq, ] 
#             cur_dt = dt[:, i-1:i]

#             half_life = hclip(self.base ** (all_feature @ self.theta))
#             # half_life = self.base ** (all_feature @ self.theta)
#             p_all = pclip(2. ** (-np.log(cur_dt)/half_life)) # TODO how to give the dt the right temperature
#             p_item = p_all[np.arange(num_seq), cur_item]

#             success = (p_item>=0.5)*1
#             fail = (p_item<0.5)*1

#             all_feature[np.arange(num_seq), cur_item, 0] += 1
#             all_feature[np.arange(num_seq), cur_item, 1] += success
#             all_feature[np.arange(num_seq), cur_item, 2] += fail
#             # ipdb.set_trace()
            
#             tmp_feature = np.copy(all_feature)
#             all_features.append(tmp_feature)
#             half_lifes.append(half_life)
            
#             x[:, i, :, 0] += p_all
#             # ipdb.set_trace()

#         all_features = np.stack(all_features, 1).astype(int) # [num_seq, time_step-1, num_node, 3]
#         half_lifes = np.stack(half_lifes, 1)
#         params = {
#             'half_life': half_lifes,
#             'num_history': all_features[..., 0:1],
#             'num_success': all_features[..., 1:2],
#             'num_failure': all_features[..., 2:3],
#         }
        
        
#         return x, params
            





class SpacedRepetitionModel(object):
    """
    Spaced repetition model. Implements the following approaches:
      - 'hlr' (half-life regression; trainable)
      - 'lr' (logistic regression; trainable)
      - 'leitner' (fixed)
      - 'pimsleur' (fixed)
    """
    def __init__(self, method='hlr', omit_h_term=False, initial_weights=None, lrate=.001, hlwt=.01, l2wt=.1, sigma=1.):
        self.method = method
        self.omit_h_term = omit_h_term
        self.weights = defaultdict(float)
        if initial_weights is not None:
            self.weights.update(initial_weights)
        self.fcounts = defaultdict(int)
        self.lrate = lrate
        self.hlwt = hlwt
        self.l2wt = l2wt
        self.sigma = sigma

    def halflife(self, inst, base):
        try:
            dp = sum([self.weights[k]*x_k for (k, x_k) in inst.fv])
            return hclip(base ** dp)
        except:
            return MAX_HALF_LIFE

    def predict(self, inst, base=2.):
        if self.method == 'hlr':
            h = self.halflife(inst, base)
            p = 2. ** (-inst.t/h)
            return pclip(p), h
        elif self.method == 'leitner':
            try:
                h = hclip(2. ** inst.fv[0][1])
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2. ** (-inst.t/h)
            return pclip(p), h
        elif self.method == 'pimsleur':
            try:
                h = hclip(2. ** (2.35*inst.fv[0][1] - 16.46))
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2. ** (-inst.t/h)
            return pclip(p), h
        elif self.method == 'lr':
            dp = sum([self.weights[k]*x_k for (k, x_k) in inst.fv])
            p = 1./(1+math.exp(-dp))
            return pclip(p), random.random()
        else:
            raise Exception

    def train_update(self, inst):
        if self.method == 'hlr':
            base = 2.
            p, h = self.predict(inst, base)
            dlp_dw = 2.*(p-inst.p)*(LN2**2)*p*(inst.t/h)
            dlh_dw = 2.*(h-inst.h)*LN2*h
            for (k, x_k) in inst.fv:
                rate = (1./(1+inst.p)) * self.lrate / math.sqrt(1 + self.fcounts[k])
                # rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # sl(p) update
                self.weights[k] -= rate * dlp_dw * x_k
                # sl(h) update
                if not self.omit_h_term:
                    self.weights[k] -= rate * self.hlwt * dlh_dw * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma**2
                # increment feature count for learning rate
                self.fcounts[k] += 1
        elif self.method == 'leitner' or self.method == 'pimsleur':
            pass
        elif self.method == 'lr':
            p, _ = self.predict(inst)
            err = p - inst.p
            for (k, x_k) in inst.fv:
                # rate = (1./(1+inst.p)) * self.lrate   / math.sqrt(1 + self.fcounts[k])
                rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # error update
                self.weights[k] -= rate * err * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma**2
                # increment feature count for learning rate
                self.fcounts[k] += 1

    def train(self, trainset):
        if self.method == 'leitner' or self.method == 'pimsleur':
            return
        random.shuffle(trainset)
        for inst in trainset:
            self.train_update(inst)

    def losses(self, inst):
        p, h = self.predict(inst)
        slp = (inst.p - p)**2
        slh = (inst.h - h)**2
        return slp, slh, p, h

    def eval(self, testset, prefix=''):
        results = {'p': [], 'h': [], 'pp': [], 'hh': [], 'slp': [], 'slh': []}
        for inst in testset:
            slp, slh, p, h = self.losses(inst)
            results['p'].append(inst.p)     # ground truth
            results['h'].append(inst.h)
            results['pp'].append(p)         # predictions
            results['hh'].append(h)
            results['slp'].append(slp)      # loss function values
            results['slh'].append(slh)
        mae_p = mae(results['p'], results['pp'])
        mae_h = mae(results['h'], results['hh'])
        cor_p = spearmanr(results['p'], results['pp'])
        cor_h = spearmanr(results['h'], results['hh'])
        total_slp = sum(results['slp'])
        total_slh = sum(results['slh'])
        total_l2 = sum([x**2 for x in self.weights.values()])
        total_loss = total_slp + self.hlwt*total_slh + self.l2wt*total_l2
        if prefix:
            sys.stderr.write('%s\t' % prefix)
        sys.stderr.write('%.1f (p=%.1f, h=%.1f, l2=%.1f)\tmae(p)=%.3f\tcor(p)=%.3f\tmae(h)=%.3f\tcor(h)=%.3f\n' % \
            (total_loss, total_slp, self.hlwt*total_slh, self.l2wt*total_l2, \
            mae_p, cor_p, mae_h, cor_h))

    def dump_weights(self, fname):
        with open(fname, 'wb') as f:
            for (k, v) in self.weights.iteritems():
                f.write('%s\t%.4f\n' % (k, v))

    def dump_predictions(self, fname, testset):
        with open(fname, 'wb') as f:
            f.write('p\tpp\th\thh\tlang\tuser_id\ttimestamp\n')
            for inst in testset:
                pp, hh = self.predict(inst)
                f.write('%.4f\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\n' % (inst.p, pp, inst.h, hh, inst.lang, inst.uid, inst.ts))

    def dump_detailed_predictions(self, fname, testset):
        with open(fname, 'wb') as f:
            f.write('p\tpp\th\thh\tlang\tuser_id\ttimestamp\tlexeme_tag\n')
            for inst in testset:
                pp, hh = self.predict(inst)
                for i in range(inst.right):
                    f.write('1.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n' % (pp, inst.h, hh, inst.lang, inst.uid, inst.ts, inst.lexeme))
                for i in range(inst.wrong):
                    f.write('0.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n' % (pp, inst.h, hh, inst.lang, inst.uid, inst.ts, inst.lexeme))


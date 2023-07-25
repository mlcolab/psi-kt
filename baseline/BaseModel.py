# -*- coding: UTF-8 -*-

import torch
from tqdm import tqdm
from sklearn.metrics import *
import numpy as np
import torch.nn.functional as F
import os
import ipdb
from utils.utils import get_feed_general
from utils.utils import *
import time

from typing import List, Tuple, Dict

##########################################################################################
# Base Model for all KT
##########################################################################################

class BaseModel(torch.nn.Module):
    runner = 'KTRunner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser, model_name='BaseModel'):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        return parser

    @staticmethod
    def pred_evaluate_method(
        y_pred: np.ndarray, 
        y_true: np.ndarray, 
        metrics: List[str],
    ) -> dict: 
        """
        Compute evaluation metrics for a set of predictions.

        Args:
            y_pred: The predicted values as a NumPy array.
            y_true: The ground truth values as a NumPy array.
            metrics: A list of evaluation metrics to compute.

        Returns:
            A dictionary containing the evaluation metrics and their values.
        """
        # Flatten the arrays to one dimension
        y_pred = np.ravel(y_pred)
        y_true = np.ravel(y_true)
        
        # Convert the predictions to binary values based on a threshold of 0.5
        y_pred_binary = (y_pred > 0.5).astype(int)
        evaluation_funcs = {
            'rmse': mean_squared_error,
            'mae': mean_absolute_error,
            'auc': roc_auc_score,
            'f1': f1_score,
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score
        }
        
        # Define the evaluation functions for each metric
        evaluations = {}
        for metric in metrics:
            if metric in evaluation_funcs:
                evaluations[metric] = evaluation_funcs[metric](
                    y_true, 
                    y_pred_binary if metric in ['f1', 'accuracy', 'precision', 'recall'] else y_pred
                )
                
        return evaluations


    @staticmethod
    def init_weights(
        m: torch.nn.Module
    ) -> None:
        """
        Initialize weights and biases of the neural network module.

        Args:
            m (torch.nn.Module): The neural network module to initialize.

        Returns:
            None: The method modifies the weights and biases of the input module in-place.
        """
        
        # TODO: add more initialization methods
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif isinstance(m, torch.nn.RNNCell) or isinstance(m, torch.nn.GRUCell) or isinstance(m, torch.nn.RNN):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(m, torch.nn.LSTMCell) or isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name or 'weight_ch' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    
                    
    @staticmethod
    def batch_to_gpu(batch, device):
        if torch.cuda.device_count() > 0:
            for key in batch:
                batch[key] = batch[key].to(device)
        return batch


    def __init__(self, model_path='../model/Model/Model_{}_{}.pt'):
        super(BaseModel, self).__init__()
        self.model_path = model_path
        self._init_weights()
        self.optimizer = None


    def _init_weights(
        self
    ):
        pass


    def forward(
        self, 
        feed_dict
    ):
        pass


    def get_feed_dict(
        self, 
        corpus, 
        data, 
        batch_start: int,
        batch_size: int,
        phase: str,
    ):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        
        feed_dict_keys = {
            'skill_seq': 'skill_seq',
            'label_seq': 'correct_seq',
            'time_seq': 'time_seq',
            'problem_seq': 'problem_seq',
            'num_history': 'num_history',
            'num_success': 'num_success',
            'num_failure': 'num_failure',
            'user_id': 'user_id'
            }
        
        feed_dict = get_feed_general(
            keys=feed_dict_keys, 
            data=data, 
            start=batch_start, 
            batch_size=real_batch_size, 
        ) # [batch_size, seq_len]
        
        return feed_dict


    def prepare_batches(
        self, 
        corpus, 
        data: List[Tuple], 
        batch_size: int, 
        phase: str, 
    ) -> List:
        """
        Prepare the data into batches for training/validation/test.

        Args:
            corpus: the corpus object
            data: the training/validation/test data which needs to be batched
            batch_size: the batch size
            phase: the current training phase ('train', 'valid', or 'test')

        Returns:
            A list of batches of the input data
        """
        num_examples = len(data)
        total_batches = (num_examples + batch_size - 1) // batch_size
        assert num_examples > 0

        # Prepare the batches using a list comprehension
        batches = []
        for batch in tqdm(range(total_batches), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self.get_feed_dict(corpus, data, batch * batch_size, batch_size, phase))
            
        return batches


    def count_variables(
        self
    ):
        """
        Counts the number of trainable parameters in a PyTorch model.

        Args:
            model: A PyTorch model.

        Returns:
            The total number of trainable parameters in the model.
        """
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters
    

    def save_model(
        self, 
        epoch: int, 
        mini_epoch: int = 0,
        model_path: str = None
    ) -> None:
        """
        Save the model to a file.

        Args:
            epoch: the current epoch number
            model_path: the path to save the model to
        """
        if model_path is None:
            model_path = self.model_path
        model_path = model_path.format(epoch, mini_epoch)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.state_dict(), model_path)
        self.logs.write_to_log_file('Save model to ' + model_path)

        # self.model_path = model_path


    def load_model(self, model_path: str = None) -> None:
        """
        Load the model from a file.

        Args:
            model_path: the path to load the model from
        """
        if model_path is None:
            model_path = self.model_path

        self.load_state_dict(torch.load(model_path))
        self.eval()
        self.logs.write_to_log_file('Load model from ' + model_path)


    def customize_parameters(self) -> List[Dict]:
        """
        Customize the optimizer settings for different parameters.

        Returns:
            A list of dictionaries specifying the optimization settings for each parameter group
        """
        weight_p, bias_p = [], []
        # Find parameters that require gradient
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()): 
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)

        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        return optimize_dict


    def actions_before_train(
        self
    ):
        total_parameters = self.count_variables()
        self.logs.write_to_log_file('#params: %d' % total_parameters)


    def actions_after_train(
        self
    ): # TODO: add more actions
        end_time = time.time()
        train_time = end_time - self.start_time
        final_loss = self.logs.get_last_loss()
        self.logs.write_to_log_file('Training time: {:.2f} seconds'.format(train_time))
        self.logs.write_to_log_file('Final training loss: {:.4f}'.format(final_loss))



##########################################################################################
# Learner Model
##########################################################################################


class BaseLearnerModel(BaseModel):
    def __init__(self, 
                 mode, 
                 device='cpu', 
                 logs=None):
        super(BaseLearnerModel, self).__init__()
        self.mode = mode
        self.device = device
        self.logs = logs
        self.optimizer = None
        
        self.model_path = os.path.join(logs.args.log_path, 'Model/Model_{}_{}.pt')
        
        
    @staticmethod
    def _find_whole_stats(
        all_feature: torch.Tensor,
        t: torch.Tensor,
        items: torch.Tensor,
        num_node: int,
    ):
        '''
        Args:
            all_feature: [bs, 1, num_step, 3]
            items/t: [bs, num_step]
        '''
        all_feature = all_feature.long()
        device = all_feature.device
        num_seq, num_step = t.shape
        
        # Allocate memory without initializing tensors
        whole_stats = torch.zeros((num_seq, num_node, num_step, 3), device=device, dtype=torch.int64)
        whole_last_time = torch.zeros((num_seq, num_node, num_step+1), device=device, dtype=torch.int64)
        
        # Precompute index tensor
        seq_indices = torch.arange(num_seq, device=device)
        
        # Set initial values for whole_last_time
        whole_last_time[seq_indices, items[:,0], 1] = t[:, 0]

        # Loop over time steps
        for i in range(1, num_step):
            cur_item = items[:, i] # [num_seq, ] 
            cur_feat = all_feature[:,0,i] # [bs, 1, 3] 
            
            # Accumulate whole_stats
            whole_stats[:,:,i] = whole_stats[:,:,i-1] # whole_stats[:,:,i-1] # 
            whole_stats[seq_indices, cur_item, i] = cur_feat
            
            whole_last_time[:,:,i+1] = whole_last_time[:,:,i] # + whole_last_time[seq_indices,:,i]
            whole_last_time[seq_indices, cur_item, i+1] = t[:, i]

        return whole_stats, whole_last_time


    @staticmethod
    def _compute_all_features(
        num_seq: int,
        num_node: int,
        time_step: int,
        device: torch.device,
        stats_cal_on_fly: bool = False,
        items: torch.Tensor = None,
        stats: torch.Tensor = None,
    ):
        if stats_cal_on_fly or items is None:
            item_start = items[:, 0]
            all_feature = torch.zeros((num_seq, num_node, 3), device=device)
            all_feature[torch.arange(0, num_seq), item_start, 0] += 1
            all_feature[torch.arange(0, num_seq), item_start, 2] += 1
            all_feature = all_feature.unsqueeze(-2).tile((1, 1, time_step, 1))
        else:
            all_feature = stats.float()  # [num_seq/bs, num_node, num_time_step, 3]
        return all_feature


    @staticmethod
    def _initialize_parameter(
        shape: Tuple,
        device: torch.device,
    ):
        """
        A static method to initialize a PyTorch parameter tensor with Xavier initialization.

        Args:
            shape (Tuple): A tuple specifying the shape of the parameter tensor.
            device (torch.device): A PyTorch device object specifying the device where the parameter tensor will be created.

        Returns:
            param (nn.Parameter): A PyTorch parameter tensor with the specified shape, initialized using Xavier initialization.

        """
        param = torch.nn.Parameter(torch.empty(shape, device=device))  # create a parameter tensor with the specified shape on the specified device
        torch.nn.init.xavier_uniform_(param)  # apply Xavier initialization to the parameter tensor
        return param  # return the initialized parameter tensor

    
    def forward(
        self,
        feed_dict: Dict[str, torch.Tensor],
    ):
        skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]

        bs, _ = labels.shape
        self.num_seq = bs
        
        x0 = torch.zeros((bs, self.num_node)).to(labels.device)
        if self.num_node > 1:
            x0[torch.arange(bs), skills[:,0]] += labels[:, 0]
            items = skills
        else: 
            x0[:, 0] += labels[:, 0]
            items = None
        
        stats = torch.stack([feed_dict['num_history'], feed_dict['num_success'], feed_dict['num_failure']], dim=-1)
        stats = stats.unsqueeze(1)
        
        out_dict = self.simulate_path(
            x0=x0, 
            t=times, 
            items=items,
            user_id=feed_dict['user_id'],
            stats=stats,
        )
        
        out_dict.update({
            'prediction': out_dict['x_item_pred'],
            'label': labels.unsqueeze(1) # [bs, 1, time]
        })
        
        return out_dict
    
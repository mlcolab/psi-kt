# -*- coding: UTF-8 -*-

import torch
from tqdm import tqdm
from sklearn.metrics import *
import numpy as np
import torch.nn.functional as F
import os
import ipdb
from utils.utils import *
import time

from typing import List, Tuple, Dict
class BaseModel(torch.nn.Module):
    runner = 'KTRunner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser, model_name='BaseModel'):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        return parser

    @staticmethod
    def pred_evaluate_method(y_pred, y_true, metrics): # -> unit test
        y_pred = np.ravel(y_pred)
        y_true = np.ravel(y_true)
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
        evaluations = {}
        for metric in metrics:
            if metric in evaluation_funcs:
                evaluations[metric] = evaluation_funcs[metric](
                    y_true, 
                    y_pred_binary if metric in ['f1', 'accuracy', 'precision', 'recall'] else y_pred
                )
        return evaluations

    @staticmethod
    def init_weights(m): # TO-DO: add more initialization methods
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

    def _init_weights(self):
        pass

    def forward(self, feed_dict):
        pass

    def get_feed_dict(self, corpus, data, batch_start, batch_size, train):
        pass


    def prepare_batches(self, corpus, data: List[Tuple], batch_size: int, phase: str) -> List:
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


    def count_variables(self):
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


    def actions_before_train(self):
        total_parameters = self.count_variables()
        self.logs.write_to_log_file('#params: %d' % total_parameters)


    def actions_after_train(self): # TO-DO: add more actions
        end_time = time.time()
        train_time = end_time - self.start_time
        final_loss = self.logs.get_last_loss()
        self.logs.write_to_log_file('Training time: {:.2f} seconds'.format(train_time))
        self.logs.write_to_log_file('Final training loss: {:.4f}'.format(final_loss))


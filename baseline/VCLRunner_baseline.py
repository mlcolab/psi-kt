import gc, pickle
import copy
import os

from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import utils
import ipdb
        
import ipdb
from KTRunner import KTRunner

OPTIMIZER_MAP = {
    'gd': optim.SGD,
    'adagrad': optim.Adagrad,
    'adadelta': optim.Adadelta,
    'adam': optim.Adam
}


class VCLRunner(KTRunner):
    """ 
    Class implementing GVCL approach
    """

    def __init__(
        self, 
        args,
        logs,
    ):
        '''
        Args:
            args:
            logs
        '''
        super().__init__(args=args, logs=logs)
        
        self.max_time_step = args.max_step
        

    def _eva_termination(
        self, 
        model: torch.nn.Module,
    ):
        """
        A private method that determines whether the training should be terminated based on the results.

        Args:
        - self: the object itself
        - model: the trained model

        Returns:
        - True if the training should be terminated, False otherwise
        """

        # Extract the validation results from the logs
        valid = list(self.logs.train_results[self.metrics[0]]) # 

        # Check if the last 10 validation results have not improved
        if len(valid) > 20 and utils.non_increasing(valid[-10:]):
            return True

        # Check if the maximum validation result has not improved for the past 20 epochs
        elif len(valid) - valid.index(max(valid)) > 20:
            return True

        # Otherwise, return False to continue the training
        return False
    
    
    def train(
        self, 
        model, 
        corpus
    ):
        '''
        Trains the KT model instance with parameters.

        Args:
            model: the KT model instance with parameters to train
            corpus: data
        '''
        # Build the optimizer if it hasn't been built already.
        if model.module.optimizer is None:
            model.module.optimizer, model.module.scheduler = self._build_optimizer(model)
            
        assert(corpus.data_df['train'] is not None)
        self._check_time(start=True)
        
        if self.overfit > 0:
            epoch_whole_data = copy.deepcopy(corpus.data_df['whole'][:self.overfit])
        else:
            epoch_whole_data = copy.deepcopy(corpus.data_df['whole'])

        # Return a random sample of items from an axis of object.
        epoch_whole_data = epoch_whole_data.sample(frac=1).reset_index(drop=True) 
        whole_batches = model.module.prepare_batches(corpus, epoch_whole_data, self.eval_batch_size, phase='whole')
                
        max_time_step = self.args.max_step
        
        training_resources = []
        for time in range(1, max_time_step-1):
            
            try:
                gc.collect()
                model.train()

                self._check_time()
                self.fit(model, whole_batches, time_step=time)
            
            except KeyboardInterrupt:
                self.logs.write_to_log_file("Early stop manually")
                exit_here = input("Exit completely without evaluation? (y/n) (default n):")
                if exit_here.lower().startswith('y'):
                    self.logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                    exit(1)

        # Find the best validation result across iterations
        best_test_epoch = self.logs.test_results[self.metrics[0]].argmax() # 
        test_res_dict = dict()
        
        for metric in self.metrics:
            test_res_dict[metric] = self.logs.test_results[metric][best_test_epoch]
        self.logs.write_to_log_file("\nBest Iter(test)=  %5d\t test=(%s) [%.1f s] "
                     % (best_test_epoch + 1,
                        utils.format_metric(test_res_dict),
                        self.time[1] - self.time[0]))
        self.logs.create_log(   
            args=self.args,
            model=model,
            optimizer=model.module.optimizer,
            final_test=True,
            test_results=self.logs.test_results,
        )


    def fit(
        self, 
        model, 
        batches,
        time_step: int = 0,
    ): 
        """
        Trains the given model on the given batches of data.

        Args:
            model: The model to train.
            batches: A list of data, where each element is a batch to train.
            epoch_train_data: A pandas DataFrame containing the training data.
            epoch: The current epoch number.

        Returns:
            A dictionary containing the training losses.
        """
         
        model.train()
        
        train_losses = defaultdict(list)
        test_losses = defaultdict(list)
        
        for mini_epoch in range(0, self.epoch): 
            
            # Iterate through each batch.
            for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Time %5d' % time_step + ' MiniEpoch %5d' % mini_epoch):
                
                model.module.optimizer.zero_grad(set_to_none=True)
                
                output_dict = model.module.forward_cl(batch, idx=time_step)
                loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
                loss_dict['loss_total'].backward()
                
                # torch.nn.utils.clip_grad_norm(model.module.parameters(),100)
                
                model.module.optimizer.step()
                # model.module.scheduler.step()
    
                # Append the losses to the train_losses dictionary.

                training_time = self._check_time()
                train_loss_dict = {k: v.item() for k, v in loss_dict.items() if 'cl' not in k}
                train_loss_dict['train_time'] = training_time
                test_loss_dict = {k[3:]: v.item() for k, v in loss_dict.items() if 'cl' in k}

                train_losses = self.logs.append_batch_losses(train_losses, train_loss_dict)
                test_losses = self.logs.append_batch_losses(test_losses, test_loss_dict)
                
            # Save the model.
            if mini_epoch % self.args.save_every == 0:
                model.module.save_model(epoch=time_step, mini_epoch=mini_epoch)
            
            # Log the training losses.
            train_string = self.logs.result_string("train", time_step, train_losses, t=time_step, mini_epoch=mini_epoch) # TODO
            self.logs.write_to_log_file(train_string)
            test_string = self.logs.result_string("test", time_step, test_losses, t=time_step, mini_epoch=mini_epoch) # TODO
            self.logs.write_to_log_file(test_string)
            self.logs.append_epoch_losses(train_losses, 'train')
            self.logs.append_epoch_losses(test_losses, 'test')
            
            self.logs.draw_loss_curves()
            
            # # Evaluate the model on the set.
            # if self._eva_termination(model) and self.early_stop:
            #     self.logs.write_to_log_file("Early stop at time %d epoch %d based on result." % (time_step, mini_epoch))
            #     break
            
        model.eval()
        return self.logs.train_results['loss_total'][-1]





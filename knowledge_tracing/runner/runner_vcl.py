import gc
import copy
import os
from time import time
from collections import defaultdict

from tqdm import tqdm

import torch

from knowledge_tracing.utils import utils
from knowledge_tracing.runner.runner import KTRunner


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
            epoch_whole_data['user_id'] = np.arange(self.overfit)
        else:
            epoch_whole_data = copy.deepcopy(corpus.data_df['whole'])

        # Return a random sample of items from an axis of object.
        epoch_whole_data = epoch_whole_data.sample(frac=1).reset_index(drop=True) 
        whole_batches = model.module.prepare_batches(corpus, epoch_whole_data, self.eval_batch_size, phase='whole')
                
        max_time_step = 100 # time_step
        
        try:
            for time in range(max_time_step):
                gc.collect()
                model.train()
                
                self._check_time()
                self.fit(model, whole_batches, epoch=time, time_step=time)
                self.test(model, whole_batches, epoch=time, time_step=time)
                training_time = self._check_time()

        except KeyboardInterrupt:
            self.logs.write_to_log_file("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                self.logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)


    def fit(
        self, 
        model, 
        batches, 
        epoch: int = 0, 
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
         
        model.module.train()
        train_losses = defaultdict(list)
        
        for mini_epoch in range(0, 1): # self.epoch): 
            
            # Iterate through each batch.
            for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Epoch %5d' % epoch + ' Time %5d' % mini_epoch):
                
                # Reset gradients.
                model.module.optimizer.zero_grad(set_to_none=True)
                
                # Predictive model before optimization
                s_tilde_dist, z_tilde_dist = model.module.predictive_model(feed_dict=batch, idx=time_step)
                s_post_dist, z_post_dist = model.module.inference_model(feed_dict=batch, idx=time_step)
                
                # # for DEBUG use
                # s_tilde_dist, z_tilde_dist = model.module.predictive_model(feed_dict=batch, idx=time_step+1)
                # s_post_dist, z_post_dist = model.module.inference_model(feed_dict=batch, idx=time_step+1)
                
                # Calculate loss and perform backward pass.
                output_dict = model.module.objective_function(batch, idx=time_step, pred_dist=[s_tilde_dist, z_tilde_dist], post_dist=[s_post_dist, z_post_dist])
                loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
                loss_dict['loss_total'].backward()
                
                # Update parameters.
                torch.nn.utils.clip_grad_norm(model.module.parameters(),100)
                model.module.optimizer.step()
                # model.module.scheduler.step()
                
                with torch.no_grad():
                    # Update after optimization
                    _, _ = model.module.inference_model(feed_dict=batch, idx=time_step, update=True, eval=False)
                    _, _ = model.module.predictive_model(feed_dict=batch, idx=time_step, update=True, eval=False)
                    
    
                if time_step != 0 and time_step!=self.max_time_step-2:
                    with torch.no_grad():
                        comparison = model.module.comparison_function(batch, idx=time_step)
                
                    loss_dict.update(comparison)
                # Append the losses to the train_losses dictionary.
                train_losses = self.logs.append_batch_losses(train_losses, loss_dict)
                
            if mini_epoch % 10 == 0:
                model.module.save_model(epoch=epoch, mini_epoch=mini_epoch)
            self.logs.draw_loss_curves()
                        
            string = self.logs.result_string("train", epoch, train_losses, t=epoch, mini_epoch=mini_epoch) # TODO
            self.logs.write_to_log_file(string)
            self.logs.append_epoch_losses(train_losses, 'train')
        
        model.eval()
            
        return self.logs.train_results['loss_total'][-1]





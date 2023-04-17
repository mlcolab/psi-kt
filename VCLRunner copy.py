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

from models.new_learner_model import HierachicalSSM
from models.learner_hssm_model import HSSM
        
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

    def __init__(self, args, logs,):
        '''
        Args:
            args:
            logs
        '''
        super().__init__(args=args, logs=logs)
        
        self.equalize_epochs = True
        

    def train(self, model, corpus):
        '''
        Trains the KT model instance with parameters.

        Args:
            model: the KT model instance with parameters to train
            corpus: data
        '''
        assert(corpus.data_df['train'] is not None)
        self._check_time(start=True)
        
        if self.overfit > 0:
            epoch_whole_data = copy.deepcopy(corpus.data_df['whole'][:self.overfit])
        else:
            epoch_whole_data = copy.deepcopy(corpus.data_df['whole'])

        # Return a random sample of items from an axis of object.
        epoch_whole_data = epoch_whole_data.sample(frac=1).reset_index(drop=True) 
        whole_batches = model.module.prepare_batches(corpus, epoch_whole_data, self.eval_batch_size, phase='whole')
        # Move the batch to the GPU.
        whole_batches = [model.module.batch_to_gpu(batch, model.module.device) for batch in whole_batches]
                
        time_step = whole_batches[0]['skill_seq'].shape[-1]
        s_post_dist, z_post_dist = None, None
        max_time_step = 10 # time_step
        
        try:
            for epoch in range(self.epoch):
                gc.collect()
                model.train()
                
                self._check_time()
                loss = self.fit(model, whole_batches, epoch_whole_data, epoch=epoch+1)
                training_time = self._check_time()

                ##### output validation and write to logs
                if (self.args.validate) & (epoch % self.args.validate_every == 0):
                    with torch.no_grad():
                        valid_result = self.evaluate(model, corpus, 'val', val_batches, whole_batches, epoch=epoch+1)
                        test_result = self.evaluate(model, corpus, 'test', test_batches, whole_batches, epoch=epoch+1)
                    testing_time = self._check_time()
                    
                    self.logs.append_epoch_losses(test_result, 'test')
                    self.logs.append_epoch_losses(valid_result, 'val')
                    self.logs.write_to_log_file("Epoch {:<3} loss={:<.4f} [{:<.1f} s]\t valid=({}) test=({}) [{:<.1f} s] ".format(
                                epoch + 1, loss, training_time, utils.format_metric(valid_result),
                                utils.format_metric(test_result), testing_time))
                                
                    if max(self.logs.valid_results[self.metrics[0]]) == valid_result[self.metrics[0]]:
                        # ipdb.set_trace()
                        model.module.save_model(epoch=epoch)
                    if self._eva_termination(model) and self.early_stop:
                        self.logs.write_to_log_file("Early stop at %d based on validation result." % (epoch + 1))
                        break
                else:
                    if epoch % 10 == 0:
                        model.module.save_model(epoch=epoch)
                    
                self.logs.draw_loss_curves()

        except KeyboardInterrupt:
            self.logs.write_to_log_file("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                self.logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best validation result across iterations
        best_valid_epoch = self.logs.valid_results[self.metrics[0]].argmax()
        valid_res_dict, test_res_dict = dict(), dict()
        
        for metric in self.metrics:
            valid_res_dict[metric] = self.logs.valid_results[metric][best_valid_epoch]
            test_res_dict[metric] = self.logs.test_results[metric][best_valid_epoch]
        self.logs.write_to_log_file("\nBest Iter(val)=  %5d\t valid=(%s) test=(%s) [%.1f s] "
                     % (best_valid_epoch + 1,
                        utils.format_metric(valid_res_dict),
                        utils.format_metric(test_res_dict),
                        self.time[1] - self.time[0]))

        best_test_epoch = self.logs.test_results[self.metrics[0]].argmax()
        for metric in self.metrics:
            valid_res_dict[metric] = self.logs.valid_results[metric][best_test_epoch]
            test_res_dict[metric] = self.logs.test_results[metric][best_test_epoch]
        self.logs.write_to_log_file("Best Iter(test)= %5d\t valid=(%s) test=(%s) [%.1f s] \n"
                     % (best_test_epoch + 1,
                        utils.format_metric(valid_res_dict),
                        utils.format_metric(test_res_dict),
                        self.time[1] - self.time[0]))
                        
        # model.load_model() #???



    def fit(self, model, batches, epoch_train_data, epoch=-1): 
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
        # Build the optimizer if it hasn't been built already.
        if model.module.optimizer is None:
            model.module.optimizer, model.module.scheduler = self._build_optimizer(model)
            
        model.train()
        train_losses = defaultdict(list)

        
        for t in range(0, max_time_step): # time_step
            
            # Iterate through each batch.
            for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Epoch %5d' % epoch):
                
                # Reset gradients.
                model.module.optimizer.zero_grad(set_to_none=True)
                
                # Predictive Model
                s_tilde_dist, z_tilde_dist = model.module.predictive_model(feed_dict=batch, idx=t)
                s_post_dist, z_post_dist = model.module.inference_model(feed_dict=batch, idx=t)
                
                # Calculate loss and perform backward pass.
                output_dict = model.module.objective_function(batch, idx=t, pred_dist=[s_tilde_dist, z_tilde_dist], post_dist=[s_post_dist, z_post_dist])
                loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
                loss_dict['loss_total'].backward()
                
                # Update parameters.
                model.module.optimizer.step()
                model.module.scheduler.step()
                
                if t != 0 and t!=max_time_step - 2:
                    with torch.no_grad():
                        comparison = model.module.comparison_function(batch, idx=t)
                
                    loss_dict.update(comparison)
                # Append the losses to the train_losses dictionary.
                train_losses = self.logs.append_batch_losses(train_losses, loss_dict)
                
            string = self.logs.result_string("train", epoch, train_losses, t=epoch) # TODO
            self.logs.write_to_log_file(string)
            self.logs.append_epoch_losses(train_losses, 'train')
        
        model.eval()
            
            
        return self.logs.train_results['loss_total'][-1]




    def train_epoch(self,t,x,y):
        self.model.train()
        
        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        train_samples = 10
        
        epoch_class_loss = 0
        epoch_kl_loss = 0
        epoch_total_loss = 0
        total_hits = 0

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch] # batch size 64
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=False) # [bs, 1, 28, 28]
            targets=torch.autograd.Variable(y[b],volatile=False) # [bs]

            task_labels = int(t) * torch.ones_like(targets)

            # Forward current model
            ipdb.set_trace()
            outputs=self.model(images, task_labels, tasks = [t], num_samples = train_samples)
            output=outputs[t] # [s, bs, 2]
            
            #calculate loss for every MC sample
            stacked_targets = targets.repeat([train_samples]) # [bs*s]
            flattened_output = output.view(-1, output.shape[-1]) # [bs*s, 2]
            class_loss = F.cross_entropy(flattened_output, stacked_targets, reduction = 'mean')
            
            #scale kl term by beta and dataset size
            kl_term = self.beta * self.model.get_kl(lamb = self.lamb)/(x.shape[0])
            loss = class_loss + kl_term
            ipdb.set_trace()
            
            #for calculating the accuracy
            probs = F.softmax(output, dim=2).mean(dim = 0)
            _,pred=probs.max(1)
            hits=(pred==targets).float()
            total_hits+=hits.sum().data.cpu().numpy().item()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            epoch_total_loss += loss.detach().data.item()
            epoch_class_loss += class_loss.detach().data.item()
            epoch_kl_loss += kl_term.detach().data.item()

        return epoch_class_loss/i, epoch_kl_loss/i, epoch_total_loss/i, total_hits/x.shape[0]

    def eval(self,t,x,y):
        with torch.no_grad():
            total_loss=0
            total_acc=0
            total_num=0
            self.model.eval()

            r=np.arange(x.size(0))
            r=torch.LongTensor(r).cuda()

            # Loop batches
            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images=torch.autograd.Variable(x[b],volatile=True)
                targets=torch.autograd.Variable(y[b],volatile=True)

                task_labels = int(t) * torch.ones_like(targets)

                # Forward
                outputs=self.model(images, task_labels, tasks = [t], num_samples = 20)
                output=outputs[t]
                
                probs = F.softmax(output, dim=2).mean(dim = 0)
                _,pred=probs.max(1)
                hits=(pred==targets).float()

                # Log
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=len(b)

            #not measuring loss for test set, just accuracy, so return -1 for loss
            return -1, total_acc/total_num

    def criterion(self,t,output,targets):
        return 0
    
    def ce_crit(self, t, output, targets):
        return 0


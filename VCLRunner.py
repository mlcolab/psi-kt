import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F

import utils
import ipdb
from KTRunner import *

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
        Args:
            model: KT model instance with parameters to train
            corpus: data
        '''
        
        assert(corpus.data_df['train'] is not None)
        self._check_time(start=True)
        
        ipdb.set_trace()

        ##### prepare training data (if needs quick test then specify overfit arguments in the args);
        ##### prepare the batches of training data; this is specific to different KT models (different models may require different features)
        if self.overfit > 0:
            epoch_train_data = copy.deepcopy(corpus.data_df['train'])[:self.overfit] # Index(['user_id', 'skill_seq', 'correct_seq', 'time_seq', 'problem_seq'], dtype='object')
        else:
            epoch_train_data = copy.deepcopy(corpus.data_df['train'])
        epoch_train_data = epoch_train_data.sample(frac=1).reset_index(drop=True) # Return a random sample of items from an axis of object.
        batches = model.module.prepare_batches(corpus, epoch_train_data, self.batch_size, phase='train')

        ipdb.set_trace()
        try:
            for t in range():
                if t != 0:
                    #update posterior to prior for everything except the first task
                    self.model.add_task_body_params([t-1])
            


        except KeyboardInterrupt:
            self.logs.write_to_log_file("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                self.logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)



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
        
        # Iterate through each batch.
        out_dicts = []
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Epoch %5d' % epoch):
            
            # Move the batch to the GPU.
            batch = model.module.batch_to_gpu(batch, model.module.device)
            
            # Reset gradients.
            model.module.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass.
            output_dict = model(batch)
            out_dicts.append(output_dict)
            
            # Calculate loss and perform backward pass.
            loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
            loss_dict['loss_total'].backward()
            
            # Update parameters.
            model.module.optimizer.step()
            model.module.scheduler.step()
            
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


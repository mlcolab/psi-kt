import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F

import utils
import ipdb
from KTRunner import *


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
        # model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,
        # lr_factor=3,lr_patience=5,clipgrad=100,lamb = 1, beta = 1, use_film = False):
        super().__init__(args=args, logs=logs)
        self.equalize_epochs = True
        
        # self.model=model
        # self.model_old=None
        # self.fisher=None
        # self.nepochs=nepochs
        # self.sbatch=sbatch
        # self.lr=lr
        # self.lr_min=lr_min
        # self.lr_factor=lr_factor
        # self.lr_patience=lr_patience
        # self.clipgrad=clipgrad
        # self.beta = beta
        # self.lamb = lamb
        # self.exp = args.experiment
        
        # self.ce=torch.nn.CrossEntropyLoss()
        # self.optimizer=self._get_optimizer()
            
    def _get_optimizer(self, parameters = None, lr=None):
        if lr is None: lr=self.lr
        if parameters is None: parameters = self.model.parameters()
        return torch.optim.Adam(parameters, lr = self.lr)


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




        if t != 0:
            #update posterior to prior for everything except the first task
            self.model.add_task_body_params([t-1])

        #making sure every dataset has the same # of gradient passes irrespective of dataset size
        if t == 0:
            self.first_train_size = len(xtrain)
            num_epochs_to_train = self.nepochs

            #correction if the task order is permuted (for mixture)
            if 'mixture' == self.exp:
                self.first_train_size = 20600 #size of facescrub
                num_epochs_to_train = int(round(self.nepochs * self.first_train_size/len(xtrain)))
        if t > 0 and self.equalize_epochs:
            num_epochs_to_train = int(round(self.nepochs * self.first_train_size/len(xtrain)))

        print('training for {} epochs'.format(num_epochs_to_train))

        # Loop epochs
        for e in range(num_epochs_to_train):
            # Train
            clock0=time.time()
            class_loss, kl_loss, total_loss, train_acc  = self.train_epoch(t,xtrain,ytrain)
            clock1=time.time()

            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms| Train: class_loss={:.3f}  kl_loss={:.3f}  total_loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),class_loss, kl_loss, total_loss,100*train_acc))

        return

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


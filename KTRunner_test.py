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


OPTIMIZER_MAP = {
    'gd': optim.SGD,
    'adagrad': optim.Adagrad,
    'adadelta': optim.Adadelta,
    'adam': optim.Adam
}


class KTRunner(object):
    '''
    This implements the training loop, testing & validation, optimization etc. 
    '''

    def __init__(self, args, logs):
        '''
        Args:
            args: the global arguments
            logs: the Logger instance for logging information
        '''
        self.overfit = args.overfit # TODO debug args
        self.time = None

        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.batch_size_multiGPU 
        self.eval_batch_size = args.eval_batch_size
        
        self.metrics = args.metric.strip().lower().split(',')
        for i in range(len(self.metrics)):
            self.metrics[i] = self.metrics[i].strip()

        self.early_stop = args.early_stop
        self.logs = logs


    def _check_time(self, start=False):
        """
        Check the time to compute the training/test/val time.

        Returns:
        The elapsed time since the last call to this method or the start time.
        """
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        else:
            tmp_time = self.time[1]
            self.time[1] = time()
            return self.time[1] - tmp_time


    def _build_optimizer(self, model):
        '''
        Choose the optimizer based on the optimizer name in the global arguments.
        The optimizer has the setting of weight decay, and learning rate decay which can be modified in global arguments.

        Args:
            model: the training KT model
        '''
        optimizer_name = self.args.optimizer.lower()
        lr = self.args.lr
        weight_decay = self.args.l2
        lr_decay = self.args.lr_decay
        lr_decay_gamma = self.args.gamma

        if optimizer_name not in OPTIMIZER_MAP:
            raise ValueError("Unknown optimizer: " + optimizer_name)

        optimizer_class = OPTIMIZER_MAP[optimizer_name]
        self.logs.write_to_log_file(f"Optimizer: {optimizer_name}")
        # name =  [param[0] for param in list(model.module.named_parameters())]
        # ipdb.set_trace()
        inference_params = []
        graph_params = []
        generative_params = []
        for param_group in list(model.module.named_parameters()):
            if 'node_' in param_group[0] and param_group[1].requires_grad:
                graph_params.append(param_group[1])
            elif 'infer_' in param_group[0] and param_group[1].requires_grad:
                inference_params.append(param_group[1])
            elif param_group[1].requires_grad:
                generative_params.append(param_group[1])

        self.graph_params = graph_params
        self.infer_params = inference_params
        self.gen_params = generative_params
        
        optimizer_infer = optimizer_class(inference_params, lr=lr, weight_decay=weight_decay)
        optimizer_graph = optimizer_class(graph_params, lr=lr, weight_decay=weight_decay)
        optimizer_gen = optimizer_class(generative_params, lr=lr, weight_decay=weight_decay)

        scheduler_infer = lr_scheduler.StepLR(optimizer_infer, step_size=lr_decay, gamma=lr_decay_gamma)
        scheduler_graph = lr_scheduler.StepLR(optimizer_graph, step_size=lr_decay, gamma=lr_decay_gamma)
        scheduler_gen = lr_scheduler.StepLR(optimizer_gen, step_size=lr_decay, gamma=lr_decay_gamma)
        
        return [optimizer_gen, optimizer_infer, optimizer_graph], [scheduler_gen, scheduler_infer, scheduler_graph]


    def _print_res(self, model, corpus): # TODO
        '''
        Print the model prediction on test data set.
        This is used in main function to compare the performance of model before and after training.
        Args:
            model: KT model instance
            corpus: data containing test dataset
        '''
        set_name = 'test'
        result = self.evaluate(model, corpus, set_name)
        res_str = utils.format_metric(result)
        return res_str


    def _eva_termination(self, model):
        valid = list(self.logs.valid_results[self.metrics[0]])
        if len(valid) > 20 and utils.non_increasing(valid[-10:]):
            return True
        elif len(valid) - valid.index(max(valid)) > 20:
            return True
        return False
    

    def train(self, model, corpus):
        '''
        Trains the KT model instance with parameters.

        Args:
            model: the KT model instance with parameters to train
            corpus: data
        '''
        assert(corpus.data_df['train'] is not None)
        self._check_time(start=True)
        
        ##### prepare training data (if needs quick test then specify overfit arguments in the args);
        ##### prepare the batches of training data; this is specific to different KT models (different models may require different features)
        set_name = ['train', 'val', 'test', 'whole']
        if self.overfit > 0:
            epoch_train_data, epoch_val_data, epoch_test_data, epoch_whole_data = [
                copy.deepcopy(corpus.data_df[key][:self.overfit]) for key in set_name
            ]
        else:
            epoch_train_data, epoch_val_data, epoch_test_data, epoch_whole_data = [
                copy.deepcopy(corpus.data_df[key]) for key in set_name
            ]

        # Return a random sample of items from an axis of object.
        epoch_train_data = epoch_train_data.sample(frac=1).reset_index(drop=True) 
        train_batches = model.module.prepare_batches(corpus, epoch_train_data, self.batch_size, phase='train')
        val_batches = None
        test_batches = None
        whole_batches = None
        
        if self.args.validate:
            val_batches = model.module.prepare_batches(corpus, epoch_val_data, self.eval_batch_size, phase='val')
            test_batches = model.module.prepare_batches(corpus, epoch_test_data, self.eval_batch_size, phase='test')
            
            if isinstance(model.module, HierachicalSSM) or isinstance(model.module, HSSM):
                whole_batches = model.module.prepare_batches(corpus, epoch_whole_data, self.eval_batch_size, phase='whole')
            else: whole_batches = None
        
        
        try:
            for epoch in range(self.epoch):
                gc.collect()
                model.train()
                
                self._check_time()
                
                for mini_epoch in range(10):
                    loss = self.fit(model, train_batches, epoch_train_data, epoch=epoch+1, mini_epoch=mini_epoch, phase='infer')
                for mini_epoch in range(5):
                    loss = self.fit(model, train_batches, epoch_train_data, epoch=epoch+1, mini_epoch=mini_epoch, phase='graph')
                for mini_epoch in range(5):
                    loss = self.fit(model, train_batches, epoch_train_data, epoch=epoch+1, mini_epoch=mini_epoch, phase='gen')
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


    def fit(self, model, batches, epoch_train_data, epoch=-1, mini_epoch=-1, phase='infer'): 
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
            opt, sch = self._build_optimizer(model)
            model.module.optimizer_gen, model.module.optimizer_infer, model.module.optimizer_graph = opt
            model.module.scheduler_gen, model.module.scheduler_infer, model.module.scheduler_graph = sch
            model.module.optimizer = model.module.optimizer_infer
            
        model.train()
        train_losses = defaultdict(list)
        
        # Iterate through each batch.
        if phase == 'infer':
            opt = model.module.optimizer_infer
            sch = model.module.scheduler_infer
            for param in self.gen_params + self.graph_params:
                param.requires_grad = False
            for param in self.infer_params:
                param.requires_grad = True
        elif phase == 'gen':
            opt = model.module.optimizer_gen
            sch = model.module.scheduler_gen
            for param in self.infer_params + self.graph_params:
                param.requires_grad = False
            for param in self.infer_params:
                param.requires_grad = True
        else:
            opt = model.module.optimizer_graph
            sch = model.module.scheduler_graph
            for param in self.gen_params + self.infer_params:
                param.requires_grad = False
            for param in self.infer_params:
                param.requires_grad = True
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Epoch %5d' % epoch):
            model.module.optimizer_infer.zero_grad(set_to_none=True)
            model.module.optimizer_graph.zero_grad(set_to_none=True)
            model.module.optimizer_gen.zero_grad(set_to_none=True)
            output_dict = model(batch)
            loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
            loss_dict['loss_total'].backward()
            opt.step()
            if mini_epoch == 4:
                sch.step()  
            
            # Append the losses to the train_losses dictionary.
            train_losses = self.logs.append_batch_losses(train_losses, loss_dict)
            
        string = self.logs.result_string("train", epoch, train_losses, t=epoch) # TODO
        self.logs.write_to_log_file(string)
        self.logs.append_epoch_losses(train_losses, 'train')
        
        model.eval()
    
            
        # TODO DEBUG: to visualize the difference of synthetic data adj
        if 'synthetic' in self.args.dataset and epoch%2 == 0:
            import matplotlib.patches as mpatches
            gt_adj = batch['gt_adj']
            _, probs, pred_adj = model.module.var_dist_A.sample_A(num_graph=100)
            print(torch.mean(probs, 0))
            # ipdb.set_trace()
            mat_diff = gt_adj-pred_adj[0,0] 
            mat_diff = mat_diff.int().cpu().detach().numpy()
            im = plt.imshow(mat_diff, interpolation='none', cmap='Blues',aspect='auto',alpha=0.5)

            values = np.unique(mat_diff.ravel())
            colors = [im.cmap(im.norm(value)) for value in values]
            patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values))]

            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
            plt.savefig(os.path.join(self.args.plotdir, 'adj_diff_epoch{}.png'.format(epoch)))
            
        return self.logs.train_results['loss_total'][-1]


    def predict(self, model, corpus, set_name, data_batches=None, whole_batches=None, epoch=None):
        '''
        Args:
            model: 
        '''
        model.eval()
        
        predictions, labels = [], []
        out_dicts = []

        if isinstance(model.module, HierachicalSSM) or isinstance(model.module, HSSM):
            for batch in tqdm(whole_batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
                batch = model.module.batch_to_gpu(batch)
                
                out_dict = model.module.predictive_model(batch)
                out_dicts.append(out_dict)
                
                prediction, label = out_dict['prediction'], out_dict['label']
                predictions.extend(prediction.detach().cpu().data.numpy())
                labels.extend(label.detach().cpu().data.numpy())
        
        else:
            for batch in tqdm(data_batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
                batch = model.module.batch_to_gpu(batch)
            
                out_dict = model(batch)
                out_dicts.append(out_dict)
                
                prediction, label = out_dict['prediction'], out_dict['label']
                predictions.extend(prediction.detach().cpu().data.numpy())
                labels.extend(label.detach().cpu().data.numpy())
        
        # Save the output dict for visualization
        if self.args.vis_val & epoch % 5 == 0:
            flat_outdicts = {}
            for key in out_dicts[0].keys():
                if key not in ['elbo', 'iwae', 'initial_likelihood', 'sequence_likelihood', 
                               'st_entropy', 'zt_entropy',
                               'log_prob_yt', 'log_prob_zt', 'log_prob_st']:
                    flat_outdicts[key] = torch.cat([out[key] for out in out_dicts], 1)
            
            with open(os.path.join(self.args.visdir, set_name+'_out_dict_epoch_{}.pkg'.format(epoch)), 'wb') as f:
                pickle.dump(flat_outdicts, f)
                
        return np.array(predictions), np.array(labels)


    def evaluate(self, model, corpus, set_name, data_batches=None, whole_batches=None, epoch=None):
        '''
        Evaluate the results for an input set.

        Args:
            model: The trained model to evaluate.
            corpus: The Corpus object that holds the input data.
            set_name: The name of the dataset to evaluate (e.g. 'train', 'valid', 'test').
            data_batches: The list of batches containing the input data (optional).
            whole_batches: The list of whole batches containing the input data (optional).
            epoch: The epoch number (optional).

        Returns:
            The evaluation results as a dictionary.
        '''

        # Get the predictions and labels from the predict() method.
        predictions, labels = self.predict(model, corpus, set_name, data_batches, whole_batches, epoch=epoch)

        # Get the lengths of the sequences in the input dataset.
        lengths = np.array(list(map(lambda lst: len(lst) - 1, corpus.data_df[set_name]['skill_seq'])))

        # Concatenate the predictions and labels into arrays.
        concat_pred, concat_label = [], []
        for pred, label, length in zip(predictions, labels, lengths):
            concat_pred.append(pred)
            concat_label.append(label)
        concat_pred = np.concatenate(concat_pred)
        concat_label = np.concatenate(concat_label)

        # Evaluate the predictions and labels using the pred_evaluate_method of the model.
        return model.module.pred_evaluate_method(concat_pred, concat_label, self.metrics)

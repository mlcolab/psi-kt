import gc, copy, os
from collections import defaultdict

from tqdm import tqdm
import numpy as np

import torch
from torch.optim import lr_scheduler

from knowledge_tracing.utils import utils
from knowledge_tracing.runner import OPTIMIZER_MAP
from knowledge_tracing.runner.runner import KTRunner
from knowledge_tracing.data.data_loader import DataReader
from knowledge_tracing.groupkt.groupkt import * 


class GroupKTRunner(KTRunner):
    '''
    This implements the training loop, testing & validation, optimization etc. 
    '''

    def __init__(self, args, logs):
        super().__init__(args, logs)


    def _build_optimizer(
        self, 
        model
    ):
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
        
        if not self.args.em_train:
            optimizer = optimizer_class(model.module.customize_parameters(), lr=lr, weight_decay=weight_decay) # TODO some parameters are not initialized
            scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=lr_decay_gamma)
            return optimizer, scheduler
        
        else:
            generative_params = []
            inference_params = []
            graph_params = []
            for param_group in list(model.module.named_parameters()):
                if param_group[1].requires_grad:
                    if 'node_' in param_group[0]:
                        graph_params.append(param_group[1])
                    elif 'infer_' in param_group[0] and param_group[1].requires_grad:
                        inference_params.append(param_group[1])
                    elif param_group[1].requires_grad:
                        generative_params.append(param_group[1])

            self.graph_params = graph_params
            self.inference_params = inference_params
            self.generative_params = generative_params

            optimizer_infer = optimizer_class(inference_params, lr=lr, weight_decay=weight_decay)
            optimizer_graph = optimizer_class(graph_params, lr=lr, weight_decay=weight_decay)
            optimizer_gen = optimizer_class(generative_params, lr=lr, weight_decay=weight_decay)

            scheduler_infer = lr_scheduler.StepLR(optimizer_infer, step_size=lr_decay, gamma=lr_decay_gamma)
            scheduler_graph = lr_scheduler.StepLR(optimizer_graph, step_size=lr_decay, gamma=lr_decay_gamma)
            scheduler_gen = lr_scheduler.StepLR(optimizer_gen, step_size=lr_decay, gamma=lr_decay_gamma)

            return [optimizer_infer, optimizer_gen, optimizer_graph], [scheduler_infer, scheduler_gen, scheduler_graph]
    

    def train(
        self, 
        model: torch.nn.Module,
        corpus: DataReader,
    ):
        '''
        Trains the KT model instance with parameters.

        Args:
            model: the KT model instance with parameters to train
            corpus: data
        '''
        assert(corpus.data_df['train'] is not None)
        self._check_time(start=True)
        
        # prepare the batches of training data; this is specific to different KT models (different models may require different features)
        set_name = ['train', 'val', 'test', 'whole']
        epoch_train_data, epoch_val_data, epoch_test_data, epoch_whole_data = [
            copy.deepcopy(corpus.data_df[key]) for key in set_name
        ]

        # Return a random sample of items from an axis of object.
        train_batches = model.module.prepare_batches(corpus, epoch_train_data, self.batch_size, phase='train')
        val_batches, test_batches, whole_batches = None, None, None
        
        if self.args.test:
            test_batches = model.module.prepare_batches(corpus, epoch_test_data, self.eval_batch_size, phase='test')
            whole_batches = model.module.prepare_batches(corpus, epoch_whole_data, self.eval_batch_size, phase='whole')
        if self.args.validate:
            val_batches = model.module.prepare_batches(corpus, epoch_val_data, self.eval_batch_size, phase='val')

        try:
            for epoch in range(self.epoch):
                gc.collect()
                model.module.train()
                
                self._check_time()
                
                if not self.args.em_train: 
                    loss = self.fit(model=model, batches=train_batches, epoch=epoch)
                    self.test(
                        model=model, 
                        corpus=corpus, 
                        epoch=epoch, 
                        train_loss=loss,
                        test_batches=test_batches,
                        val_batches=val_batches,
                    )
                else: # TODO: this is not used currently
                    loss = self.fit_em_phases(model, corpus, epoch=epoch)

                if epoch % self.args.save_every == 0:
                    model.module.save_model(epoch=epoch)
                    
                if self.early_stop:
                    if self._eva_termination(model): 
                        self.logs.write_to_log_file("Early stop at %d based on validation result." % (epoch + 1))
                        break
                self.logs.draw_loss_curves()


        except KeyboardInterrupt:
            self.logs.write_to_log_file("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                self.logs.write_to_log_file(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best validation result across iterations
        valid_res_dict, test_res_dict = dict(), dict()

        if self.args.validate:
            best_valid_epoch = self.logs.valid_results[self.metrics[0]].argmax()
            for metric in self.metrics:
                valid_res_dict[metric] = self.logs.valid_results[metric][best_valid_epoch]
                test_res_dict[metric] = self.logs.test_results[metric][best_valid_epoch]
            self.logs.write_to_log_file("\nBest Iter(val)=  %5d\t valid=(%s) test=(%s) [%.1f s] "
                        % (best_valid_epoch + 1,
                            utils.format_metric(valid_res_dict),
                            utils.format_metric(test_res_dict),
                            self.time[1] - self.time[0]))

        if self.args.test:
            best_test_epoch = self.logs.test_results[self.metrics[0]].argmax()
            for metric in self.metrics:
                test_res_dict[metric] = self.logs.test_results[metric][best_test_epoch]
            self.logs.write_to_log_file("Best Iter(test)= %5d\t test=(%s) [%.1f s] \n"
                        % (best_test_epoch + 1,
                            utils.format_metric(test_res_dict),
                            self.time[1] - self.time[0]))
                        
        self.logs.create_log(   
            args=self.args,
            model=model,
            optimizer=model.module.optimizer,
            final_test=True if self.args.test else False,
            test_results=self.logs.test_results,
        )


    def test(
        self, 
        model: torch.nn.Module,
        corpus: DataReader,
        epoch: int = 0, 
        train_loss: float = 0.,
        test_batches: list = None,
        val_batches: list = None,
    ):
        # ipdb.set_trace()
        training_time = self._check_time()

        model.module.eval()
        if (self.args.test) & (epoch % self.args.test_every == 0):
            with torch.no_grad():
                test_result = self.evaluate(
                    model=model, 
                    corpus=corpus, 
                    set_name='test', 
                    data_batches=test_batches, 
                    epoch=epoch
                )

                if self.args.validate:
                    valid_result = self.evaluate(model, corpus, 'val', val_batches, epoch=epoch)
                    self.logs.append_epoch_losses(valid_result, 'val')

                    if max(self.logs.valid_results[self.metrics[0]]) == valid_result[self.metrics[0]]:
                        model.module.save_model(epoch=epoch)
                else:
                    valid_result = test_result

            testing_time = self._check_time()
            
            self.logs.append_epoch_losses(test_result, 'test')
            self.logs.write_to_log_file("Epoch {:<3} loss={:<.4f} [{:<.1f} s]\t valid=({}) test=({}) [{:<.1f} s] ".format(
                        epoch + 1, train_loss, training_time, utils.format_metric(valid_result),
                        utils.format_metric(test_result), testing_time))


    def fit(
        self, 
        model: torch.nn.Module,
        batches = None,
        epoch=-1,
    ): 
        """
        Trains the given model on the given batches of data.

        Args:
            model: The model to train.
            batches: A list of data, where each element is a batch to train.
            epoch: The current epoch number.

        Returns:
            A dictionary containing the training losses.
        """

        # Build the optimizer if it hasn't been built already.
        if model.module.optimizer is None:
            model.module.optimizer, model.module.scheduler = self._build_optimizer(model)
            
        model.module.train()
        train_losses = defaultdict(list)
        
        # Iterate through each batch.
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Epoch %5d' % epoch):
            # Move batches to GPU if necessary.
            batch = model.module.batch_to_gpu(batch, self.device)
            
            # Reset gradients.
            model.module.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass.
            output_dict = model(batch)
            
            # Calculate loss and perform backward pass.
            loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
            loss_dict['loss_total'].backward()
            
            # Update parameters.
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 100)
            model.module.optimizer.step()

            # Append the losses to the train_losses dictionary.
            train_losses = self.logs.append_batch_losses(train_losses, loss_dict)
            
        string = self.logs.result_string("train", epoch, train_losses, t=epoch) # TODO
        self.logs.write_to_log_file(string)
        self.logs.append_epoch_losses(train_losses, 'train')

        model.module.scheduler.step()
        model.module.eval()
            
        # # TODO DEBUG: to visualize the difference of synthetic data adj
        # if 'synthetic' in self.args.dataset and epoch%2 == 0:
        #     import matplotlib.patches as mpatches
        #     gt_adj = batch['gt_adj']
        #     _, probs, pred_adj = model.module.var_dist_A.sample_A(num_graph=100)
        #     print(torch.mean(probs, 0))
        #     # ipdb.set_trace()
        #     mat_diff = gt_adj-pred_adj[0,0] 
        #     mat_diff = mat_diff.int().cpu().detach().numpy()
        #     im = plt.imshow(mat_diff, interpolation='none', cmap='Blues',aspect='auto',alpha=0.5)

        #     values = np.unique(mat_diff.ravel())
        #     colors = [im.cmap(im.norm(value)) for value in values]
        #     patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values))]

        #     plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        #     plt.savefig(os.path.join(self.args.plotdir, 'adj_diff_epoch{}.png'.format(epoch)))
            
        return self.logs.train_results['loss_total'][-1]


    def predict(
        self, 
        model, 
        data_batches=None, 
        epoch=None
    ):
        '''
        Args:
            model: 
        '''
        model.module.eval()
        
        predictions, labels = [], []
                
        for batch in tqdm(data_batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch = model.module.batch_to_gpu(batch, self.device)
            
            out_dict = model.module.predictive_model(batch)
            prediction, label = out_dict['prediction'], out_dict['label']
            
            predictions.extend(prediction.detach().cpu().data.numpy())
            labels.extend(label.detach().cpu().data.numpy())
        
                
        return np.array(predictions), np.array(labels)


    def evaluate(
        self, 
        model, 
        corpus, 
        set_name, 
        data_batches=None, 
        epoch=None
    ):
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
        predictions, labels = self.predict(model, data_batches, epoch=epoch)

        # Get the lengths of the sequences in the input dataset.
        lengths = np.array(list(map(lambda lst: len(lst) - 1, corpus.data_df[set_name]['skill_seq'])))

        # # Concatenate the predictions and labels into arrays.
        # concat_pred, concat_label = [], []
        # for pred, label, length in zip(predictions, labels, lengths):
        #     concat_pred.append(pred)
        #     concat_label.append(label)
        # concat_pred = np.concatenate(concat_pred)
        # concat_label = np.concatenate(concat_label)
        # ipdb.set_trace()
        concat_pred = predictions
        concat_label = labels
        
        # Evaluate the predictions and labels using the pred_evaluate_method of the model.
        return model.module.pred_evaluate_method(concat_pred, concat_label, self.metrics)


    def fit_em_phases(self, model, corpus, epoch=-1):

        if model.module.optimizer is None:
            opt, sch = self._build_optimizer(model)
            model.module.optimizer_infer, model.module.optimizer_gen, model.module.optimizer_graph = opt
            model.module.scheduler_infer, model.module.scheduler_gen, model.module.scheduler_graph = sch
            model.module.optimizer = model.module.optimizer_infer

        for phase in ['infer', 'gen_graph']: # 'model', 'graph', 'infer', 'gen'

            model.module.train()
            
            if phase == 'model':
                opt = [model.module.optimizer_infer, model.module.optimizer_gen]
                for param in self.graph_params: 
                    param.requires_grad = False
                for param in self.generative_params + self.inference_params:
                    param.requires_grad = True
            elif phase == 'graph':
                opt = [model.module.optimizer_graph]
                for param in self.generative_params + self.inference_params:
                    param.requires_grad = False
                for param in self.graph_params:
                    param.requires_grad = True
            elif phase == 'infer':
                opt = [model.module.optimizer_infer]
                for param in self.generative_params + self.graph_params:
                    param.requires_grad = False
                for param in self.inference_params:
                    param.requires_grad = True
            elif phase == 'gen':
                opt = [model.module.optimizer_gen]
                for param in self.inference_params + self.graph_params:
                    param.requires_grad = False
                for param in self.generative_params:
                    param.requires_grad = True
            elif phase == 'infer_graph':
                opt = [model.module.optimizer_infer, model.module.optimizer_graph]
                for param in self.generative_params:
                    param.requires_grad = False
                for param in self.inference_params + self.graph_params:
                    param.requires_grad = True
            elif phase == 'gen_graph':
                opt = [model.module.optimizer_gen, model.module.optimizer_graph]
                for param in self.inference_params:
                    param.requires_grad = False
                for param in self.generative_params + self.graph_params:
                    param.requires_grad = True

            for i in range(5): # TODO: 5 is a hyperparameter
                loss = self.fit_one_phase(model, epoch=epoch, mini_epoch=i, opt=opt)

            self.test(model, corpus, train_loss=loss)

        model.module.scheduler_infer.step()
        model.module.scheduler_graph.step()
        model.module.scheduler_gen.step()

        model.module.eval()

        return self.logs.train_results['loss_total'][-1]
    

    def fit_one_phase(
        self, 
        model, 
        epoch=-1, 
        mini_epoch=-1, 
        phase='infer',
        opt=None,
    ): 

        train_losses = defaultdict(list)
                
        for batch in tqdm(self.whole_batches, leave=False, ncols=100, mininterval=1, desc='Epoch %5d' % epoch): # TODO

            model.module.optimizer_infer.zero_grad(set_to_none=True)
            model.module.optimizer_graph.zero_grad(set_to_none=True)
            model.module.optimizer_gen.zero_grad(set_to_none=True)

            output_dict = model(batch)
            loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
            loss_dict['loss_total'].backward()

            # ipdb.set_trace()
            torch.nn.utils.clip_grad_norm_(model.module.parameters(),100)

            for o in opt:
                o.step()

            # Append the losses to the train_losses dictionary.
            train_losses = self.logs.append_batch_losses(train_losses, loss_dict)
            
        string = self.logs.result_string("train", epoch, train_losses, t=epoch, mini_epoch=mini_epoch) # TODO
        self.logs.write_to_log_file(string)
        self.logs.append_epoch_losses(train_losses, 'train')
        
        return self.logs.train_results['loss_total'][-1]
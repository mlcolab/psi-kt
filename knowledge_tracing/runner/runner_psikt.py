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
from knowledge_tracing.psikt.psikt import *


class PSIKTRunner(KTRunner):
    """
    This implements the training loop, testing & validation, optimization etc.
    """

    def __init__(self, args, logs):
        KTRunner.__init__(args, logs)

    def _partition_parameters(self, model: torch.nn.Module):
        """
        Partitions the model's parameters into generative, inference, and graph-related groups for separate optimization.

        Args:
            model: The KT model instance whose parameters are to be partitioned.

        Returns:
            Three lists of parameters corresponding to generative, inference, and graph components of the model.
        """
        generative_params = []
        inference_params = []
        graph_params = []
        for param_group in list(model.module.named_parameters()):
            if param_group[1].requires_grad:
                if "node_" in param_group[0]:
                    graph_params.append(param_group[1])
                elif "infer_" in param_group[0] and param_group[1].requires_grad:
                    inference_params.append(param_group[1])
                else:
                    generative_params.append(param_group[1])

        return generative_params, inference_params, graph_params

    def _build_optimizer(
        self,
        model: torch.nn.Module,
    ) -> tuple:
        """
        Choose the optimizer based on the optimizer name in the global arguments.
        The optimizer has the setting of weight decay, and learning rate decay which can be modified in global arguments.

        Args:
            model: the training KT model
        """

        # Retrieve optimizer settings from global arguments
        optimizer_name = self.args.optimizer.lower()
        lr = self.args.lr
        weight_decay = self.args.l2
        lr_decay = self.args.lr_decay
        lr_decay_gamma = self.args.gamma

        # Ensure the specified optimizer is supported
        if optimizer_name not in OPTIMIZER_MAP:
            raise ValueError("Unknown optimizer: " + optimizer_name)

        optimizer_class = OPTIMIZER_MAP[optimizer_name]
        self.logs.write_to_log_file(f"Optimizer: {optimizer_name}")

        # For models not using EM training, create a single optimizer and scheduler
        if not self.args.em_train:
            optimizer = optimizer_class(
                model.module.customize_parameters(), lr=lr, weight_decay=weight_decay
            )
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=lr_decay, gamma=lr_decay_gamma
            )
            return optimizer, scheduler

        else:
            # For EM training, separate the model's parameters into different groups
            (
                generative_params,
                inference_params,
                graph_params,
            ) = self._partition_parameters(model)

            optimizer_infer = optimizer_class(
                inference_params, lr=lr, weight_decay=weight_decay
            )
            optimizer_graph = optimizer_class(
                graph_params, lr=lr, weight_decay=weight_decay
            )
            optimizer_gen = optimizer_class(
                generative_params, lr=lr, weight_decay=weight_decay
            )

            scheduler_infer = lr_scheduler.StepLR(
                optimizer_infer, step_size=lr_decay, gamma=lr_decay_gamma
            )
            scheduler_graph = lr_scheduler.StepLR(
                optimizer_graph, step_size=lr_decay, gamma=lr_decay_gamma
            )
            scheduler_gen = lr_scheduler.StepLR(
                optimizer_gen, step_size=lr_decay, gamma=lr_decay_gamma
            )

            return [optimizer_infer, optimizer_gen, optimizer_graph], [
                scheduler_infer,
                scheduler_gen,
                scheduler_graph,
            ]

    def train(
        self,
        model: torch.nn.Module,
        corpus: DataReader,
    ) -> None:
        """
        Trains the KT model instance with parameters.

        Args:
            model: the KT model instance with parameters to train
            corpus: data
        """
        assert corpus.data_df["train"] is not None
        self._check_time(start=True)

        # prepare the batches of training data; this is specific to different KT models (different models may require different features)
        set_name = ["train", "val", "test", "whole"]
        epoch_train_data, epoch_val_data, epoch_test_data, epoch_whole_data = [
            copy.deepcopy(corpus.data_df[key]) for key in set_name
        ]

        # Return a random sample of items from an axis of object.
        train_batches = model.module.prepare_batches(
            corpus, epoch_train_data, self.batch_size, phase="train"
        )
        val_batches, test_batches = None, None

        if self.args.test:
            test_batches = model.module.prepare_batches(
                corpus, epoch_test_data, self.eval_batch_size, phase="test"
            )
        if self.args.validate:
            val_batches = model.module.prepare_batches(
                corpus, epoch_val_data, self.eval_batch_size, phase="val"
            )

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
                else:
                    loss = self.fit_em_phases(model, corpus, epoch=epoch)

                if epoch % self.args.save_every == 0:
                    model.module.save_model(epoch=epoch)

                if self.early_stop:
                    if self._eva_termination(model):
                        self.logs.write_to_log_file(
                            "Early stop at %d based on validation result." % (epoch + 1)
                        )
                        break
                self.logs.draw_loss_curves()

        except KeyboardInterrupt:
            self.logs.write_to_log_file("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith("y"):
                self.logs.write_to_log_file(
                    os.linesep + "-" * 45 + " END: " + utils.get_time() + " " + "-" * 45
                )
                exit(1)

        # Find the best validation result across iterations
        valid_res_dict, test_res_dict = dict(), dict()

        if self.args.validate:
            best_valid_epoch = self.logs.val_results[self.metrics[0]].argmax()
            for metric in self.metrics:
                valid_res_dict[metric] = self.logs.val_results[metric][best_valid_epoch]
                test_res_dict[metric] = self.logs.test_results[metric][best_valid_epoch]
            self.logs.write_to_log_file(
                "\nBest Iter(val)=  %5d\t valid=(%s) test=(%s) [%.1f s] "
                % (
                    best_valid_epoch + 1,
                    utils.format_metric(valid_res_dict),
                    utils.format_metric(test_res_dict),
                    self.time[1] - self.time[0],
                )
            )

        if self.args.test:
            best_test_epoch = self.logs.test_results[self.metrics[0]].argmax()
            for metric in self.metrics:
                test_res_dict[metric] = self.logs.test_results[metric][best_test_epoch]
            self.logs.write_to_log_file(
                "Best Iter(test)= %5d\t test=(%s) [%.1f s] \n"
                % (
                    best_test_epoch + 1,
                    utils.format_metric(test_res_dict),
                    self.time[1] - self.time[0],
                )
            )

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
        train_loss: float = 0.0,
        test_batches: list = None,
        val_batches: list = None,
    ) -> None:
        """
        Evaluate the model's performance on the test and validation datasets.

        Args:
            model (torch.nn.Module): The neural network model to evaluate.
            corpus (DataReader): The data reader object containing test and validation datasets.
            epoch (int, optional): The current epoch number. Defaults to 0.
            train_loss (float, optional): The training loss. Defaults to 0.0.
            test_batches (list, optional): List of batches for the test dataset. Defaults to None.
            val_batches (list, optional): List of batches for the validation dataset. Defaults to None.

        Returns:
            None
        """
        # Measure training time
        training_time = self._check_time()

        # Set the model to evaluation mode
        model.module.eval()

        # If testing is enabled and the current epoch is a multiple of test_every
        if (self.args.test) & (epoch % self.args.test_every == 0):
            with torch.no_grad():
                test_result = self.evaluate(
                    model=model,
                    corpus=corpus,
                    set_name="test",
                    data_batches=test_batches,
                    epoch=epoch,
                )

                if self.args.validate:
                    valid_result = self.evaluate(
                        model, corpus, "val", val_batches, epoch=epoch
                    )
                    self.logs.append_epoch_losses(valid_result, "val")

                    if (
                        max(self.logs.val_results[self.metrics[0]])
                        == valid_result[self.metrics[0]]
                    ):
                        model.module.save_model(epoch=epoch)
                else:
                    valid_result = test_result

            # Measure testing time
            testing_time = self._check_time()

            # Append test results to logs
            self.logs.append_epoch_losses(test_result, "test")

            # Write log entry
            self.logs.write_to_log_file(
                "Epoch {:<3} loss={:<.4f} [{:<.1f} s]\t valid=({}) test=({}) [{:<.1f} s] ".format(
                    epoch + 1,
                    train_loss,
                    training_time,
                    utils.format_metric(valid_result),
                    utils.format_metric(test_result),
                    testing_time,
                )
            )

    def fit(
        self,
        model: torch.nn.Module,
        batches: list = None,
        epoch: int = -1,
    ) -> dict:
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
            model.module.optimizer, model.module.scheduler = self._build_optimizer(
                model
            )

        model.module.train()
        train_losses = defaultdict(list)

        # Iterate through each batch.
        for batch in tqdm(
            batches, leave=False, ncols=100, mininterval=1, desc="Epoch %5d" % epoch
        ):
            # Move batches to GPU if necessary.
            batch = model.module.batch_to_gpu(batch, self.device)

            # Reset gradients.
            model.module.optimizer.zero_grad(set_to_none=True)

            # Forward pass.
            output_dict = model(batch)

            # Calculate loss and perform backward pass.
            loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
            loss_dict["loss_total"].backward()

            # Update parameters.
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 100)
            model.module.optimizer.step()

            # Append the losses to the train_losses dictionary.
            train_losses = self.logs.append_batch_losses(train_losses, loss_dict)

        string = self.logs.result_string("train", epoch, train_losses, t=epoch)
        self.logs.write_to_log_file(string)
        self.logs.append_epoch_losses(train_losses, "train")

        model.module.scheduler.step()
        model.module.eval()

        return self.logs.train_results["loss_total"][-1]

    def predict(
        self, model: torch.nn.Module, data_batches: list = None, epoch: int = -1
    ):
        """
        Predict labels using the trained model.

        Args:
            model: The trained neural network model.
            data_batches (list, optional): List of data batches for prediction. Defaults to None.
            epoch (int, optional): The current epoch number. Defaults to None.

        Returns:
            tuple: A tuple containing two numpy arrays: (predictions, labels).

        Note:
            This method assumes that the model has already been trained and is in evaluation mode.

        """
        # Set the model to evaluation mode
        model.module.eval()

        # Initialize lists to store predictions and labels
        predictions, labels = [], []

        # Iterate over data batches for prediction
        for batch in tqdm(
            data_batches, leave=False, ncols=100, mininterval=1, desc="Predict"
        ):
            # Move batch to GPU
            batch = model.module.batch_to_gpu(batch, self.device)

            # Get predictions from the model
            out_dict = model.module.predictive_model(batch)
            prediction, label = out_dict["prediction"], out_dict["label"]

            # Convert predictions and labels to numpy arrays and store
            predictions.extend(prediction.detach().cpu().data.numpy())
            labels.extend(label.detach().cpu().data.numpy())

        # Return predictions and labels as numpy arrays
        return np.array(predictions), np.array(labels)

    def evaluate(
        self,
        model: torch.nn.Module,
        corpus: DataReader,
        set_name: str,
        data_batches: list = None,
        epoch: int = -1,
    ) -> dict:
        """
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
        """

        # Get the predictions and labels from the predict() method.
        concat_pred, concat_label = self.predict(model, data_batches, epoch=epoch)

        # Evaluate the predictions and labels using the pred_evaluate_method of the model.
        return model.module.pred_evaluate_method(
            concat_pred, concat_label, self.metrics
        )

    def fit_em_phases(
        self, model: torch.nn.Module, corpus: DataReader, epoch: int = -1
    ) -> float:
        """
        Perform training using the EM algorithm in multiple phases.

        Args:
            model: The neural network model to train.
            corpus: The data reader object containing the training data.
            epoch (int, optional): The current epoch number. Defaults to -1.

        Returns:
            float: The total loss after training.

        Note:
            This method assumes that the model's optimizer and scheduler have been initialized.

        """
        # Initialize optimizer and scheduler if not already done
        if model.module.optimizer is None:
            opt, sch = self._build_optimizer(model)
            (
                model.module.optimizer_infer,
                model.module.optimizer_gen,
                model.module.optimizer_graph,
            ) = opt
            (
                model.module.scheduler_infer,
                model.module.scheduler_gen,
                model.module.scheduler_graph,
            ) = sch
            model.module.optimizer = model.module.optimizer_infer

        # Iterate over training phases
        for phase in ["infer", "gen_graph"]:  # 'model', 'graph', 'infer', 'gen'
            model.module.train()

            if phase == "model":
                opt = [model.module.optimizer_infer, model.module.optimizer_gen]
                for param in self.graph_params:
                    param.requires_grad = False
                for param in self.generative_params + self.inference_params:
                    param.requires_grad = True
            elif phase == "graph":
                opt = [model.module.optimizer_graph]
                for param in self.generative_params + self.inference_params:
                    param.requires_grad = False
                for param in self.graph_params:
                    param.requires_grad = True
            elif phase == "infer":
                opt = [model.module.optimizer_infer]
                for param in self.generative_params + self.graph_params:
                    param.requires_grad = False
                for param in self.inference_params:
                    param.requires_grad = True
            elif phase == "gen":
                opt = [model.module.optimizer_gen]
                for param in self.inference_params + self.graph_params:
                    param.requires_grad = False
                for param in self.generative_params:
                    param.requires_grad = True
            elif phase == "infer_graph":
                opt = [model.module.optimizer_infer, model.module.optimizer_graph]
                for param in self.generative_params:
                    param.requires_grad = False
                for param in self.inference_params + self.graph_params:
                    param.requires_grad = True
            elif phase == "gen_graph":
                opt = [model.module.optimizer_gen, model.module.optimizer_graph]
                for param in self.inference_params:
                    param.requires_grad = False
                for param in self.generative_params + self.graph_params:
                    param.requires_grad = True

            # Perform training for the current phase
            for i in range(5):  # TODO: 5 is a hyperparameter
                loss = self.fit_one_phase(model, epoch=epoch, mini_epoch=i, opt=opt)

            # Evaluate the model after each phase
            self.test(model, corpus, train_loss=loss)

        # Update schedulers
        model.module.scheduler_infer.step()
        model.module.scheduler_graph.step()
        model.module.scheduler_gen.step()

        # Set model to evaluation mode
        model.module.eval()

        # Return the total loss after training
        return self.logs.train_results["loss_total"][-1]

    def fit_one_phase(
        self,
        model: torch.nn.Module,
        epoch: int = -1,
        mini_epoch: int = -1,
        phase: str = "infer",
        opt: list = None,
    ) -> float:
        """
        Perform one training phase for the given model.

        Args:
            model (torch.nn.Module): The neural network model to train.
            epoch (int, optional): The current epoch number. Defaults to -1.
            mini_epoch (int, optional): The current mini-epoch number. Defaults to -1.
            phase (str, optional): The phase of training. Defaults to "infer".
            opt (list, optional): List of optimizers to use for training. Defaults to None.

        Returns:
            float: The total loss after the training phase.

        """
        # Dictionary to store training losses
        train_losses = defaultdict(list)

        # Iterate over batches for training
        for batch in tqdm(
            self.whole_batches,
            leave=False,
            ncols=100,
            mininterval=1,
            desc="Epoch %5d" % epoch,
        ):
            # Zero out gradients for all optimizers
            model.module.optimizer_infer.zero_grad(set_to_none=True)
            model.module.optimizer_graph.zero_grad(set_to_none=True)
            model.module.optimizer_gen.zero_grad(set_to_none=True)

            # Forward pass
            output_dict = model(batch)
            loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)

            # Backward pass and optimization
            loss_dict["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 100)
            for o in opt:
                o.step()

            # Append the losses to the train_losses dictionary
            train_losses = self.logs.append_batch_losses(train_losses, loss_dict)

        # Generate result string and write to log file
        string = self.logs.result_string(
            "train", epoch, train_losses, t=epoch, mini_epoch=mini_epoch
        )
        self.logs.write_to_log_file(string)
        self.logs.append_epoch_losses(train_losses, "train")

        # Return the total loss after the training phase
        return self.logs.train_results["loss_total"][-1]


class VCLRunner(KTRunner):
    """
    Class implementing GVCL approach
    """

    def __init__(
        self,
        args,
        logs,
    ):
        """
        Args:
            args:
            logs
        """
        KTRunner.__init__(args=args, logs=logs)

        self.max_time_step = args.max_step

    def train(self, model: torch.nn.Module, corpus: DataReader) -> None:
        """
        Trains the KT model instance with parameters.

        Args:
            model: the KT model instance with parameters to train
            corpus: data
        """
        # Build the optimizer if it hasn't been built already.
        if model.module.optimizer is None:
            model.module.optimizer, model.module.scheduler = self._build_optimizer(
                model
            )

        assert corpus.data_df["train"] is not None
        self._check_time(start=True)

        if self.num_learner > 0:
            epoch_whole_data = copy.deepcopy(
                corpus.data_df["whole"][: self.num_learner]
            )
            epoch_whole_data["user_id"] = np.arange(self.num_learner)
        else:
            epoch_whole_data = copy.deepcopy(corpus.data_df["whole"])

        # Return a random sample of items from an axis of object.
        epoch_whole_data = epoch_whole_data.sample(frac=1).reset_index(drop=True)
        train_batches = model.module.prepare_batches(
            corpus, epoch_whole_data, self.batch_size, phase="whole"
        )
        eval_batches = model.module.prepare_batches(
            corpus, epoch_whole_data, self.eval_batch_size, phase="whole"
        )

        max_time_step = 100  # time_step

        try:
            for time in range(max_time_step):
                gc.collect()
                model.train()

                self._check_time()
                self.fit(model, train_batches, epoch=time, time_step=time)
                self.test(model, eval_batches, epoch=time, time_step=time)

        except KeyboardInterrupt:
            self.logs.write_to_log_file("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith("y"):
                self.logs.write_to_log_file(
                    os.linesep + "-" * 45 + " END: " + utils.get_time() + " " + "-" * 45
                )
                exit(1)

    def fit(
        self,
        model: torch.nn.Module,
        batches: list = None,
        epoch: int = 0,
        time_step: int = 0,
    ) -> dict:
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

        for mini_epoch in range(10):  # self.epoch):
            # Iterate through each batch.
            for batch in tqdm(
                batches,
                leave=False,
                ncols=100,
                mininterval=1,
                desc="Epoch %5d" % epoch + " Time %5d" % mini_epoch,
            ):
                # Move the batch to the GPU.
                batch = model.module.batch_to_gpu(batch, self.device)

                # Reset gradients.
                model.module.optimizer.zero_grad(set_to_none=True)

                # Predictive model before optimization
                s_tilde_dist, z_tilde_dist = model.module.predictive_model(
                    feed_dict=batch, idx=time_step
                )
                s_post_dist, z_post_dist = model.module.inference_model(
                    feed_dict=batch, idx=time_step
                )

                # Calculate loss and perform backward pass.
                output_dict = model.module.objective_function(
                    batch,
                    idx=time_step,
                    pred_dist=[s_tilde_dist, z_tilde_dist],
                    post_dist=[s_post_dist, z_post_dist],
                )
                loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
                loss_dict["loss_total"].backward()

                # Update parameters.
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), 100)
                model.module.optimizer.step()

                with torch.no_grad():
                    # Update after optimization
                    _, _ = model.module.inference_model(
                        feed_dict=batch, idx=time_step, update=True, eval=False
                    )
                    _, _ = model.module.predictive_model(
                        feed_dict=batch, idx=time_step, update=True, eval=False
                    )

                # Append the losses to the train_losses dictionary.
                train_losses = self.logs.append_batch_losses(train_losses, loss_dict)

            if mini_epoch % 10 == 0:
                model.module.save_model(epoch=epoch, mini_epoch=mini_epoch)
            self.logs.draw_loss_curves()

            string = self.logs.result_string(
                "train", epoch, train_losses, t=epoch, mini_epoch=mini_epoch
            )
            self.logs.write_to_log_file(string)
            self.logs.append_epoch_losses(train_losses, "train")

        return self.logs.train_results["loss_total"][-1]

    def test(
        self,
        model: torch.nn.Module,
        test_batches: list = None,
        epoch: int = 0,
        time_step: int = 0,
    ) -> None:
        """
        Evaluate the KT model instance on the test dataset.

        Args:
            model (torch.nn.Module): The KT model instance to evaluate.
            test_batches (list, optional): List of batches for the test dataset. Defaults to None.
            epoch (int, optional): The current epoch number. Defaults to 0.
            time_step (int, optional): The current time step. Defaults to 0.

        """
        # Dictionary to store test losses
        test_losses = defaultdict(list)

        # Set the model to evaluation mode
        model.module.eval()

        # If testing is enabled and the current epoch is a multiple of test_every
        if self.args.test and epoch % self.args.test_every == 0:
            with torch.no_grad():
                # Iterate over test batches and evaluate
                for batch in test_batches:
                    batch = model.module.batch_to_gpu(batch, self.device)
                    loss_dict = model.module.eval_model(batch, time_step)
                    test_losses = self.logs.append_batch_losses(test_losses, loss_dict)

            # Generate result string and write to log file
            string = self.logs.result_string("test", epoch, test_losses, t=epoch)
            self.logs.write_to_log_file(string)
            self.logs.append_epoch_losses(test_losses, "test")

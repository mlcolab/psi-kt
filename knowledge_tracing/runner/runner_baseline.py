import gc, copy, os, argparse
from collections import defaultdict

from tqdm import tqdm
import numpy as np

import torch

from knowledge_tracing.utils import utils, logger
from knowledge_tracing.data.data_loader import DataReader
from knowledge_tracing.runner.runner import KTRunner


class BaselineKTRunner(KTRunner):
    """
    This implements the training loop, testing & validation, optimization etc.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        logs: logger.Logger,
    ) -> None:
        """
        Initialize the BaselineKTRunner instance.

        Args:
            args (argparse.Namespace): Global arguments provided as a namespace.
            logs (logger.Logger): The Logger instance for logging information.
        """
        KTRunner.__init__(self, args, logs)
        self.time = None

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
        # Ensure that training data is available
        assert corpus.data_df["train"] is not None
        # Mark the start time of the training
        self.start_time = self._check_time(start=True)

        # Prepare training data (if needs quick test then specify num_learner arguments in the args);
        set_name = ["train", "val", "test"]
        if self.num_learner > 0:
            # If specified, limit the data to the first 'num_learner' entries
            epoch_train_data, epoch_val_data, epoch_test_data = [
                copy.deepcopy(corpus.data_df[key][: self.num_learner])
                for key in set_name
            ]
        else:
            epoch_train_data, epoch_val_data, epoch_test_data = [
                copy.deepcopy(corpus.data_df[key]) for key in set_name
            ]

        # Shuffle training data
        epoch_train_data = epoch_train_data.sample(frac=1).reset_index(drop=True)
        # Prepare data batches for training and optionally for validation and testing
        self.train_batches = model.module.prepare_batches(
            corpus, epoch_train_data, self.batch_size, phase="train"
        )
        self.val_batches = None
        self.test_batches = None

        # Prepare validation and test batches if respective flags are set
        if self.args.test:
            self.test_batches = model.module.prepare_batches(
                corpus, epoch_test_data, self.eval_batch_size, phase="test"
            )
        if self.args.validate:
            self.val_batches = model.module.prepare_batches(
                corpus, epoch_val_data, self.eval_batch_size, phase="val"
            )

        try:
            for epoch in range(self.epoch):
                # Collect garbage to free up memory
                gc.collect()

                # Set model to training mode
                model.module.train()

                # Check and print the elapsed time
                self._check_time()

                # Perform a single epoch of training
                loss = self.fit(model, epoch=epoch)
                # Optionally perform testing after each epoch
                _ = self.test(model, corpus, epoch, loss)

                # Save the model periodically
                if epoch % self.args.save_every == 0:
                    model.module.save_model(epoch=epoch)

                # Early stopping based on validation performance
                if self.early_stop:
                    if self._eva_termination(
                        model, self.metrics, self.logs.val_results
                    ):
                        self.logs.write_to_log_file(
                            "Early stop at %d based on validation result." % (epoch)
                        )
                        break

                # Draw loss curves
                self.logs.draw_loss_curves()

        except KeyboardInterrupt:
            # Handle manual interruption of training
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
    ) -> tuple:
        """
        Perform testing on the model.

        Args:
            model: The torch.nn.Module model to test.
            corpus: The DataReader object containing the test data.
            epoch: The current epoch number.
            train_loss: The training loss.

        Returns:
            Tuple containing the test and validation results (test_result, val_result).
        """

        training_time = self._check_time()
        val_result, test_result = None, None

        with torch.no_grad():
            if self.args.validate:
                val_result = self.evaluate(
                    model, corpus, "val", self.val_batches, epoch=epoch
                )

                self.logs.append_epoch_losses(val_result, "val")
                if (
                    max(self.logs.val_results[self.metrics[0]])
                    == val_result[self.metrics[0]]
                ):
                    model.module.save_model(epoch=epoch)

                val_time = self._check_time()
                self.logs.write_to_log_file(
                    "Epoch {:<3} loss={:<.4f} [{:<.1f} s]\t valid=({}) [{:<.1f} s] ".format(
                        epoch,
                        train_loss,
                        training_time,
                        utils.format_metric(val_result),
                        val_time,
                    )
                )

            if (
                (self.args.test)
                & (epoch % self.args.test_every == 0)
                & (epoch >= self.early_stop)
            ):
                test_result = self.evaluate(
                    model, corpus, "test", self.test_batches, epoch=epoch
                )

                testing_time = self._check_time()
                self.logs.append_epoch_losses(test_result, "test")
                self.logs.write_to_log_file(
                    "Epoch {:<3} loss={:<.4f} [{:<.1f} s]\t test=({}) [{:<.1f} s] ".format(
                        epoch,
                        train_loss,
                        training_time,
                        utils.format_metric(test_result),
                        testing_time,
                    )
                )

        return test_result, val_result

    def fit(
        self,
        model: torch.nn.Module,
        epoch: int = 0,
    ) -> dict:
        """
        Trains the given model on the given batches of data.

        Args:
            model: The model to train.
            epoch_train_data: A pandas DataFrame containing the training data.
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

        outputs = []
        # Iterate through each batch.
        for batch in tqdm(
            self.train_batches,
            leave=False,
            ncols=100,
            mininterval=1,
            desc="Epoch %5d" % epoch,
        ):
            batch = model.module.batch_to_gpu(batch, self.device)

            # Reset gradients.
            model.module.optimizer.zero_grad(set_to_none=True)

            # Forward pass.
            output_dict = model(batch)
            outputs.append(output_dict)

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
        self,
        model: torch.nn.Module,
        corpus: DataReader,
        set_name: str,
        data_batches: list = None,
        epoch: int = 0,
    ) -> tuple:
        """
        Performs prediction on a specified dataset using the provided model.

        Args:
            model: The trained model instance used for making predictions.
            corpus: The data container from which the data batches are prepared.
            set_name: A string indicating the name of the dataset to predict on (e.g., "train", "val", "test").
            data_batches: A list of data batches prepared for prediction. If None, batches will be prepared based on set_name.
            epoch: The current epoch number, used for logging or tracking. Not directly used in prediction.

        Returns:
            predictions: A numpy array containing the predicted values.
            labels: A numpy array containing the true labels corresponding to the predictions.
        """
        # Ensure the model is in evaluation mode to disable dropout or batch normalization effects during inference
        model.module.eval()

        # Initialize lists to store predictions, labels, and raw outputs
        predictions, labels, outputs = [], [], []

        # Iterate over each batch in the dataset
        for batch in tqdm(
            data_batches, leave=False, ncols=100, mininterval=1, desc="Predict"
        ):
            # Move the batch data to the appropriate device (GPU or CPU)
            batch = model.module.batch_to_gpu(batch, self.device)
            # Perform prediction using the model
            out_dict = model.module.predictive_model(batch)
            # Collect the raw outputs for further analysis or debugging
            outputs.append(out_dict)

            # Extract predictions and labels from the output dictionary
            prediction, label = out_dict["prediction"], out_dict["label"]
            # Detach predictions and labels from the computation graph and move to CPU, converting to numpy arrays
            predictions.extend(prediction.detach().cpu().data.numpy())
            labels.extend(label.detach().cpu().data.numpy())

        # Convert lists of predictions and labels to numpy arrays for easier handling
        return np.array(predictions), np.array(labels)

    def evaluate(
        self,
        model: torch.nn.Module,
        corpus: DataReader,
        set_name: str,
        data_batches: list = None,
        epoch: int = 0,
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
        predictions, labels = self.predict(
            model, corpus, set_name, data_batches, epoch=epoch
        )

        # Get the lengths of the sequences in the input dataset.
        lengths = np.array(
            list(map(lambda lst: len(lst) - 1, corpus.data_df[set_name]["skill_seq"]))
        )

        # Concatenate the predictions and labels into arrays.
        concat_pred, concat_label = [], []
        for pred, label, _ in zip(predictions, labels, lengths):
            concat_pred.append(pred)
            concat_label.append(label)
        concat_pred = np.concatenate(concat_pred)
        concat_label = np.concatenate(concat_label)

        # Evaluate the predictions and labels using the pred_evaluate_method of the model.
        return model.module.pred_evaluate_method(
            concat_pred, concat_label, self.metrics
        )


class BaselineContinualRunner(BaselineKTRunner):
    """
    This implements the training loop, testing & validation, optimization etc.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        logs: logger.Logger,
    ) -> None:
        """
        Initialize the BaselineKTRunner instance.

        Args:
            args (argparse.Namespace): Global arguments provided as a namespace.
            logs (logger.Logger): The Logger instance for logging information.
        """

        self.time = None

        # number of data to train
        self.num_learner = args.num_learner

        # training options
        self.epoch = args.epoch
        self.batch_size = args.batch_size_multiGPU
        self.eval_batch_size = args.eval_batch_size

        # list of evaluation metrics to use during training
        self.metrics = args.metric.strip().lower().split(",")
        for i in range(len(self.metrics)):
            self.metrics[i] = self.metrics[i].strip()

        self.args = args
        self.early_stop = args.early_stop
        self.logs = logs
        self.device = args.device

    def train(
        self,
        model: torch.nn.Module,
        corpus: DataReader,
    ) -> None:
        """
        Initiates the training process for the KT model with the given dataset.

        This method sets up the optimizer if it hasn't been already, prepares the data,
        and iterates through a training loop to fit the model on the entire dataset for
        a specified number of epochs or until early stopping is triggered.

        Args:
            model: The KT model instance to be trained.
            corpus: The dataset wrapper used for training.
        """
        # Build the optimizer if it hasn't been built already.
        if model.module.optimizer is None:
            model.module.optimizer, model.module.scheduler = self._build_optimizer(
                model
            )

        # Ensure the training data exists
        assert corpus.data_df["train"] is not None

        # Record the start time for the training process
        self._check_time(start=True)

        # Prepare the dataset. If num_learner is specified, use a subset of the data.
        if self.num_learner > 0:
            epoch_whole_data = copy.deepcopy(
                corpus.data_df["whole"][: self.num_learner]
            )
            epoch_whole_data["user_id"] = np.arange(self.num_learner)
        else:
            epoch_whole_data = copy.deepcopy(corpus.data_df["whole"])

        # Shuffle the dataset
        epoch_whole_data = epoch_whole_data.sample(frac=1).reset_index(drop=True)
        # Prepare batches from the shuffled dataset for training
        whole_batches = model.module.prepare_batches(
            corpus, epoch_whole_data, self.eval_batch_size, phase="whole"
        )

        try:
            # Main training loop
            for time in range(1, 100):
                # Clean up memory
                gc.collect()
                # Set model to training mode
                model.train()

                # Check and log the time
                self._check_time()
                # Fit the model on the current batch
                self.fit(model, whole_batches, epoch=time, time_step=time)
                # Perform testing using the same batches
                self.test(model, whole_batches, epoch=time, time_step=time)

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
        batches: list,
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

                output_dict = model.module.forward_cl(batch, idx=time_step)
                loss_dict = model.module.loss(batch, output_dict, metrics=self.metrics)
                loss_dict["loss_total"].backward()

                # Update parameters.
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), 100)
                model.module.optimizer.step()

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
        Evaluates the model on a test dataset.

        This method is called at specified epochs to evaluate the model's performance on the test dataset.
        It computes the loss and other metrics defined in the initialization of the runner. The results are
        logged for each epoch.

        Args:
            model: The KT model instance to be evaluated.
            test_batches: A list of data batches for testing. Each batch is a portion of the test dataset.
            epoch: The current epoch of the training process. Used for logging and to decide if testing should occur.
            time_step: The current time step in the training process. Used for continual learning evaluations.

        Returns:
            None: This method logs the test results and does not return any values.
        """
        # Dictionary to store test losses and metrics
        test_losses = defaultdict(list)

        # Set the model to evaluation mode
        model.module.eval()

        # Check if testing is enabled and it's the correct epoch for testing
        if (self.args.test) and (epoch % self.args.test_every == 0):
            with torch.no_grad():  # Disable gradient computation for evaluation
                for batch in test_batches:  # Iterate through each test batch
                    # Move the batch to the appropriate device (GPU/CPU)
                    batch = model.module.batch_to_gpu(batch, self.device)
                    # Evaluate the model on the batch and obtain loss/metrics
                    loss_dict = model.module.evaluate_cl(batch, time_step, self.metrics)
                    # Append batch losses/metrics to the test losses
                    test_losses = self.logs.append_batch_losses(test_losses, loss_dict)

            # Generate a result string for the current epoch's test performance
            string = self.logs.result_string("test", epoch, test_losses, t=epoch)
            self.logs.write_to_log_file(string)
            self.logs.append_epoch_losses(test_losses, "test")

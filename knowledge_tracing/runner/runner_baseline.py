import sys

sys.path.append("..")

import gc, copy, os, argparse
from time import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from knowledge_tracing.utils import utils, logger
from knowledge_tracing.data.data_loader import DataReader
from knowledge_tracing.runner.ktrunner import KTRunner

OPTIMIZER_MAP = {
    "gd": optim.SGD,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "adam": optim.Adam,
}


class BaselineKTRunner(KTRunner):
    """
    This implements the training loop, testing & validation, optimization etc.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        logs: logger.Logger,
    ):
        """
        Initialize the BaselineKTRunner instance.

        Args:
            args (argparse.Namespace): Global arguments provided as a namespace.
            logs (logger.Logger): The Logger instance for logging information.
        """

        self.time = None

        # number of data to train
        self.overfit = args.overfit

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

    def _eva_termination(
        self,
        model: torch.nn.Module,
        metrics_list: list = None,
        metrics_log: dict = None,
    ) -> bool:
        """
        Determine whether the training should be terminated based on the validation results.

        Returns:
        - True if the training should be terminated, False otherwise
        """

        for m in metrics_list:
            valid = list(metrics_log[m])

            # Check if the last 10 validation results have not improved
            if not (len(valid) > 10 and utils.non_increasing(valid[-10:])):
                return False
            # Check if the maximum validation result has not improved for the past 10 epochs
            elif not (len(valid) - valid.index(max(valid)) > 10):
                return False

        return True

    def train(
        self,
        model: torch.nn.Module,
        corpus: DataReader,
    ):
        """
        Trains the KT model instance with parameters.

        Args:
            model: the KT model instance with parameters to train
            corpus: data
        """
        assert corpus.data_df["train"] is not None
        self.start_time = self._check_time(start=True)

        # Prepare training data (if needs quick test then specify overfit arguments in the args);
        set_name = ["train", "val", "test", "whole"]
        if self.overfit > 0:
            epoch_train_data, epoch_val_data, epoch_test_data, epoch_whole_data = [
                copy.deepcopy(corpus.data_df[key][: self.overfit]) for key in set_name
            ]
        else:
            epoch_train_data, epoch_val_data, epoch_test_data, epoch_whole_data = [
                copy.deepcopy(corpus.data_df[key]) for key in set_name
            ]

        # Return a random sample of items from an axis of object.
        epoch_train_data = epoch_train_data.sample(frac=1).reset_index(drop=True)
        self.train_batches = model.module.prepare_batches(
            corpus, epoch_train_data, self.batch_size, phase="train"
        )
        self.val_batches = None
        self.test_batches = None

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
                gc.collect()
                model.module.train()

                self._check_time()

                loss = self.fit(model, epoch=epoch)
                test = self.test(model, corpus, epoch, loss)

                if epoch % self.args.save_every == 0:
                    model.module.save_model(epoch=epoch)

                if self.early_stop:
                    if self._eva_termination(
                        model, self.metrics, self.logs.val_results
                    ):
                        self.logs.write_to_log_file(
                            "Early stop at %d based on validation result." % (epoch)
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
    ):
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

            if (self.args.test) & (epoch % self.args.test_every == 0):
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
    ):
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

    def predict(self, model, corpus, set_name, data_batches=None, epoch=None):
        """
        Args:
            model:
        """
        model.module.eval()

        predictions, labels, outputs = [], [], []

        for batch in tqdm(
            data_batches, leave=False, ncols=100, mininterval=1, desc="Predict"
        ):
            batch = model.module.batch_to_gpu(batch, self.device)
            out_dict = model.module.predictive_model(batch)
            outputs.append(out_dict)

            prediction, label = out_dict["prediction"], out_dict["label"]
            predictions.extend(prediction.detach().cpu().data.numpy())
            labels.extend(label.detach().cpu().data.numpy())

        # import pickle
        # import ipdb
        # filehandler = open("/mnt/qb/work/mlcolab/hzhou52/0iclr_exp3_mi_learner_and_emb/10-20/akt_junyi15_1000_2023_260_test.obj","wb")
        # pickle.dump(outputs,filehandler)
        # filehandler.close()

        return np.array(predictions), np.array(labels)

    def evaluate(self, model, corpus, set_name, data_batches=None, epoch=None):
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
        for pred, label, length in zip(predictions, labels, lengths):
            concat_pred.append(pred)
            concat_label.append(label)
        concat_pred = np.concatenate(concat_pred)
        concat_label = np.concatenate(concat_label)

        # Evaluate the predictions and labels using the pred_evaluate_method of the model.
        return model.module.pred_evaluate_method(
            concat_pred, concat_label, self.metrics
        )

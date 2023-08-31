import time, os, argparse, math, itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

import torch

import ipdb


class Logger:
    def __init__(
        self,
        args: argparse.Namespace
    ):
        self.args = args
        
        self.train_results = pd.DataFrame()
        self.test_results = pd.DataFrame()
        self.val_results = pd.DataFrame()
        
        if args.create_logs:
            self.create_log_path(args)


    @staticmethod 
    def append_batch_losses(
        losses_list: dict,
        losses: dict,
    ):
        """
        # TODO is it necesary to make this static?
        Appends the losses dictionary to the losses_list.

        Args:
            losses_list -- the list of losses to append to
            losses -- the dictionary of losses to append
        """
        for loss, value in losses.items():
            if isinstance(value, float):
                losses_list[loss].append(value)
            elif isinstance(value, defaultdict):
                if not losses_list[loss]:
                    losses_list[loss] = defaultdict(list)
                for idx, elem in value.items():
                    losses_list[loss][idx].append(elem)
            else:
                losses_list[loss].append(value.item())
        return losses_list
    
    
    def create_log_path(
        self, 
        args: argparse.Namespace,
        add_path_var: str = "",
    ):
        """
        Creates a log path for saving files related to the experiment.

        Args:
            args: An object containing various arguments.
            add_path_var: Additional path variable to be included in the log path.
        """
        args.log_path = os.path.join(args.save_folder, add_path_var, args.model_name, args.dataset,
                                     f"{args.time}_{args.expername}_overfit_{args.overfit}")
        os.makedirs(args.log_path, exist_ok=True)

        self.log_file = os.path.join(args.log_path, "log.txt")
        self.write_to_log_file(args)

        args.plotdir = os.path.join(args.log_path, "plots")
        os.makedirs(args.plotdir, exist_ok=True)
        args.visdir = os.path.join(args.log_path, "out_dict")
        os.makedirs(args.visdir, exist_ok=True)


    def write_to_log_file(
        self, 
        string: str,
    ):
        """
        Write given string in log-file and print as terminal output
        """
        if not isinstance(string, str):
            string = str(string)
        cur_file = open(self.log_file, "a")
        cur_file.write(string)
        cur_file.write("\n")
        cur_file.close()


    def append_epoch_losses(
        self, 
        loss_dict: dict,
        phase: str = 'train',
    ):
        '''
        Append loss results to corresponding data frame
        '''
        if phase == 'train':
            results_df = self.train_results
        elif phase == 'val':
            results_df = self.val_results
        elif phase == 'test':
            results_df = self.test_results
        else:
            raise ValueError('Invalid result type: ' + phase)

        results_idx = len(results_df)
        for k, v in loss_dict.items():
            results_df.at[str(results_idx), k] = np.mean(v)
        
        
    def result_string(
        self, 
        phase: str, 
        epoch: int, 
        losses: dict, 
        t: float = None, 
        mini_epoch: int = None,
    ): 
        """
        Generates a string representation of the results.

        Args:
            phase: Specifies whether it is for training, validation, or testing.
            epoch: The current epoch number.
            losses: The dictionary of losses.
            t: Optional time value for performance measurement.
            mini_epoch: Optional mini-epoch identifier.

        Returns:
            The string representation of the results.
        """
        
        string = ""
        
        if phase == "test":
            string += (
                "-------------------------------- \n"
                "--------Testing----------------- \n"
                "-------------------------------- \n"
            )
        else:
            string += str(epoch) + " " + str(mini_epoch) + phase + "\t \t"

        for loss, value in losses.items():
            if type(value) == defaultdict:
                string += loss + " "
                for idx, elem in sorted(value.items()):
                    string += str(idx) + ": {:.10f} \t".format(
                        np.mean(list(itertools.chain.from_iterable(elem)))
                    )
            elif np.mean(value) != 0 and not math.isnan(np.mean(value)):
                string += loss + " {:.10f} \t".format(np.mean(value))

        if t is not None:
            string += "time: {:.4f}s \t".format(time.time() - t)

        return string
    
    
    def draw_loss_curves(self):
        """
        Draw loss curves for train, validation, and test results.

        This method plots the loss curves based on the train, validation, and test results stored in the class.

        """

        for i in self.train_results.columns:
            plt.figure()
            plt.plot(self.train_results[i], "-b", label="train " + i)

            if self.val_results is not None and i in self.val_results:
                plt.plot(self.val_results[i], "-r", label="val " + i)

            if self.test_results is not None and i in self.test_results:
                plt.plot(self.test_results[i], "-g", label="test " + i)

            plt.xlabel("epoch")
            plt.ylabel(i)
            plt.legend(loc="upper right")

            # save image
            filename = f'train_{i}.png'
            filepath = os.path.join(self.args.plotdir, filename)
            plt.savefig(filepath)
            plt.close()
            

    def save_checkpoint(
        self, 
        args: argparse.Namespace,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        specifier: str = "",
    ):
        """
        Save model and optimizer checkpoints at specified path and specifier.

        Args:
            args: object containing relevant training parameters
            model: trained PyTorch model
            optimizer: optimizer used during training
            specifier: optional string specifier to differentiate checkpoints
        """
        # Set file paths for saving model and optimizer state dicts
        model_file_path = os.path.join(args.log_path, f"model_{specifier}.pt")
        optimizer_file_path = os.path.join(args.log_path, f"optimizer_{specifier}.pt")

        # Save model state dict if model exists
        if model is not None:
            torch.save(model.state_dict(), model_file_path)

        # Save optimizer state dict if optimizer exists
        if optimizer is not None:
            torch.save(optimizer.state_dict(), optimizer_file_path)

            
    def create_log( 
        self,
        args: argparse.Namespace,
        accuracy: float = None,
        model: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        final_test: bool = False,
        test_results: pd.DataFrame = None,
        specifier: str = "",
    ):
        """
        Create a log file and save the model and results.

        This method saves the losses throughout training, accuracy, and test results in a log file.
        It also saves the model checkpoint.

        """

        print("Saving model and log-file to " + args.log_path)

        # Save losses throughout training and plot
        self.train_results.to_pickle(os.path.join(self.args.log_path, "out_dict", "train_loss"))

        if self.val_results is not None:
            self.val_results.to_pickle(os.path.join(self.args.log_path, "out_dict", "val_loss"))

        if self.test_results is not None:
            self.test_results.to_pickle(os.path.join(self.args.log_path, "out_dict", "test_loss"))
            
        if accuracy is not None:
            np.save(os.path.join(self.args.log_path, "accuracy"), accuracy)

        if final_test:
            pd_test_results = pd.DataFrame(
                [
                    [k] + [np.mean(v)]
                    for k, v in test_results.items()
                    if type(v) != defaultdict
                ],
                columns=["loss", "score"],
            )
            pd_test_results.to_pickle(os.path.join(self.args.log_path, "out_dict", "test_results"))

            pd_test_results_per_influenced = pd.DataFrame(
                list(
                    itertools.chain(
                        *[
                            [
                                [k]
                                + [idx]
                                + [np.mean(list(itertools.chain.from_iterable(elem)))]
                                for idx, elem in sorted(v.items())
                            ]
                            for k, v in test_results.items()
                            if type(v) == defaultdict
                        ]
                    )
                ),
                columns=["loss", "num_influenced", "score"],
            )
            pd_test_results_per_influenced.to_pickle(
                os.path.join(args.log_path, "out_dict", "test_loss_per_influenced")
            )
            specifier = "final"

        # Save the model checkpoint
        self.save_checkpoint(args, model, optimizer, specifier=specifier)
    

    # def draw_val_curve(self):
    #     for i in self.val_results.columns:
    #         plt.figure()
    #         plt.plot(self.train_results[i], "-b", label="train " + i)

    #         if self.val_results is not None and i in self.val_results:
    #             plt.plot(self.val_results[i], "-r", label="val " + i)

    #         plt.xlabel("epoch")
    #         plt.ylabel("loss")
    #         plt.legend(loc="upper right")

    #         # save image
    #         plt.savefig(os.path.join(self.args.log_path, 'train_' + i + ".png"))
    #         plt.close()
    
    # def draw_tta_curves(self):
    #     for i in self.val_results.columns:
    #         if 'tta_ori_' in i:
    #             plt.figure()
    #             plt.plot(self.val_results[i], "-b", label="val " + i)
    #             plt.plot(self.val_results[i.replace('_ori_', '_')], "-r", label="val " + i.replace('_ori_', '_'))

    #             plt.xlabel("epoch")
    #             plt.ylabel("loss")
    #             plt.legend(loc="upper right")

    #             # save image
    #             plt.savefig(os.path.join(self.args.log_path, i + ".png"))
    #             plt.close()
    #         if 'tta' not in i:
    #             plt.figure()
    #             plt.plot(self.val_results[i], "-r", label="val " + i)

    #             plt.xlabel("epoch")
    #             plt.ylabel("loss")
    #             plt.legend(loc="upper right")

    #             # save image
    #             plt.savefig(os.path.join(self.args.log_path, 'val_' + i + ".png"))
    #             plt.close()
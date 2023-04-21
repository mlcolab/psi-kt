import time
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from collections import defaultdict
import itertools
import ipdb


class Logger:
    def __init__(self, args):
        self.args = args
        self.num_models_to_keep = 1
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"
        
        self.train_results = pd.DataFrame()
        self.test_results = pd.DataFrame()
        self.valid_results = pd.DataFrame()
        
        self.create_log_path(args)


    @staticmethod
    def append_batch_losses(losses_list, losses):
        """
        Appends the losses dictionary to the losses_list.

        Arguments:
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
    
    
    def create_log_path(self, args, add_path_var=""):
        args.log_path = os.path.join(args.save_folder, add_path_var, args.model_name, args.dataset,
                                     f"{args.time}_{args.expername}_overfit_{args.overfit}")
        os.makedirs(args.log_path, exist_ok=True)

        self.log_file = os.path.join(args.log_path, "log.txt")
        self.write_to_log_file(args)

        args.optimizer_file = os.path.join(args.log_path, "optimizer.pt")

        args.plotdir = os.path.join(args.log_path, "plots")
        os.makedirs(args.plotdir, exist_ok=True)
        args.visdir = os.path.join(args.log_path, "out_dict")
        os.makedirs(args.visdir, exist_ok=True)


    def write_to_log_file(self, string):
        """
        Write given string in log-file and print as terminal output
        """
        if not isinstance(string, str):
            string = str(string)
        cur_file = open(self.log_file, "a")
        cur_file.write(string)
        cur_file.write("\n")
        cur_file.close()


    def append_epoch_losses(self, loss_dict, phase='train'):
        '''
        Append loss results to corresponding data frame
        '''
        if phase == 'train':
            results_df = self.train_results
        elif phase == 'val':
            results_df = self.valid_results
        elif phase == 'test':
            results_df = self.test_results
        else:
            raise ValueError('Invalid result type: ' + phase)

        # ipdb.set_trace()
        results_idx = len(results_df)
        for k, v in loss_dict.items():
            results_df.at[str(results_idx), k] = np.mean(v)
        
        
    def result_string(self, trainvaltest, epoch, losses, t=None, mini_epoch=''): # TODO
        string = ""
        if trainvaltest == "test":
            string += (
                "-------------------------------- \n"
                "--------Testing----------------- \n"
                "-------------------------------- \n"
            )
        else:
            string += str(epoch) + " " + str(mini_epoch) + trainvaltest + "\t \t"

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
        # ipdb.set_trace()
        for i in self.train_results.columns:
            plt.figure()
            plt.plot(self.train_results[i], "-b", label="train " + i)

            if self.valid_results is not None and i in self.valid_results:
                plt.plot(self.valid_results[i], "-r", label="val " + i)

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
            
    
    def save_checkpoint(self, args, optimizer, specifier=""):
        args.decoder_file = os.path.join(args.log_path, "decoder" + specifier + ".pt")
        args.optimizer_file = os.path.join(
            args.log_path, "optimizer" + specifier + ".pt"
        )
        if decoder is not None:
            torch.save(decoder.state_dict(), args.decoder_file)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), args.optimizer_file)
            
            
    def create_log( 
        self,
        args,
        accuracy=None,
        model=None,
        optimizer=None,
        final_test=False,
        test_results=None,
        specifier="",
    ):

        print("Saving model and log-file to " + args.log_path)

        # Save losses throughout training and plot
        self.train_results.to_pickle(os.path.join(self.args.log_path, "out_dict", "train_loss"))

        if self.valid_results is not None:
            self.valid_results.to_pickle(os.path.join(self.args.log_path, "out_dict", "val_loss"))

        if self.test_results is not None:
            self.test_results.to_pickle(os.path.join(self.args.log_path, "out_dict", "test_loss"))
            
        if accuracy is not None:
            np.save(os.path.join(self.args.log_path, "accuracy"), accuracy)

        # specifier = ""
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
        self.save_checkpoint(args, optimizer, specifier=specifier)
    
    


    # def draw_val_curve(self):
    #     for i in self.valid_results.columns:
    #         plt.figure()
    #         plt.plot(self.train_results[i], "-b", label="train " + i)

    #         if self.valid_results is not None and i in self.valid_results:
    #             plt.plot(self.valid_results[i], "-r", label="val " + i)

    #         plt.xlabel("epoch")
    #         plt.ylabel("loss")
    #         plt.legend(loc="upper right")

    #         # save image
    #         plt.savefig(os.path.join(self.args.log_path, 'train_' + i + ".png"))
    #         plt.close()
    
    # def draw_tta_curves(self):
    #     for i in self.valid_results.columns:
    #         if 'tta_ori_' in i:
    #             plt.figure()
    #             plt.plot(self.valid_results[i], "-b", label="val " + i)
    #             plt.plot(self.valid_results[i.replace('_ori_', '_')], "-r", label="val " + i.replace('_ori_', '_'))

    #             plt.xlabel("epoch")
    #             plt.ylabel("loss")
    #             plt.legend(loc="upper right")

    #             # save image
    #             plt.savefig(os.path.join(self.args.log_path, i + ".png"))
    #             plt.close()
    #         if 'tta' not in i:
    #             plt.figure()
    #             plt.plot(self.valid_results[i], "-r", label="val " + i)

    #             plt.xlabel("epoch")
    #             plt.ylabel("loss")
    #             plt.legend(loc="upper right")

    #             # save image
    #             plt.savefig(os.path.join(self.args.log_path, 'val_' + i + ".png"))
    #             plt.close()


    
    

        
        
    # def append_train_loss(self, loss):
    #     for k, v in loss.items():
    #         self.train_results.at[str(self.train_results_idx), k] = np.mean(v)
    #     self.train_results_idx += 1

    # def append_val_loss(self, val_loss):
    #     for k, v in val_loss.items():
    #         self.valid_results.at[str(self.valid_results_idx), k] = np.mean(v)
    #     self.valid_results_idx += 1

    # def append_test_loss(self, test_loss):
    #     for k, v in test_loss.items():
    #         if type(v) != defaultdict:
    #             self.test_results.at[str(self.test_results_idx), k] = np.mean(v)
    #     self.test_results_idx += 1










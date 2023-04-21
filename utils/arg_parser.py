import argparse
import datetime
import numpy as np


def parse_args(parser):
    '''
    All of the general arguments defined here.
    Model-specific arguments are defined in corresponding files.
    '''
    ############## TODO debugging experiment ############## 
    parser.add_argument('--overfit', type=int, default=100, help='whether to overfit and debug (0/>0); if overfit>0, the number of data to train')
    parser.add_argument('--gt_adj_path', type=str)

    ############## distributed training ############## 
    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
    parser.add_argument('--distributed', default=1, type=int, help='')

    ############## global ############## from main.py
    parser.add_argument('--random_seed', type=int, default=2022,)
    parser.add_argument("--GPU_to_use", type=int, default=None, help="GPU to use for training")
    parser.add_argument('--gpu', type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')

    ############## data loader ##############
    parser.add_argument('--data_dir', type=str, default='/mnt/lustre/mlcolab/hzhou52/kt', help='Input data dir.')
    parser.add_argument('--dataset', type=str, help='[junyi, assistment12, ]')
    parser.add_argument('--sep', type=str, default='\t', help='sep of csv file.')
    parser.add_argument('--kfold', type=int, default=5, help='K-fold number.')
    parser.add_argument('--max_step', type=int, default=50, help='Max time steps per sequence.')
    parser.add_argument('--regenerate_corpus', action="store_true", default=False)

    ############## logger ##############
    parser.add_argument("--save_folder", type=str, default="/mnt/lustre/mlcolab/hzhou52/kt/logs",)
    parser.add_argument("--save_every", type=int, default=10,)
    parser.add_argument("--expername", type=str, default="",)

    ############## KTRunner ##############
    parser.add_argument('--train', type=int, default=1, help='To train the model or not.')
    parser.add_argument(
                '--train_mode', type=str, default='train_split_time', 
                help= 'simple_split_time' + 'simple_split_learner' 
                + 'ls_split_time' 
                + 'ns_split_time' + 'ns_split_learner'
                + 'ln_split_time', 
            ) #
    parser.add_argument("--test", default=0, type=int, help="test results throughout training.")
    parser.add_argument("--test_every", action="store_true", default=5, help="test results throughout training.")
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--early_stop', type=int, default=1, help='whether to early-stop.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size during training.')
    parser.add_argument('--eval_batch_size', type=int, default=512, help='Batch size during testing.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for each deep layer')
    parser.add_argument('--l2', type=float, default=1e-5, help='Weight of l2_regularize in loss.')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer: GD, Adam, Adagrad, Adadelta')
    parser.add_argument('--metric', type=str, default='F1, Accuracy, Recall, Precision, AUC', 
                        help='metrics: AUC, F1, Accuracy, Recall, Presicion;'
                                'The first one will be used to determine whether to early stop')
    parser.add_argument('--fold', type=int, default=0, help='Select a fold to run.')
    
    
    ############## load and save model ##############
    parser.add_argument('--load', type=int, default=0, help='Whether load model and continue to train')
    parser.add_argument(
        "--load_folder",
        type=str,
        default="",
        help="Where to load pre-trained model if finetuning/evaluating. "
        + "Leave empty to train from scratch",
    )
    

    ############## training hyperparameter ##############
    parser.add_argument("--lr_decay", type=int, default=5000, help="After how epochs to decay LR by a factor of gamma.",)
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor.")
    parser.add_argument("--var", type=float, default=5e-2, help="Output variance.") # TODO
    
    return parser 

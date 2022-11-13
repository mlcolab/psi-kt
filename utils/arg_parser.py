import argparse
import torch
import datetime
import numpy as np


def parse_args(parser):
    # parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', type=int, default=100,)

    parser.add_argument('--world_size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    # parser.add_argument('--distributed', default=1, type=int, 
    #                     help='')


    ############## global ############## from main.py
    parser.add_argument('--random_seed', type=int, default=2022,)
    parser.add_argument("--GPU_to_use", type=int, default=None, help="GPU to use for training")
    parser.add_argument('--gpu', type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')

    ############## data loader ##############
    parser.add_argument('--data_dir', type=str, default='/mnt/qb/work/mlcolab/hzhou52/kt', help='Input data dir.')
    parser.add_argument('--dataset', type=str, default='test', help='[junyi, assistment12, ]')
    parser.add_argument('--sep', type=str, default='\t', help='sep of csv file.')
    parser.add_argument('--kfold', type=int, default=5, help='K-fold number.')
    parser.add_argument('--max_step', type=int, default=50, help='Max time steps per sequence.')
    parser.add_argument('--quick_test', action="store_true",)

    ############## logger ##############
    # parser.add_argument('--model_name', type=str, default='hkt', help='Choose a model to run.')
    parser.add_argument("--save_folder", type=str, default="/mnt/qb/work/mlcolab/hzhou52/kt/logs",)
    parser.add_argument("--expername", type=str, default="",)

    ############## KTRunner ##############
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--early_stop', type=int, default=1, help='whether to early-stop.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size during training.')
    parser.add_argument('--eval_batch_size', type=int, default=512, help='Batch size during testing.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for each deep layer')
    parser.add_argument('--l2', type=float, default=0., help='Weight of l2_regularize in loss.')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer: GD, Adam, Adagrad, Adadelta')
    parser.add_argument('--metric', type=str, default='F1, AUC, Accuracy, Recall, Precision',
                        help='metrics: AUC, F1, Accuracy, Recall, Presicion;'
                                'The first one will be used to determine whether to early stop')
    
    
    
    parser.add_argument('--load', type=int, default=1,
                        help='Whether load model and continue to train')
    parser.add_argument('--fold', type=int, default=0,
                        help='Select a fold to run.')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files.')
    ############## training hyperparameter ##############
    parser.add_argument("--lr_decay", type=int, default=50, help="After how epochs to decay LR by a factor of gamma.",)
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor.")



    parser.add_argument(
        "--load_folder",
        type=str,
        default="",
        help="Where to load pre-trained model if finetuning/evaluating. "
        + "Leave empty to train from scratch",
    )



    parser.add_argument(
        "--validate", action="store_true", default=True, help="validate results throughout training."
    )

    parser.add_argument("--var", type=float, default=5e-2, help="Output variance.") # TODO
    

    # args.cuda = not args.no_cuda and torch.cuda.is_available()

    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)



    return parser # args

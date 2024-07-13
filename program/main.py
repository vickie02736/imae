import argparse
import os
import sys

sys.path.append('..')
import yaml
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
from utils import int_or_string, str2bool

SEED = 3409
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#-------------------------------------------------------------------------------------------


def get_args_parser():

    parser = argparse.ArgumentParser(
        description='Irregular time series forecasting')

    # parser.add_argument('--Train',
    #                     type=bool,
    #                     default=True,
    #                     help='Train the model')
    parser.add_argument('--resume-epoch',
                        type=int_or_string,
                        default=1,
                        help='start epoch after last training')
    parser.add_argument('--database',
                        type=str,
                        default='shallow_water',
                        help='Database name')
    parser.add_argument('--model-name',
                        type=str,
                        default='imae',
                        help='Model name')
    parser.add_argument('--interpolation',
                        type=str,
                        default=None,
                        choices=['linear', 'gaussian', None],
                        help='Interpolation method')

    train_group = parser.add_argument_group()
    train_group.add_argument('--epochs',
                             type=int,
                             default=2,
                             help='Number of epochs')
    train_group.add_argument('--save-frequency', type=int, default=2, help='Save once after how many epochs of training')
    train_group.add_argument('--mask-flag', type=str2bool, default=True, help='Mask flag')

    test_group = parser.add_argument_group()
    # test_group.add_argument(
    #     '--task',
    #     choices=['basic', 'outer', 'basic_rollout', 'outer_rollout'],
    #     default='inner',
    #     help='Task type')
    # test_group.add_argument('--mask-ratio',
    #                         type=float,
    #                         default=0.1,
    #                         help='Mask ratio')
    test_group.add_argument('--test-flag', type=bool, default=False, help='Test flag')

    return parser


#-------------------------------------------------------------------------------------------


def main():

    parser = get_args_parser()
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    end_epoch = args.resume_epoch + args.epochs

    # load config
    # data_config = yaml.load(open("../database/shallow_water/config.yaml", "r"), Loader=yaml.FullLoader)
    # config = yaml.load(open("../configs/sw_train_config.yaml", "r"), Loader=yaml.FullLoader)

    # load database 
    # from database.shallow_water.dataset import seq_DataBuilder, fra_DataBuilder
    # valid_dataset = seq_DataBuilder(data_config['valid_file'],
    #                                 config['seq_length'],
    #                                 config['valid']['rollout_times'],
    #                                 timestep=100)
    # if args.model_name == 'cae':s
    #     train_dataset = fra_DataBuilder(data_config['train_file'],
    #                                     timestep=100)
    # else:
    #     train_dataset = seq_DataBuilder(data_config['train_file'],
    #                                     config['seq_length'],
    #                                     config['train']['rollout_times'],
    #                                     timestep=100)

    if args.model_name == 'imae': 
        from engines import ImaeTrainer
        engine = ImaeTrainer(rank, args)

    elif args.model_name == 'convlstm':
        from engines import ConvLstmTrainer
        engine = ConvLstmTrainer(rank, args)

    elif args.model_name == 'cae':
        from engines import CaeTrainer
        engine = CaeTrainer(rank, args)

    elif args.model_name == 'cae_lstm':
        from engines import CaeLstmTrainer
        engine = CaeLstmTrainer(rank, args)

    else:
        pass

    for epoch in tqdm(range(args.resume_epoch, end_epoch),
                        desc="Epoch progress"):
        engine.train_epoch(epoch)
        engine.evaluate_epoch(epoch)


if __name__ == "__main__":
    main()
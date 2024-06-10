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
from program.utils.tools import int_or_string

SEED = 3409
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#-------------------------------------------------------------------------------------------


def get_args_parser():

    parser = argparse.ArgumentParser(
        description='Irregular time series forecasting')

    parser.add_argument('--Train',
                        type=bool,
                        default=True,
                        help='Train the model')
    parser.add_argument('--resume-epoch',
                        type=int_or_string,
                        default=0,
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
                             default=200,
                             help='Number of epochs')

    test_group = parser.add_argument_group()
    test_group.add_argument(
        '--task',
        choices=['inner', 'outer', 'inner_rollout', 'outer_rollout'],
        default='inner',
        help='Task type')
    test_group.add_argument('--mask-ratio',
                            type=float,
                            default=0.1,
                            help='Mask ratio')

    return parser


#-------------------------------------------------------------------------------------------


def main():

    parser = get_args_parser()
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    # load database
    if args.database == 'shallow_water':
        config = yaml.load(open("../configs/sw_train_config.yaml", "r"),
                           Loader=yaml.FullLoader)
        data_config = yaml.load(open("../database/shallow_water/config.yaml",
                                     "r"),
                                Loader=yaml.FullLoader)
        from database.shallow_water.dataset import seq_DataBuilder, fra_DataBuilder
        valid_dataset = seq_DataBuilder(data_config['valid_file'],
                                        config['seq_length'],
                                        config['valid']['rollout_times'],
                                        timestep=100)
        if args.model_name == 'cae':
            print(args.model_name)
            train_dataset = fra_DataBuilder(data_config['train_file'],
                                            timestep=100)
        else:
            train_dataset = seq_DataBuilder(data_config['train_file'],
                                            config['seq_length'],
                                            config['train']['rollout_times'],
                                            timestep=100)
    elif args.database == 'moving_mnist':
        config = yaml.load(open("../configs/mm_train_config.yaml", "r"),
                           Loader=yaml.FullLoader)
        data_config = yaml.load(open("../database/moving_mnist/config.yaml",
                                     "r"),
                                Loader=yaml.FullLoader)
        from database.moving_mnist.dataset import seq_DataBuilder
        train_dataset = seq_DataBuilder(config['seq_length'],
                                        config['train']['rollout_times'],
                                        'train')
        valid_dataset = seq_DataBuilder(config['seq_length'],
                                        config['valid']['rollout_times'],
                                        'valid')
    else:
        pass

    if args.model_name == 'imae':
        from models import VisionTransformer
        from engines import ImaeTrainer
        os.makedirs(config['imae']['save_checkpoint'], exist_ok=True)
        os.makedirs(config['imae']['save_reconstruct'], exist_ok=True)
        os.makedirs(config['imae']['save_loss'], exist_ok=True)

        model = VisionTransformer(args.database,
                                  data_config['channels'],
                                  data_config['image_size'],
                                  config['patch_size'],
                                  num_layers=config['imae']['num_layers'],
                                  nhead=config['imae']['nhead'])
        engine = ImaeTrainer(rank, config, train_dataset, valid_dataset, model,
                             args.epochs, args.resume_epoch)
        end_epoch = args.resume_epoch + args.epochs
        for epoch in tqdm(range(args.resume_epoch, end_epoch),
                          desc="Epoch progress"):
            engine.train_epoch(epoch)
            engine.evaluate_epoch(epoch)

    elif args.model_name == 'convlstm':
        from models import ConvLSTM
        from engines import ConvLstmTrainer
        os.makedirs(os.path.join(config['convlstm']['save_checkpoint'],
                                 args.interpolation),
                    exist_ok=True)
        os.makedirs(os.path.join(config['convlstm']['save_reconstruct'],
                                 args.interpolation),
                    exist_ok=True)
        os.makedirs(os.path.join(config['convlstm']['save_loss'],
                                 args.interpolation),
                    exist_ok=True)
        model = ConvLSTM(config['channels'], config['convlstm']['hidden_dim'],
                         tuple(config['convlstm']['kernel_size']),
                         config['convlstm']['num_layers'])
        engine = ConvLstmTrainer(rank, config, train_dataset, valid_dataset,
                                 model, args.epochs, args.resume_epoch,
                                 args.interpolation)
        end_epoch = args.resume_epoch + args.epochs
        for epoch in tqdm(range(args.resume_epoch, end_epoch),
                          desc="Epoch progress"):
            engine.train_epoch(epoch)
            engine.evaluate_epoch(epoch)

    elif args.model_name == 'cae':
        from models import ConvAutoencoder
        from engines import CaeTrainer
        os.makedirs(config['cae']['save_checkpoint'], exist_ok=True)
        os.makedirs(config['cae']['save_reconstruct'], exist_ok=True)
        os.makedirs(config['cae']['save_loss'], exist_ok=True)
        model = ConvAutoencoder(config['cae']['latent_dim'],
                                config['channels'])
        engine = CaeTrainer(rank, config, train_dataset, valid_dataset, model,
                            args.epochs, args.resume_epoch)
        end_epoch = args.resume_epoch + args.epochs
        for epoch in tqdm(range(args.resume_epoch, end_epoch),
                          desc="Epoch progress"):
            engine.train_epoch(epoch)
            engine.evaluate_epoch(epoch)
    else:
        pass


if __name__ == "__main__":
    main()

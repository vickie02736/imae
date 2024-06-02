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

from model import VisionTransformer
from engine import Trainer, Evaluator
from program.utils.tools import int_or_string


SEED = 3409
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


#-------------------------------------------------------------------------------------------

def get_args_parser():

    parser = argparse.ArgumentParser(description='Irregular time series forecasting')

    parser.add_argument('--test', type=bool, default=False, help='Test the model')
    parser.add_argument('--resume-epoch', type=int_or_string, default=0, help='start epoch after last training')
    parser.add_argument('--database', type=str, default='shallow_water', help='Database name')
    parser.add_argument('--model', type=str, default='imae', help='Model name')

    train_group = parser.add_argument_group()
    train_group.add_argument('--epochs', type=int, default=200, help='Number of epochs')

    test_group = parser.add_argument_group()
    test_group.add_argument('--task', choices=['inner', 'outer', 'inner_rollout', 'outer_rollout'],
                            default='inner', help='Task type')
    test_group.add_argument('--mask-ratio', type=float, default=0.1, help='Mask ratio')

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
        # shallow water
        config = yaml.load(open("../configs/imae_sw.yaml", "r"), Loader=yaml.FullLoader)
        data_config = yaml.load(open("../database/shallow_water/config.yaml","r"), Loader=yaml.FullLoader)
        from database.shallow_water.dataset import seq_DataBuilder
        train_dataset = seq_DataBuilder(data_config['train_file'], config['seq_length'], config['train']['rollout_times'], timestep=100)
        valid_dataset = seq_DataBuilder(data_config['valid_file'], config['seq_length'], config['valid']['rollout_times'], timestep=100)

    elif args.database == 'moving_mnist':
        # moving mnist
        config = yaml.load(open("../configs/imae_mm.yaml.yaml", "r"), Loader=yaml.FullLoader)
        data_config = yaml.load(open("../database/moving_mnist/config.yaml","r"), Loader=yaml.FullLoader)
        from database.moving_mnist.dataset import seq_DataBuilder
        train_dataset = seq_DataBuilder(config['seq_length'], config['train']['rollout_times'], 'train')
        valid_dataset = seq_DataBuilder(config['seq_length'], config['valid']['rollout_times'], 'valid')

    else:
        pass

    model = VisionTransformer(args.database, 
                              data_config['channels'], data_config['image_size'], config['patch_size'],
                              num_layers = config['model']['num_layers'], nhead = config['model']['nhead'])

    trainer = Trainer(rank, config, train_dataset, model, args.epochs, args.resume_epoch)
    evalutor = Evaluator(rank, config, valid_dataset, model)

    end_epoch = args.resume_epoch + args.epochs
    
    for epoch in tqdm(range(args.resume_epoch, end_epoch), desc="Epoch progress"): 
        trainer.train_epoch(epoch)
        evalutor.evaluate_epoch(epoch)

if __name__ == "__main__":
    main()
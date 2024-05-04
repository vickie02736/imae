import argparse
import copy
import json
import os
import yaml
import random
import numpy as np
from tqdm import tqdm

from model import VisionTransformer
# from utils import plot_rollout, save_losses
# from utils import RMSELoss
from utils import int_or_string
from dataset import DataBuilder
from engine import Trainer, Evaluator


import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

from utils import int_or_string

# import wandb
# wandb.login()
# 

SEED = 3409
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


#-------------------------------------------------------------------------------------------


def get_args_parser():

    parser = argparse.ArgumentParser(description='Train Vision Transformer')

    parser.add_argument('--train', type=bool, default=False, help='Train the model')
    parser.add_argument('--resume-epoch', type=int_or_string, default=0, help='start epoch after last training')

    train_group = parser.add_argument_group()
    train_group.add_argument('--epochs', type=int, default=200, help='Number of epochs')

    eval_group = parser.add_argument_group()
    eval_group.add_argument('--task', choices=['inner', 'outer', 'inner_rollout', 'outer_rollout'],
                            default='inner', help='Task type')
    eval_group.add_argument('--mask-ratio', type=float, default=0.1, help='Mask ratio')

    return parser

#-------------------------------------------------------------------------------------------



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    config = yaml.load(open("SW_config.yaml", "r"), Loader=yaml.FullLoader)
    os.makedirs(config['save_path']['checkpoint'], exist_ok=True)
    os.makedirs(config['save_path']['reconstruct'], exist_ok=True)

    model = VisionTransformer(config['channels'], config['image_size'], config['patch_size'])

    train_dataset = DataBuilder(config['dataset']['train'], config['seq_length'], config['train']['rollout_times'])
    trainer = Trainer(rank, config, train_dataset, model, args.epochs)
    valid_dataset = DataBuilder(config['dataset']['valid'], config['seq_length'], config['train']['rollout_times'])
    evalutor = Evaluator(rank, config, valid_dataset, model, args.epochs)

    if args.resume_epoch != 0: 
        trainer.setup()
        trainer.load_checkpoint(args.resume_epoch)

    end_epoch = args.resume_epoch + args.epochs
    
    for epoch in tqdm(range(args.resume_epoch, end_epoch), desc="Epoch progress"): 
        train_loss = trainer.train_epoch(epoch)
        valid_loss = evalutor.evaluate_epoch(epoch) # dictionary

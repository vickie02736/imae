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


# import wandb
# wandb.login()


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
    parser.add_argument('--database', type=str, default='shallow_water', help='Database name')

    train_group = parser.add_argument_group()
    train_group.add_argument('--epochs', type=int, default=200, help='Number of epochs')

    eval_group = parser.add_argument_group()
    eval_group.add_argument('--task', choices=['inner', 'outer', 'inner_rollout', 'outer_rollout'],
                            default='inner', help='Task type')
    eval_group.add_argument('--mask-ratio', type=float, default=0.1, help='Mask ratio')

    return parser

#-------------------------------------------------------------------------------------------


def main():

    parser = get_args_parser()
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    if args.database == 'shallow_water':
        # shallow water
        config = yaml.load(open("../program/imae/config_sw.yaml", "r"), Loader=yaml.FullLoader)
        data_config = yaml.load(open("../database/shallow_water/config.yaml","r"), Loader=yaml.FullLoader)
        from database.shallow_water.dataset import DataBuilder
        train_dataset = DataBuilder(data_config['train_file'], config['seq_length'], config['train']['rollout_times'], timestep=100)
        valid_dataset = DataBuilder(data_config['valid_file'], config['seq_length'], config['valid']['rollout_times'], timestep=100)

        # if rank == 0: 
        #     wandb.init(project="imae_sw", config ={
        #         "database": args.database,
        #         "batch_size": config['batch_size'],
        #         "epochs": args.epochs,
        #     })

    elif args.database == 'moving_mnist':
        # moving mnist
        config = yaml.load(open("../program/imae/config_mm.yaml", "r"), Loader=yaml.FullLoader)
        data_config = yaml.load(open("../database/moving_mnist/config.yaml","r"), Loader=yaml.FullLoader)
        from database.moving_mnist.dataset import DataBuilder
        train_dataset = DataBuilder(config['seq_length'], config['train']['rollout_times'], 'train')
        valid_dataset = DataBuilder(config['seq_length'], config['valid']['rollout_times'], 'valid')

        # if rank == 0: 
        #     wandb.init(project="imae_mm", config ={
        #         "database": args.database,
        #         "batch_size": config['batch_size'],
        #         "epochs": args.epochs,
        #     })
    else:
        pass


    os.makedirs(config['train']['save_checkpoint'], exist_ok=True)
    os.makedirs(config['valid']['save_reconstruct'], exist_ok=True)

    model = VisionTransformer(args.database, 
                              data_config['channels'], data_config['image_size'], config['patch_size'],
                              num_layers = config['model']['num_layers'], nhead = config['model']['nhead'])
    torch.save(model.state_dict(), os.path.join(config['train']['save_checkpoint'], 'init.pth'))

    trainer = Trainer(rank, config, train_dataset, model, args.epochs, args.resume_epoch)
    evalutor = Evaluator(rank, config, valid_dataset, model)


    trainer.setup()
    evalutor.setup()
    if args.resume_epoch == 0: 
        losses = {}
        os.makedirs(config['train']['save_loss'], exist_ok=True)
        os.makedirs(config['valid']['save_loss'], exist_ok=True)
        with open(os.path.join(config['train']['save_loss'], 'train_losses.json'), 'w') as file:
            json.dump(losses, file)
        with open(os.path.join(config['valid']['save_loss'], 'valid_losses.json'), 'w') as file:
            json.dump(losses, file)
    else:

        trainer.load_checkpoint(args.resume_epoch)

    end_epoch = args.resume_epoch + args.epochs
    
    for epoch in tqdm(range(args.resume_epoch, end_epoch), desc="Epoch progress"): 
        trainer.train_epoch(epoch)
        evalutor.evaluate_epoch(epoch)



if __name__ == "__main__":
    
    main()
import argparse
import copy
import json
import os
import yaml
import random
import numpy as np
from tqdm import tqdm

from model import VisionTransformer
from utils import plot_rollout, save_losses
from utils import RMSELoss
from utils import int_or_string
from dataset import DataBuilder
from engine import Trainer


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# import wandb
# wandb.login()
# 

SEED = 3409
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


#-------------------------------------------------------------------------------------------

def int_or_string(value):
    try:
        return int(value)
    except ValueError:
        return value



def get_args_parser():

    parser = argparse.ArgumentParser(description='Train Vision Transformer')

    parser.add_argument('--train', type=bool, default=False, help='Train the model')
    parser.add_argument('--restart-epoch', type=int_or_string, default=0, help='start epoch after last training')

    train_group = parser.add_argument_group()
    train_group.add_argument('--epochs', type=int, default=200, help='Number of epochs')

    eval_group = parser.add_argument_group()
    eval_group.add_argument('--task', choices=['inner', 'outer', 'inner_rollout', 'outer_rollout'],
                            default='inner', help='Task type')
    eval_group.add_argument('--mask-ratio', type=float, default=0.1, help='Mask ratio')

    return parser

#-------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------

# def initialize_or_load_model(args, config, device):
#     # Check if starting fresh or loading from checkpoint
#     if args.start_epoch == 0:
#         # Instantiate the model with initial parameters
#         model = VisionTransformer(config['channels'], config['image_size'], config['patch_size'], device).to(device)
#         model = DDP(model, device_ids=[device.index])
#         train_losses = []
#         train_losses_rollout = []
#         valid_losses = []
#     else:
#         # Load the model from checkpoint
#         checkpoint_path = config['save_path']['checkpoint'] + f'checkpoint_{args.start_epoch - 1}.pth'
#         checkpoint = torch.load(checkpoint_path, map_location=device)

#         # Re-create the model and load state
#         model = VisionTransformer(config['channels'], config['image_size'], config['patch_size'], device)
#         model.load_state_dict(checkpoint['model'])
#         model.to(device)
#         model = DDP(model, device_ids=[device.index])

#         # Assuming optimizer, scheduler, scaler are initialized earlier in the script
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         scheduler.load_state_dict(checkpoint['scheduler'])
#         scaler.load_state_dict(checkpoint['scaler'])

#     return model, train_losses, train_losses_rollout, valid_losses

#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------

# def main(rank, args, world_size): 

#     torch.cuda.set_device(rank)
#     torch.distributed.init_process_group("nccl", rank=rank, world_size=torch.cuda.device_count())

#     config = yaml.load(open("SW_config.yaml", "r"), Loader=yaml.FullLoader)

#     model = VisionTransformer(config['channels'], config['image_size'], config['patch_size'], torch.device(f"cuda:{rank}"))

    

#     train_dataset = DataBuilder(config['dataset']['train'], config['seq_length'], config['rollout_times'])
#     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, sampler=train_sampler)

#     valid_dataset = DataBuilder(config['dataset']['valid'], config['seq_length'], config['rollout_times'])
#     valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
#     valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, sampler=valid_sampler)

#     loss_fn = nn.MSELoss()
#     T_start = args.epochs * 0.05 * len(train_dataset) // config['batch_size']
#     T_start = int(T_start)
#     learning_rate = 1e-4
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.03)
#     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_start, eta_min=1e-6, last_epoch=-1)
#     scaler = torch.cuda.amp.GradScaler()
        

#     trainer = Trainer(model, train_loader, config, loss_fn, optimizer, scheduler, scaler, rank)
    
#     end_epoch = args.start_epoch + args.epochs
#     for epoch in tqdm(range(args.start_epoch, end_epoch), desc="Epoch progress"): 
#         train_sampler.set_epoch(epoch)  # Ensures proper shuffling per epoch
#         train_losses, train_losses_rollout = trainer.train_epoch(epoch)
#         trainer.save_checkpoint(epoch)   



#         # wandb.init(project="imae",config={"epochs": args.epochs,})



#     best_loss = float('inf')

#         # torch.manual_seed(epoch)

#         # if args.train: 
#         #     # trainer(model, device, train_loader, loss_fn, optimizer, scaler, scheduler, **config)
#         #     running_loss = 0
#         #     running_loss_rollout = 0
#         #     train_losses = []
#         #     train_losses_rollout = []

#         #     model.train()

#         #     for _, sample in enumerate(dataloader): 
                
#         #         origin = sample["Input"].float().to(device)
#         #         target = sample["Target"].float().to(device)
#         #         target_chunks = torch.chunk(target, 2, dim=1)

#         #         mask_ratio = torch.rand(1).item()
#         #         num_mask = int(mask_ratio * config['seq_length'])

#         #         with torch.autocast(device_type=device.type):
#         #             output = model(origin, num_mask)
#         #             predic_loss = loss_fn(output, target_chunks[0])

#         #             output = model(output, 0)
#         #             rollout_loss = loss_fn(output, target_chunks[1])

#         #             loss = predic_loss + rollout_loss

#         #         optimizer.zero_grad()
#         #         scaler.scale(loss).backward(retain_graph=True) # Scales loss
#         #         scaler.unscale_(optimizer)
#         #         nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
#         #         scaler.step(optimizer) 
#         #         scaler.update() 
#         #         scheduler.step()
                
#         #         running_loss += predic_loss.item()
#         #         running_loss_rollout += rollout_loss.item()
                    
#         #         train_loss = running_loss / len(dataloader.dataset)
#         #         train_losses.append(train_loss)
#         #         rollout_loss = running_loss_rollout / len(dataloader.dataset)
#         #         train_losses_rollout.append(train_loss)

#         #     save_dict = {
#         #         'model': model.state_dict(),
#         #         'optimizer': optimizer.state_dict(),
#         #         'scheduler': scheduler.state_dict(),
#         #         'scaler': scaler.state_dict(),
#         #         'epoch': epoch
#         #     }

#         #     torch.save(save_dict, os.path.join(config['save_path']['checkpoint'], f'checkpoint_{epoch}.tar'))
#         #     torch.save(save_dict, os.path.join(config['save_path']['checkpoint'], f'checkpoint_{epoch}.pth'))
#         #     torch.save(save_dict, os.path.join(config['save_path']['checkpoint'], f'checkpoint_{epoch}.pth.tar'))   

#     #----------------------------------------------------------------------------------------------------
            
#         # model.eval()
#         # running_losses = [0 for _ in range(2)]

#         # with torch.no_grad():

#         #     for i, sample in enumerate(valid_loader):

#         #         origin = sample["Input"].float().to(device)
#         #         origin_copy = copy.deepcopy(origin)
#         #         target = sample["Target"].float().to(device)
#         #         target_chunks = torch.chunk(target, config['rollout_times'], dim=1)
                
#         #         output_chunks = []

#         #         for j, chunk in enumerate(target_chunks):

#         #             if j == 0: 
#         #                 mask_ratio = torch.rand(1).item()
#         #                 num_mask = int(mask_ratio * config['seq_length'])
#         #                 output = model(origin_copy, num_mask)
#         #             else: 
#         #                 output = model(origin_copy, 0)
                    
#         #             output_chunks.append(output)
#         #             loss = loss_fn(output, chunk)
#         #             running_losses[j] += loss.item()
#         #             origin_copy = copy.deepcopy(output)

#         #         if i == 1:
#         #             plot_rollout(origin, output_chunks, target_chunks, epoch, config['save_path']['reconstruct'])
        
#         # chunk_losses = []
#         # for running_loss in running_losses:
#         #     valid_loss = running_loss / len(valid_loader.dataset)
#         #     chunk_losses.append(valid_loss)

#         # current_loss = chunk_losses[0]
#         # if best_loss > current_loss:
#         #     best_loss = current_loss
#             # torch.save(save_dict, os.path.join(config['save_path']['checkpoint'], f'best_checkpoint.tar'))
#             # torch.save(save_dict, os.path.join(config['save_path']['checkpoint'], f'best_checkpoint.pth'))
#             # torch.save(save_dict, os.path.join(config['save_path']['checkpoint'], f'best_checkpoint..pth.tar'))

#         # valid_losses.append(chunk_losses)

#         # save_losses(config['save_path']['checkpoint'], train_losses, train_losses_rollout, valid_losses)

#         # wandb.log({"train_losses": train_losses, "train_losses_rollout": train_losses_rollout, "valid_losses": valid_losses})



# if __name__ == "__main__":
#     parser = get_args_parser()
#     args = parser.parse_args()
#     world_size = torch.cuda.device_count()
#     torch.multiprocessing.spawn(main, args=(args, world_size,), nprocs=world_size)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    config = yaml.load(open("SW_config.yaml", "r"), Loader=yaml.FullLoader)
    os.makedirs(config['save_path']['checkpoint'], exist_ok=True)
    os.makedirs(config['save_path']['reconstruct'], exist_ok=True)

    train_dataset = DataBuilder(config['dataset']['train'], config['seq_length'], config['rollout_times'])
    model = VisionTransformer(config['channels'], config['image_size'], config['patch_size'])
    trainer = Trainer(rank, config, train_dataset, model, args.epochs)
    # args.restart_epoch = 6
    # start_epoch = 2
    trainer.setup()
    trainer.load_checkpoint(args.restart_epoch)

    end_epoch = args.restart_epoch + args.epochs
    for epoch in tqdm(range(args.restart_epoch, end_epoch), desc="Epoch progress"): 
        trainer.train_epoch(epoch)
    # end_epoch = start_epoch + args.epochs
    # for epoch in tqdm(range(start_epoch, end_epoch), desc="Epoch progress"): 
    #     trainer.train_epoch(epoch)
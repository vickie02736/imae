import argparse
import copy
import json
import os
import random
import numpy as np
from tqdm import tqdm

from model import VisionTransformer
from utils import plot_rollout, save_losses
from dataset import DataBuilder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


SEED = 3409
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


### Argparse
parser = argparse.ArgumentParser(description='Train Vision Transformer')

parser.add_argument('--mask-ratio', type=float, default=0.9, help='Masking ratio')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--rollout-times', type=int, default=1, help='Rollout times')
parser.add_argument('--start-epoch', type=int, default=0, help='start epoch after last training')

args = parser.parse_args()
### End of Argparse

epochs = args.epochs
mask_ratio = args.mask_ratio
file_number = int(mask_ratio * 10)
batch_size = args.batch_size
rollout_times = args.rollout_times
start_epoch = args.start_epoch
end_epoch = start_epoch + epochs


### Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(3, 16, 128, device)
model = model.to(device)
###


### Load data
train_dataset = DataBuilder('../data/train_file.csv',10, 1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = DataBuilder('../data/valid_file.csv',10, rollout_times)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
###


### Set Parameters
T_start = args.epochs * 0.05 * len(train_dataset) // batch_size
T_start = int(T_start)

loss_fn = nn.MSELoss()
learning_rate = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.03)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_start, eta_min=1e-6, last_epoch=-1)
scaler = torch.cuda.amp.GradScaler()
###


### Checkpoint
if start_epoch == 0:
    epochs = range(0, args.epochs)
    train_losses = []
    valid_losses = []
    valid_losses_rollout = []

else: 
    # Load the checkpoint
    checkpoint_path = "../data/Vit_checkpoint/{file_number}/epoch_{i}.pth".format(file_number=file_number,i=args.start_epoch-1)
    checkpoint = torch.load(checkpoint_path)
    # Update model and optimizer with the loaded state dictionaries
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])

    # Now, you can also access the train and evaluation losses if you need
    with open("../data/Vit_checkpoint/{file_number}", "r") as f: 
        loss_data = json.load(f)
    train_losses = loss_data.get('train_losses', [])
    valid_losses = loss_data.get('valid_losses', [])
###


### Set save_path
checkpoint_save_path = "../data/Vit_checkpoint/{file_number}".format(file_number=file_number)
if not os.path.exists(checkpoint_save_path):
    os.makedirs(checkpoint_save_path)
rec_save_path = "../data/Vit_rec/{file_number}".format(file_number=file_number)
if not os.path.exists(rec_save_path): 
    os.makedirs(rec_save_path)
rollout_rec_save_path = "../data/Vit_rec/{file_number}_rollout".format(file_number=file_number)
if not os.path.exists(rollout_rec_save_path): 
    os.makedirs(rollout_rec_save_path)
###


for epoch in tqdm(range(start_epoch, end_epoch), desc="Epoch progress"):

    model.train()
    
    running_loss = 0
    for _, sample in enumerate(train_loader): 
        optimizer.zero_grad()

        with torch.autocast(device_type = device.type):
            target = sample["Target"].float().to(device)
            origin = sample["Input"].float().to(device)
            output = model(origin, mask_ratio)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward(retain_graph=True) # Scales loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer) 
        scaler.update() 
        scheduler.step()
        running_loss += loss.item()
        
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch
    }
    torch.save(save_dict, os.path.join(checkpoint_save_path, f'checkpoint_{epoch}.tar'))
    torch.save(save_dict, os.path.join(checkpoint_save_path, f'checkpoint_{epoch}.pth'))
    torch.save(save_dict, os.path.join(checkpoint_save_path, f'checkpoint_{epoch}.pth.tar'))
    torch.save(model, os.path.join(checkpoint_save_path, f'checkpoint_{epoch}.pth'))


#####################################################

    model.eval()

    running_losses = [0 for _ in range(rollout_times)]
    with torch.no_grad():

        for i, sample in enumerate(val_loader):

            origin = sample["Input"].float().to(device)
            origin_copy = copy.deepcopy(origin)
            target = sample["Target"].float().to(device)
            target_chunks = torch.chunk(target, rollout_times, dim=1)
            
            output_chunks = []
            for j, chunk in enumerate(target_chunks):
                if j == 0: 
                    output = model(origin_copy, mask_ratio)
                    output_chunks.append(output)
                    loss = loss_fn(output, chunk)
                    running_losses[j] += loss.item()
                    origin_copy = copy.deepcopy(output)
                else: 
                    output = model(origin_copy, 0)
                    output_chunks.append(output)
                    loss = loss_fn(output, chunk)
                    running_losses[j] += loss.item()
                    origin_copy = copy.deepcopy(output)

            if i == 1:
                a = output_chunks[0].unsqueeze(0)
                b = target_chunks[0].unsqueeze(0)
                plot_rollout(origin, a, b, epoch, rec_save_path)
                plot_rollout(origin, output_chunks, target_chunks, epoch, rollout_rec_save_path)
    
    chunk_losses = []
    for running_loss in running_losses:
        valid_loss = running_loss / len(val_loader.dataset)
        chunk_losses.append(valid_loss)

    valid_losses.append(chunk_losses)

    save_losses(checkpoint_save_path, train_losses, valid_losses)
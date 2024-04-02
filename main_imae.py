import sys
sys.path.append("..")
import os
import json

import torch
import torch.nn as nn
from torch import optim

# from dataset import DataBuilder
from dataset_imae import DataBuilder
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from model_imae import VisionTransformer, train_model


### Argparse
parser = argparse.ArgumentParser(description='Train Vision Transformer')

parser.add_argument('--mask_ratio', type=float, default=0.9, help='Masking ratio')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--rollout_times', type=int, default=1, help='Rollout times')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch after last training')

args = parser.parse_args()
### End of Argparse


### Initialise model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionTransformer(3, 16, 128, device)
model = model.to(device)
###


### Load data
batch_size = args.batch_size

train_dataset = DataBuilder('../data/train_file.csv',10, args.rollout_times)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = DataBuilder('../data/valid_file.csv',10, args.rollout_times)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

file_number = int(args.mask_ratio * 10)
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


if args.start_epoch == 0:

    epochs = range(0, args.epochs)
    train_losses = []
    valid_losses = []

else: 
    
    # Path to the checkpoint file
    checkpoint_path = "../data/Vit_checkpoint/{file_number}/epoch_{i}.pth".format(file_number=file_number,i=args.start_epoch-1)
    
    # Load the checkpoint
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
    
    epochs = range(args.start_epoch, args.start_epoch + args.epochs)



for epoch in tqdm(epochs): 

    rec_save_path = "../data/Vit_rec/{file_number}".format(file_number=file_number)
    if not os.path.exists(rec_save_path): 
        os.makedirs(rec_save_path)
    checkpoint_save_path = "../data/Vit_checkpoint/{file_number}".format(file_number=file_number)
    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)

    train_model(model, optimizer, scheduler, scaler, args.mask_ratio, 
                train_losses, valid_losses, loss_fn, 
                train_loader, val_loader, epoch, checkpoint_save_path, rec_save_path, device)
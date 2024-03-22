import sys
sys.path.append(".")
import os

import torch
import torch.nn as nn
from torch import optim

# from dataset import DataBuilder
from Project.imae.dataset import DataBuilder

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from model_imae import VisionTransformer, train, eval


### Argparse
parser = argparse.ArgumentParser(description='Train Vision Transformer')

parser.add_argument('--mask_ratio', type=float, default=0.9, help='Masking ratio')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')

args = parser.parse_args()
### End of Argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionTransformer(3, 16, 128, device)
model = model.to(device)

batch_size = 128
# train_dataset = DataBuilder('/home/uceckz0/Scratch/imae/train_file.csv',20, 10)
train_dataset = DataBuilder('/home/uceckz0/Scratch/imae/train_file.csv',10, 1)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataset = DataBuilder('/home/uceckz0/Scratch/imae/valid_file.csv',20, 10)
val_dataset = DataBuilder('/home/uceckz0/Scratch/imae/valid_file.csv',10, 1)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

epochs = range(0, args.epochs)

learning_rate = 1e-4
T_start = args.epochs * 0.05 * len(train_dataset) // batch_size
T_start = int(T_start)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.03)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_start, eta_min=1e-6, last_epoch=-1, verbose=False)
scaler = torch.cuda.amp.GradScaler()

train_loss = []
eval_loss = []

# Path to the checkpoint file
# checkpoint_path = "/home/uceckz0/Scratch/imae/Vit_checkpoint/epoch_{epoch}.pth".format(epoch=101)

# # Load the checkpoint
# checkpoint = torch.load(checkpoint_path)

# # Update model and optimizer with the loaded state dictionaries
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# scaler.load_state_dict(checkpoint['scaler'])

# # Now, you can also access the train and evaluation losses if you need
# train_loss = checkpoint['train_loss']
# eval_loss = checkpoint['eval_loss']



loss_fn = nn.MSELoss()

file_number = int(args.mask_ratio * 10)

for epoch in tqdm(epochs): 

    print("epoch:", epoch)
    
    train_loss_epoch = train(model, optimizer, scheduler, scaler, train_loader, args.mask_ratio)
    train_loss.append(train_loss_epoch)
        

    rec_save_path = "/home/uceckz0/Scratch/imae/Vit_rec_{file_number}".format(file_number=file_number)
    if not os.path.exists(rec_save_path):
        os.makedirs(rec_save_path)
    eval_loss_epoch = eval(model, val_loader, args.mask_ratio, epoch, 
                           save_path=rec_save_path)
    eval_loss.append(eval_loss_epoch)

    print("Train Loss:", train_loss_epoch, "eval_loss:", eval_loss_epoch)

    checkpoint = {'epoch': epoch, 'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 
                  'train_loss': train_loss, 'eval_loss': eval_loss}
    checkpoint_save_path = "/home/uceckz0/Scratch/imae/Vit_checkpoint_{file_number}".format(file_number=file_number)
    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)
    torch.save(checkpoint, checkpoint_save_path+"/epoch_{epoch}.pth".format(epoch=epoch))
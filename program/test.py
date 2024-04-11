import argparse
import copy
import json
import os
import random
import numpy as np

from model import VisionTransformer
from utils import plot_rollout
from dataset import DataBuilder

import torch
import torch.nn as nn
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
parser.add_argument('--load-epoch', type=int, default=200, help='Load epoch')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--rollout-times', type=int, default=1, help='Rollout times')

args = parser.parse_args()
### End of Argparse

load_epoch = args.load_epoch
mask_ratio = args.mask_ratio
file_number = int(mask_ratio * 10)
batch_size = args.batch_size
rollout_times = args.rollout_times
checkpoint_save_path = checkpoint_path = f"../data/Vit_checkpoint/{file_number}"
rollout_rec_save_path = f"../data/Vit_test/{file_number}"

### Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(3, 16, 128, device)
model = model.to(device)
###

### Load data
val_dataset = DataBuilder('../data/valid_file.csv',10, rollout_times)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
###

model = VisionTransformer(3, 16, 128, device)
checkpoint = torch.load(os.path.join(checkpoint_save_path, f"checkpoint_{load_epoch}.tar"))
model.load_state_dict(checkpoint['model'])
model = model.to(device)

loss_fn = nn.MSELoss()
test_losses = []

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
            output = model(origin_copy, mask_ratio)
            output_chunks.append(output)
            loss = loss_fn(output, chunk)
            running_losses[j] += loss.item()
            origin_copy = output

        if i == 1:
            plot_rollout(origin, output_chunks, target_chunks, load_epoch, rollout_rec_save_path)

chunk_losses = []
for running_loss in running_losses:
    valid_loss = running_loss / len(val_loader.dataset)
    chunk_losses.append(valid_loss)

test_losses.append(chunk_losses)

with open(os.path.join(rollout_rec_save_path, 'test_losses.json'), 'w') as f:
    json.dump(test_losses, f)
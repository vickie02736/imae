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
parser.add_argument('--checkpoint-num', type=int, default=0, help='Checkpoint number')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--rollout-times', type=int, default=64, help='Rollout times')
parser.add_argument('--sequence-length', type=int, default=10, help='Sequence length')

args = parser.parse_args()
### End of Argparse

checkpoint_num = args.checkpoint_num
batch_size = args.batch_size
rollout_times = args.rollout_times
sequence_length = args.sequence_length
checkpoint_path = f"../data/Vit_test/checkpoint_{checkpoint_num}.pth"

rollout_rec_save_path = f"../data/Vit_test/"
os.makedirs(rollout_rec_save_path, exist_ok=True)

### Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(3, 16, 128, device)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.to(device)
###

### Load data
val_dataset = DataBuilder('../data/inner_test_file.csv',sequence_length, rollout_times)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
###

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

            if j == 0: 
                mask_ratio = torch.rand(1).item()
                num_mask = int(mask_ratio * sequence_length)
                output = model(origin_copy, num_mask)
            else: 
                output = model(origin_copy, 0)
            
            output_chunks.append(output)
            loss = loss_fn(output, chunk)
            running_losses[j] += loss.item()

        if i == 1: 
            plot_rollout(origin, output_chunks, target_chunks, i, rollout_rec_save_path)


chunk_losses = []
for running_loss in running_losses:
    valid_loss = running_loss / len(val_loader.dataset)
    chunk_losses.append(valid_loss)

test_losses.append(chunk_losses)

with open(os.path.join(rollout_rec_save_path, 'test_losses.json'), 'w') as f:
    json.dump(test_losses, f)
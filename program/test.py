import argparse
import copy
import json
import os
import random
import numpy as np

from model import VisionTransformer
# from utils import plot_rollout
from dataset import DataBuilder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


SEED = 3409
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def int_or_string(value):
    if value == "best":
        return value
    else:
        return int(value)

### Argparse
parser = argparse.ArgumentParser(description='Test Vision Transformer')
parser.add_argument('--checkpoint', type=int_or_string, default=0, help='Checkpoint ')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--rollout-times', type=int, default=64, help='Rollout times')
parser.add_argument('--sequence-length', type=int, default=10, help='Sequence length')
parser.add_argument('--mask-ratio', type=float, default=0.1, help='Mask ratio')
parser.add_argument('--task', choices=['inner', 'outer', 'inner_rollout', 'outer_rollout'],
                    default='inner', help='Task type')
parser.add_argument('--mask-type', choices=['random', 'consecutive'],
                    default='random', help='Mask type')

args = parser.parse_args()
## End of Argparse


rollout_times = args.rollout_times
sequence_length = args.sequence_length
mask_ratio = args.mask_ratio
mask_type = args.mask_type
task = args.task
checkpoint = args.checkpoint


num_mask = int(sequence_length * mask_ratio)
test_csv = f'../dataset_split/csv/{task}_test_file.csv'
output_dir = f'/home/uceckz0/Project/imae/data/rec_{mask_type}/{task}'

rollout_rec_save_path = output_dir + f"/{num_mask}"
os.makedirs(rollout_rec_save_path, exist_ok=True)


if checkpoint == "best":
    checkpoint_path = f'../data/checkpoint_{mask_type}/{task}_best_checkpoint.tar'
else: 
    checkpoint_path = f"../data/checkpoint_{mask_type}/checkpoint_{checkpoint}.pth"

### Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    model = VisionTransformer(3, 16, 128, device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
else:
    model = VisionTransformer(3, 16, 128, device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.to(device)
###

### Load data
val_dataset = DataBuilder(test_csv, sequence_length, rollout_times)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
###

mse_loss_fn = nn.MSELoss()
mse_losses = []

mae_loss_fn = nn.L1Loss()
mae_losses = []


model.eval()

mse_running_loss = [0 for _ in range(rollout_times)]
mae_running_loss = [0 for _ in range(rollout_times)]


with torch.no_grad():

    for i, sample in enumerate(val_loader):
        
        origin = sample["Input"].float().to(device)
        origin_copy = copy.deepcopy(origin)
        target = sample["Target"].float().to(device)
        target_chunks = torch.chunk(target, rollout_times, dim=1)
        pos = sample["Pos"][0]
        R = sample["R"][0]
        Hp = sample["Hp"][0]
        
        output_chunks = []

        for j, chunk in enumerate(target_chunks):

            if j == 0: 
                output = model(origin_copy, num_mask, mask_type)
                masked_origin = copy.deepcopy(origin_copy)
            else: 
                output = model(origin_copy, 0, mask_type)
            
            output_chunks.append(output)

            mse_loss = mse_loss_fn(output, chunk)
            mse_running_loss[j] += mse_loss.item()

            mae_loss = mae_loss_fn(output, chunk)
            mae_running_loss[j] += mae_loss.item()

            origin_copy = copy.deepcopy(output)
        
        if i == 9 or i == 10: 

            _, ax = plt.subplots(rollout_times*2+2, 10, figsize=(20, rollout_times*4+2))

            for m in range(10): 
                # visualise input
                ax[0][m].imshow(origin[0][m][0].cpu().detach().numpy())
                ax[0][m].set_xticks([])
                ax[0][m].set_yticks([])
                # ax[0][m].set_title("Timestep {timestep} (Input)".format(timestep=m+1), fontsize=10)
                ax[0][m].set_title("Timestep {timestep} (Input)".format(timestep=int(pos[m]+1)), fontsize=10)
                
                ax[1][m].imshow(masked_origin[0][m][0].cpu().detach().numpy())
                ax[1][m].set_xticks([])
                ax[1][m].set_yticks([])
                ax[1][m].set_title("Timestep {timestep} (Input)".format(timestep=int(pos[m])+1), fontsize=10)

            for k in range(rollout_times): 

                for m in range(10):
                    # visualise output
                    ax[2*k+2][m].imshow(output_chunks[k][0][m][0].cpu().detach().numpy())
                    ax[2*k+2][m].set_xticks([])
                    ax[2*k+2][m].set_yticks([])
                    ax[2*k+2][m].set_title("Timestep {timestep} (Prediction)".format(timestep=int(pos[m])+11+k*10), fontsize=10)
                    # visualise target
                    ax[2*k+3][m].imshow(target_chunks[k][0][m][0].cpu().detach().numpy())
                    ax[2*k+3][m].set_xticks([])
                    ax[2*k+3][m].set_yticks([])
                    ax[2*k+3][m].set_title("Timestep {timestep} (Target)".format(timestep=int(pos[m])+11+k*10), fontsize=10)
                
            plt.tight_layout()
            plt.savefig(os.path.join(rollout_rec_save_path + f"/{i}.png"))
            plt.close()


mse_chunk_losses = []
for loss in mse_running_loss:
    valid_loss = loss / len(val_loader.dataset)
    mse_chunk_losses.append(valid_loss)
mse_losses.append(mse_chunk_losses)

with open(os.path.join(rollout_rec_save_path, 'mse_losses.json'), 'w') as f:
    json.dump(mse_losses, f)

mae_chunk_losses = []
for loss in mae_running_loss:
    valid_loss = loss / len(val_loader.dataset)
    mae_chunk_losses.append(valid_loss)
mae_losses.append(mae_chunk_losses)

with open(os.path.join(rollout_rec_save_path, 'mae_losses.json'), 'w') as f:
    json.dump(mae_losses, f)
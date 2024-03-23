import sys
sys.path.append(".")
import os

import torch
import torch.nn as nn

from dataset_imae import DataBuilder
from torch.utils.data import DataLoader
from model_imae import VisionTransformer

from matplotlib import pyplot as plt

import copy
import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = VisionTransformer(3, 16, 128, device)

checkpoint_path = "/home/uceckz0/Project/imae/data/Vit_checkpoint_1//epoch_{epoch}.pth".format(epoch=198)
rec_save_path = "/home/uceckz0/Project/imae/data/Vit_rec_1_rollout_2"
if not os.path.exists(rec_save_path):
    os.makedirs(rec_save_path)


checkpoint = torch.load(checkpoint_path, map_location="cpu")  
model.load_state_dict(checkpoint["model"])

loss_fn = nn.MSELoss()

test_loss = []

rollout_times = 2
seq_length = 10
mask_ratio = 0.1

dataset = DataBuilder("data/valid_file.csv", seq_length, rollout_times)
# dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

running_loss = []

# Iterating over the training dataset
for i, sample in enumerate(tqdm(dataset)): 

    origin = sample["Input"].float().to(device)
    origin = origin.unsqueeze(0)
    origin_copy = copy.deepcopy(origin)
    target = sample["Target"].float().to(device)
    target = target.unsqueeze(0)
    
    # Divide the target into chunks for each rollout
    target_chunks = torch.chunk(target, rollout_times, dim=1)
    output_chunks = []

    chunk_loss = []
    
    with torch.no_grad():
        for chunk in target_chunks:
            output = model(origin, mask_ratio)
            output_chunks.append(output)
            loss = loss_fn(output, chunk)
            chunk_loss.append(loss.item())
    
    running_loss.append(chunk_loss)

    fig, ax = plt.subplots(rollout_times*2+1, seq_length, figsize=(20, rollout_times*4+2))

    for j in range(seq_length): 
        # visualise input
        ax[0][j].imshow(origin_copy[0][j][0].cpu().detach().numpy())
        ax[0][j].set_xticks([])
        ax[0][j].set_yticks([])
        ax[0][j].set_title("Timestep {timestep} (Input)".format(timestep=j+1), fontsize=10)
        
    for k in range(rollout_times):
        for j in range(seq_length):
            # visualise output
            ax[2*k+1][j].imshow(output_chunks[k][0][j][0].cpu().detach().numpy())
            ax[2*k+1][j].set_xticks([])
            ax[2*k+1][j].set_yticks([])
            ax[2*k+1][j].set_title("Timestep {timestep} (Prediction)".format(timestep=j+11+k*10), fontsize=10)
            # visualise target
            ax[2*k+2][j].imshow(target_chunks[k][0][j][0].cpu().detach().numpy())
            ax[2*k+2][j].set_xticks([])
            ax[2*k+2][j].set_yticks([])
            ax[2*k+2][j].set_title("Timestep {timestep} (Target)".format(timestep=j+11+k*10), fontsize=10)

        
    fig.suptitle("Rollout Loss: {chunk_loss}".format(chunk_loss=chunk_loss))
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("{save_path}/{i}.png".format(save_path = rec_save_path, i = i))
    plt.close()
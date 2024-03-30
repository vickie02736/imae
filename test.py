import sys
sys.path.append("..")
import os

import torch
import torch.nn as nn

from dataset_imae import DataBuilder
from torch.utils.data import DataLoader
from model_imae import VisionTransformer

from matplotlib import pyplot as plt

import copy
import argparse


parser = argparse.ArgumentParser(description='Train Vision Transformer')

parser.add_argument('--category', type=int, default=3, help='Trained category')
parser.add_argument('--mask_ratio', type=float, default=0.1, help='Masking ratio')
parser.add_argument('--rollout_times', type=int, default=1, help='Rollout times')
parser.add_argument('--load_epoch', type=int, default=0, help='Load which checkpoint')

args = parser.parse_args()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
# device = torch.device("cpu")
model = VisionTransformer(3, 16, 128, device)

rollout_times = args.rollout_times
seq_length = 10
mask_ratio = args.mask_ratio
file_number = int(args.mask_ratio * 10)

checkpoint_path = "../data/Vit_checkpoint/5/epoch_{epoch}.pth".format(epoch=args.load_epoch)
rec_save_path = "../data/Vit_rec/rollout/ceiling_{category}/maskratio_{file_number}/rollout_{rollout_times}".format(category=args.category, file_number=file_number)
if not os.path.exists(rec_save_path):
    os.makedirs(rec_save_path)


checkpoint = torch.load(checkpoint_path, map_location=device)  
model.load_state_dict(checkpoint["model"])
model = model.to(device)

loss_fn = nn.MSELoss()

test_loss = []

dataset = DataBuilder("../data/inner_test_file.csv", seq_length, rollout_times)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

running_loss = []

# Iterating over the training dataset
for i, sample in enumerate(dataloader):

    model.eval()

    with torch.no_grad():

        if i == 1:  

            origin = sample["Input"].float().to(device)
            target = sample["Target"].float().to(device)
            origin_copy = copy.deepcopy(origin)

            target_chunks = torch.chunk(target, rollout_times, dim=1)
            output_chunks = []

            chunk_loss = []

            # for chunk in target_chunks:
            for a, chunk in enumerate(target_chunks):
                print(a)
                output = model(origin_copy, 0.3)
                output_chunks.append(output)
                loss = loss_fn(output, chunk)
                chunk_loss.append(loss.item())
                origin_copy = output

            # Plot

            fig, ax = plt.subplots(rollout_times*2+1, seq_length, figsize=(20, rollout_times*4+2))

            for j in range(seq_length): 
                # visualise input
                ax[0][j].imshow(origin[0][j][0].cpu().detach().numpy())
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

                
            # fig.suptitle("Rollout Loss: {chunk_loss}".format(chunk_loss=chunk_loss))
            
            plt.tight_layout()
            # # plt.show()
            plt.savefig("{save_path}/{i}.png".format(save_path = rec_save_path, i = i))
            plt.close()
            
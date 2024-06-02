import os
import json
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

def int_or_string(value):
    if value == "best":
        return value
    else:
        return int(value)


def plot(origin, masked_origin, output_chunks, target_chunks, seq_len, idx, save_path):
    rollout_times = len(output_chunks)
    _, ax = plt.subplots(rollout_times*2+2, seq_len, figsize=(seq_len*2, rollout_times*4+2))
    for j in range(seq_len): 
        # visualise input
        ax[0][j].imshow(origin[0][j][0].cpu().detach().numpy())
        ax[0][j].set_xticks([])
        ax[0][j].set_yticks([])
        ax[0][j].set_title("Timestep {timestep} (Input)".format(timestep=j+1), fontsize=10)
        # visualise masked input
        ax[1][j].imshow(masked_origin[0][j][0].cpu().detach().numpy())
        ax[1][j].set_xticks([])
        ax[1][j].set_yticks([])
        ax[1][j].set_title("Timestep {timestep} (Masked Input)".format(timestep=j+1), fontsize=10)
    for k in range(rollout_times): 
        for j in range(seq_len):
            # visualise output
            ax[2*k+2][j].imshow(output_chunks[k][0][j][0].cpu().detach().numpy())
            ax[2*k+2][j].set_xticks([])
            ax[2*k+2][j].set_yticks([])
            ax[2*k+2][j].set_title("Timestep {timestep} (Prediction)".format(timestep=j+(k+1)*seq_len), fontsize=10)
            # visualise target
            ax[2*k+3][j].imshow(target_chunks[k][0][j][0].cpu().detach().numpy())
            ax[2*k+3][j].set_xticks([])
            ax[2*k+3][j].set_yticks([])
            ax[2*k+3][j].set_title("Timestep {timestep} (Target)".format(timestep=j+(k+1)*seq_len), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{idx}.png"))
    plt.close()



def save_losses(epoch, loss_data, save_path, save_name):
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, save_name), 'r') as f: 
            losses = json.load(f)
        losses[epoch]=loss_data
        with open(os.path.join(save_path, save_name), 'w') as f: 
            json.dump(losses, f)


def mask(x, mask_mtd = "zeros", test_flag=False, mask_ratio=None): 
    seq_lenth = len(x[0])
    if test_flag == False: 
        mask_ratio = torch.rand(1).item()
    else:
        mask_ratio = mask_ratio
    num_mask = int(1 + mask_ratio * (seq_lenth - 2))
    weights = torch.ones(x.shape[1]).expand(x.shape[0], -1)
    idx = torch.multinomial(weights, num_mask, replacement=False)
    if mask_mtd == "zeros":
        masked_tensor = torch.zeros(x.shape[2], x.shape[3], x.shape[4]).to(x.device)
    elif mask_mtd == "random":
        masked_tensor = torch.rand(x.shape[2], x.shape[3], x.shape[4]).to(x.device)
    batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand(-1, num_mask)
    x[batch_indices, idx] = masked_tensor
    return x, idx
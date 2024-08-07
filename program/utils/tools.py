import os
import json
import torch
import argparse

def int_or_string(value):
    if value == "best":
        return value
    else:
        return int(value)

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_losses(epoch, loss_data, save_path):
    with open(os.path.join(save_path), 'r') as f:
        losses = json.load(f)
    losses[epoch] = loss_data
    with open(os.path.join(save_path), 'w') as f:
        json.dump(losses, f)


def mask(x, mask_mtd="zeros", test_flag=False, mask_ratio=None):
    seq_lenth = len(x[0])
    if test_flag == False:
        mask_ratio = torch.rand(1).item()
    else:
        mask_ratio = mask_ratio
    num_mask = int(1 + mask_ratio * (seq_lenth - 2))
    weights = torch.ones(x.shape[1]).expand(x.shape[0], -1)
    idx = torch.multinomial(weights, num_mask, replacement=False)
    if mask_mtd == "zeros":
        masked_tensor = torch.zeros(x.shape[2], x.shape[3],
                                    x.shape[4]).to(x.device)
    elif mask_mtd == "random":
        masked_tensor = torch.rand(x.shape[2], x.shape[3],
                                   x.shape[4]).to(x.device)
    batch_indices = torch.arange(x.shape[0],
                                 device=x.device).unsqueeze(1).expand(
                                     -1, num_mask)
    x[batch_indices, idx] = masked_tensor
    return x, idx

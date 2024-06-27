import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import sys

sys.path.append("..")


class seq_DataBuilder(Dataset):

    def __init__(self, dataset,
                 input_length,
                 rollout_times,
                 inner = True, 
                 transform=None):
        '''
        input_length: the length of input
        rollout_times * input_length = the length of target
        dataset: train valid test
        '''

        self.input_length = input_length
        self.rollout_times = rollout_times
        self.total_length = input_length * (rollout_times + 1)

        ### calculate min max for normalization
        self.min = np.min(train_data)
        self.max = np.max(train_data)

        self.all_clips = []
        if dataset == 'train':
            list = np.load('../database/moving_mnist/data/train_data.npy', allow_pickle=True)
        elif dataset == 'valid':
            list = np.load('../database/moving_mnist/data/val_data.npy', allow_pickle=True)
        elif dataset == 'test':
            list = np.load('../database/moving_mnist/data/test_data.npy', allow_pickle=True)
        else:
            pass

        for i in range(len(list)):
            each_sequence = list[i]
            clips = self.cut_clips(each_sequence)
            self.all_clips.extend(clips)

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        clip = self.all_clips[idx]

        input_sequence = clip[0:self.input_length]
        input_sequence = self.normalize(input_sequence, self.min, self.max)
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        input_sequence = input_sequence.unsqueeze(1)  # add channel dimension

        target_sequence = clip[self.input_length:self.total_length]
        target_sequence = self.normalize(target_sequence, self.min, self.max)
        target_sequence = torch.tensor(target_sequence, dtype=torch.float32)
        target_sequence = target_sequence.unsqueeze(1)

        return {"Input": input_sequence, "Target": target_sequence}

    def cut_clips(self, sorted_df):
        clips = []
        for start in range(len(sorted_df) - self.total_length + 1):
            end = start + self.total_length
            clip = sorted_df[start:end]
            clips.append(clip)
        return clips  # all clips from the same sequence

    def normalize(self, arr, mins, maxs):
        normalized_arr = (arr - mins) / (maxs - mins)
        return normalized_arr


class fra_DataBuilder(Dataset):
    def __init__(self, dataset, transform=None):
        list = np.load('../database/moving_mnist/data/mnist_test_seq.npy',
                       allow_pickle=True)
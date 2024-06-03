import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import sys

sys.path.append("..")


class seq_DataBuilder(Dataset):

    def __init__(self,
                 input_length,
                 rollout_times,
                 dataset,
                 train_ratio=0.8,
                 transform=None):
        '''
        input_length: the length of input
        rollout_times * input_length = the length of target
        dataset: train valid test
        '''

        list = np.load('../database/moving_mnist/data/mnist_test_seq.npy',
                       allow_pickle=True)
        transposed_list = np.transpose(list, (1, 0, 2, 3))

        self.input_length = input_length
        self.rollout_times = rollout_times
        self.total_length = input_length * (rollout_times + 1)

        test_size = 1 - train_ratio

        train_data, temp_data = train_test_split(transposed_list,
                                                 test_size=test_size,
                                                 random_state=42)
        valid_data, test_data = train_test_split(temp_data,
                                                 test_size=0.5,
                                                 random_state=42)

        ### calculate min max for normalization
        self.min = np.min(train_data)
        self.max = np.max(train_data)

        all_clips = []
        if dataset == 'train':
            for i in range(len(train_data)):
                each_sequence = train_data[i]
                clips = self.cut_clips(each_sequence)
                all_clips.extend(clips)
            self.all_clips = all_clips
        elif dataset == 'valid':
            for i in range(len(valid_data)):
                each_sequence = valid_data[i]
                clips = self.cut_clips(each_sequence)
                all_clips.extend(clips)
            self.all_clips = all_clips
        elif dataset == 'test':
            for i in range(len(test_data)):
                each_sequence = test_data[i]
                clips = self.cut_clips(each_sequence)
                all_clips.extend(clips)
            self.all_clips = all_clips
        else:
            pass

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

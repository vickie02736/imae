import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import ast
import torch
import sys

sys.path.append("..")
import yaml

config = yaml.load(open("../database/shallow_water/config.yaml", "r"),
                   Loader=yaml.FullLoader)


class seq_DataBuilder(Dataset):

    def __init__(self,
                 dataset,
                 clip_length,
                 rollout_times,
                 timestep,
                 transform=None):
        '''
        dataset_path: config['train_file'], config['valid_file'], config['inner_test_file'], config['outer_test_file']
        clip_length: the length of input
        rollout_times * clip_length = the length of target
        '''

        self.clip_length = clip_length
        self.rollout_times = rollout_times
        self.total_length = clip_length * (rollout_times + 1)

        df = pd.read_csv(
            f"../database/shallow_water/dataset_split/{timestep}timestep.csv")
        df = df[df["Key"].isin(dataset)]
        unique_keys = df["Key"].unique()

        all_clips = []
        for i in range(len(unique_keys)):
            filtered_df = df[df["Key"] == unique_keys[i]]  # Take each video
            # sorted each sequence
            sorted_df = filtered_df.sort_values(by=["Pos"])
            clips = self.cut_clips(sorted_df)
            all_clips.extend(clips)
        self.all_clips = all_clips

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):

        clip = self.all_clips[idx]
        param_R = torch.tensor(clip["R"].to_numpy(), dtype=torch.float32)
        param_Hp = torch.tensor(clip["Hp"].to_numpy(), dtype=torch.float32)
        param_Pos = torch.tensor(clip["Pos"].to_numpy(), dtype=torch.float32)
        param_Label = clip["Label"].values
        param_Label = [
            ast.literal_eval(param_Label[i]) for i in range(len(param_Label))
        ]
        param_Label = torch.tensor(param_Label)

        image_address = clip["Address"].iloc[
            0]  # the images in the same clip should have the same address
        full_sequence = np.load(image_address,
                                allow_pickle=True,
                                mmap_mode='r')

        mins = config['min']
        maxs = config['max']

        sub_sequance = full_sequence[clip["Pos"]]
        sub_sequance = self.normalize(sub_sequance, mins, maxs)
        sub_sequance = torch.tensor(sub_sequance, dtype=torch.float32)

        input_idx = clip["Pos"].iloc[0:self.clip_length]
        target_idx = clip["Pos"].iloc[self.clip_length:self.total_length]

        input_sequence = full_sequence[input_idx]
        input_sequence = self.normalize(input_sequence, mins, maxs)
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)

        target_sequence = full_sequence[target_idx]
        target_sequence = self.normalize(target_sequence, mins, maxs)
        target_sequence = torch.tensor(target_sequence, dtype=torch.float32)

        return {
            "R": param_R,
            "Hp": param_Hp,
            "Pos": param_Pos,
            "Label": param_Label,
            "Input": input_sequence,
            "Target": target_sequence,
            "Sub_S": sub_sequance
        }

    def cut_clips(self, sorted_df):
        clips = []
        for start in range(len(sorted_df) - self.total_length + 1):
            end = start + self.total_length
            clip = sorted_df.iloc[start:end]
            clips.append(clip)
        return clips  # all clips from the same sequence

    def normalize(self, arr, mins, maxs):
        normalized_arr = np.empty_like(arr, dtype=np.float32)
        for c in range(arr.shape[1]):
            normalized_arr[:, c, :, :] = 2 * ((arr[:, c, :, :] - mins[c]) /
                                              (maxs[c] - mins[c])) - 1
        return normalized_arr


class fra_DataBuilder(Dataset):

    def __init__(self, dataset, timestep, transform=None):
        '''
        dataset_path: config['train_file'], config['valid_file'], config['inner_test_file'], config['outer_test_file']
        '''
        df = pd.read_csv(
            f"../database/shallow_water/dataset_split/{timestep}timestep.csv")

        self.all_imgs = df[df["Key"].isin(dataset)]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img = self.all_imgs.iloc[idx]
        param_R = torch.tensor(img["R"], dtype=torch.float32)
        param_Hp = torch.tensor(img["Hp"], dtype=torch.float32)
        param_Pos = torch.tensor(img["Pos"], dtype=torch.float32)
        param_Label = img["Label"]
        image_address = img["Address"]
        all_img = np.load(image_address, allow_pickle=True, mmap_mode='r')
        img = all_img[int(param_Pos)]

        mins = config['min']
        maxs = config['max']

        normalized_img = self.normalize(img, mins, maxs)

        return {
            "R": param_R,
            "Hp": param_Hp,
            "Pos": param_Pos,
            "Label": param_Label,
            "Frame": normalized_img
        }

    def normalize(self, img, mins, maxs):
        normalized_img = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[0]):
            normalized_img[c, :, :] = 2 * ((img[c, :, :] - mins[c]) /
                                           (maxs[c] - mins[c])) - 1
        return normalized_img


# file = config['train_file']
# dataset = img_DataBuilder(file, timestep=100)

# from torch.utils.data import DataLoader
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# for i, data in enumerate(dataloader):
#     print(data["Frame"].shape)
#     break

import random
import pandas as pd
import numpy as np

import os
import sys
import yaml

sys.path.append("..")

SEED = 3409
random.seed(SEED)

# data_dir = '../data/shallow_water_simulation'
config = yaml.load(open("../database/shallow_water/config.yaml", "r"), Loader=yaml.FullLoader)

R_list = config['R']
Hp_list = config['Hp']
all_keys = {f"R_{R}_Hp_{Hp}" for R in R_list for Hp in Hp_list}


# outer_test
outer_test_Rs = {R_list[0], R_list[-1]}
outer_test_Hps = {Hp_list[0], Hp_list[-1]}
outer_test_pairs = {(R, Hp) for R in outer_test_Rs for Hp in Hp_list}.union({(R, Hp) for Hp in outer_test_Hps for R in R_list})
outer_test_keys = {f"R_{R}_Hp_{Hp}" for R, Hp in outer_test_pairs}


# inner_test
inner_test_pair_set = set()
for i in range(len(R_list) - 2):
    if i + 2 < len(R_list):
        inner_test_pair_set.add((R_list[i + 1], Hp_list[i + 1]))
        inner_test_pair_set.add((R_list[i + 1], Hp_list[len(Hp_list)-1 - i]))
inner_test_keys = {f"R_{R}_Hp_{Hp}" for R, Hp in inner_test_pair_set}


# train and valid
remaining_keys = all_keys - inner_test_keys - outer_test_keys
outer_test_list = list(outer_test_keys)
inner_test_list = list(inner_test_keys)
remaining_list = list(remaining_keys)
random.shuffle(remaining_list)
split_point = int(len(remaining_list) * 0.80)

train_list = remaining_list[:split_point]
valid_list = remaining_list[split_point:]


file_data_pairs = {
    'inner_test_file': inner_test_list,
    'outer_test_file': outer_test_list,
    'valid_file': valid_list,
    'train_file': train_list, 
}



# for filename, data_dict in file_data_pairs.items():

#     with open(f"./json/{filename}.json", 'w') as file:
#         files = list(data_dict)
#         file_names = [os.path.splitext(i)[0] for i in files]
#         file_paths = [os.path.join(data_dir, i+".npy") for i in files]
#         file_list = dict(zip(file_names, file_paths))
#         json.dump(file_list, file)

#     with open(f"./txt/{filename}.txt", 'w') as file:
#         for item in data_dict:
#             file.write(item + '\n')

#     data = pd.DataFrame(list(file_list.items()), columns=['Key', 'Address'])
#     data[['R', 'Hp']] = data['Key'].str.extract(r'R_(\d+)_Hp_(\d+)')
#     data['R'] = pd.to_numeric(data['R'])
#     data['Hp'] = pd.to_numeric(data['Hp'])
#     new_rows = [row.tolist() + [i] for _, row in data.iterrows() for i in range(0, 100)]
#     data = pd.DataFrame(new_rows, columns=['Key', 'Address', 'R', 'Hp', 'Pos'])
#     data['Label'] = [[a, b, c] for a, b, c in zip(data['R'], data['Hp'], data['Pos'])]
#     data.to_csv(f"./csv/{filename}.csv", index=False)
    


df = pd.read_csv("../database/shallow_water/dataset_split/100timestep.csv")
train_df = df[df["Key"].isin(train_list)]

# calculate the min and max of the training dataset, for nomalization
def calculate_min_max(df):
    mins, maxs = [], []
    arr = []
    for i in range(len(df)):
        full_sequence = np.load(df["Address"].iloc[i], allow_pickle=True, mmap_mode='r')
        image = full_sequence[df["Pos"].iloc[i]]
        arr.append(image)
    arr = np.stack(arr)

    # normalized_arr = np.empty_like(arr)
    for c in range(arr.shape[1]):
        min_val = arr[:, c, :, :].min()
        max_val = arr[:, c, :, :].max()
        # normalized_arr = 2 * ((arr[:, c, :, :] - min_val) / (max_val - min_val)) - 1
        mins.append(min_val)
        maxs.append(max_val)
    return mins, maxs

mins, maxs = calculate_min_max(train_df)
min_max = {'min': [float(x) for x in mins], 'max': [float(x) for x in maxs]}


with open('../database/shallow_water/config.yaml', 'r') as yaml_file:
    config_data = yaml.safe_load(yaml_file) or {}
config_data.update(file_data_pairs)
config_data.update(min_max)
with open('../database/shallow_water/config.yaml', 'w') as yaml_file:
    yaml.dump(config_data, yaml_file, default_flow_style=False)
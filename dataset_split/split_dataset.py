import json
import random
import pandas as pd

import os
import sys
sys.path.append("..")

SEED = 3409
random.seed(SEED)

data_dir = '../data/shallow_water_simulation'

R_list = [36, 49, 64, 72, 81, 90, 100, 110, 121, 132, 144, 150, 160, 169, 180, 196]
Hp_list = [x for x in range(2, 22)]
all_keys = {f"R_{R}_Hp_{Hp}" for R in R_list for Hp in Hp_list}


# outer_test
outer_test_Rs = {36, 196}
outer_test_Hps = {2, 21}
outer_test_pairs = {(R, Hp) for R in outer_test_Rs for Hp in Hp_list}.union({(R, Hp) for Hp in outer_test_Hps for R in R_list})
outer_test_keys = {f"R_{R}_Hp_{Hp}" for R, Hp in outer_test_pairs}


# inner_test
inner_test_pair_set = set()
for i in range(len(R_list) - 2):
    if i + 4 < len(Hp_list):
        inner_test_pair_set.add((R_list[i + 1], Hp_list[i + 3]))
    index_hp = 17 - i
    if index_hp < len(Hp_list):
        inner_test_pair_set.add((R_list[i + 1], Hp_list[index_hp]))
inner_test_keys = {f"R_{R}_Hp_{Hp}" for R, Hp in inner_test_pair_set}


# train and valid
remaining_keys = all_keys - inner_test_keys - outer_test_keys
remaining_list = list(remaining_keys)
random.shuffle(remaining_list)
split_point = int(len(remaining_list) * 0.80)

train_keys = set(remaining_list[:split_point])
valid_keys = set(remaining_list[split_point:])


file_data_pairs = {
    'inner_test_file': inner_test_keys,
    'outer_test_file': outer_test_keys,
    'valid_file': valid_keys,
    'train_file': train_keys
}

for filename, data_dict in file_data_pairs.items():

    with open(filename+".json", 'w') as outfile:
        files = list(data_dict)
        file_names = [os.path.splitext(i)[0] for i in files]
        file_paths = [os.path.join(data_dir, i+".npy") for i in files]
        file_list = dict(zip(file_names, file_paths))
        json.dump(file_list, outfile)

    data = pd.DataFrame(list(file_list.items()), columns=['Key', 'Address'])
    data[['R', 'Hp']] = data['Key'].str.extract(r'R_(\d+)_Hp_(\d+)')
    data['R'] = pd.to_numeric(data['R'])
    data['Hp'] = pd.to_numeric(data['Hp'])
    new_rows = [row.tolist() + [i] for _, row in data.iterrows() for i in range(0, 200)] # 200 is the number of timesteps
    data = pd.DataFrame(new_rows, columns=['Key', 'Address', 'R', 'Hp', 'Pos'])
    data['Label'] = [[a, b, c] for a, b, c in zip(data['R'], data['Hp'], data['Pos'])]
    data.to_csv(filename+".csv", index=False)
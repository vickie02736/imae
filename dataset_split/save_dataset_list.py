import os
import json
import pandas as pd

import sys
sys.path.append("..")


data_dir = '../data/shallow_water_simulation_rollout_test'
save_json_path = './inner_rollout_test.json'
save_csv_path = './inner_rollout_test.csv'


files = os.listdir(data_dir)

def extract_number(file_name):
    try:
        return int(file_name.split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return float('inf')  # Return a large number to put this file at the end
files = sorted(files, key=extract_number)


file_names = [os.path.splitext(i)[0] for i in files]
file_paths = [os.path.join(data_dir, i) for i in files]
file_list = dict(zip(file_names, file_paths))

with open(save_json_path, 'w') as outfile:
    json.dump(file_list, outfile)
    

# Columns = ['Key', 'Address']
data = pd.DataFrame(list(file_list.items()), columns=['Key', 'Address'])


# Columns = ['Key', 'Address', 'R', 'Hp']
data[['R', 'Hp']] = data['Key'].str.extract(r'R_(\d+)_Hp_(\d+)')
# Converting columns to numeric type
data['R'] = pd.to_numeric(data['R'])
data['Hp'] = pd.to_numeric(data['Hp'])


# Columns = ['Key', 'Address', 'R', 'Hp', 'Pos']
new_rows = [row.tolist() + [i] for _, row in data.iterrows() for i in range(0, 200)] # 200 is the number of timesteps
data = pd.DataFrame(new_rows, columns=['Key', 'Address', 'R', 'Hp', 'Pos'])

# Columns = ['Key', 'Address', 'R', 'Hp', 'Pos', 'Label']
# Label = [R, Hp, Pos]
data['Label'] = [[a, b, c] for a, b, c in zip(data['R'], data['Hp'], data['Pos'])]


data.to_csv(save_csv_path, index=False)
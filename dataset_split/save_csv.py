import pandas as pd
import numpy as np
import json

json_file = './rollout_example.json'
save_path = './rollout_example.csv'

with open(json_file, 'r') as file:
    data = json.load(file)

# Columns = ['Key', 'Address']
data = pd.DataFrame(list(data.items()), columns=['Key', 'Address'])


# Columns = ['Key', 'Address', 'R', 'Hp']
data[['R', 'Hp']] = data['Key'].str.extract(r'R_(\d+)_Hp_(\d+)')
# Converting columns to numeric type
data['R'] = pd.to_numeric(data['R'])
data['Hp'] = pd.to_numeric(data['Hp'])


# Columns = ['Key', 'Address', 'R', 'Hp', 'Pos']
new_rows = [row.tolist() + [i] for _, row in data.iterrows() for i in range(0, 200)]
data = pd.DataFrame(new_rows, columns=['Key', 'Address', 'R', 'Hp', 'Pos'])

# Columns = ['Key', 'Address', 'R', 'Hp', 'Pos', 'Label']
# Label = [R, Hp, Pos]
data['Label'] = [[a, b, c] for a, b, c in zip(data['R'], data['Hp'], data['Pos'])]



data.to_csv(save_path, index=False)
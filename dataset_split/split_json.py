import json


all_file= "./all_file.json"

with open(all_file, 'r') as infile:
    data = json.load(infile) # len(data) = 266

position_twin = ['R_36_Hp_2']
valid_keys = ['R_49_Hp_3','R_90_Hp_3',
              'R_49_Hp_5','R_72_Hp_5','R_90_Hp_5','R_121_Hp_5',
              'R_49_Hp_7','R_90_Hp_7',
              'R_49_Hp_9','R_72_Hp_9','R_90_Hp_9','R_121_Hp_9',
              'R_49_Hp_11','R_90_Hp_11',
              'R_49_Hp_13','R_72_Hp_13','R_90_Hp_13','R_121_Hp_13',
              'R_49_Hp_15','R_90_Hp_15',
              'R_49_Hp_17','R_72_Hp_17','R_90_Hp_17','R_121_Hp_17']
inner_test_keys = ['R_72_Hp_3','R_121_Hp_3',
                   'R_72_Hp_7','R_121_Hp_7',
                   'R_72_Hp_11','R_121_Hp_11',
                   'R_72_Hp_15','R_121_Hp_15',]
outer_test_keys = ['R_36_Hp_19','R_49_Hp_19','R_64_Hp_19','R_72_Hp_19','R_81_Hp_19', 
                   'R_90_Hp_19','R_100_Hp_19','R_121_Hp_19','R_132_Hp_19','R_144_Hp_19',
                   'R_144_Hp_2','R_144_Hp_3','R_144_Hp_4','R_144_Hp_5','R_144_Hp_6',
                   'R_144_Hp_7','R_144_Hp_8','R_144_Hp_9','R_144_Hp_10','R_144_Hp_11',
                   'R_144_Hp_12','R_144_Hp_13','R_144_Hp_14','R_144_Hp_15','R_144_Hp_16',
                   'R_144_Hp_17']
all_included_keys = set(position_twin + valid_keys + inner_test_keys + outer_test_keys)
train_keys = set(data.keys()) - all_included_keys

train_dict = {key: data[key] for key in train_keys}
valid_dict = {key: data[key] for key in valid_keys}
inner_test_dict = {key: data[key] for key in inner_test_keys}
outer_test_dict = {key: data[key] for key in outer_test_keys} 

file_data_pairs = {
    'inner_test_file.json': inner_test_dict,
    'outer_test_file.json': outer_test_dict,
    'valid_file.json': valid_dict,
    'train_file.json': train_dict
}

for filename, data_dict in file_data_pairs.items():
    with open(filename, 'w') as outfile:
        json.dump(data_dict, outfile)
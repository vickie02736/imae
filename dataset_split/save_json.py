import os
import json
import pandas as pd
import random

import sys
sys.path.append("..")


data_dir = '../data/shallow_water_simulation_rollout_test'
save_path = './rollout_example.json'

def save_json(data_dir, save_path, should_sort=True):
    files = os.listdir(data_dir)
    
    if should_sort:
        def extract_number(file_name):
            try:
                return int(file_name.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                return float('inf')  # Return a large number to put this file at the end

        files = sorted(files, key=extract_number)

    file_names = [os.path.splitext(i)[0] for i in files]
    file_paths = [os.path.join(data_dir, i) for i in files]
    file_list = dict(zip(file_names, file_paths))

    with open(save_path, 'w') as outfile:
        json.dump(file_list, outfile)


save_json(data_dir, save_path)
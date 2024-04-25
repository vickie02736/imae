import os
import pandas as pd

directory_path = '../data/shallow_water_simulation_inner_rollout'
save_name = "inner_rollout_test_file"

file_names = []

# Read all file names in the directory
for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)
    file_names.append(file_path)
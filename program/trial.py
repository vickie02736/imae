import itertools

model_list = ['imae', 'convlstm', 'cae_lstm']
rollout_list = [2, 3, 4]
mask_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
timestep_list = [100, 120]
bound_list = ['inner_test_file', 'outer_test_file']

# Generate all combinations
combinations = list(itertools.product(model_list, rollout_list, mask_ratio, timestep_list, bound_list))

# Print all combinations
for combination in combinations:
    print(combination[0])
    break

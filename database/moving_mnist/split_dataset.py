import numpy as np

# Load the data
data = np.load('./data/mnist_test_seq.npy', allow_pickle=True)

# Permute the array to match the desired shape
data = np.transpose(data, (1, 0, 2, 3))

# Add an extra dimension
data = np.expand_dims(data, axis=2)

# Determine the split sizes
num_samples = data.shape[0]
train_size = int(num_samples * 0.8)
val_size = int(num_samples * 0.1)
test_size = num_samples - train_size - val_size  # To handle any rounding issues

# Split the data
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Save the split data to .npy files
np.save('./data/train_data.npy', train_data)
np.save('./data/val_data.npy', val_data)
np.save('./data/test_data.npy', test_data)
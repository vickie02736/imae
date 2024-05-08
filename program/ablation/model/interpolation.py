import numpy as np
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

sequence = [np.random.rand(3, 128, 128) if i not in [2, 3, 7] else None for i in range(10)]


def linear_interpolation(sequence):
    length = len(sequence)
    filled_sequence = sequence.copy()

    i = 0
    while i < length:
        if filled_sequence[i] is None:
            start = i
            
            # Find the end of the block of missing frames
            while i < length and filled_sequence[i] is None:
                i += 1
            end = i

            # Get the previous and next frames if available
            prev_frame = filled_sequence[start - 1] if start > 0 else None
            next_frame = filled_sequence[end] if end < length else None

            # Handle cases where missing frames are at the start or end
            if prev_frame is None:
                filled_sequence[start:end] = [next_frame] * (end - start)
            elif next_frame is None:
                filled_sequence[start:end] = [prev_frame] * (end - start)
            else:
                # Interpolate across the block
                step = (next_frame - prev_frame) / (end - start + 1)
                for j in range(end - start):
                    filled_sequence[start + j] = prev_frame + step * (j + 1)
        else:
            i += 1

    return filled_sequence



data = np.array([1, 2, None, 4, None, 6, 7, None, 9, 10])
x_observed = np.array([i for i, v in enumerate(data) if v is not None])
y_observed = np.array([v for v in data if v is not None])

# Reshape for scikit-learn
X_observed = x_observed.reshape(-1, 1)
y_observed = y_observed.reshape(-1, 1)

# Define a kernel with parameters to be learned
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))

# Create and fit Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_observed, y_observed)

# Predict missing values
x_missing = np.array([i for i, v in enumerate(data) if v is None])
X_missing = x_missing.reshape(-1, 1)
y_pred, sigma = gp.predict(X_missing, return_std=True)

# Output results
for i, val in zip(x_missing, y_pred.flatten()):
    print(f"Predicted value at position {i}: {val} +/- {sigma[i - len(x_observed)]}")




# def kriging_interpolation():
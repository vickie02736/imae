import torch
import torch.nn.functional as F

def linear_interpolation(batch_frames, known_mask_idx):
    batch_size, seq_len, channels, height, width = batch_frames.shape
    interpolated_frames = batch_frames.clone()

    for b in range(batch_size):
        frames = batch_frames[b]
        known_indices = [i for i in range(seq_len) if i not in known_mask_idx[b]]
        missing_indices = known_mask_idx[b].tolist()

        for idx in missing_indices:
            prev_idx = max([i for i in known_indices if i < idx], default=None)
            next_idx = min([i for i in known_indices if i > idx], default=None)

            if prev_idx is None:
                interpolated_frames[b, idx] = frames[next_idx]
            elif next_idx is None:
                interpolated_frames[b, idx] = frames[prev_idx]
            else:
                alpha = (idx - prev_idx) / (next_idx - prev_idx)
                interpolated_frames[b, idx] = (1 - alpha) * frames[prev_idx] + alpha * frames[next_idx]

    return interpolated_frames



def gaussian_kernel1d(size, sigma):
    coords = torch.arange(size).float() - (size - 1) / 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g /= g.sum()
    return g



def gaussian_interpolation(batch_frames, known_mask_idx, sigma=1.0):
    batch_size, seq_len, channels, height, width = batch_frames.shape
    interpolated_frames = batch_frames.clone()

    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = gaussian_kernel1d(kernel_size, sigma).view(1, 1, -1).to(batch_frames.device)
    
    for b in range(batch_size):
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    values = batch_frames[b, :, c, h, w]
                    known_mask = torch.ones(seq_len, device=batch_frames.device)
                    known_mask[known_mask_idx[b]] = 0
                    
                    # Convolve only known values and normalize by known mask convolved
                    smoothed_values = F.conv1d(values.view(1, 1, -1) * known_mask.view(1, 1, -1), kernel, padding=kernel_size//2)
                    smoothed_known = F.conv1d(known_mask.view(1, 1, -1), kernel, padding=kernel_size//2)
                    
                    smoothed_values /= smoothed_known + 1e-10  # Avoid division by zero
                    interpolated_frames[b, :, c, h, w] = smoothed_values.view(-1)

    return interpolated_frames


# Example usage
# x = torch.rand(2, 10, 3, 128, 128)
# from tools import mask
# x, idx = mask(x)
# interpolated_frames = linear_interpolation(x, idx)
# # interpolated_frames = gaussian_interpolation(x, idx)
# print(interpolated_frames.shape)
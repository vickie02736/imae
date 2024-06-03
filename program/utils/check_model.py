import torch

checkpoint = torch.load(
    '/home/uceckz0/Project/imae/data/shallow_water/ckpt/checkpoint_0.pth',
    map_location="cpu")
print(checkpoint['model'].keys())

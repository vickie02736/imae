import torch
import torch.nn as nn
from piqa import SSIM


class RMSELoss(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class PSNRLoss(nn.Module):

    def __init__(self, max_pixel=1.0):
        super(PSNRLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.max_pixel = max_pixel

    def forward(self, img1, img2):
        mse = self.mse_loss(img1, img2)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 20 * torch.log10(self.max_pixel / torch.sqrt(mse))


class SSIMLoss():

    def __init__(self, device):
        self.ssim_loss = SSIM().to(device)
        self.device = device

    def forward(self, output, chunk):
        output = (output - output.min()) / (output.max() - output.min())
        chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
        if output.shape[2] == 1:
            output = output.repeat(1, 1, 3, 1, 1)
            chunk = chunk.repeat(1, 1, 3, 1, 1)
        ssim_values = torch.zeros(len(output),
                                  len(output[0]),
                                  device=self.device)
        for i in range(len(output[0])):
            ssim_values[:, i] = self.ssim_loss(output[:, i], chunk[:, i])
        loss = ssim_values.mean(dim=1)
        return loss.sum()

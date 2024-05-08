import torch
import torch.nn as nn
import math


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
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
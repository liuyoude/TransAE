import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ASDLoss(nn.Module):
    def __init__(self, w_mse, w_ce):
        super(ASDLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.w_ce = w_ce
        self.w_mse = w_mse

    def forward(self, out, logit, target, label):
        mse_loss = self.mse_loss(out, target)
        ce_loss = self.ce_loss(logit, label)
        loss = self.w_mse * mse_loss + self.w_ce * ce_loss
        return loss, mse_loss, ce_loss
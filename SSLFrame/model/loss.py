# --*-- conding:utf-8 --*--
# @Time  : 2024/6/10
# @Author: weibo
# @Email : csbowei@gmail.com
# @Description: 损失函数

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        # NCE (Noise Contrastive Estimation)
        nce_loss = F.cross_entropy(outputs, targets.argmax(dim=1))

        # RCE (Reverse Cross-Entropy)
        prob = F.softmax(outputs, dim=1)
        log_prob = torch.log(prob + 1e-10)  # add small value to prevent log(0)
        rce_loss = -torch.mean(torch.sum(log_prob * targets, dim=1))

        # Combined Loss
        loss = self.alpha * nce_loss + self.beta * rce_loss
        return loss

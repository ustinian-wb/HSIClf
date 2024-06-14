# --*-- conding:utf-8 --*--
# @Time  : 2024/6/3
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : cnn.py
# @Description: 自定义ResNet结构

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter
import common_utils
from SSLFrame.model.loss import CombinedLoss
from tqdm import tqdm


# 自定义残差块
class CustomResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CustomResidualBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样层，用于调整维度
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x

        # 第一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层卷积
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果有下采样层，则调整输入维度
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class CustomResNet(nn.Module):
    def __init__(self, in_channels, num_class):
        super(CustomResNet, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 各个残差块层
        self.layer1 = self._resnet_block(CustomResidualBlock, 64, 64, 3, stride=1)
        self.layer2 = self._resnet_block(CustomResidualBlock, 64, 128, 3, stride=2)
        self.layer3 = self._resnet_block(CustomResidualBlock, 128, 256, 3, stride=2)
        self.layer4 = self._resnet_block(CustomResidualBlock, 256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_class)

    def _resnet_block(self, block, input_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(input_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, mode='train'):
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if mode == 'feature':
            return x

        x = self.fc1(x)
        x = self.fc2(x)
        return x

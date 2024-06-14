# --*-- conding:utf-8 --*--
# @Time  : 2024/6/10
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : hetcnn.py
# @Description:

import torch.nn as nn
import torch.nn.functional as F


class HetConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HetConv3D, self).__init__()
        self.spectral_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 7), stride=(1, 1, 1),
                                       padding=(0, 0, 3))
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), stride=(1, 1, 1),
                                      padding=(1, 1, 0))
        self.spectral_bn = nn.BatchNorm3d(out_channels)
        self.spatial_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        spectral_out = F.relu(self.spectral_bn(self.spectral_conv(x)))
        spatial_out = F.relu(self.spatial_bn(self.spatial_conv(x)))
        out = spectral_out + spatial_out
        return out


class HSI3DNet(nn.Module):
    def __init__(self, num_classes=16, fc1_in=768):
        super(HSI3DNet, self).__init__()
        self.hetconv1 = HetConv3D(1, 24)
        self.hetconv2 = HetConv3D(24, 24)
        self.hetconv3 = HetConv3D(24, 24)
        self.avgpool = nn.AvgPool3d((1, 9, 9))
        self.dropout = nn.Dropout(0.5)

        # IndianPines: 1536; PaviaUniversity, SalinasScene:768
        self.fc1 = nn.Linear(fc1_in, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, mode='train'):
        x = self.hetconv1(x)
        x = self.hetconv2(x)
        x = self.hetconv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if mode == 'feature':
            return x

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

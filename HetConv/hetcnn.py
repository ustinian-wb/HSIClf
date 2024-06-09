# --*-- conding:utf-8 --*--
# @Time  : 2024/6/10
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : hetcnn.py
# @Description:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from loss import CombinedLoss
import numpy as np
import common_utils

from collections import Counter

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
    def __init__(self, num_classes=16):
        super(HSI3DNet, self).__init__()
        self.hetconv1 = HetConv3D(1, 24)
        self.hetconv2 = HetConv3D(24, 24)
        self.hetconv3 = HetConv3D(24, 24)
        self.avgpool = nn.AvgPool3d((1, 9, 9))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1536, 128)
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

    def test(self, train_dataset, test_dataset, epochs=50, lr=0.0005, batch_size=128, save_model=False, save_path=None):
        X_train, y_train = train_dataset
        X_test, y_test = test_dataset

        # 将数据转换为 PyTorch 张量，并调整形状
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).cuda()
        # 构造 TensorDataset 和 DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        # 定义损失函数和优化器
        criterion = CombinedLoss(0.5, 0.5)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # 训练网络
        train_losses, test_losses, test_accuracies = [], [], []
        best_accuracy = 0.0  # 记录最佳准确率
        best_model_path = save_path  # 保存最佳模型的路径

        for epoch in range(epochs):
            # 训练模型
            self.train()
            epoch_train_losses = []
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())

            train_loss = np.mean(epoch_train_losses)
            train_losses.append(train_loss)

            # 在测试集上评估模型
            self.eval()
            correct, total = 0, 0
            epoch_test_losses = []
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    outputs = self(batch_data)
                    loss = criterion(outputs, batch_labels)
                    epoch_test_losses.append(loss.item())

                    _, predicted = torch.max(outputs, 1)  # 获取每行最大值的索引
                    _, batch_labels = torch.max(batch_labels, 1)  # 将目标值转换为一维张量

                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                test_loss = np.mean(epoch_test_losses)
                test_losses.append(test_loss)
            test_accuracy = correct / total
            test_accuracies.append(test_accuracy)

            # 保存当前最佳模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                if save_model:
                    torch.save(self.state_dict(), best_model_path)

            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        common_utils.plot_metrics(train_losses, test_losses, test_accuracies)
        print(f"Best Test Accuracy: {best_accuracy:.4f}")

    def assign_majority_labels(self, y_train, superpixels):
        """
            统计每类超像素中样本标签的多数值，并将属于该超像素的所有样本标签都设置为这个值
            :param y_train: one-hot编码的标签数组，形状为 (num_samples, num_classes)
            :param superpixels: 每个样本的超像素标签，形状为 (num_samples,)
            :return: 处理后的标签数组
            """
        # 将one-hot编码转换为标签索引
        labels = np.argmax(y_train, axis=1)

        # 获取所有唯一的超像素标签
        unique_superpixels = np.unique(superpixels)

        # 遍历每个超像素标签
        for sp in unique_superpixels:
            # 找到属于该超像素的所有样本索引
            indices = np.where(superpixels == sp)[0]

            # 获取这些样本的标签
            sp_labels = labels[indices]

            # 统计标签出现的次数，找到多数标签
            majority_label = Counter(sp_labels).most_common(1)[0][0]

            # 将这些样本的标签都设置为多数标签
            labels[indices] = majority_label

        # 将标签索引转换回one-hot编码
        y_train_majority = np.zeros_like(y_train)
        y_train_majority[np.arange(len(labels)), labels] = 1

        return y_train_majority

    def train1(self, train_dataset, superpixels, epochs=50, lr=0.0005, batch_size=128):
        # 构造 TensorDataset 和 DataLoader
        X_train, y_train = train_dataset

        # 根据superpixels来设置标签
        # y_train = self.assign_majority_labels(y_train, superpixels)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()
        criterion = CombinedLoss(0.5, 0.5)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # 训练网络
        self.train()
        for epoch in range(epochs):
            # 训练模型
            epoch_train_losses = []
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_data)
                # loss = criterion(outputs, batch_labels.argmax(dim=1))
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())
            train_loss = np.mean(epoch_train_losses)

            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}")

    def extract_feature(self, data, batch_size=128):
        # 将模型设置为评估模式
        self.eval()

        outputs = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                data_batch = data[i:i + batch_size]
                data_tensor = torch.tensor(data_batch, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()
                batch_outputs = self(data_tensor, mode='feature')
                outputs.append(batch_outputs.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def cls(self, data, batch_size=128):
        # 将模型设置为评估模式
        self.eval()

        outputs = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                data_batch = data[i:i + batch_size]
                data_tensor = torch.tensor(data_batch, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()
                batch_outputs = self(data_tensor)
                outputs.append(batch_outputs.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)


def load_model(model, model_path):
    """
    加载保存的最佳模型权重，并将其应用到模型实例上。

    :param model: 要加载权重的模型实例
    :param model_path: 模型权重的路径
    :return: 加载权重后的模型实例
    """
    # 检查模型权重文件是否存在
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU mode.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # 加载模型权重
    model.load_state_dict(state_dict)
    model.to(device)

    return model





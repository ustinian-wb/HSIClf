# --*-- conding:utf-8 --*--
# @Time  : 2024/6/3
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : cnn.py
# @Description:

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import common_utils
from loss import CombinedLoss
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

    def test(self, train_dataset, test_dataset, epochs=50, lr=0.0005, batch_size=512, save_model=False, save_path=None):
        common_utils.logger.info(f"> Training stage 2")

        X_train, y_train = train_dataset
        X_test, y_test = test_dataset

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).cuda()

        # 构造 TensorDataset 和 DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()
        criterion = CombinedLoss(0.5, 0.5)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # 训练网络
        train_losses, test_losses, test_accuracies = [], [], []
        best_accuracy = 0.0  # 记录最佳准确率
        best_epoch = 0
        best_model_path = save_path  # 保存最佳模型的路径

        final_train_loss, final_test_accuracy = None, None

        for epoch in range(epochs):
            # 训练模型
            self.train()
            epoch_train_losses = []
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch [{epoch + 1}/{epochs}] Training")
                for batch_data, batch_labels in tepoch:
                    optimizer.zero_grad()
                    outputs = self(batch_data)
                    # loss = criterion(outputs, batch_labels.argmax(dim=1))
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    epoch_train_losses.append(loss.item())

                    tepoch.set_postfix(train_loss=loss.item())

            train_loss = np.mean(epoch_train_losses)
            train_losses.append(train_loss)

            # 在测试集上评估模型
            self.eval()
            correct, total = 0, 0
            epoch_test_losses = []
            with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                    tepoch.set_description(f"Epoch [{epoch + 1}/{epochs}] Testing")
                    for batch_data, batch_labels in tepoch:
                        outputs = self(batch_data)
                        # loss = criterion(outputs, batch_labels.argmax(dim=1))
                        loss = criterion(outputs, batch_labels)
                        epoch_test_losses.append(loss.item())

                        _, predicted = torch.max(outputs, 1)  # 获取每行最大值的索引
                        _, batch_labels = torch.max(batch_labels, 1)  # 将目标值转换为一维张量

                        total += batch_labels.size(0)
                        correct += (predicted == batch_labels).sum().item()
                        test_accuracy = correct / total

                        tepoch.set_postfix(accuracy=test_accuracy, test_loss=loss.item())
                        tepoch.update()

            test_loss = np.mean(epoch_test_losses)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            # 保存当前最佳模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch + 1
                if save_model:
                    torch.save(self.state_dict(), best_model_path)

            final_train_loss, final_test_accuracy = train_loss, test_accuracy
            # if (epoch + 1) % 1 == 0:
            #     print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        common_utils.plot_metrics(train_losses, test_losses, test_accuracies)
        common_utils.logger.info(
            f"Stage 2 finished. epoch: {epochs}, train_loss: {final_train_loss:.4f}, test accuracy: {final_test_accuracy:.4f}")
        common_utils.logger.info(f"Best Test Accuracy: {best_accuracy:.4f} in epoch {best_epoch}\n")

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

    def train1(self, train_dataset, superpixels, epochs=50, lr=0.0005, batch_size=512):
        common_utils.logger.info(f"> Training stage 1")
        # 构造 TensorDataset 和 DataLoader
        X_train, y_train = train_dataset

        # 根据superpixels来设置标签
        # y_train = self.assign_majority_labels(y_train, superpixels)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()
        criterion = CombinedLoss(0.5, 0.5)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # 训练网络
        self.train()
        final_loss = None
        for epoch in range(epochs):
            # 训练模型
            epoch_train_losses = []

            # 使用 tqdm 包装 train_loader 以显示进度条
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch [{epoch + 1}/{epochs}]")
                for batch_data, batch_labels in tepoch:
                    optimizer.zero_grad()
                    outputs = self(batch_data)
                    # loss = criterion(outputs, batch_labels.argmax(dim=1))
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    epoch_train_losses.append(loss.item())

                    # 更新进度条中的损失信息
                    tepoch.set_postfix(train_loss=loss.item())

            train_loss = np.mean(epoch_train_losses)
            final_loss = train_loss

        # 打印每个 epoch 的损失
        common_utils.logger.info(f"Stage 1 finished. epoch: {epochs}, loss: {final_loss:.4f} \n")

    def extract_feature(self, data, batch_size=512):
        # 将模型设置为评估模式
        self.eval()

        outputs = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                data_batch = data[i:i + batch_size]
                data_tensor = torch.tensor(data_batch, dtype=torch.float32).cuda()
                batch_outputs = self(data_tensor, mode='feature')
                outputs.append(batch_outputs.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def cls(self, data, batch_size=512):
        # 将模型设置为评估模式
        self.eval()

        outputs = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                data_batch = data[i:i + batch_size]
                data_tensor = torch.tensor(data_batch, dtype=torch.float32).cuda()
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(train_losses, test_losses, test_accuracies):
    """
    绘制loss曲线
    :param train_losses: 训练loss
    :param test_losses: 测试loss
    :param test_accuracies: 正确率列表
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(test_losses, label='Testing Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Testing Loss over Epochs')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(test_accuracies, label='Testing Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Testing Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.show()


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


def train(data, label, in_channels, num_class, epochs=200, lr=0.00005, batch_size=32, save_path='./models',
          save_model=True):
    data_tensor = torch.tensor(data, dtype=torch.float32).cuda()
    label_tensor = torch.tensor(label, dtype=torch.float32).cuda()

    # 构造 TensorDataset和DataLoader
    dataset = TensorDataset(data_tensor, label_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 构建模型
    model = CustomResNet(in_channels=in_channels, num_class=num_class).cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练网络
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        for batch_data, batch_labels in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels.argmax(dim=1))
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            if save_model:
                torch.save(model.state_dict(), f'{save_path}/cnn_{epoch + 1}_{train_loss:.4f}')
    if save_model:
        torch.save(model.state_dict(), f'models/cnn_last.pth')

    return model


def train_2(model, data, label, epochs=5, lr=0.001, batch_size=32, save_path='./models',
            save_model=True):
    data_tensor = torch.tensor(data, dtype=torch.float32).cuda()
    label_tensor = torch.tensor(label, dtype=torch.float32).cuda()

    # 构造 TensorDataset和DataLoader
    dataset = TensorDataset(data_tensor, label_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练网络
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        for batch_data, batch_labels in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels.argmax(dim=1))
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}")
        if save_model:
            torch.save(model.state_dict(), save_path + f'/cnn_stage2_{epoch + 1}_{train_losses[-1]:.4f}.pth')
    return model


def _test(data, label, in_channels, num_class, test_size=0.3, epochs=200, lr=0.001, batch_size=128, save_model=False,
          save_path='./models'):
    data_tensor = torch.tensor(data, dtype=torch.float32).cuda()
    label_tensor = torch.tensor(label, dtype=torch.float32).cuda()

    # 划分训练集和测试集
    dataset = TensorDataset(data_tensor, label_tensor)
    train_size = int(len(dataset) * (1 - test_size))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    # 构造 TensorDataset和DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 构建模型
    model = CustomResNet(in_channels=in_channels, num_class=num_class).cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练网络
    train_losses, test_losses, test_accuracies = [], [], []
    for epoch in range(epochs):
        # 训练模型
        model.train()
        epoch_train_losses = []
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels.argmax(dim=1))
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)

        # 在测试集上评估模型
        model.eval()
        correct, total = 0, 0
        epoch_test_losses = []
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels.argmax(dim=1))
                epoch_test_losses.append(loss.item())

                _, predicted = torch.max(outputs, 1)  # 获取每行最大值的索引
                _, batch_labels = torch.max(batch_labels, 1)  # 将目标值转换为一维张量

                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            test_loss = np.mean(epoch_test_losses)
            test_losses.append(test_loss)
        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        if (epoch + 1) % 1 == 0:
            if save_model:
                torch.save(model.state_dict(), f'{save_path}/cnn_{epoch + 1}')
    if save_model:
        torch.save(model.state_dict(), f'models/cnn_last.pth')
    plot_metrics(train_losses, test_losses, test_accuracies)
    return model


def feature_extract(model_input, data, in_channels, num_class, batch_size=32):
    if isinstance(model_input, str):
        # 如果传入的为保存的模型路径，则加载已经训练好的模型参数
        model = CustomResNet(in_channels=in_channels, num_class=num_class).cuda()
        model.load_state_dict(torch.load(model_input))
    else:
        # 如果传入的为模型
        model = model_input
    # 将模型设置为评估模式
    model.eval()

    outputs = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            data_batch = data[i:i + batch_size]
            data_tensor = torch.tensor(data_batch, dtype=torch.float32).cuda()
            batch_outputs = model(data_tensor, mode='feature')
            outputs.append(batch_outputs.detach().cpu().numpy())

    return model, np.concatenate(outputs, axis=0)


def cls(model, data, batch_size=32):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            data_batch = data[i:i + batch_size]
            data_tensor = torch.tensor(data_batch, dtype=torch.float32).cuda()
            batch_outputs = model(data_tensor)
            outputs.append(batch_outputs.detach().cpu().numpy())

    result = np.concatenate(outputs, axis=0)
    # 找出每一行中最大元素的索引
    max_indices = np.argmax(result, axis=1)
    # 创建一个与Z形状相同，但全部填充0的矩阵
    final_result = np.zeros_like(result, dtype=np.float32)
    # 在每一行的最大元素索引处设置为1
    final_result[np.arange(result.shape[0]), max_indices] = 1
    return final_result


def load_model_eval(path, in_channels, num_class):
    model = CustomResNet(in_channels=in_channels, num_class=num_class).cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

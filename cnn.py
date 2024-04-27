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


class CNN(nn.Module):
    def __init__(self, d, c):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 调整池化层参数
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, c)

    def forward(self, x, mode='train'):
        # 训练模式
        if mode == 'train':
            x = x.view(x.size(0), -1)
            x = x.unsqueeze(2)
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = torch.relu(x)
            # x = self.pool(x)

            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = torch.relu(x)
            # x = self.pool(x)

            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = torch.relu(x)
            # x = self.pool(x)

            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            return x
        elif mode == 'feature':
            # 提取特征模式
            x = x.view(x.size(0), -1)
            x = x.unsqueeze(2)
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = torch.relu(x)
            # x = self.pool(x)

            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = torch.relu(x)
            # x = self.pool(x)

            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = torch.relu(x)
            # x = self.pool(x)
            # 返回特征向量
            return x.view(x.size(0), -1)


def cnn_train(data, label, in_dims, c, epochs=200, lr=0.00005, batch_size=1024, save_path='./models', save_model=True):
    """
    训练cnn网络
    :param data: 数据
    :param label: 标签
    :param in_dims: 数据维度
    :param c: 标签类别数
    :param epochs: 训练轮数
    :param lr: 学习率
    :param batch_size: 批量大小
    :param save_path: 模型保存路径
    :param save_model: 模型保存开关
    :return: 模型
    """
    # 转换数据格式
    data_tensor = torch.tensor(data, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.float32)
    data_tensor = data_tensor.unsqueeze(2)

    # 构建dataloader
    dataset = TensorDataset(data_tensor, label_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(d=in_dims, c=c).to(device)

    # 损失函数，优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for epoch in range(epochs):
        # 训练模型
        model.train()
        epoch_train_losses = []
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, mode='train')
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}")
        if (epoch + 1) % 20 == 0:
            if save_model:
                torch.save(model.state_dict(), f'{save_path}/cnn_{epoch + 1}_{train_loss:.4f}.pth')
    if save_model:
        torch.save(model.state_dict(), f'models/cnn_last.pth')
    model.cpu()
    return model


def cnn_feature(model, data):
    """
    使用训练好的模型提取特征
    :param model: 模型
    :param data: 数据
    :return: 数据的特征向量
    """
    # 处理格式
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = data_tensor.to(device)
    model.to(device)  # 将模型移动到对应设备上

    # 提取特征
    outputs = model(data_tensor, mode='feature')
    return outputs.detach().cpu().numpy()


def cnn_test(data, label, test_size=0.3):
    """
    用于测试网络结构
    :param data: 数据
    :param label: 标签
    :param test_size: 测试集比例
    :return: None
    """
    # 网络参数设置
    batch_size = 1024
    num_epochs = 300
    lr = 0.00005

    # 处理数据格式
    data_tensor = torch.tensor(data, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.float32)
    data_tensor = data_tensor.unsqueeze(2)

    # 划分训练集和测试集
    dataset = TensorDataset(data_tensor, label_tensor)
    train_size = int(len(dataset) * (1 - test_size))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    # 构建dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(d=64, c=17).to(device)

    # 损失函数、优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    train_losses, test_losses, test_accuracies = [], [], []
    for epoch in range(num_epochs):
        # 训练模型
        model.train()
        epoch_train_losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_test_losses.append(loss.item())

                _, predicted = torch.max(outputs, 1)  # 获取每行最大值的索引
                _, targets = torch.max(targets, 1)  # 将目标值转换为一维张量

                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            test_loss = np.mean(epoch_test_losses)
            test_losses.append(test_loss)
        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f'models/cnn_{epoch + 1}_{test_accuracy:.4f}.pth')

    plot_metrics(train_losses, test_losses, test_accuracies)


def load_model(in_dims, c, path):
    """
    加载模型
    :param in_dims: 数据维度
    :param c: 类别个数
    :param path: 模型路径
    :return: 加载好的模型
    """
    # 创建模型实例
    model = CNN(d=in_dims, c=c)

    # 加载已经训练好的模型参数
    model.load_state_dict(torch.load(path))

    # 将模型设置为评估模式
    model.eval()
    return model

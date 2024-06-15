# --*-- conding:utf-8 --*--
# @Time  : 2024/6/14
# @Author: weibo
# @Email : csbowei@gmail.com
# @Description: 模型工具类，包括加载模型、训练、分类、提取特征方法等操作

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

from SSLFrame.model.loss import CombinedLoss
import numpy as np
import common_utils

from collections import Counter


def load_model(model, model_path):
    """
    加载保存的模型权重
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


def train_1(model_name, model, train_dataset, superpixels, epoch=20, lr=0.0005, batch_size=128):
    """
    Stage1 初始化训练
    :param model_name: 模型名称
    :param model: 模型
    :param train_dataset: 训练数据集
    :param superpixels: 超像素
    :param epoch: 超参数
    :param lr: 超参数
    :param batch_size: 超参数
    """
    common_utils.logger.info(f"> Training stage 1")
    # 构造 TensorDataset 和 DataLoader
    X_train, y_train = train_dataset

    # 根据superpixels来设置标签
    # y_train = assign_majority_labels(y_train, superpixels)

    # 针对不同模型做数据调整
    if model_name == 'resnet':
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
    elif model_name == 'hetconv' or model_name == 'new':
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    criterion = CombinedLoss(0.5, 0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练网络
    model.train()
    final_loss = None
    for iepoch in range(epoch):
        # 训练模型
        epoch_train_losses = []

        # 使用 tqdm 包装 train_loader 以显示进度条
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch [{iepoch + 1}/{epoch}]")
            for batch_data, batch_labels in tepoch:
                optimizer.zero_grad()
                outputs = model(batch_data)
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
    common_utils.logger.info(f"Stage 1 finished. epoch: {epoch}, loss: {final_loss:.4f} \n")


def train_2(model_name, model, train_dataset, val_dataset, epoch=30, lr=0.0005, batch_size=128, save_model=False,
            save_path=None):
    """
    Stage2 第二次训练
    :param model_name: 模型名称
    :param model: 模型
    :param train_dataset: 训练数据集
    :param val_dataset: 验证数据集
    :param epoch: 超参数
    :param lr: 超参数
    :param batch_size: 超参数
    :param save_model: 是否保存模型
    :param save_path: 模型保存路径
    """
    common_utils.logger.info(f"> Training stage 2")

    X_train, y_train = train_dataset
    X_val, y_val = val_dataset

    # 将数据转换为 PyTorch 张量，并调整形状
    # 针对不同模型做数据调整
    if model_name == 'resnet':
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).cuda()
    elif model_name == 'hetconv' or model_name == 'new':
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).cuda()

    # 构造 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    criterion = CombinedLoss(0.5, 0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练网络
    train_losses, val_losses, val_accuracies = [], [], []
    best_accuracy = 0.0  # 记录最佳准确率
    best_epoch = 0
    best_model_path = save_path  # 保存最佳模型的路径

    final_train_loss, final_val_accuracy = None, None

    for iepoch in range(epoch):
        # 训练模型
        model.train()
        epoch_train_losses = []
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch [{iepoch + 1}/{epoch}] Training")
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())
                tepoch.set_postfix(train_loss=loss.item())
                tepoch.update()

        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)

        # 在测试集上评估模型
        model.eval()
        correct, total = 0, 0
        epoch_val_losses = []
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch [{iepoch + 1}/{epoch}] Validating")
                for batch_data, batch_labels in tepoch:
                    outputs = model(batch_data)
                    # loss = criterion(outputs, batch_labels.argmax(dim=1))
                    loss = criterion(outputs, batch_labels)
                    epoch_val_losses.append(loss.item())

                    _, predicted = torch.max(outputs, 1)  # 获取每行最大值的索引
                    _, batch_labels = torch.max(batch_labels, 1)  # 将目标值转换为一维张量

                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    val_accuracy = correct / total

                    tepoch.set_postfix(accuracy=val_accuracy, val_loss=loss.item())
                    tepoch.update()

        val_loss = np.mean(epoch_val_losses)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 保存当前最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = iepoch + 1
            if save_model:
                torch.save(model.state_dict(), best_model_path)

        final_train_loss, final_val_accuracy = train_loss, val_accuracy

        # if (epoch + 1) % 1 == 0:
        #     print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    common_utils.plot_metrics(train_losses, val_losses, val_accuracies)
    common_utils.logger.info(
        f"Stage 2 finished. epoch: {epoch}, train_loss: {final_train_loss:.4f}, val accuracy: {final_val_accuracy:.4f}")
    common_utils.logger.info(f"Best Val Accuracy: {best_accuracy:.4f} in epoch {best_epoch}\n")


def assign_majority_labels(y_train, superpixels):
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


def extract_feature(model_name, model, data, batch_size=51):
    """
    仅使用模型的提取特征部分来将data转换为特征向量
    :param model_name: 模型名称
    :param model: 模型
    :param data: 要提取特征的数据
    :param batch_size: 超参数
    :return: 提取后的特征向量
    """
    # 将模型设置为评估模式
    model.eval()

    outputs = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            data_batch = data[i:i + batch_size]
            # 针对不同模型做数据调整
            if model_name == 'resnet':
                data_tensor = torch.tensor(data_batch, dtype=torch.float32).cuda()
            elif model_name == 'hetconv' or model_name == 'new':
                data_tensor = torch.tensor(data_batch, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()
            batch_outputs = model(data_tensor, mode='feature')
            outputs.append(batch_outputs.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def cls(model_name, model, data, batch_size=128):
    """
    使用模型进行分类
    :param model_name: 模型名称
    :param model: 模型
    :param data: 要分类的数据（测试集）
    :param batch_size: 超参数
    :return: 分类结果
    """
    # 将模型设置为评估模式
    model.eval()

    outputs = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            data_batch = data[i:i + batch_size]

            if model_name == 'resnet':
                data_tensor = torch.tensor(data_batch, dtype=torch.float32).cuda()
            elif model_name == 'hetconv' or model_name == 'new':
                data_tensor = torch.tensor(data_batch, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()

            batch_outputs = model(data_tensor)
            outputs.append(batch_outputs.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)

# --*-- conding:utf-8 --*--
# @Time  : 2024/6/12
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : tester.py
# @Description:

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

from modelTester.model.loss import CombinedLoss
import numpy as np
import common_utils


def train_and_test(model_name, model, train_dataset, val_dataset, epoch=30, lr=0.0005, batch_size=128, save_model=False,
                   save_path=None):
    X_train, y_train = train_dataset
    X_val, y_val = val_dataset

    # 将数据转换为 PyTorch 张量，并调整形状
    # 针对不同模型做数据调整
    if model_name == 'hetconv' or model_name == 'new':
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
        f"finished. epoch: {epoch}, train_loss: {final_train_loss:.4f}, val accuracy: {final_val_accuracy:.4f}")
    common_utils.logger.info(f"Best Val Accuracy: {best_accuracy:.4f} in epoch {best_epoch}\n")


def cls(model_name, model, data, batch_size=128):
    # 将模型设置为评估模式
    model.eval()

    outputs = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            data_batch = data[i:i + batch_size]

            if model_name == 'hetconv' or model_name == 'new':
                data_tensor = torch.tensor(data_batch, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1).cuda()

            batch_outputs = model(data_tensor)
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

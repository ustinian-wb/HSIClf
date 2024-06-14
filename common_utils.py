# --*-- conding:utf-8 --*--
# @Time  : 2024/6/2
# @Author: weibo
# @Email : csbowei@gmail.com
# @Description: 公共工具类

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

import logging

logger = None


def setup_logger(name, log_file, level=logging.INFO):
    """
    设置一个新的日志记录器
    :param name: 日志记录器名称
    :param log_file: 日志保存路径
    :param level: 日志级别
    :return: 日志记录器
    """
    global logger
    if logger is None:
        # 创建一个文件处理器，以追加模式打开日志文件
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # 创建一个控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

        # 获取并配置日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def print_evaluation(y_gt, y_pred, msg=None):
    """
    打印评估结果
    :param y_gt: 标签真值
    :param y_pred: 标签预测值
    :param msg: 提示信息
    :return: [oa, aa, kappa, conf_matrix]
    """

    def evaluate(y_gt, y_pred):
        """
        评价结果，计算正确率、平均精度、Kappa系数和混淆矩阵
        :param y_gt: 标签真值
        :param y_pred: 标签预测值
        :return: 正确率, 平均精度, Kappa系数, 混淆矩阵
        """
        if len(y_gt.shape) > 1:
            y_gt = np.argmax(y_gt, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_gt, y_pred)

        # 计算总体准确率 (OA)
        oa = accuracy_score(y_gt, y_pred)

        # 计算平均准确率 (AA)，避免除零
        class_counts = np.sum(conf_matrix, axis=1)
        valid_classes = class_counts > 0
        class_accuracies = np.zeros_like(class_counts, dtype=float)
        class_accuracies[valid_classes] = np.diag(conf_matrix)[valid_classes] / class_counts[valid_classes]
        aa = np.mean(class_accuracies)

        # 计算Kappa系数
        kappa = cohen_kappa_score(y_gt, y_pred)

        # 可视化混淆矩阵
        # plot_confusion_matrix(confusion_mat, round(accuracy, 4), noise_ratio, save=save)
        return [oa, aa, kappa, conf_matrix]

    def plot_confusion_matrix(confusion_mat, x_label='y_pred', y_label='y_gt'):
        """
        绘制混淆矩阵
        :param confusion_mat: 混淆矩阵
        :param x_label: X轴标签
        :param y_label: Y轴标签
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'Confusion Matrix')
        plt.show()

    # 评价
    oa, aa, kappa, conf_matrix = evaluate(y_gt, y_pred)
    logger.info(f"{msg}"
                + f"\nOverall Accuracy: {oa * 100:.2f}%"
                + f"\nAverage Accuracy: {aa * 100:.2f}%"
                + f"\nKappa: {kappa:.4f}\n")

    # 绘制混淆矩阵
    plot_confusion_matrix(conf_matrix)
    return [oa, aa, kappa, conf_matrix]


def plot_metrics(train_losses, val_losses, val_accuracies):
    """
    绘制loss曲线
    :param train_losses: 训练loss
    :param val_losses: 验证loss
    :param val_accuracies: 正确率列表
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validating Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validating Loss over Epochs')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(val_accuracies, label='Validating Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validating Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.show()

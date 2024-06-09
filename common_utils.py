# --*-- conding:utf-8 --*--
# @Time  : 2024/6/2
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : dataset_config.yaml
# @Description: 数据集配置文件

import logging
import yaml
import os
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import logging
import matplotlib.pyplot as plt

import ERSModule
import cv2
import pywt
from scipy.ndimage import uniform_filter
from scipy.sparse.linalg import cg
from scipy.sparse import identity
import time
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import seaborn as sns
import scipy.io
from sklearn.model_selection import StratifiedShuffleSplit

# logging配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_evaluation(y_gt, y_pred, msg=None):
    """
    打印评估结果
    :param y_gt: 标签真值
    :param y_pred: 标签预测值
    :param msg: 提示信息
    :return: None
    """

    import numpy as np
    from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

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
        :return: 无
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'Confusion Matrix')
        plt.show()

    # 评价
    oa, aa, kappa, conf_matrix = evaluate(y_gt, y_pred)
    if msg:
        print(msg)
    print("------------------------------")
    print("Accuracy: {:.2f}%".format(oa * 100))
    print("Average Accuracy: {:.2f}%".format(aa * 100))
    print("Kappa: {:.4f}".format(kappa))
    print("------------------------------")

    # 绘制混淆矩阵
    plot_confusion_matrix(conf_matrix)

    return [oa, aa, kappa, conf_matrix]


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

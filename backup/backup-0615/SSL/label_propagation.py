# --*-- conding:utf-8 --*--
# @Time  : 2024/6/9
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : label_propagation.py
# @Description:
import copy

import numpy as np

from scipy.sparse.linalg import cg
from scipy.sparse import identity
import time
import common_utils


def propagation(data, feature_data, unlabelled_ratio=0.3, iters=100):
    """
    输入：data(用于看传播前后的噪声样本数，获取超像素)，训练样本提取出来的特征、一些配置参数
    输出：标签传播后的训练样本的标签（直接放在data的属性里？）
    :return:
    """

    # y_gt = np.argmax(y_gt, axis=1)
    temp_labels = data.train_dataset[1]
    num_classes = np.max(temp_labels)  # 确定类别数
    temp_one_hot = np.zeros((len(temp_labels), num_classes), dtype=int)  # 创建one-hot矩阵
    temp_one_hot[np.arange(len(temp_labels)), temp_labels - 1] = 1  # 减1并设置对应位置为1

    before_noisy_num = np.sum(np.any(temp_one_hot != data.train_dataset_noisy[1], axis=1))

    # 构造SSPTM
    A = sparse_affinity_matrix(feature_data, data.superpixels)
    SSPTM = generate_SSPTM(A)

    # 预测伪标签
    # 暂时采用RLPA中的MAV方式
    y_noisy = data.train_dataset_noisy[1]
    y_pred = np.zeros_like(y_noisy)

    start_time = time.time()
    for i in range(iters):
        # 随机选取unlabelled_ratio比例的数据标签置0
        num_samples = len(y_noisy)
        num_unlabelled = int(unlabelled_ratio * num_samples)
        indices = np.arange(num_samples)
        unlabelled_indices = np.random.choice(indices, num_unlabelled, replace=False)
        y_with_unlabelled = y_noisy.copy()
        y_with_unlabelled[unlabelled_indices] = 0

        pseudo_label, cg_solution, info = diffusion_learning(SSPTM, y_noisy, alpha=0.7, verbose=False)
        y_pred += pseudo_label
    end_time = time.time()

    y_final = np.zeros_like(y_pred)
    y_final[np.arange(y_pred.shape[0]), np.argmax(y_pred, axis=1)] = 1

    after_noisy_num = np.sum(np.any(temp_one_hot != y_final, axis=1))

    common_utils.logger.info(f"> Label propagation"
                             + f"\nTime taken[CG, iters={iters}]: {end_time - start_time:.4f} seconds"
                             + f"\nNoisy samples: {before_noisy_num} -> {after_noisy_num}\n")

    data.train_dataset_propagated = copy.copy(data.train_dataset)
    data.train_dataset_propagated[1] = y_final


def sparse_affinity_matrix(data, superpixel, sim_type='gauss'):
    """
    根据超像素来构造表示不同样本间相似度的稀疏亲和矩阵A
    :param data: 样本
    :param superpixel: 超像素
    :param sim_type: 相似度度量 dot:向量点积
    :return: 稀疏亲和矩阵 A
    """
    num_samples, _ = data.shape
    # 将数组数值归一化至0~1
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # print3d(normalized_data, id='normalized_data', superpixel=superpixel)

    # 计算不同超像素区域的方差
    variances = compute_superpixel_variances(normalized_data, superpixel)

    # 初始化稀疏亲和矩阵 A
    A = np.zeros((num_samples, num_samples))

    # 计算相似度矩阵
    for sp in np.unique(superpixel):  # 对每个超像素值进行迭代
        mask = superpixel == sp  # 创建一个布尔掩码，标记出所有值等于当前超像素值的位置
        variance = variances[sp]  # 获取当前超像素区域的方差

        # 从归一化后的数据中获取当前超像素区域的所有像素值，其中每行表示一个像素
        masked_data = normalized_data[mask]

        # 相似度度量方式为高斯核 - RLPA
        if sim_type == 'gauss':
            # 计算当前超像素区域内每对像素之间的点积，得到一个二维数组
            distances_square = np.sum(np.square(masked_data[:, np.newaxis] - masked_data), axis=2)
            similarities = np.exp(-(distances_square / (2 * variance)))

        # 相似度度量方式为余弦距离
        # 暂未启用 - 待确认
        elif sim_type == 'cos':
            # 计算当前超像素区域内每对像素之间的点积，得到一个二维数组
            dot_matrix = np.dot(masked_data, masked_data.T)

            # 计算 masked_data 中每一行的模长
            norms = np.linalg.norm(masked_data, axis=1)
            # 计算 masked_data 中每一列的模长
            norms_T = np.linalg.norm(masked_data.T, axis=0)
            # 外积得到模长的乘积矩阵，用于归一化相似度矩阵
            norms_matrix = np.outer(norms, norms_T)

            # 归一化相似度矩阵，得到余弦相似度矩阵
            similarities = dot_matrix / norms_matrix

        # 相似度度量方式为欧几里得距离
        # 暂未启用 - 待确认
        elif sim_type == 'euclidean':
            distances_squre = np.sum(np.square(masked_data[:, np.newaxis] - masked_data), axis=2)
            similarities = np.exp(-(distances_squre / (2 * variance)))

        # 使用np.where(mask)获取当前超像素区域内每个像素的索引
        idxs = np.where(mask)
        # 将相似度填充到稀疏亲和矩阵A的对应位置
        A[idxs[0][:, np.newaxis], idxs[0][np.newaxis, :]] = similarities
    return A


def generate_SSPTM(A):
    """
    生成SSPTM
    :param A: 亲和矩阵
    :return: SSPTM
    """
    # 将矩阵A与其转置相加，得到对称矩阵W
    W = A + A.T

    # 对W的每一行进行归一化，使其和为1
    normalized_W = W / np.sum(W, axis=1)[:, None]
    return normalized_W


def compute_superpixel_variances(data, superpixel):
    """
    计算不同超像素区域的方差
    :param data: 样本数据
    :param superpixel: 超像素区域标签
    :return: 不同超像素区域的方差数组
    """
    unique_superpixels = np.unique(superpixel)
    variances = {}

    for sp in unique_superpixels:
        mask = superpixel == sp
        data_mask = data[mask]
        if len(data_mask) == 1:
            variances[sp] = 1
            continue

            # 计算样本对之间的欧几里得距离的平方
        pairwise_distances_squared = np.sum((data_mask[:, np.newaxis, :] - data_mask[np.newaxis, :, :]) ** 2, axis=-1)

        # 排除对角线上的元素，因为对角线上的元素对应的是同一个样本与自身的距离为0
        np.fill_diagonal(pairwise_distances_squared, 0)

        # 计算平均值
        num_samples = len(data_mask)
        sigma_squared = np.sum(pairwise_distances_squared) / (num_samples * (num_samples - 1))

        # 计算 sigma的平方
        variances[sp] = sigma_squared

    return variances


def diffusion_learning(SSPTM, Y, alpha=0.5, verbose=False):
    """
    使用共轭梯度法(CG)来求解线性方程组：(I - alpha * W) * Z = Y， 推断伪标签
    :param SSPTM: 概率转移矩阵
    :param one_hot_gt: one-hot类型标签 nxc
    :param alpha: 学习率
    :param verbose: 是否输出迭代信息
    :return: 伪标签label_pseudo
    """
    num_n, num_c = Y.shape

    # print("共轭梯度法求解开始：")

    # 构建方程 (I - alpha * T)Z = Y
    I = identity(num_n)
    A = I - alpha * SSPTM

    # 共轭梯度法求解
    # 初始化伪标签矩阵Z
    Z = np.zeros((num_n, num_c))
    # 对Y的每一列使用共轭梯度法求解
    for i in range(Y.shape[1]):
        iter_count = [0]  # 迭代次数计数器
        Z_guess = np.zeros(num_n)  # 初始猜测解

        # 修改后的callback函数
        def callback(xk):
            iter_count[0] += 1
            residual_norm = np.linalg.norm(A.dot(xk) - Y[:, i])  # 计算残差范数
            if verbose:
                print(f'列 {i}，迭代次数: {iter_count[0]}，残差范数: {residual_norm}')

        Z[:, i], info = cg(A, Y[:, i], x0=Z_guess, callback=callback)
        if info != 0:
            print(f"求解失败，列 {i}，info =", info)
            return
    # 输出结果
    # print("Z_solution shape:", Z.shape)

    # 找出每一行中最大元素的索引
    max_indices = np.argmax(Z, axis=1)
    # 创建一个与Z形状相同，但全部填充0的矩阵
    one_hot_Z = np.zeros_like(Z, dtype=np.int32)
    # 在每一行的最大元素索引处设置为1
    one_hot_Z[np.arange(Z.shape[0]), max_indices] = 1

    return one_hot_Z, Z, info

import os

import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import ERSModule
import cv2

from scipy.sparse.linalg import cg
from scipy.sparse import identity
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def to_one_hot(labels, num_classes):
    """
    将标签转换为one-hot编码
    :param labels: 标签矩阵
    :param num_classes: 类别数
    :return: one-hot编码的labels
    """
    return np.eye(num_classes)[labels.astype(int)].astype(int)


def ers(img_3d, num_superpixel, conn8=1, lamb=0.5, sigma=5.0, show_image=True, save_image=True):
    """
    ERS超像素分割
    :param img_3d: 3维HSI图像
    :param num_superpixel: 超像素个数
    :param conn8: 连接性参数，指定超像素之间的连接方式。在这里，conn8设置为1，表示使用8邻域连接（即每个像素与其周围8个像素相连）
    :param lamb: 平衡项参数，用于控制超像素的大小和形状。较小的值会生成更大的超像素，较大的值会生成更小的超像素
    :param sigma: 高斯核的标准差，用于计算相似性度量。较大的值会使相似性度量更加平滑，较小的值会更敏感
    :param show_image: 是否展示图像，默认值True
    :param save_image: 是否保存图像，默认值True
    :return: label数组
    """
    # 将像素值缩放到0-255之间
    img_float = img_3d.astype(np.float32)
    img_normalized = cv2.normalize(img_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # python版本ers接口以list作为输出，需要将array展平为list
    img_list = img_normalized.flatten().tolist()

    # 超像素分割
    height, width = img_3d.shape[0], img_3d.shape[1]
    label_list = ERSModule.ERS(img_list, height, width, num_superpixel, conn8, lamb, sigma)
    label = np.reshape(np.asarray(label_list), (height, width))

    # 随机为每个超像素生成颜色，生成色图
    colors = np.uint8(np.random.rand(num_superpixel, 3) * 255)
    img_colormap = colormap(label, colors)

    # 绘制超像素边界
    img_border = np.zeros_like(img_normalized)
    for y in range(height):
        for x in range(width):
            if x < width - 1 and label[y, x] != label[y, x + 1]:
                img_border[y, x] = [255, 255, 255]  # 设置边界像素为白色
            if y < height - 1 and label[y, x] != label[y + 1, x]:
                img_border[y, x] = [255, 255, 255]
    # 将绘制了超像素边界的空白图像与原图像叠加
    img_overlay = cv2.addWeighted(img_normalized, 0.7, img_border, 0.3, 0)

    # 保存结果图像
    if save_image:
        result_dir = 'ers_results'
        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(result_dir, f'img_normalized_T={num_superpixel}_conn8={conn8}_lamb={lamb}_sigma={sigma}.png'),
            img_normalized)
        cv2.imwrite(
            os.path.join(result_dir, f'img_normalized_T={num_superpixel}_conn8={conn8}_lamb={lamb}_sigma={sigma}.png'),
            img_normalized)
        cv2.imwrite(
            os.path.join(result_dir, f'img_colormap_T={num_superpixel}_conn8={conn8}_lamb={lamb}_sigma={sigma}.png'),
            img_colormap)
        cv2.imwrite(
            os.path.join(result_dir, f'img_overlay_T={num_superpixel}_conn8={conn8}_lamb={lamb}_sigma={sigma}.png'),
            img_overlay)

    # 显示结果: 原图像、超像素分割色图、叠加图像
    if show_image:
        space = np.ones((img_normalized.shape[0], 10, 3), dtype=np.uint8) * 220
        combined_image = np.hstack((img_normalized, space, img_colormap, space, img_overlay))
        cv2.imshow("ERS result", combined_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return label


def colormap(input, colors):
    """
    为超像素分割生成色图
    :param input: 超像素分割结果标签
    :param colors: 颜色数组
    :return: 色图结果
    """
    height, width = input.shape[:2]
    output = np.zeros([height, width, 3], np.uint8)
    for y in range(height):
        for x in range(width):
            label_id = int(input[y, x])
            for k in range(3):
                output[y, x, k] = colors[label_id, k]
    return output


def show_image_projection(data, cmap='gray'):
    """
    展示HSI向二维平面投影图像
    :param data: HSI图像
    :param cmap: 配色
    :return: None
    """
    projection = data.mean(axis=2)  # 可以尝试其他投影方法，如最大值投影等
    plt.imshow(projection, cmap=cmap)
    plt.axis('off')
    plt.title('Projection')
    plt.show()


def pca(input_data, dimension):
    """
    PCA降维
    :param input_data: 输入ndarray
    :param dimension:  降维后的维度
    :return: 降维后的数据
    """
    pca_model = PCA(n_components=dimension)
    reduced_data_array = np.reshape(pca_model.fit_transform(np.reshape(input_data, (-1, input_data.shape[2]))),
                                    input_data.shape[:-1] + (dimension,))
    return reduced_data_array


def split_data(data, one_hot_gt, unlabelled_ratio):
    """
    将data和one_hot_gt随机划分为干净数据和unlabelled数据，并返回划分后的数据和标记矩阵。
    :param data: 降维后的数据，形状为(N, M, D)
    :param one_hot_gt: 标签的one-hot编码，形状为(N, M, C)
    :param unlabelled_ratio: unlabelled数据所占比例，取值范围为[0, 1]
    :return: clean_data: 干净数据，形状为(N, M, D)
             clean_labels: 干净数据的标签，形状为(N, M, C)，one-hot编码
             unlabelled_mask: 标记矩阵，形状为(N, M)，其中1表示对应位置为unlabelled数据，0表示为干净数据
    """
    N, M, _ = data.shape
    total_samples = N * M
    num_unlabelled = int(total_samples * unlabelled_ratio)

    # 将数据和标签展平为一维数组
    flattened_data = data.reshape((total_samples, -1))
    flattened_labels = one_hot_gt.reshape((total_samples, -1))

    # 随机选择unlabelled数据的索引
    unlabelled_indices = np.random.choice(total_samples, num_unlabelled, replace=False)

    # 创建标记矩阵
    unlabelled_mask = np.zeros((N, M), dtype=bool)
    unlabelled_mask.flat[unlabelled_indices] = True

    # 根据unlabelled_indices将数据和标签分为有标签数据和无标签数据
    clean_data = flattened_data[~unlabelled_mask.flatten()]
    clean_labels = flattened_labels[~unlabelled_mask.flatten()]
    unlabelled_data = flattened_data[unlabelled_mask.flatten()]

    return clean_data, clean_labels, unlabelled_data, unlabelled_mask


def diffusion_learning(SSPTM, one_hot_gt, alpha=0.5, verbose=False):
    """
    使用共轭梯度法(CG)来求解线性方程组：(I - alpha * W) * Z = Y， 推断伪标签
    :param SSPTM: 概率转移矩阵
    :param one_hot_gt: one-hot类型标签 nxc
    :param alpha: 学习率
    :param verbose: 是否输出迭代信息
    :return: 伪标签label_pseudo
    """
    Y = one_hot_gt.reshape(-1, one_hot_gt.shape[2])
    num_n, num_c = Y.shape

    # print("共轭梯度法求解开始：")
    start_time = time.time()

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

    end_time = time.time()

    # 输出结果
    # print("Z_solution shape:", Z.shape)
    # print("Time taken[CG]:", end_time - start_time, "seconds")

    # 找出每一行中最大元素的索引
    max_indices = np.argmax(Z, axis=1)
    # 创建一个与Z形状相同，但全部填充0的矩阵
    one_hot_Z = np.zeros_like(Z, dtype=np.uint16)
    # 在每一行的最大元素索引处设置为1
    one_hot_Z[np.arange(Z.shape[0]), max_indices] = 1

    return one_hot_Z, Z, info


def plot_confusion_matrix(confusion_mat, iters, accu):
    """
    绘制混淆矩阵
    :param confusion_mat: 混交矩阵
    :param iters: 迭代次数 - 用于标题
    :param accu: 正确率 - 用于标题
    :return:
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix [iters: {iters}, accuracy: {accu}]')
    path = f'./results/conf_mat[iters_{iters}, accuracy_{accu}].png'
    plt.savefig(path)
    plt.show()


def evaluate(ground_truth, predictions, iters):
    """
    评价结构，计算正确率和混淆矩阵
    :param ground_truth: 标签真值
    :param predictions: 标签预测值
    :param iters: 迭代次数 - 用于绘图标题
    :return: 正确率, 混淆矩阵
    """
    # 计算预测的正确率
    accuracy = accuracy_score(ground_truth.argmax(axis=1), predictions.argmax(axis=1))

    # 计算混淆矩阵
    confusion_mat = confusion_matrix(ground_truth.argmax(axis=1), predictions.argmax(axis=1))

    # 可视化混淆矩阵
    plot_confusion_matrix(confusion_mat, iters, round(accuracy, 4))
    return accuracy, confusion_mat


def add_noise_to_ground_truth(ground_truth, noise_ratio):
    """
    向标签中增加噪声
    :param ground_truth: 标签真值
    :param noise_ratio: 噪声比例
    :return: 带有噪声的标签值
    """
    num_samples, num_classes = ground_truth.shape
    noisy_ground_truth = np.copy(ground_truth)

    # 计算要添加噪声的样本数量
    num_noisy_samples = int(num_samples * noise_ratio)

    # 为这些样本添加噪声
    for _ in range(num_noisy_samples):
        # 随机选择一个样本和一个要更改为 1 的类别
        sample_idx = np.random.randint(num_samples)
        class_idx = np.random.randint(num_classes)
        # 将该样本的标签更改为对应的 one-hot 编码
        noisy_ground_truth[sample_idx] = np.eye(num_classes)[class_idx]

    return noisy_ground_truth


def majority_vote(label_presudo_list):
    """
    MAV投票
    :param label_presudo_list: 每轮训练的标签预测值的列表
    :return: 投票后的标签预测值
    """
    # 将所有 label_presudo 相加
    label_presudo_list = np.array(label_presudo_list)
    sum_label_presudo = np.sum(label_presudo_list, axis=0)
    # 将每行最大值的位置设置为 1，其他位置设置为 0
    vote_result = np.zeros_like(sum_label_presudo)
    vote_result[np.arange(sum_label_presudo.shape[0]), np.argmax(sum_label_presudo, axis=1)] = 1
    return vote_result

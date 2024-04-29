import os

import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import ERSModule
import cv2
import pywt
from scipy.ndimage import uniform_filter
from scipy.sparse.linalg import cg
from scipy.sparse import identity
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def load_dataset(X_path, y_path):
    """
    加载数据集
    :param X_path: 数据路径
    :param y_path: 标签路径
    :return: 加载的数据
    """
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y


# TODO: 增加更多参数
def preprocess(data, label, num_class, pca_dim, T):
    # 标准化: 均值为0，标准差为1
    normalized_data = z_score(data)

    # TODO: 平均滤波的效果有待测试，可以尝试保留边缘信息的滤波器，例如小波变换
    # 平均滤波
    # filtered_data = filter(normalized_data, window=3)
    filtered_data = normalized_data

    # pca降维
    reduced_data = pca(filtered_data, pca_dim)

    # 超像素分割
    superpixels = ers(pca(filtered_data, 3), T, conn8=1, lamb=0.5, sigma=5, show_image=False, save_image=True)

    # 标签转换为one-hot编码
    one_hot_label = to_one_hot(label, num_class)
    return reduced_data, one_hot_label, superpixels


def filter(img, window=3):
    """
    平均滤波
    :param img: 输入图像
    :param window: 平均滤波器的窗口大小
    :return: 滤波后的图像
    """
    filtered_img = np.zeros_like(img)
    for i in range(img.shape[2]):
        filtered_img[:, :, i] = uniform_filter(img[:, :, i], size=window, mode='reflect')
    return filtered_img


def z_score(data):
    """
    （在特征维度）标准化为均值为0，方差为1
    :param data: 输入数据
    :return: 标准化后的数据
    """
    # 计算均值和标准差
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))

    # 标准化
    normalized_data = (data - mean) / std
    return normalized_data


def to_one_hot(labels, num_classes):
    """
    将标签转换为one-hot编码
    :param labels: 标签矩阵
    :param num_classes: 类别数
    :return: one-hot编码的labels
    """
    return np.eye(num_classes)[labels.astype(int)].astype(int)


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


def ers(img_3d, num_superpixel, conn8=1, lamb=0.5, sigma=5.0, show_image=True, save_image=False):
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


def add_noise_to_label(label, noise_ratio):
    """
    向标签中增加噪声
    :param label: 标签真值
    :param noise_ratio: 噪声比例
    :return: 带有噪声的标签值
    """
    M, N, c = label.shape
    label = label.reshape(-1, label.shape[-1])
    num_samples, num_classes = label.shape
    noisy_label = np.copy(label)

    # 计算要添加噪声的样本数量
    num_noisy_samples = int(num_samples * noise_ratio)

    # 为这些样本添加噪声
    for _ in range(num_noisy_samples):
        # 随机选择一个样本和一个要更改为 1 的类别
        sample_idx = np.random.randint(num_samples)
        class_idx = np.random.randint(num_classes)
        # 将该样本的标签更改为对应的 one-hot 编码
        noisy_label[sample_idx] = np.eye(num_classes)[class_idx]

    noisy_label = noisy_label.reshape(M, N, c)
    print(1)
    return noisy_label


def split_dataset(data, one_hot_label, dirty_ratio):
    """
    按照dirty_ratio划分数据集为clean和dirty
    :param data: 数据集
    :param one_hot_label: one-hot编码标签
    :param dirty_ratio: 脏数据比例
    :return: clean_data, dirty_data, clean_label, dirty_label, dirty_mask矩阵
    """
    total_samples, block_N, block_M, _ = data.shape
    N, M, c = one_hot_label.shape
    num_dirty = int(total_samples * dirty_ratio)

    # 将标签展平为一维数组
    flattened_labels = one_hot_label.reshape((total_samples, -1))

    # 随机选择dirty数据的索引
    dirty_indices = np.random.choice(total_samples, num_dirty, replace=False)

    # 创建标记矩阵 True: dirty; False: clean
    dirty_mask = np.zeros((N, M), dtype=bool)
    dirty_mask.flat[dirty_indices] = True

    # 根据dirty_indices将数据和标签分为有标签数据和无标签数据
    clean_data = data[~dirty_mask.flatten()]
    clean_label = flattened_labels[~dirty_mask.flatten()]
    dirty_data = data[dirty_mask.flatten()]
    dirty_label = flattened_labels[dirty_mask.flatten()]

    return clean_data, dirty_data, clean_label, dirty_label, dirty_mask


# TODO: 待验证
def extract_samples(img, window_size=7):
    # 反射填充
    padded_img = np.pad(img, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2), (0, 0)),
                        mode='reflect')
    samples = []
    for i in range(window_size // 2, padded_img.shape[0] - window_size // 2):
        for j in range(window_size // 2, padded_img.shape[1] - window_size // 2):
            # 提取以当前像素为中心的窗口
            sample = padded_img[i - window_size // 2:i + window_size // 2 + 1,
                     j - window_size // 2:j + window_size // 2 + 1, :]
            samples.append(sample)
    return np.array(samples)


if __name__ == '__main__':
    # 做测试用

    # 生成示例数据
    data = np.array([[[1, 2], [3, 4], [5, 6]],
                     [[7, 8], [9, 10], [11, 12]],
                     [[13, 14], [15, 16], [17, 18]]])

    normalized_data = z_score(data)
    print(normalized_data)

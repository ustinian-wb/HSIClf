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
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import seaborn as sns
import scipy.io


def load_dataset(X_path, X_name, y_path, y_name):
    """
    加载数据集
    :param X_path: 数据路径
    :param y_path: 标签路径
    :return: 加载的数据
    """
    X = np.array(scipy.io.loadmat(X_path)[X_name])
    y = np.array(scipy.io.loadmat(y_path)[y_name])
    return X, y


# TODO: 增加更多参数
def preprocess(data, label, num_class, pca_dim, T):
    """
    预处理函数：包括标准化、滤波、PCA降维、超像素分割、转为one-hot编码
    :param data: 数据
    :param label: 标签
    :param num_class: 标签类别数
    :param pca_dim: PCA降维维度
    :param T: 超像素数量
    :return: 降维后的数据、one-hot编码标签、超像素
    """
    # 标准化: 均值为0，标准差为1
    normalized_data = z_score(data)

    # TODO: 平均滤波的效果一般，可以尝试保留边缘信息的滤波器，例如小波变换
    # 平均滤波
    filtered_data = filter(normalized_data, window=3)
    # filtered_data = normalized_data

    # pca降维
    reduced_data = pca(filtered_data, pca_dim)

    # 超像素分割
    superpixels = ers(pca(filtered_data, 3), T, conn8=1, lamb=0.5, sigma=5, show_image=False, save_image=False)

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
    noisy_indices = np.random.choice(num_samples, size=num_noisy_samples, replace=False)
    for idx in noisy_indices:
        # 随机选择一个要更改为 1 的类别
        class_idx = np.random.randint(num_classes)
        # 将该样本的标签更改为对应的 one-hot 编码
        noisy_label[idx] = np.eye(num_classes)[class_idx]

    noisy_label = noisy_label.reshape(M, N, c)
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


def gen_label_for_propagation(noisy_label, mask):
    """
    将noisy_label中unlabelled的样本标签置为0
    :param noisy_label: 噪声标签
    :param mask: unlabelled掩码矩阵
    :return: 处理后的标签
    """
    new_label = np.copy(noisy_label)
    new_label[mask] = 0
    return new_label


def extract_samples(img, window_size=7):
    """
    按样本块提取像素
    :param img: HSI图像
    :param window_size: 样本块窗口大小
    :return: 提取好的样本(num_samples, window_size, window_size, feature_dim)
    """
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
    print("Time taken[CG]:", end_time - start_time, "seconds")

    # 找出每一行中最大元素的索引
    max_indices = np.argmax(Z, axis=1)
    # 创建一个与Z形状相同，但全部填充0的矩阵
    one_hot_Z = np.zeros_like(Z, dtype=np.float32)
    # 在每一行的最大元素索引处设置为1
    one_hot_Z[np.arange(Z.shape[0]), max_indices] = 1

    return one_hot_Z, Z, info


def plot_confusion_matrix(confusion_mat, accu, noise_ratio, save=True):
    """
    绘制混淆矩阵
    :param confusion_mat: 混交矩阵
    :param accu: 正确率 - 用于标题
    :param noise_ratio: 噪声比例 - 用于绘制标题
    :return:
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix [accuracy: {accu}]')
    path = f'./results/conf_mat[noise_ratio_{noise_ratio}, accuracy_{accu}].png'
    if save:
        plt.savefig(path)
    plt.show()


def evaluate(ground_truth, predictions, noise_ratio, save=False):
    """
    评价结果，计算正确率、平均精度、Kappa系数和混淆矩阵
    :param ground_truth: 标签真值
    :param predictions: 标签预测值
    :param noise_ratio: 噪声比例 - 用于绘制标题
    :return: 正确率, 平均精度, Kappa系数, 混淆矩阵
    """
    # 计算预测的正确率 OA
    accuracy = accuracy_score(ground_truth.argmax(axis=1), predictions.argmax(axis=1))

    # 计算混淆矩阵
    confusion_mat = confusion_matrix(ground_truth.argmax(axis=1), predictions.argmax(axis=1))

    # 计算平均精度 AA
    AA = np.mean(np.diag(confusion_mat) / confusion_mat.sum(axis=1))

    # 计算Kappa系数
    kappa = cohen_kappa_score(ground_truth.argmax(axis=1), predictions.argmax(axis=1))

    # 可视化混淆矩阵
    plot_confusion_matrix(confusion_mat, round(accuracy, 4), noise_ratio, save=save)
    return [accuracy, AA, kappa, confusion_mat]

def print_evaluation(evaluation_list, msg=''):
    """
    打印评估结果
    :param evaluation_list: 指标
    :return: None
    """
    if msg != '':
        print(msg)
    accuracy, AA, kappa, confusion_mat = evaluation_list
    print("------------------------------")
    print("Accuracy:", accuracy)
    print("Average Accuracy:", AA)
    print("Kappa:", kappa)
    print("------------------------------")
    # print("Confusion Matrix:\n", confusion_mat)



if __name__ == '__main__':
    # 做测试用

    # 生成示例数据
    data = np.array([[[1, 2], [3, 4], [5, 6]],
                     [[7, 8], [9, 10], [11, 12]],
                     [[13, 14], [15, 16], [17, 18]]])

    normalized_data = z_score(data)
    print(normalized_data)

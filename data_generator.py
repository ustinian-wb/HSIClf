# --*-- conding:utf-8 --*--
# @Time  : 2024/6/2
# @Author: weibo
# @Email : csbowei@gmail.com
# @Description: 存储数据的类及其操作

import yaml
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import copy
import ERSModule
import cv2
from scipy.ndimage import uniform_filter
import scipy.io

global logger


class DataGenerator():
    def __init__(self, dataset_name, config_path="./dataset_config.yaml", dataset_type="mat", seed=0):
        self.dataset_name = dataset_name
        self.config_path = config_path
        self.dataset_type = dataset_type
        self.seed = seed

        # 读入的数据/预处理后的数据
        self.dataset = []
        self.superpixels = []
        self.ers_results = []

        # 参数
        self.pca_dim = None
        self.T = None
        self.num_classes = None

        # 划分训练集、测试集、验证集
        self.train_dataset = []
        self.test_dataset = []
        self.val_dataset = []
        self.mask = None

        # 添加了噪声后的训练集
        self.train_dataset_noisy = []

        # 标签传播后的训练集
        self.train_dataset_propagated = []

    def load_dataset(self):
        """
        按照指定配置文件加载数据集
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)[self.dataset_name]
            root_path = Path(config['root_path'])
            X_path, y_path = root_path / config['X_path'], root_path / config['y_path']

            self.pca_dim = config['pca_dim']
            self.T = config['T']
            self.num_classes = config['c']

            if self.dataset_type == "mat":
                # 读取mat文件
                if X_path.suffix != ".mat" or y_path.suffix != ".mat":
                    raise ValueError("mat")
                X = np.array(scipy.io.loadmat(X_path)[config['X_name']])
                y = np.array(scipy.io.loadmat(y_path)[config['y_name']])
            elif self.dataset_type == "npy":
                # 读取numpy文件
                raise TypeError(self.dataset_type)
                pass

        except TypeError as e:
            logger.error(f"Unsupported data file type：{e}")
        except ValueError as e:
            logger.error(f"Unmatched file type：{e}")
        except Exception as e:
            logger.error(f"Error while loading dataset：{e}")

        self.dataset = [X, y]

    def preprocess(self, patch=False, show_ers_image=False, ers_params=None):
        """
        数据预处理，包括: 标准化、滤波、PCA降维、超像素分割，去除标签为0的数据
        :param patch: 是否提取样本块
        :param show_ers_image: 是否展示超像素分割结果
        :param ers_params: 保留, 为超像素分割设置参数
        """

        def z_score(data):
            """
            （在特征维度）标准化为均值为0，方差为1
            :param data: 输入数据
            :return: 标准化后的数据
            """
            mean = np.mean(data, axis=(0, 1))
            std = np.std(data, axis=(0, 1))
            normalized_data = (data - mean) / std
            return normalized_data

        def mean_filter(img, window=3):
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

        def ers(img_3d, num_superpixel, conn8=1, lamb=0.5, sigma=5, show_image=False):
            """
            ERS超像素分割
            :param img_3d: 3维HSI图像
            :param num_superpixel: 超像素个数
            :param conn8: 连接性参数，指定超像素之间的连接方式。在这里，conn8设置为1，表示使用8邻域连接（即每个像素与其周围8个像素相连）
            :param lamb: 平衡项参数，用于控制超像素的大小和形状。较小的值会生成更大的超像素，较大的值会生成更小的超像素
            :param sigma: 高斯核的标准差，用于计算相似性度量。较大的值会使相似性度量更加平滑，较小的值会更敏感
            :param show_image: 是否展示图像，默认值False
            :return: label数组, [[conn8, lamb, sigma], [原图像、超像素分割色图、叠加图像]]
            """

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

            # 将像素值缩放到0-255之间
            img_float = img_3d.astype(np.float32)
            img_normalized = cv2.normalize(img_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_8U)

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

            # 显示结果: 原图像、超像素分割色图、叠加图像
            if show_image:
                space = np.ones((img_normalized.shape[0], 10, 3), dtype=np.uint8) * 220
                combined_image = np.hstack((img_normalized, space, img_colormap, space, img_overlay))
                cv2.imshow("ERS result", combined_image)
                cv2.waitKey()
                cv2.destroyAllWindows()

            return label, [[num_superpixel, conn8, lamb, sigma], [img_normalized, img_colormap, img_overlay]]

        def extract_patch(X, window_size=9):
            """
            按样本块提取像素
            :param X: HSI图像
            :param window_size: 样本块窗口大小
            :return: 提取好的样本(num_samples, window_size, window_size, feature_dim)
            """
            # 反射填充
            padded_img = np.pad(X, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2), (0, 0)),
                                mode='reflect')
            patches = []
            for i in range(window_size // 2, padded_img.shape[0] - window_size // 2):
                for j in range(window_size // 2, padded_img.shape[1] - window_size // 2):
                    # 提取以当前像素为中心的窗口
                    sample = padded_img[i - window_size // 2:i + window_size // 2 + 1,
                             j - window_size // 2:j + window_size // 2 + 1, :]
                    patches.append(sample)
            extracted_patches = np.array(patches)

            return extracted_patches

        X, y = self.dataset
        # 标准化: 均值为0，标准差为1
        normalized_X = z_score(X)

        # TODO: 平均滤波的效果一般，可以尝试保留边缘信息的滤波器，例如小波变换
        # 平均滤波
        filtered_X = mean_filter(normalized_X, window=3)
        # filtered_X = normalized_X

        # pca降维
        reduced_X = pca(filtered_X, self.pca_dim)

        # 超像素分割
        superpixels, ers_results = ers(pca(filtered_X, 3), self.T, show_image=show_ers_image)

        # 去除标签为0的数据: 噪声/背景
        # 展开(reduce)数据
        if patch:
            flat_X = extract_patch(reduced_X)
        else:
            flat_X = reduced_X.reshape(-1, reduced_X.shape[-1])  # 形状 (num_samples, pca_dim)
        flat_y = y.flatten()
        flat_superpixels = superpixels.flatten()

        # 根据y的值过滤数据，去除y == 0的条目
        mask = flat_y != 0

        filtered_X = flat_X[mask]
        filtered_y = flat_y[mask]
        filtered_superpixels = flat_superpixels[mask]

        self.dataset = [filtered_X, filtered_y]
        self.superpixels = filtered_superpixels
        self.ers_results = ers_results

    def split_dataset(self, train_num):
        """
        划分训练集、测试集和验证集。
        :param train_num: 当取值(0, 1]时，表示每个类别选取train_num比例的数据；当>1时，表示每个类别取train_num个样本。
        """
        X, y = self.dataset
        unique_classes = np.unique(y)
        train_indices = []
        test_indices = []

        # 逐类别处理
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            num_cls_samples = len(cls_indices)

            if 0 < train_num <= 1:
                num_train_samples = int(num_cls_samples * train_num)
            elif train_num > 1:
                num_train_samples = int(train_num)
                num_train_samples = min(num_train_samples, num_cls_samples)
            else:
                raise ValueError("train_num must be greater than 0")

            np.random.shuffle(cls_indices)
            train_indices.extend(cls_indices[:num_train_samples])
            test_indices.extend(cls_indices[num_train_samples:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # 划分训练集和初步的测试集
        X_train = X[train_indices]
        y_train = y[train_indices]

        # 进一步划分测试集为测试集和验证集：将剩余的数据80%为测试集，20%为验证集
        val_indices = []
        new_test_indices = []

        for cls in unique_classes:
            cls_test_indices = test_indices[y[test_indices] == cls]
            num_cls_test_samples = len(cls_test_indices)

            num_val_samples = int(0.2 * num_cls_test_samples)
            np.random.shuffle(cls_test_indices)
            val_indices.extend(cls_test_indices[:num_val_samples])
            new_test_indices.extend(cls_test_indices[num_val_samples:])

        val_indices = np.array(val_indices)
        new_test_indices = np.array(new_test_indices)

        X_test = X[new_test_indices]
        y_test = y[new_test_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]

        self.superpixels = self.superpixels[train_indices]

        mask = np.zeros(y.shape, dtype=bool)
        mask[train_indices] = True

        self.train_dataset = [X_train, y_train]
        self.test_dataset = [X_test, y_test]
        self.val_dataset = [X_val, y_val]
        self.mask = mask

    def add_noise(self, noise_ratio):
        """
        向标签中增加噪声
        :param noise_ratio: 噪声比例
        """
        self.train_dataset_noisy = copy.copy(self.train_dataset)
        y = self.train_dataset[1]
        num_samples = y.shape[0]
        noisy_y = np.copy(y)

        y_min = y.min()
        y_max = y.max()

        # 计算要添加噪声的样本数量
        num_noisy_samples = int(num_samples * noise_ratio)

        # 为这些样本添加噪声
        noisy_indices = np.random.choice(num_samples, size=num_noisy_samples, replace=False)
        for idx in noisy_indices:
            original_class = y[idx]
            # 随机选择一个不同于原标签的类别
            new_class = np.random.randint(y_min, y_max + 1)
            while new_class == original_class:
                new_class = np.random.randint(y_min, y_max + 1)
            noisy_y[idx] = new_class

        # common_utils.print_evaluation(y, noisy_y, msg="> The accuracy of labels with noise: ")
        self.train_dataset_noisy[1] = noisy_y

    def one_hot_encode(self):
        """
        将部分数据的标签转为one-hot编码
        """
        for idataset in [self.train_dataset_noisy, self.test_dataset, self.val_dataset]:
            labels = idataset[1]
            if labels.ndim > 1:
                continue
            num_classes = np.max(labels)  # 确定类别数
            one_hot = np.zeros((len(labels), num_classes), dtype=int)  # 创建one-hot矩阵
            one_hot[np.arange(len(labels)), labels - 1] = 1  # 减1并设置对应位置为1
            idataset[1] = one_hot

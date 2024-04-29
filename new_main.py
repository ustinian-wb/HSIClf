import new_utils
import numpy as np

# 数据集类别数
num_classes_dict = {
    'Indian_Pines': 16,
}

dataset_dict = {
    'Indian_Pines': {
        'shape': [145, 145],
        'dim': 200,
        'num_class': 16 + 1,
        'data_path': './datasets/IndianPines/IndianPines.npy',
        'label_path': './datasets/IndianPines/IndianPines_gt.npy'
    },
}

# 噪声水平
noise_ratio = 0.1

# 扩散学习相关参数
T = 162  # 超像素数量
pca_dim = 64  # PCA降维维度(%, dim): (0.999, 69) (0.99, 25) (0.95, 5)
dirty_ratio = 0.4  # unlabelled数据所占比例
alpha = 0.7  # 扩散程度

# cnn参数
epochs = 200
lr = 0.00005
batch_size = 1024

if __name__ == "__main__":
    # 加载数据集
    dataset = dataset_dict['Indian_Pines']
    data, label = new_utils.load_dataset(dataset['data_path'], dataset['label_path'])

    # 预处理
    processed_data, one_hot_label, superpixels = new_utils.preprocess(data, label, dataset['num_class'], pca_dim, T)

    # 向标签中增加噪声
    noisy_label = new_utils.add_noise_to_label(one_hot_label, noise_ratio)

    # 以7x7块提取出样本
    extracted_data = new_utils.extract_samples(processed_data, window_size=7)

    # 划分数据集
    clean_data, dirty_data, clean_label, dirty_label, mask = new_utils.split_dataset(extracted_data, noisy_label,
                                                                                     dirty_ratio)

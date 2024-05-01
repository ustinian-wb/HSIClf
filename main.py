import utils
import numpy as np

import cnn
import time
import ssptm

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

# cnn训练参数 - 暂时直接写入了代码
# epochs = 200
# lr = 0.00005
# batch_size = 1024

if __name__ == "__main__":
    # 加载数据集
    print("\n> 加载数据集并预处理...")
    dataset = dataset_dict['Indian_Pines']
    data, label = utils.load_dataset(dataset['data_path'], dataset['label_path'])

    # 预处理
    processed_data, one_hot_label, superpixels = utils.preprocess(data, label, dataset['num_class'], pca_dim, T)

    # 提取出样本块
    extracted_data = utils.extract_samples(processed_data, window_size=9)

    # # 测试模型本身在正确数据集上的性能：
    # print("\n> 测试模型本身在正确数据集上的性能...")
    # cnn._test(extracted_data, one_hot_label.reshape(-1, dataset['num_class']), pca_dim, dataset['num_class'],
    #               test_size=0.3, epochs=30,
    #               lr=0.001, batch_size=128, save_model=False)

    print("\n> 构造带有噪声的数据集...")
    # 向标签中增加噪声
    noisy_label = utils.add_noise_to_label(one_hot_label, noise_ratio)
    # 查看当前标签的正确率
    accuracy, confusion_mat = utils.evaluate(one_hot_label.reshape((data.shape[0] * data.shape[1], -1)),
                                             noisy_label.reshape((data.shape[0] * data.shape[1], -1)),
                                             noise_ratio=noise_ratio, save=False)
    print(f"The accuracy of labels with noise: {accuracy}")

    # 划分数据集(带有脏标签)
    clean_data, dirty_data, clean_label, dirty_label, mask = utils.split_dataset(extracted_data, noisy_label,
                                                                                 dirty_ratio)
    # 训练模型
    print("\n> 模型初始化训练中...")
    model = cnn.train(clean_data, clean_label, pca_dim, dataset['num_class'], epochs=10, save_model=False)

    # 提取特征
    print("\n> 提取特征中...")
    # 可以采用直接加载模型的方式，注释掉上述的训练模型部分
    # model = './models/cnn_last.pth'
    model, feature_data = cnn.feature_extract(model, extracted_data, pca_dim, dataset['num_class'])

    # 标签传播过程
    print("\n> 标签传播:")
    # 构建SSPTM
    print("构建SSPTM...")
    A = ssptm.sparse_affinity_matrix(feature_data.reshape(data.shape[0], data.shape[1], -1), superpixels)
    SSPTM = ssptm.generate_SSPTM(A)

    # 预测伪标签
    print("预测伪标签...")
    pseudo_label, cg_solution, info = utils.diffusion_learning(SSPTM, noisy_label, alpha=alpha, verbose=False)

    accuracy, confusion_mat = utils.evaluate(one_hot_label.reshape((data.shape[0] * data.shape[1], -1)),
                                             pseudo_label,
                                             noise_ratio=noise_ratio)
    print(f"直接以伪标签作为结果的正确率为：{accuracy}")

    print("\n> 模型第二阶段训练...")
    model = cnn.train_2(model, extracted_data, pseudo_label, epochs=10, lr=0.001, batch_size=32, save_model=False)

    # 可以直接加载训练两次后的模型来进行预测，需要只是掉提取样本块后的代码
    # model = cnn.load_model_eval(model_path, pca_dim, dataset['num_class'])
    result_label = cnn.cls(model, extracted_data, batch_size=32)
    accuracy, confusion_mat = utils.evaluate(one_hot_label.reshape((data.shape[0] * data.shape[1], -1)),
                                             result_label,
                                             noise_ratio=noise_ratio)
    print(f"二阶段模型在整个数据集上Accuracy: {accuracy}")

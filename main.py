import utils
import numpy as np

import cnn
import time
import ssptm

# 数据集
dataset_dict = {
    # epoch_1=10 epoch_2=10 pca_dim=64 lr_1 = 0.00005 lr_2 = 0.001
    # (%, dim): (0.999, 69) (0.99, 25) (0.95, 5)
    'Indian_Pines': {
        'shape': [145, 145],
        'dim': 200,
        'num_class': 16 + 1,
        'data_path': './datasets/IndianPines/Indian_pines_corrected.mat',
        'data_name': 'indian_pines_corrected',
        'label_path': './datasets/IndianPines/Indian_pines_gt.mat',
        'label_name': 'indian_pines_gt',
        'T': 162
    },
    # epoch_1= epoch_2= pca_dim=
    # (%, dim): (0.999, 16) (0.99, 4) (0.95, 3)
    'Pavia_University': {
        'shape': [610, 340],
        'dim': 103,
        'num_class': 9 + 1,
        'data_path': './datasets/PaviaUniversity/PaviaU.mat',
        'data_name': 'paviaU',
        'label_path': './datasets/PaviaUniversity/PaviaU_gt.mat',
        'label_name': 'paviaU_gt',
        'T': 132
    },
    # epoch_1= epoch_2= pca_dim=64
    # (%, dim): (0.999, 6) (0.99, 3) (0.95, 2)
    'Salinas_Scene': {
        'shape': [512, 217],
        'dim': 204,
        'num_class': 16 + 1,
        'data_path': './datasets/SalinasScene/Salinas_corrected.mat',
        'data_name': 'salinas_corrected',
        'label_path': './datasets/SalinasScene/Salinas_gt.mat',
        'label_name': 'salinas_gt',
        'T': 102
    },
}

# 噪声水平
noise_ratio = 0.5

# 扩散学习相关参数
pca_dim = 64  # PCA降维维度
dirty_ratio = 0.3  # unlabelled数据所占比例
alpha = 0.75  # 扩散程度

# cnn训练参数
batch_size = 4096

epoch_test = 50
lr_test = 0.0001

epochs_1 = 10
lr_1 = 0.0001

epochs_2 = 20
lr_2 = 0.0001

if __name__ == "__main__":
    # 加载数据集
    test_dataset = 'Pavia_University'
    print("\n> 加载数据集并预处理...")
    dataset = dataset_dict[test_dataset]
    data, label = utils.load_dataset(dataset['data_path'], dataset['data_name'], dataset['label_path'],
                                     dataset['label_name'])

    # 预处理
    processed_data, one_hot_label, superpixels = utils.preprocess(data, label, dataset['num_class'], pca_dim,
                                                                  dataset['T'])

    # 提取出样本块
    extracted_data, one_hot_label, superpixels = utils.extract_samples(processed_data, one_hot_label, superpixels,
                                                                       window_size=9, dataset_size=0.1)

    # 测试模型本身在正确数据集上的性能：
    # print("\n> 测试模型本身在正确数据集上的性能...")
    # cnn._test(extracted_data, one_hot_label, pca_dim, dataset['num_class'],
    #           test_size=0.3, epochs=epoch_test,
    #           lr=lr_test, batch_size=batch_size, save_model=False)

    print(f"\n> 构造带有噪声的数据集[ratio={noise_ratio}]...")
    # 向标签中增加噪声
    noisy_label = utils.add_noise_to_label(one_hot_label, noise_ratio)

    # 查看当前标签的正确率
    evaluation_list = utils.evaluate(one_hot_label, noisy_label, noise_ratio=noise_ratio, save=False)
    utils.print_evaluation(evaluation_list, '> The accuracy of labels with noise: ')

    # 划分数据集(带有脏标签)
    clean_data, dirty_data, clean_label, dirty_label, mask = utils.split_dataset(extracted_data, noisy_label,
                                                                                 dirty_ratio)

    # 训练模型
    print("\n> 模型初始化训练中...")
    model = cnn.train(clean_data, clean_label, pca_dim, dataset['num_class'], batch_size=batch_size, epochs=epochs_1,
                      lr=lr_1, save_model=True)

    # 提取特征
    print("\n> 提取特征中...")
    # 可以采用直接加载模型的方式，注释掉上述的训练模型部分
    # model = './models/cnn_last.pth'
    model, feature_data = cnn.feature_extract(model, extracted_data, pca_dim, dataset['num_class'])

    # 标签传播过程
    print("\n> 标签传播:")
    # 构建SSPTM
    print("构建SSPTM...")
    A = ssptm.sparse_affinity_matrix(feature_data, superpixels)
    SSPTM = ssptm.generate_SSPTM(A)
    # # 计算最大值
    # max_val = np.max(A)
    #
    # # 计算最小值
    # min_val = np.min(A)
    #
    # # 计算均值
    # mean_val = np.mean(A)
    #
    # # 计算中位数
    # median_val = np.median(A)
    #
    # print("最大值:", max_val)
    # print("最小值:", min_val)
    # print("均值:", mean_val)
    # print("中位数:", median_val)

    # 预测伪标签
    print("预测伪标签...")
    remove_noise_label = utils.gen_label_for_propagation(noisy_label, mask)
    pseudo_label, cg_solution, info = utils.diffusion_learning(SSPTM, remove_noise_label, alpha=alpha, verbose=True)

    evaluation_list = utils.evaluate(one_hot_label, pseudo_label, noise_ratio=noise_ratio, save=False)
    utils.print_evaluation(evaluation_list, '> 直接以伪标签作为结果:')

    print("\n> 模型第二阶段训练...")
    model = cnn.train_2(model, extracted_data, pseudo_label, epochs=epochs_2, lr=lr_2, batch_size=batch_size,
                        save_model=False)

    # 可以直接加载训练两次后的模型来进行预测，需要只是掉提取样本块后的代码
    # model = cnn.load_model_eval(model_path, pca_dim, dataset['num_class'])
    result_label = cnn.cls(model, extracted_data, batch_size=batch_size)
    evaluation_list = utils.evaluate(one_hot_label, result_label, noise_ratio=noise_ratio, save=False)
    utils.print_evaluation(evaluation_list, '> 二阶段模型在整个数据集上的评价结果:')

# TODO: 1. 划分数据集，完成多数据集测试
# TODO: 2. 尝试设置阈值

import RLPAutils
import numpy as np
import time
import RLPAssptm

# 数据集
dataset_dict = {
    'Indian_Pines': {
        'shape': [145, 145],
        'dim': 200,
        'num_class': 16 + 1,
        'data_path': '../datasets/IndianPines/Indian_pines_corrected.mat',
        'data_name': 'indian_pines_corrected',
        'label_path': '../datasets/IndianPines/Indian_pines_gt.mat',
        'label_name': 'indian_pines_gt',
        'T': 162
    },
    'Pavia_University': {
        'shape': [610, 340],
        'dim': 103,
        'num_class': 9 + 1,
        'data_path': '../datasets/PaviaUniversity/PaviaU.mat',
        'data_name': 'paviaU',
        'label_path': '../datasets/PaviaUniversity/PaviaU_gt.mat',
        'label_name': 'paviaU_gt',
        'T': 132
    },
    'Salinas_Scene': {
        'shape': [512, 217],
        'dim': 204,
        'num_class': 16 + 1,
        'data_path': '../datasets/SalinasScene/Salinas_corrected.mat',
        'data_name': 'salinas_corrected',
        'label_path': '../datasets/SalinasScene/Salinas_gt.mat',
        'label_name': 'salinas_gt',
        'T': 102
    },
}

# 噪声水平
noise_ratio = 0.5

# 扩散学习相关参数
pca_dim = 64  # PCA降维维度(%, dim): (0.999, 69) (0.99, 25) (0.95, 5)
dirty_ratio = 0.3  # unlabelled数据所占比例
alpha = 0.75  # 扩散程度

mav_iters = 1

if __name__ == "__main__":
    # 加载数据集
    test_dataset = 'Pavia_University'
    print("\n> 加载数据集并预处理...")
    dataset = dataset_dict[test_dataset]
    data, label = RLPAutils.load_dataset(dataset['data_path'], dataset['data_name'], dataset['label_path'],
                                         dataset['label_name'])

    # 预处理
    processed_data, one_hot_label, superpixels = RLPAutils.preprocess(data, label, dataset['num_class'], pca_dim,
                                                                      dataset['T'])

    print(f"\n> 构造带有噪声的数据集[ratio={noise_ratio}]...")
    # 向标签中增加噪声
    noisy_label = RLPAutils.add_noise_to_label(one_hot_label, noise_ratio)
    # # 查看当前标签的正确率
    # evaluation_list = RLPAutils.evaluate(one_hot_label.reshape((data.shape[0] * data.shape[1], -1)),
    #                                              noisy_label.reshape((data.shape[0] * data.shape[1], -1)),
    #                                              noise_ratio=noise_ratio, save=False)
    # RLPAutils.print_evaluation(evaluation_list, '> The accuracy of labels with noise: ')

    pred_label = np.zeros((noisy_label.shape[0] * noisy_label.shape[1], dataset['num_class']))
    for i in range(mav_iters):
        print(f"> 标签传播-第{i + 1}次迭代：")
        # 划分数据集(带有脏标签)
        clean_data, dirty_data, clean_label, dirty_label, mask = RLPAutils.split_dataset(processed_data, noisy_label,
                                                                                         dirty_ratio)
        remove_noise_label = RLPAutils.gen_label_for_propagation(noisy_label, mask)

        # 标签传播过程
        # 构建SSPTM
        print("构建SSPTM...")
        A = RLPAssptm.sparse_affinity_matrix(processed_data, superpixels)
        SSPTM = RLPAssptm.generate_SSPTM(A)

        # 预测伪标签
        print("预测伪标签...")
        pseudo_label, cg_solution, info = RLPAutils.diffusion_learning(SSPTM, remove_noise_label, alpha=alpha,
                                                                       verbose=False)

        evaluation_list = RLPAutils.evaluate(one_hot_label.reshape((data.shape[0] * data.shape[1], -1)),
                                             pseudo_label,
                                             noise_ratio=noise_ratio, save=False)
        pred_label += pseudo_label
        RLPAutils.print_evaluation(evaluation_list, '> 本次迭代RLPA的评价结果:')
        print("-------------------------------------------")

    max_indices = np.argmax(pred_label, axis=1)
    pred_label_onehot = np.zeros_like(pred_label)
    pred_label_onehot[np.arange(pred_label.shape[0]), max_indices] = 1
    evaluation_list = RLPAutils.evaluate(one_hot_label.reshape((data.shape[0] * data.shape[1], -1)),
                                         pred_label_onehot,
                                         noise_ratio=noise_ratio, save=False)
    RLPAutils.print_evaluation(evaluation_list, '> RLPA最终MAV的评价结果:')

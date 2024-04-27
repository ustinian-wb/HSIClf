import utils
import ssptm

import numpy as np
import time
import cnn

# 数据集类别数
num_classes_dict = {
    'Indian_Pines': 16,
}

# 噪声水平
noise_ratio = 0.4

# 扩散学习相关参数
vote_times = 100  # 投票次数
T = 162  # 超像素数量
pca_dim = 64  # PCA降维维度(%, dim): (0.999, 69) (0.99, 25) (0.95, 5)
unlabelled_ratio = 0.5  # unlabelled数据所占比例
alpha = 0.7  # 扩散程度

# cnn参数
epochs = 10
lr = 0.00005
batch_size = 1024

if __name__ == "__main__":
    # 加载数据集
    data = np.load('./datasets/IndianPines/IndianPines.npy')
    data_gt = np.load('./datasets/IndianPines/IndianPines_gt.npy')
    num_classes = num_classes_dict['Indian_Pines'] + 1
    N, M, _ = data.shape
    total_samples = N * M

    # 转换data_gt为one-hot编码
    one_hot_gt = utils.to_one_hot(data_gt.flatten(), num_classes).reshape(data_gt.shape + (num_classes,))

    # 增加噪声
    noisy_one_hot_gt = utils.add_noise_to_ground_truth(one_hot_gt.reshape((total_samples, -1)), noise_ratio).reshape(N,
                                                                                                                     M,
                                                                                                                     -1)

    # 对数据进行PCA降维 (%, dim)
    reduced_data = utils.pca(data, pca_dim)
    # 归一化
    reduced_data = (reduced_data - np.min(reduced_data)) / (np.max(reduced_data) - np.min(reduced_data))

    # 超像素分割
    superpixel = utils.ers(utils.pca(data, 3), T, conn8=1, lamb=0.5, sigma=5, show_image=False, save_image=True)

    # 训练
    label_presudo_list = []
    total_time = 0
    accuracy_list = []
    for i in range(vote_times):
        print(f"\n======================== 第{i + 1}轮训练 ========================")
        start_time = time.time()
        # 随机划分数据(clean 与 unlabeled)
        clean_data, clean_labels, unlabelled_data, unlabelled_mask = utils.split_data(reduced_data,
                                                                                      noisy_one_hot_gt,
                                                                                      unlabelled_ratio)

        # cnn训练
        print("> cnn训练:")
        cnn_time_start = time.time()
        model = cnn.cnn_train(clean_data, clean_labels, pca_dim, num_classes, epochs=epochs, lr=lr,
                              batch_size=batch_size,
                              save_path='./models', save_model=False)

        # model = cnn.load_model(pca_dim, num_classes, './models/cnn_last.pth')

        # 将所有data输入网络，构造特征向量
        feature_data = cnn.cnn_feature(model, reduced_data.reshape((total_samples, -1)))
        feature_data = feature_data.reshape(N, M, -1)
        cnn_time_end = time.time()
        print(f"Time taken[cnn]: {cnn_time_end - cnn_time_start:.2f} seconds")

        # 标签传播过程
        print("\n> 标签传播:")
        d_time_start = time.time()
        # 构建SSPTM
        print("构建SSPTM...")
        A = ssptm.sparse_affinity_matrix(feature_data, superpixel)
        SSPTM = ssptm.generate_SSPTM(A)

        # 预测伪标签
        print("预测伪标签...")
        label_presudo, cg_solution, info = utils.diffusion_learning(SSPTM, noisy_one_hot_gt, alpha=alpha, verbose=False)
        # 我认为这里将mva投票的依据设置为cg_solution可以提高投票的有效性
        # 但是我的内存太小，不足以支撑尝试
        # 实际上现在的情况中，并不需要MVA投票
        label_presudo_list.append(label_presudo)
        d_time_end = time.time()
        print(f"Time taken[diffusion]: {d_time_end - d_time_start:.2f} seconds")

        end_time = time.time()
        print(f"\n> Time taken[train {i + 1}]: {end_time - start_time:.2f} seconds")
        total_time += end_time - start_time

        final_predict_label = utils.majority_vote(label_presudo_list)
        accuracy, confusion_mat = utils.evaluate(one_hot_gt.reshape((total_samples, -1)), final_predict_label,
                                                 iters=i + 1)
        accuracy_list.append(round(accuracy, 4))

        print("\n> 投票结果:")
        print(f"Accuracy[vote:{i + 1}]: {accuracy:.4f}")
        print(f"History accuracy: {accuracy_list}")
        print(f"Total time taken: {total_time:.2f} seconds")
    print("\n\n ======================== 最终结果 ========================")
    # MVA投票
    final_predict_label = utils.majority_vote(label_presudo_list)
    accuracy, confusion_mat = utils.evaluate(one_hot_gt.reshape((total_samples, -1)), final_predict_label,
                                             iters=vote_times)
    print(f"Accuracy[vote:{vote_times}]: {accuracy}")

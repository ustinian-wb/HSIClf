# --*-- conding:utf-8 --*--
# @Time  : 2024/6/3
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : main.py
# @Description: SSL

import yaml
import common_utils
import numpy as np
import cnn
import label_propagation

dataset_config_path = "../../../dataset_config.yaml"
# datasets = ['IndianPines', 'PaviaUniversity', 'SalinasScene']
# datasets = ['IndianPines']
# 扩散0.7
datasets = ['SalinasScene']
# noisy_ratios = np.linspace(0.1, 0.9, 9)
# noisy_ratios = np.linspace(0.3, 0.4, 1)
# noisy_ratios = np.array([0, 0.1, 0.3])
# noisy_ratios = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, ])
# noisy_ratios = np.array([0.5, 0.5, 0.5])
noisy_ratios = np.array([0.2])
train_num_dict = {
    'IndianPines': 0.1,
    'PaviaUniversity': 50,
    'SalinasScene': 50
}

epochs_dict = {
    'IndianPines': [10, 20],
    'PaviaUniversity': [10, 20],
    'SalinasScene': [10, 20]
}

# 设置日志记录器
common_utils.setup_logger('ssl(resnet)', 'ssl(resnet).log')

if __name__ == "__main__":
    for dataset in datasets:
        # 获取数据集相关配置
        with open(dataset_config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)[dataset]

        # 加载数据集
        data = data_generator.DataGenerator(dataset, dataset_config_path)
        data.load_dataset()

        common_utils.logger.info("===============================================================")
        # TODO: 两次训练的lr等参数
        common_utils.logger.info("> Parameter settings"
                                 + f"\ndataset: {dataset}"
                                 + f"\nseed: {data.seed}"
                                 + f"\npca_dim: {data.pca_dim}"
                                 + f"\nsuperpixels: {data.T}"
                                 + f"\ntrain_num: {train_num_dict[dataset]}"
                                 + f"\nepoch: {epochs_dict[dataset]}\n")

        # 预处理
        data.preprocess(patch=True)

        # 划分数据集
        data.split_dataset(train_num=train_num_dict[dataset])

        for noisy_ratio in noisy_ratios:
            common_utils.logger.info(f"------------------------------------> Test noisy_ratio: {noisy_ratio}")
            # 增加噪声
            data.add_noise(noisy_ratio)
            data.one_hot_encode()

            # # 测试模型
            # print(f"\n> 测试模型在噪声水平: {noisy_ratio} 的性能...")
            # test_model = cnn.CustomResNet(data.pca_dim, data.num_classes).cuda()
            # test_model.test(data.train_dataset_noisy, data.test_dataset, epochs=20)
            # pred = test_model.cls(data.test_dataset[0])
            # common_utils.print_evaluation(pred, data.test_dataset[1], msg="> [original model]The accuracy of labels: ")

            if noisy_ratio == 0:
                continue

            # SSL模型
            ssl_model = cnn.CustomResNet(data.pca_dim, data.num_classes).cuda()

            # stage 1
            ssl_model.train1(data.train_dataset_noisy, data.superpixels, epochs=epochs_dict[dataset][0])

            # 提取特征
            feature_data = ssl_model.extract_feature(data.train_dataset_noisy[0])
            # 标签传播 - 使用model提取的特征来构造ssptm
            label_propagation.propagation(data, feature_data)

            best_model_path = f'./models/{dataset}/best_model_{noisy_ratio}.pth'
            ssl_model.test(data.train_dataset_propagated, data.test_dataset, epochs=epochs_dict[dataset][1],
                           save_model=True,
                           save_path=best_model_path)
            ssl_model = cnn.load_model(cnn.CustomResNet(data.pca_dim, data.num_classes).cuda(), best_model_path)
            pred = ssl_model.cls(data.test_dataset[0])
            common_utils.print_evaluation(pred, data.test_dataset[1],
                                          msg=f"> [noisy_ratio: {noisy_ratio}]The accuracy of labels: ")

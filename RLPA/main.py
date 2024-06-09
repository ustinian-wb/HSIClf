# --*-- conding:utf-8 --*--
# @Time  : 2024/6/3
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : main.py
# @Description: SSL

import yaml
import common_utils
import numpy as np
import data_generator
import label_propagation
import elm

dataset_config_path = "../dataset_config.yaml"
# datasets = ['IndianPines', 'PaviaUniversity', 'SalinasScene']
datasets = ['PaviaUniversity']

# noisy_ratios = np.linspace(0.1, 0.9, 9)
# noisy_ratios = np.linspace(0.3, 0.4, 1)
# noisy_ratios = np.array([0, 0.1, 0.3])
# noisy_ratios = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, ])
# noisy_ratios = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
noisy_ratios = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, ])
# elm正则化参数
lambdaa = 2 ** np.array([8, 10, 12, 14])
if __name__ == "__main__":
    for dataset in datasets:
        # print(f"dataset: {dataset}, noisy_ratio: {noisy_ratio}")

        # 获取数据集相关配置
        with open(dataset_config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)[dataset]

        # 加载数据集
        data = data_generator.DataGenerator(dataset, dataset_config_path)

        data.load_dataset()
        # 预处理
        data.preprocess(patch=False)

        # 划分数据集
        data.split_dataset(train_num=50)

        for noisy_ratio in noisy_ratios:
            # 增加噪声
            data.add_noise(noisy_ratio)
            data.one_hot_encode()

            # 标签传播 - 使用model提取的特征来构造ssptm
            label_propagation.propagation(data, data.train_dataset_noisy[0])

            # 创建ELM分类器
            elm_classifier = elm.ELMClassifier(input_dim=data.pca_dim, output_dim=data.num_classes, hidden_neurons=1000,
                                               activation_function="sigm")

            # 训练模型
            elm_classifier.train(data.train_dataset_propagated, lambdaa)

            # 分类
            Y_pred = elm_classifier.classify(data.test_dataset[0])

            # 打印预测结果
            # print(Y_pred)
            common_utils.print_evaluation(Y_pred, data.test_dataset[1], msg="> [ssl model]The accuracy of labels: ")

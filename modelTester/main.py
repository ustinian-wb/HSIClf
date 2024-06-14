# --*-- conding:utf-8 --*--
# @Time  : 2024/6/3
# @Author: weibo
# @Email : csbowei@gmail.com
# @Description: 测试模型本身性能

import common_utils
import numpy as np

import data_generator
from modelTester.model import hetcnn
import test_utils
from datetime import datetime

dataset_config_path = "../dataset_config.yaml"
# datasets = ['IndianPines', 'PaviaUniversity', 'SalinasScene']
# datasets = ['IndianPines']
datasets = ['SalinasScene']

# noisy_ratios = np.array([0, 0, 0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5])
noisy_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

train_num_dict = {
    'IndianPines': 0.1,
    'PaviaUniversity': 50,
    'SalinasScene': 50
}
epoch = 30
lr = 0.005
batch_size = 128

selected_model = 'hetconv'

# 设置日志记录器
common_utils.setup_logger(f'modelTester_{selected_model}', f'./log/modelTester_{selected_model}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.log')

if __name__ == "__main__":
    for dataset in datasets:
        # 加载数据集
        data = data_generator.DataGenerator(dataset, dataset_config_path)
        data.load_dataset()
        common_utils.logger.info("===============================================================")
        common_utils.logger.info("> Parameter settings"
                                 + f"\ndataset: {dataset}"
                                 + f"\nseed: {data.seed}"
                                 + f"\npca_dim: {data.pca_dim}"
                                 + f"\ntrain_num: {train_num_dict[dataset]}"
                                 + f"\nlr: {lr}"
                                 + f"\nepoch: {epoch}\n")

        data.preprocess(patch=True)  # 预处理
        data.split_dataset(train_num=train_num_dict[dataset])  # 划分数据集

        for noisy_ratio in noisy_ratios:
            common_utils.logger.info(f"------------------------------------> Test noisy_ratio: {noisy_ratio}")
            # 增加噪声
            data.add_noise(noisy_ratio)
            data.one_hot_encode()

            # 测试模型
            best_model_path = f'./ckpt/{selected_model}/{dataset}/best_model_{noisy_ratio}_epoch_{epoch}_lr_{lr}.pth'
            # 需要根据不同的数据集来设置fc1的参数
            fc1_in = 1536 if dataset == 'IndianPines' else 768
            test_model = hetcnn.HSI3DNet(data.num_classes, fc1_in).cuda()
            test_utils.train_and_test(selected_model, test_model, data.train_dataset_noisy, data.val_dataset,
                                      epoch=epoch, lr=lr, batch_size=batch_size,
                                      save_model=True,
                                      save_path=best_model_path)

            # 加载最佳模型测试
            best_model = test_utils.load_model(test_model, best_model_path)
            pred = test_utils.cls(selected_model, best_model, data.test_dataset[0])
            common_utils.print_evaluation(pred, data.test_dataset[1],
                                          msg=f"> [noisy_ratio: {noisy_ratio}] Best results: ")

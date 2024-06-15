# --*-- conding:utf-8 --*--
# @Time  : 2024/6/14
# @Author: weibo
# @Email : csbowei@gmail.com
# @Description: SSL框架主函数

import yaml
import common_utils
import numpy as np
import data_generator
from model import resnet
from model import hetcnn
from model import new
import label_propagation
import model_utils
from datetime import datetime

dataset_config_path = "../dataset_config.yaml"
# datasets = ['IndianPines', 'PaviaUniversity', 'SalinasScene']
# datasets = ['SalinasScene'] # 扩散0.7
datasets = ['SalinasScene']

# noisy_ratios = np.array([0.5])
noisy_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# selected_model = 'hetconv'
selected_model = 'new'

# 设置日志记录器
log_file = f'./log/SSL({selected_model})_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.log'
common_utils.setup_logger('ssl', log_file)

if __name__ == "__main__":
    for dataset in datasets:
        # 读入配置信息
        with open('./train_config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        # 加载数据集
        data = data_generator.DataGenerator(dataset, dataset_config_path)
        data.load_dataset()
        common_utils.logger.info("===============================================================")
        common_utils.logger.info("> Parameter settings"
                                 + f"\ndataset: {dataset}"
                                 + f"\nseed: {data.seed}"
                                 + f"\npca_dim: {data.pca_dim}"
                                 + f"\nsuperpixels: {data.T}"
                                 + f"\ntrain_num: {config['train_num_dict'][dataset]}"
                                 + f"\nepoch: {config['epochs_dict'][dataset]}"
                                 + f"\nbatch_size: {config['batch_size']}"
                                 + f"\nlr: {config['lr']}\n")
        data.preprocess(patch=True)  # 预处理
        data.split_dataset(train_num=config['train_num_dict'][dataset])  # 划分数据集

        # 遍历不同级别的噪声
        for noisy_ratio in noisy_ratios:
            common_utils.logger.info(f"------------------------------------> Test noisy_ratio: {noisy_ratio}")
            data.add_noise(noisy_ratio)  # 增加噪声
            data.one_hot_encode()  # 标签one-hot编码

            # 加载模型
            model = None
            if selected_model == 'resnet':
                model = resnet.CustomResNet(data.pca_dim, data.num_classes).cuda()
            elif selected_model == 'hetconv' or selected_model == 'new':
                # 需要根据不同的数据集来设置fc1的参数
                fc1_in = 1536 if dataset == 'IndianPines' else 768
                model = hetcnn.HSI3DNet(data.num_classes, fc1_in).cuda()

            # Stage 1
            model_utils.train_1(selected_model, model, data.train_dataset_noisy, data.superpixels,
                                epoch=config['epochs_dict'][dataset][0], lr=config['lr'][0],
                                batch_size=config['batch_size'][0])

            # 提取特征
            feature_data = model_utils.extract_feature(selected_model, model, data.train_dataset_noisy[0])

            # 标签传播 - 使用model提取的特征来构造ssptm
            label_propagation.propagation(data, feature_data, alpha=config['alpha'][dataset], unlabelled_ratio=0.3,
                                          iters=100)

            # Stage 2
            best_model_path = f'./ckpt/{selected_model}/{dataset}/best_model_{noisy_ratio}_epoch{config["epochs_dict"][dataset]}_lr{config["lr"]}.pth'
            model_utils.train_2(selected_model, model, data.train_dataset_propagated, data.val_dataset,
                                epoch=config['epochs_dict'][dataset][1], lr=config['lr'][1],
                                batch_size=config['batch_size'][1],
                                save_model=True,
                                save_path=best_model_path)

            # 验证模型性能
            best_model = model_utils.load_model(model, best_model_path)
            pred = model_utils.cls(selected_model, best_model, data.test_dataset[0])
            common_utils.print_evaluation(pred, data.test_dataset[1],
                                          msg=f"> [noisy_ratio: {noisy_ratio}]The accuracy of labels: ")

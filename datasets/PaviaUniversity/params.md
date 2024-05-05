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

# 划分数据集
dataset_size=0.1
random_state=666
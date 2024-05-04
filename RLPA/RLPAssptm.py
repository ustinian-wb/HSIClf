import numpy as np


def compute_superpixel_variances(data, superpixel):
    """
    计算不同超像素区域的方差
    :param data: 样本数据
    :param superpixel: 超像素区域标签
    :return: 不同超像素区域的方差数组
    """
    unique_superpixels = np.unique(superpixel)
    variances = []

    for sp in unique_superpixels:
        mask = superpixel == sp
        data_mask = data[mask]
        # variances.append(np.var(data_mask, axis=0))

        # 计算样本对之间的欧几里得距离的平方
        pairwise_distances_squared = np.sum((data_mask[:, np.newaxis, :] - data_mask[np.newaxis, :, :]) ** 2, axis=-1)

        # 排除对角线上的元素，因为对角线上的元素对应的是同一个样本与自身的距离为0
        np.fill_diagonal(pairwise_distances_squared, 0)

        # 计算平均值
        num_samples = len(data_mask)
        sigma_squared = np.sum(pairwise_distances_squared) / (num_samples * (num_samples - 1))

        # 计算 sigma的平方
        variances.append(sigma_squared)

    return np.array(variances)


def sparse_affinity_matrix(data, superpixel, sim_type='gauss'):
    """
    根据超像素来构造表示不同样本间相似度的稀疏亲和矩阵A
    :param data: 样本
    :param superpixel: 超像素
    :param sim_type: 相似度度量 dot:向量点积
    :return: 稀疏亲和矩阵 A
    """

    # 将数组数值归一化至0~1
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # print3d(normalized_data, id='normalized_data', superpixel=superpixel)

    # 计算不同超像素区域的方差
    variances = compute_superpixel_variances(normalized_data, superpixel)

    # 初始化稀疏亲和矩阵 A
    A = np.zeros((data.shape[0] * data.shape[1], data.shape[0] * data.shape[1]))

    # 计算相似度矩阵
    for sp in np.unique(superpixel):  # 对每个超像素值进行迭代
        mask = superpixel == sp  # 创建一个布尔掩码，标记出所有值等于当前超像素值的位置
        variance = variances[sp]  # 获取当前超像素区域的方差

        # 从归一化后的数据中获取当前超像素区域的所有像素值，并将其重塑为二维数组，其中每行表示一个像素
        masked_data = normalized_data[mask].reshape(-1, normalized_data.shape[2])

        # 相似度度量方式为高斯核 - RLPA
        if sim_type == 'gauss':
            # 计算当前超像素区域内每对像素之间的点积，得到一个二维数组
            distances_square = np.sum(np.square(masked_data[:, np.newaxis] - masked_data), axis=2)
            similarities = np.exp(-(distances_square / (2 * variance)))

        # 相似度度量方式为余弦距离
        # 暂未启用 - 待确认
        elif sim_type == 'cos':
            # 计算当前超像素区域内每对像素之间的点积，得到一个二维数组
            dot_matrix = np.dot(masked_data, masked_data.T)

            # 计算 masked_data 中每一行的模长
            norms = np.linalg.norm(masked_data, axis=1)
            # 计算 masked_data 中每一列的模长
            norms_T = np.linalg.norm(masked_data.T, axis=0)
            # 外积得到模长的乘积矩阵，用于归一化相似度矩阵
            norms_matrix = np.outer(norms, norms_T)

            # 归一化相似度矩阵，得到余弦相似度矩阵
            similarities = dot_matrix / norms_matrix

        # 相似度度量方式为欧几里得距离
        # 暂未启用 - 待确认
        elif sim_type == 'euclidean':
            distances_squre = np.sum(np.square(masked_data[:, np.newaxis] - masked_data), axis=2)
            similarities = np.exp(-(distances_squre / (2 * variance)))

        # 使用np.where(mask)获取当前超像素区域内每个像素的索引
        idxs = np.where(mask)
        # 将相似度填充到稀疏亲和矩阵A的对应位置
        for i, idx in enumerate(idxs[0]):
            x = idx * data.shape[1] + idxs[1][i]
            y = mask.flatten()
            A[x, y] = similarities[i]

    return A


def generate_SSPTM(A):
    """
    生成SSPTM
    :param A: 亲和矩阵
    :return: SSPTM
    """
    # 将矩阵A与其转置相加，得到对称矩阵W
    W = A + A.T

    # 对W的每一行进行归一化，使其和为1
    normalized_W = W / np.sum(W, axis=1)[:, None]
    return normalized_W


if __name__ == "__main__":
    """
    此处代码用于测试、debug
    """
    import utils


    def generate_test_data(seed=42):
        # 固定随机种子 生成数组
        np.random.seed(42)
        data = np.random.randint(-10, 11, size=(3, 3, 3), dtype=np.int32)
        # 超像素数组
        superpixel = np.random.randint(0, 3, size=(3, 3), dtype=np.int32)
        return data, superpixel


    def include_real_data():
        # 读入数据
        data = np.load('./datasets/IndianPines/IndianPines.npy')
        data_gt = np.load('./datasets/IndianPines/IndianPines_gt.npy')

        # 划分超像素
        T = 50  # 超像素数量
        superpixel = utils.ers(utils.pca(data, 3), T, conn8=1, lamb=0.5, sigma=5, show_image=False, save_image=True)

        # 降维数据
        pca_dim = 16
        reduced_data = utils.pca(data, pca_dim)

        # 标签转为one-hot
        num_classes = 17
        one_hot_gt = utils.to_one_hot(data_gt.flatten(), num_classes).reshape(data_gt.shape + (num_classes,))

        return reduced_data, one_hot_gt, superpixel


    # data, superpixel = generate_test_data()
    data, one_hot_gt, superpixel = include_real_data()

    # 计算稀疏亲和矩阵 A
    A = sparse_affinity_matrix(data, superpixel)

    # shape(21025, 21025)
    SSPTM = generate_SSPTM(A)
    # print(SSPTM)

    label_presudo, cg_solution, info = utils.diffusion_learning(SSPTM, one_hot_gt, alpha=0.5)
    print(info)
    print(label_presudo)

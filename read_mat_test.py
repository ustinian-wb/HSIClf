import scipy.io

# 读取.mat文件
mat_data = scipy.io.loadmat('./datasets/SalinasScene/Salinas_corrected.mat')
# 提取数据
your_array = mat_data['salinas_corrected']

# 如果需要将数组转换为ndarray
import numpy as np
your_ndarray = np.array(your_array)
print(your_ndarray.max())
# 现在your_ndarray就是你想要的NumPy数组了

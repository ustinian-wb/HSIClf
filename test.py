import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# 假设data是你的高光谱图像数据，shape为(145, 145, 200)
# 这里简单生成一个示例数据
img = np.load('./datasets/IndianPines/IndianPines.npy')

import numpy as np
from scipy.ndimage import correlate

# TODO: 待验证
def extract_samples(img, window_size=7):
    # 反射填充
    padded_img = np.pad(img, ((window_size//2, window_size//2), (window_size//2, window_size//2), (0, 0)), mode='reflect')
    samples = []
    for i in range(window_size//2, padded_img.shape[0] - window_size//2):
        for j in range(window_size//2, padded_img.shape[1] - window_size//2):
            # 提取以当前像素为中心的窗口
            sample = padded_img[i-window_size//2:i+window_size//2+1, j-window_size//2:j+window_size//2+1, :]
            samples.append(sample)
    return np.array(samples)


# 假设img是你的(145, 145, 7)的ndarray高光谱图像
samples = extract_samples(img)

# 打印提取样本后的形状
print(samples.shape)

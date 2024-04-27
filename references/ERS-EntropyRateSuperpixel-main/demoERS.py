from ERSModule import *
import cv2
import numpy as np


# 定义一个函数，将标签映射到颜色
def colormap(input, colors):
    height = input.shape[0]
    width = input.shape[1]
    output = np.zeros([height, width, 3], np.uint8)
    for y in range(0, height):
        for x in range(0, width):
            id = int(input[y, x])
            for k in range(0, 3):
                output[y, x, k] = colors[id, k]
    return output


# 设置所需超像素的数量
nC = 100

# 读取输入图像（将“242078.jpg”替换为实际图像路径）
img = cv2.imread("242078.jpg")

# 将图像展平为一维列表以进行处理
img_list = img.flatten().tolist()

# 获取图像尺寸
h = img.shape[0]
w = img.shape[1]

# 将图像转换为灰度图
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用ERS算法以获取超像素标签
label_list = ERS(img_list, h, w, nC)
label = np.reshape(np.asarray(label_list), (h, w))

# 为每个超像素生成随机颜色
colors = np.uint8(np.random.rand(nC, 3) * 255)

# 将标签映射到颜色
output = colormap(label, colors)

# 显示原始图像和超像素分割结果
cv2.imshow("原始图像", img)
cv2.imshow("超像素分割", output)
cv2.waitKey()
cv2.destroyAllWindows()

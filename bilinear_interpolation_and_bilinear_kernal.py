"""
@author: Jerry
@time: 2021/7/27 8:53
@address: Qingdao
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm, trange

# 双线性插值
def Bilinear_interpolation(src, new_size):
    """
        :param src:   扩大前的图像
        :param new_size:   扩大后的图像尺寸
        :return:
    """

    dst_height, dst_width = new_size
    src_height, src_width = src.shape[:2]

    # 求缩放比
    scale_x = float(src_width) / dst_width
    scale_y = float(src_height) / dst_height

    # 图像的大小
    dst = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)

    for n in trange(3):
        for dst_y in trange(dst_height):
            for dst_x in range(dst_width):

                # 目标像素图上的坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 计算在原图上的四个临近点的位置
                src_x_0 = int(np.floor(src_x))
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, src_width - 1)
                src_y_1 = min(src_y_0 + 1, src_height - 1)

                value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, n] + (src_x - src_x_0) * src[src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, n] + (src_x - src_x_0) * src[src_y_1, src_x_1, n]
                dst[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return dst

# # 手动初始化转置卷积的卷积核
# og = np.orgid[:4, :4]
# print(og[0])
# print(og[1])
# filt = (1 - abs(og[0] - 2)/2) * (1 - abs(og[1] - 2) / 2)
# print(filt)


# 手动设计一个滤子
def bilinear_kernal(in_channels, out_channels, kernel_size):
    """
    :param in_channels:  输入通道数
    :param out_channels: 输出通道数
    :param kernel_size:  卷积核的大小
    :return:
    """
    factor = (kernel_size + 1) // 2
    center = kernel_size / 2

    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(in_channels), :, :] = filt

    return torch.from_numpy(weight)


if __name__ == '__main__':
    # img = cv2.imread("./Bzhan.jpg")
    # img_res = Bilinear_interpolation(img, (1000, 1000))
    # cv2.imshow("src", img)
    # cv2.imshow("dst", img_res)
    #
    # key = cv2.waitKey()
    # if key == 27:
    #     cv2.destroyWindows()
    #
    # print(img.shape)
    # print(img_res.shape)

    x = plt.imread("./Bzhan.jpg")
    print(x.shape)

    # unsqueeze(0) 增加一个 batch_size 通道
    x = torch.from_numpy(x.astype('float32')).permute(2, 0, 1).unsqueeze(0)
    print(x.shape)

    # 扩大2倍的 参数
    conv_trans = nn.ConvTranspose2d(in_channels=3, out_channels=3,
                                    kernel_size=4, stride=2, padding=1)

    conv_trans.weight.data = bilinear_kernal(in_channels=3, out_channels=3, kernel_size=4)

    y = conv_trans(x).data.squeeze().permute(1, 2, 0).numpy()
    plt.imshow(y.astype('uint8'))
    print(y.shape)

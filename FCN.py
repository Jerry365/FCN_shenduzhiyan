import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.models import vgg16_bn
import cv2
from tqdm import tqdm, trange
from bilinear_interpolation_and_bilinear_kernal import bilinear_kernal


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



pretrained_net = vgg16_bn(pretrained=False)
# print(pretrained_net.features)

# 第 0 层
# print(pretrained_net.features[0])

# 第 0 层的图像尺寸
# print(pretrained_net.features[0].weight.shape)

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        # conv + conv + pool
        self.stage1 = pretrained_net.features[:7]

        # conv + conv + pool
        self.stage2 = pretrained_net.features[7:14]

        # conv + conv + conv + pool
        self.stage3 = pretrained_net.features[14:24]

        # conv + conv + conv + pool
        self.stage4 = pretrained_net.features[24:34]

        # conv + conv + conv + pool
        self.stage5 = pretrained_net.features[34:]


        self.scores1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.scores2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.scores3 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.conv_tran1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_tran2 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernal(in_channels=num_classes, out_channels=num_classes, kernel_size=16)

        self.upsample_2x_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_2x_1.weight.data = bilinear_kernal(in_channels=512, out_channels=512, kernel_size=4)

        self.upsample_2x_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_2x_2.weight.data = bilinear_kernal(in_channels=256, out_channels=256, kernel_size=4)

    def forward(self, x): # 352, 480, 3
        s1 = self.stage1(x)  # 176, 240, 64
        s2 = self.stage2(s1) # 88, 120, 128
        s3 = self.stage3(s2) # 44, 60, 256
        s4 = self.stage4(s3) # 22, 30, 512
        s5 = self.stage5(s4) # 11, 15, 256

        scores1 = self.scores1(s5) # 11, 15, 12

        s5 = self.upsample_2x_1(s5) # 22, 30, 512
        add1 = s5 + s4              # 22, 30, 512
        scores2 = self.scores2(add1) # 22, 30, 12


        add1 = self.conv_tran1(add1) # 22, 30, 256
        add1 = self.upsample_2x_2(add1) # 44, 60, 256

        add2 = add1 + s3                # 44, 60, 265

        add2 = self.conv_tran2(add2)  # 44, 60, 12
        scores3 = self.upsample_8x(add2) # 352, 480, 12

        return scores3

if __name__ == '__main__':

    # gt 表示标签
    gt = np.random.rand(4, 352, 480)
    gt = gt.astype(np.int64)
    gt = torch.from_numpy(gt)
    print("标签图的尺寸 : ", gt.shape)

    # x 表示原图
    x = torch.randn(4, 3, 352, 480)
    print("原图的尺寸 : ",x.shape)


    net = FCN(12)
    y = net(x)
    print("网络的输出尺寸 : ", y.shape)

    out = F.log_softmax(y, dim=1)
    print("log_softmax 之后 : ", out.shape)

    certerion = nn.NLLLoss()

    loss = certerion(out, gt)
    loss.backward()
    print("损失函数值 : ", loss)


















"""
@author: Jerry
@time: 2021/7/27 8:52
@address: Qingdao
"""
#################################### 数据预处理类#####################################
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cfg

class LabelProcessor:
    def __init__(self, file_path):
        self.colormap = self.read_color_map(file_path)
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []

        # pd_label_color.index == RangeIndex(start=0, stop=12, step=1)
        for i in range(len(pd_label_color.index)):
            # 按行读取
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap


    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256**3)
        for i, cm in enumerate(colormap):
            cm2lbl[((cm[0] * 256) + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class CamvidDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        """
            :param file_path:  数据和标签的路径
            :param crop_size:  图片的尺寸
        """
        if len(file_path) != 2:
            raise ValueError("同时指定图片和标签的路径，图片路径在前，标签路径在后")

        self.img_path = file_path[0]
        self.label_path = file_path[1]

        # 原图和标签的路径
        self.imgs = self.read(self.img_path)
        self.labels = self.read(self.label_path)

        # 初始化数据处理的函数设置
        self.crop_size = crop_size


    def __getitem__(self, item):
        img = self.imgs[item]
        label = self.labels[item]

        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        # 中心裁剪
        img, label = self.center_crop(img, label, self.crop_size)

        # 图片转换
        img, label = self.img_transform(img, label)

        sample = {
            'img': img,
            'label': label
        }
        return sample

    # 返回图片的数量
    def __len__(self):
        return len(self.imgs)

    def read(self, path):
        # 取出图片的名称
        file_list = os.listdir(path)

        # 图片的完整路径
        file_path_list = [os.path.join(path, img) for img in file_list]

        # 图片排序
        file_path_list.sort()

        # 返回所有的完整路径
        return file_path_list

    def center_crop(self, img, label, crop_size):
        img = F.center_crop(img, crop_size)
        label = F.center_crop(label, crop_size)
        return img, label

    def img_transform(self, img, label):
        ###################################  img 数据处理 ########################################
        # 定义图片转换方式
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]
        )
        # 原图进行 transform
        img = transform_img(img)

        ####################################### label 数据处理 #######################################
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))

        # 标签进行编码
        label = label_processor.encode_label_img(label)
        label = torch.from_numpy(label)
        return img, label


label_processor = LabelProcessor(cfg.class_dict_path)


if __name__ == '__main__':
    BATCH_SIZE = 4
    EPOCH_NUMBER = 2
    TRAIN_ROOT = './CamVid/train'
    TRAIN_LABEL = './CamVid/train_labels'

    VAL_ROOT = './CamVid/val'
    VAL_LABEL = './CamVid/val_labels'

    TEST_ROOT = './CamVid/test'
    TEST_LABEL = './CamVid/test_labels'
    class_dict_path = 'CamVid/num_classes.csv'
    crop_size = (352, 480)

    Cam_train = CamvidDataset([TRAIN_ROOT, TRAIN_LABEL], crop_size)
    Cam_val = CamvidDataset([VAL_ROOT, VAL_LABEL], crop_size)
    Cam_test = CamvidDataset([TEST_ROOT, TEST_LABEL], crop_size)


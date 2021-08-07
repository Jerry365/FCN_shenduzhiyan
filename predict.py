import os
import pandas as pd
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
from evalution_segmentation import eval_semantic_segmentation
from FCN import FCN
import cfg
from PIL import Image
import numpy as np


# 有显卡用显卡跑，没有用 cpu 跑
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

miou_list = [0]


# 载入测试集
Cam_test = CamvidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)

test_data = DataLoader(Cam_test, batch_size=1, shuffle=False, num_workers=0)

# 网络载入到 GPU 或者 CPU
net = FCN(12)
net.to(device)
net.load_state_dict(torch.load('train_moiu = 0.32850_1.pth'))


pd_value_color = pd.read_csv('./CamVid/num_classes.csv', sep=',')
name_value = pd_value_color['name'].values
num_class = len(name_value)
colormap = []

for i in range(num_class):
   tmp = pd_value_color.iloc[i]
   color = [tmp['r'], tmp['g'], tmp['b']]
   colormap.append(color)

cm = np.array(colormap).astype('uint8')


# 图片存放的目录
dir = './result_picture/'
if not os.path.exists(dir):
    os.mkdir(dir)

for i, sample in enumerate(test_data):
    valImg = sample['img'].to(device)

    out = net(valImg)
    out = F.log_softmax(out, dim=1)

    pre_label = out.max(1)[1].squeeze().cpu().numpy()

    # 上色
    pre = cm[pre_label]


    pixel = Image.fromarray(pre)
    pixel.save(dir + str(i) + '.png')


import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
from evalution_segmentation import eval_semantic_segmentation
from FCN import FCN
import cfg


# 有显卡用显卡跑，没有用 cpu 跑
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

miou_list = [0]


# 载入测试集
Cam_test = CamvidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Cam_test, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

net = FCN(12)

net.to(device)
net.load_state_dict(torch.load('train_moiu = 0.32850_1.pth'))

train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0


for i, sample in enumerate(test_data):
    img = Variable(sample["img"].to(device))
    label = Variable(sample["label"].to(device))

    out = net(img)
    out = F.log_softmax(out, dim=1)

    # 取最大值的索引
    pre_label = out.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_label]

    true_label = label.data.cpu().numpy()
    true_label = [i for i in true_label]

    # 计算混淆矩阵
    eval_metric = eval_semantic_segmentation(pre_label, true_label)

    train_acc += eval_metric["mean_class_accuracy"]
    train_miou += eval_metric["miou"]

    # 类准确度
    train_mpa += eval_metric["pixel_accuracy"]

    if len(eval_metric['class_accuracy']) < 12:
        eval_metric['class_accuracy'] = 0
        train_class_acc += eval_metric['class_accuracy']
        error += 1
    else:
        train_class_acc += eval_metric['class_accuracy']

    # print(eval_metric['class_accuracy'], "================", i)



# 一次大循环下的结果指标
epoch_str = 'Test Acc : {:.5f}   Test Mpa : {:.5f}  Test Mean : {:.5f} \n Test class acc : {:}'.format(
    train_acc / len(test_data) - error,
    train_miou / len(test_data) - error,
    train_mpa / len(test_data) - error,
    train_class_acc / len(test_data) - error
)

if train_miou / len(test_data) - error > max(miou_list):
    miou_list.append(train_miou / (len(test_data) - error))
    print("最大的 miou 序列 = " , miou_list)
    print(epoch_str + "=============")
"""
@author: Jerry
@time: 2021/7/28 10:18
@address: Qingdao
"""
import torch
from torch.nn import functional as F
import numpy as np


gt = np.random.randn(1, 2, 3)
gt = gt.astype(np.int64)
gt = torch.from_numpy(gt)

x = torch.randn(1, 12, 2, 3)
out = F.log_softmax(x, dim=1)

print(gt)
print(x)
print(out)


pred_label = out.max(dim=1)[1].data.cpu().numpy()
pred_label = [i for i in pred_label]

gt_label = gt.data.cpu().numpy()
gt_label = [i for i in gt_label]


for pred, label in zip(pred_label, gt_label):
    print(pred.ndim, label.ndim)
    print(pred.shape, pred.shape)

pred_label, gt_label = pred_label[0], gt_label[0]
pred_label = pred_label.flatten()
gt_label = gt_label.flatten()

print(pred_label)
print(gt_label)

np.max((pred_label, gt_label))

mask = gt_label >= 0



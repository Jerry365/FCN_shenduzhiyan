from __future__ import division
import torch
import torch.nn.functional  as F
import numpy as np
import six
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore')

def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 12

    confusion = np.zeros((n_class, n_class),dtype=np.int64)  # (12，12)

    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should be same.')

        pred_label = pred_label.flatten()  # (168960，)
        gt_label = gt_label.flatten()      #(168960，)

        lb_max = np.max((pred_label, gt_label))

        if lb_max >= n_class:
            expanded_confusion = np.zeros((lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion
            n_class = lb_max + 1
            confusion = expanded_confusion
            # Count statistics from valid pixels.极度巧妙× class_nums正好使得每个ij能够对应.mask = gt_label >= 0

        # 掩码
        mask = gt_label >= 0

        confusion += np.bincount(n_class * gt_label[mask].astype(int) +
                                 pred_label[mask], minlength=n_class**2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need '
                             'to be same')

    return confusion

    # This code assumes any iterator does not contain None as its items.


def calc_semantic_segmentation_iou(confusion):
    iou_denominator = (confusion.sum(axis=1) +
                       confusion.sum(axis=0) -
                       np.diag(confusion))

    # 此行出错
    iou = np.diag(confusion) / iou_denominator

    # 包含和不包含背景
    return iou[:-1]


def eval_semantic_segmentation(pred_labels, gt_labels):
    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)

    iou = calc_semantic_segmentation_iou(confusion)  # (11，)

    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()

    # 1e-10 防止分母为 0
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)

    return {
        'iou': iou,
        'miou': np.nanmean(iou),
        'pixel_accuracy': pixel_accuracy,
        'class_accuracy': class_accuracy,
        'mean_class_accuracy': np.nanmean(class_accuracy[:-1])
    }
    # ..mean_class_accuracy' : ..np. nanmean(class_accuracy)l


# src/utils/metrics.py

import torch

def calculate_accuracy(outputs, labels):
    """
    计算准确率
    """
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels.data)
    acc = corrects.double() / labels.size(0)
    return acc
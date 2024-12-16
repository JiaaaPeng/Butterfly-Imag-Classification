# src/models/resnet50.py

import torch.nn as nn
from torchvision import models


def get_model(num_classes):
    """
    获取使用预训练权重的 ResNet-50 模型，并修改全连接层以适应 100 个类别。
    """
    # 加载预训练的 ResNet-50 模型
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # 替换全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
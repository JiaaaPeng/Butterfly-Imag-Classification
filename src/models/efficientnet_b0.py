# src/models/efficientnet_b0.py

import torch.nn as nn
from torchvision import models


def get_model(num_classes):
    """
    获取使用预训练权重的 EfficientNet-B0 模型，并修改全连接层以适应 100 个类别。
    """
    # 加载预训练的 EfficientNet-B0 模型
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # 替换分类器
    in_features = model.classifier[1].in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model
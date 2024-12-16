# src/models/mobilenet_v3_large.py

import torch.nn as nn
from torchvision import models


def get_model(num_classes):
    """
    获取使用预训练权重的 MobileNetV3-Large 模型，并修改全连接层以适应 100 个类别。
    """
    # 加载预训练的 MobileNetV3-Large 模型
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    # 替换分类器
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model

if __name__ == '__main__':
    print(get_model(100))
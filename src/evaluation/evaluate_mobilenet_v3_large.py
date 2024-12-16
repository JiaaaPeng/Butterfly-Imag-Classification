# src/evaluation/evaluate_mobilenet_v3_large.py

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.utils.data_loader import get_data_loaders
from src.models.mobilenet_v3_large import get_model
from src.utils.metrics import calculate_accuracy
from src.utils.visualization import save_metrics_to_csv


def load_model(model, model_path, device='cpu'):
    """
    加载模型权重
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def save_original_image(image_tensor, img_name, output_dir='../../outputs/feature_maps/mobilenet_v3_large/original_images'):
    """
    保存原始图像为文件，并记录到 TensorBoard
    """
    os.makedirs(output_dir, exist_ok=True)
    # 反归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    image = inv_normalize(image_tensor.cpu())
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).numpy()

    # 保存图片
    save_path = os.path.join(output_dir, f'{img_name}_original.png')
    plt.imsave(save_path, image_np)

    return image_np


def save_feature_map(feature_map, layer_name, img_name, channel_idx, block_output_dir, writer):
    """
    保存特征图为图片，并记录到 TensorBoard
    """
    os.makedirs(block_output_dir, exist_ok=True)
    feature_map = feature_map.cpu().numpy()

    # 选择特征图的指定通道
    channel = feature_map[channel_idx]

    plt.figure(figsize=(5, 5))
    plt.imshow(channel, cmap='viridis')
    plt.axis('off')
    plt.title(f'{layer_name} - Channel {channel_idx} - {img_name}')
    save_path = os.path.join(block_output_dir, f'{img_name}_{layer_name}_channel{channel_idx}.png')
    plt.savefig(save_path)
    plt.close()

    # 将特征图记录到 TensorBoard
    writer.add_image(f'{layer_name}/Channel_{channel_idx}/{img_name}', channel, dataformats='HW')


def main():
    # 定义数据路径
    train_dir = '../../data/train'
    valid_dir = '../../data/valid'

    # 获取数据加载器和类别名称
    dataloaders, class_names = get_data_loaders(train_dir, valid_dir, batch_size=8, num_workers=0)

    # 获取类别数量
    num_classes = len(class_names)

    # 获取模型
    model = get_model(num_classes)

    # 加载训练好的模型权重
    model_path = '../../experiments/mobilenet_v3_large/checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print(f"模型检查点文件 {model_path} 未找到。请先训练 MobileNetV3-Large 模型。")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, model_path, device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义需要捕获特征图的层名称
    # MobileNetV3-Large 的 blocks 通常包括 features.0, features.1, ..., features.n
    target_layers = [f'features.{i}' for i in range(len(model.features))]

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir='../../experiments/mobilenet_v3_large/features_tensorboard')

    # 定义一个字典来存储特征图
    feature_maps = {}

    def get_hook(name):
        def hook(module, input, output):
            feature_maps[name] = output

        return hook

    # 注册 Hook
    hooks = []
    for layer_name in target_layers:
        if layer_name in dict([*model.named_modules()]):
            layer = dict([*model.named_modules()])[layer_name]
            hooks.append(layer.register_forward_hook(get_hook(layer_name)))
        else:
            print(f"层 {layer_name} 不存在于模型中。")

    # 创建输出目录结构
    blocks_output_dir = '../../outputs/feature_maps/mobilenet_v3_large/features_mobilenet_v3_large'
    for layer_name in target_layers:
        os.makedirs(os.path.join(blocks_output_dir, layer_name), exist_ok=True)
    # 创建原始图像保存目录
    original_images_dir = '../../outputs/feature_maps/mobilenet_v3_large/original_images'
    os.makedirs(original_images_dir, exist_ok=True)

    # 评估模型
    running_loss = 0.0
    running_corrects = 0

    # 标记是否处理了第一张图片
    first_image_processed = False

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['valid'], desc='评估中'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += acc.item() * inputs.size(0)

            # 处理每个批次中的每个样本
            batch_size = inputs.size(0)
            for i in range(batch_size):
                img_path, _ = dataloaders['valid'].dataset.imgs[i]
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                if not first_image_processed:
                    # 保存原始图像
                    original_image_np = save_original_image(inputs[i], img_name, original_images_dir)

                    # 将原始图像记录到 TensorBoard
                    writer.add_image(f'Original Images/{img_name}', original_image_np, dataformats='HWC')

                    # 保存和可视化所有 block 的所有通道特征图
                    for layer_name in target_layers:
                        if layer_name in feature_maps:
                            feature = feature_maps[layer_name][i]
                            block_output_dir = os.path.join(blocks_output_dir, layer_name)
                            num_channels = feature.size(0)
                            for channel_idx in range(num_channels):
                                save_feature_map(feature, layer_name, img_name, channel_idx, block_output_dir, writer)

                    first_image_processed = True

    epoch_loss = running_loss / len(dataloaders['valid'].dataset)
    epoch_acc = running_corrects / len(dataloaders['valid'].dataset)

    print(f'MobileNetV3-Large 验证集 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 保存评估结果到 CSV
    results = {
        'model': ['MobileNetV3-Large'],
        'valid_loss': [epoch_loss],
        'valid_acc': [epoch_acc]
    }
    csv_path = '../../experiments/mobilenet_v3_large/results/evaluation_results.csv'
    save_metrics_to_csv(results, csv_path)
    print(f'评估结果已保存到 {csv_path}')

    # 移除 Hook
    for hook in hooks:
        hook.remove()

    # 关闭 TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()
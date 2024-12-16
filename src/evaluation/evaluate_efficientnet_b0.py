# src/evaluation/evaluate_efficientnet_b0.py

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
from src.models.efficientnet_b0 import get_model
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


def save_feature_map(feature_map, layer_name, img_name, channel, block_output_dir):
    """
    保存单个通道的特征图为图片，并记录到 TensorBoard
    """
    os.makedirs(block_output_dir, exist_ok=True)
    feature_map = feature_map.cpu().numpy()

    # 选择指定通道
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map[channel], cmap='viridis')
    plt.axis('off')
    plt.title(f'{layer_name} - {img_name} - Channel {channel}')
    save_path = os.path.join(block_output_dir, f'{img_name}_{layer_name}_channel{channel}.png')
    plt.savefig(save_path)
    plt.close()


def save_original_image(img_path, img_name, original_output_dir):
    """
    保存原始图片到文件夹中
    """
    os.makedirs(original_output_dir, exist_ok=True)
    image = Image.open(img_path).convert('RGB')
    save_path = os.path.join(original_output_dir, f'{img_name}.png')
    image.save(save_path)


def log_image_to_tensorboard(writer, tag, img_path):
    """
    将原始图片记录到 TensorBoard
    """
    image = Image.open(img_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    writer.add_image(tag, image_tensor, dataformats='CHW')


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
    model_path = '../../experiments/efficientnet_b0/checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print(f"模型检查点文件 {model_path} 未找到。请先训练 EfficientNet-B0 模型。")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, model_path, device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义需要捕获特征图的层名称
    # EfficientNet-B0 的特征层名称，根据模型结构调整
    # EfficientNet-B0 的 blocks 通常命名为 'features.0', 'features.1', ..., 'features.6'
    target_layers = [f'features.{i}' for i in range(len(model.features))]

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir='../../experiments/efficientnet_b0/features_tensorboard')

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
    blocks_output_dir = '../../outputs/feature_maps/efficientnet_b0/features'
    os.makedirs(blocks_output_dir, exist_ok=True)
    original_output_dir = '../../outputs/feature_maps/efficientnet_b0/originals'
    os.makedirs(original_output_dir, exist_ok=True)

    # 评估模型
    running_loss = 0.0
    running_corrects = 0

    # 标记是否已经处理了第一张图片
    first_image_processed = False

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders['valid'], desc='评估中')):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += acc.item() * inputs.size(0)

            # 处理第一张图片
            if not first_image_processed:
                if batch_idx * dataloaders['valid'].batch_size + 0 < len(dataloaders['valid'].dataset):
                    img_idx = batch_idx * dataloaders['valid'].batch_size + 0
                    img_path, _ = dataloaders['valid'].dataset.imgs[img_idx]
                    img_name = os.path.splitext(os.path.basename(img_path))[0]

                    # 保存原始图片
                    save_original_image(img_path, img_name, original_output_dir)

                    # 记录原始图片到 TensorBoard
                    log_image_to_tensorboard(writer, f'Original Images/{img_name}', img_path)

                    # 遍历所有 target_layers
                    for layer_name in target_layers:
                        if layer_name in feature_maps:
                            feature = feature_maps[layer_name][0]  # 第一个图像
                            num_channels = feature.shape[0]
                            block_output_dir = os.path.join(blocks_output_dir, layer_name)
                            os.makedirs(block_output_dir, exist_ok=True)

                            for c in range(num_channels):
                                # 保存每个通道的特征图
                                save_feature_map(feature, layer_name, img_name, c, block_output_dir)

                                # 将特征图记录到 TensorBoard
                                writer.add_image(f'{layer_name}/{img_name}/channel_{c}', feature[c].cpu(),
                                                 dataformats='HW')

                    first_image_processed = True

            # 处理其他图片（不需要保存特征图）
            # 如果需要对更多图片进行处理，可以扩展此部分

    epoch_loss = running_loss / len(dataloaders['valid'].dataset)
    epoch_acc = running_corrects / len(dataloaders['valid'].dataset)

    print(f'EfficientNet-B0 验证集 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 保存评估结果到 CSV
    results = {
        'model': ['EfficientNet-B0'],
        'valid_loss': [epoch_loss],
        'valid_acc': [epoch_acc]
    }
    csv_path = '../../experiments/efficientnet_b0/results/evaluation_results.csv'
    save_metrics_to_csv(results, csv_path)
    print(f'评估结果已保存到 {csv_path}')

    # 移除 Hook
    for hook in hooks:
        hook.remove()

    # 关闭 TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()
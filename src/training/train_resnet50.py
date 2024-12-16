# src/training/train_resnet50.py

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.utils.data_loader import get_data_loaders
from src.utils.metrics import calculate_accuracy
from src.utils.visualization import save_metrics_to_csv
from src.models.resnet50 import get_model

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=50, patience=10, device='cpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # 记录每个 epoch 的指标
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir='../../experiments/resnet50/logs')

    epochs_no_improve = 0

    # 检查是否存在最佳模型检查点
    checkpoint_path = '../../experiments/resnet50/checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"加载最佳模型检查点: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        best_loss = float('inf')  # 重置为极大值以便重新找到更好的模型

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个 epoch 包含训练和验证阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    acc = calculate_accuracy(outputs, labels)

                    # 反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += acc.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录指标
            if phase == 'train':
                metrics['train_loss'].append(epoch_loss)
                metrics['train_acc'].append(epoch_acc)
                writer.add_scalar('Loss/Train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
            else:
                metrics['valid_loss'].append(epoch_loss)
                metrics['valid_acc'].append(epoch_acc)
                writer.add_scalar('Loss/Valid', epoch_loss, epoch)
                writer.add_scalar('Accuracy/Valid', epoch_acc, epoch)

                # 检查是否为最佳模型
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, checkpoint_path)
                    print(f'验证集损失降低，模型已保存为 {checkpoint_path}')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f'验证集损失未降低，当前早停计数: {epochs_no_improve}/{patience}')

        # 记录 epoch
        metrics['epoch'].append(epoch + 1)

        # 检查早停条件
        if epochs_no_improve >= patience:
            print('早停条件满足，停止训练')
            break

        print()

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f} 分 {time_elapsed % 60:.0f} 秒')
    print(f'最佳验证损失: {best_loss:.4f}')

    # # 记录训练总时长
    # metrics['total_time'] = [time_elapsed]

    # 保存指标到 CSV
    csv_path = '../../experiments/resnet50/results/metrics.csv'
    save_metrics_to_csv(metrics, csv_path)
    print(f'训练指标已保存到 {csv_path}')

    # 保存 total_time 到文本文件
    with open('../../experiments/resnet50/results/total_time.txt', 'w') as f:
        f.write(f"Total training time: {time_elapsed // 60:.0f} minutes {time_elapsed % 60:.0f} seconds")
    print('训练总时长已保存到 total_time.txt')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

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

    # 冻结预训练模型的所有层，仅训练分类层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻分类层的参数以进行训练
    if isinstance(model.fc, nn.Linear):
        model.fc.requires_grad_(True)
    elif isinstance(model.fc, nn.Sequential):
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError("未知的分类器结构")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 检查并创建必要的文件夹
    os.makedirs('../../experiments/resnet50/checkpoints', exist_ok=True)
    os.makedirs('../../experiments/resnet50/results', exist_ok=True)
    os.makedirs('../../experiments/resnet50/logs', exist_ok=True)

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练模型
    trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler,
                                num_epochs=100, patience=10, device=device)


if __name__ == '__main__':
    main()
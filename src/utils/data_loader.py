# src/utils/data_loader.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, valid_dir, batch_size=8, num_workers=0):
    """
    获取训练和验证数据加载器
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(), # 随机水平翻转
            transforms.RandomRotation(20), # 随机旋转20度
            transforms.Resize((224, 224)),  # 固定大小
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]) # 归一化
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),  # 固定大小
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    return dataloaders, train_dataset.classes
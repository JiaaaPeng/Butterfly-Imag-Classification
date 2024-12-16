# src/predict.py

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.models.efficientnet_b0 import get_model as get_efficientnet_b0
from src.models.mobilenet_v3_large import get_model as get_mobilenet_v3_large
from src.models.resnet50 import get_model as get_resnet50


class ResizeAndPad:
    def __init__(self, size, fill=0):
        """
        Args:
            size (tuple or int): Desired output size. If tuple, output size will be matched to this. If int, smaller edge will be matched to this.
            fill (int or tuple): Pixel fill value for padding. Default: 0
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.fill = fill

    def __call__(self, img):
        # Get current and desired aspect ratios
        original_size = img.size  # (width, height)
        ratio = min(self.size[0] / original_size[0], self.size[1] / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        img = img.resize(new_size, Image.BICUBIC)

        # Create a new image and paste the resized on it
        new_img = Image.new("RGB", self.size, (self.fill, self.fill, self.fill))
        paste_position = ((self.size[0] - new_size[0]) // 2,
                          (self.size[1] - new_size[1]) // 2)
        new_img.paste(img, paste_position)
        return new_img


def load_model(model, model_path, device='cpu'):
    """
    加载模型权重
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    # 定义数据路径
    input_dir = '../input'
    output_dir = '../outputs/predictions'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义图像预处理
    test_transforms = transforms.Compose([
        ResizeAndPad((224, 224)),  # 使用自定义的保持纵横比缩放并填充的转换
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 定义测试集数据集类
    class TestDataset(Dataset):
        def __init__(self, img_dir, transform=None):
            self.img_dir = img_dir
            self.transform = transform
            # 按字母顺序排序
            self.img_names = sorted([f for f in os.listdir(img_dir)
                                     if os.path.isfile(os.path.join(img_dir, f))
                                     and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

        def __len__(self):
            return len(self.img_names)

        def __getitem__(self, idx):
            img_name = self.img_names[idx]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_name

    # 创建测试集数据加载器
    test_dataset = TestDataset(input_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 加载类别名称
    class_names_path = '../class_names.txt'
    if not os.path.exists(class_names_path):
        print(f"类别名称文件 {class_names_path} 未找到。请确保该文件存在并包含所有类别名称。")
        return

    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
    print(f"类别数量: {num_classes}")

    # 加载三个基础模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EfficientNet-B0
    model1 = get_efficientnet_b0(num_classes)
    model1_path = '../experiments/efficientnet_b0/checkpoints/best_model.pth'
    if not os.path.exists(model1_path):
        print(f"模型检查点文件 {model1_path} 未找到。请先训练 EfficientNet-B0 模型。")
        return
    model1 = load_model(model1, model1_path, device)

    # MobileNetV3-Large
    model2 = get_mobilenet_v3_large(num_classes)
    model2_path = '../experiments/mobilenet_v3_large/checkpoints/best_model.pth'
    if not os.path.exists(model2_path):
        print(f"模型检查点文件 {model2_path} 未找到。请先训练 MobileNetV3-Large 模型。")
        return
    model2 = load_model(model2, model2_path, device)

    # ResNet-50
    model3 = get_resnet50(num_classes)
    model3_path = '../experiments/resnet50/checkpoints/best_model.pth'
    if not os.path.exists(model3_path):
        print(f"模型检查点文件 {model3_path} 未找到。请先训练 ResNet-50 模型。")
        return
    model3 = load_model(model3, model3_path, device)

    # 定义 Softmax
    softmax = nn.Softmax(dim=1)

    for inputs, img_names in tqdm(test_loader, desc='预测中'):
        inputs = inputs.to(device)

        # EfficientNet-B0 预测
        outputs1 = model1(inputs)
        probs1 = softmax(outputs1)
        top3_probs1, top3_idxs1 = torch.topk(probs1, k=3, dim=1)

        # MobileNetV3-Large 预测
        outputs2 = model2(inputs)
        probs2 = softmax(outputs2)
        top3_probs2, top3_idxs2 = torch.topk(probs2, k=3, dim=1)

        # ResNet-50 预测
        outputs3 = model3(inputs)
        probs3 = softmax(outputs3)
        top3_probs3, top3_idxs3 = torch.topk(probs3, k=3, dim=1)

        # 遍历每个样本，保存预测结果为单独的 CSV 文件
        for i in range(len(img_names)):
            img_name = img_names[i]
            base_name = os.path.splitext(img_name)[0]
            # EfficientNet-B0
            top3_prob1 = top3_probs1[i].cpu().detach().numpy()
            top3_idx1 = top3_idxs1[i].cpu().detach().numpy()
            top3_classes1 = [class_names[idx] for idx in top3_idx1]
            other_prob1 = 1.0 - top3_prob1.sum()
            other_prob1 = max(other_prob1, 0.0)  # 防止负数
            # 保留两位小数
            top3_prob1 = [round(prob, 2) for prob in top3_prob1]
            other_prob1 = round(other_prob1, 2)

            # MobileNetV3-Large
            top3_prob2 = top3_probs2[i].cpu().detach().numpy()
            top3_idx2 = top3_idxs2[i].cpu().detach().numpy()
            top3_classes2 = [class_names[idx] for idx in top3_idx2]
            other_prob2 = 1.0 - top3_prob2.sum()
            other_prob2 = max(other_prob2, 0.0)
            top3_prob2 = [round(prob, 2) for prob in top3_prob2]
            other_prob2 = round(other_prob2, 2)

            # ResNet-50
            top3_prob3 = top3_probs3[i].cpu().detach().numpy()
            top3_idx3 = top3_idxs3[i].cpu().detach().numpy()
            top3_classes3 = [class_names[idx] for idx in top3_idx3]
            other_prob3 = 1.0 - top3_prob3.sum()
            other_prob3 = max(other_prob3, 0.0)
            top3_prob3 = [round(prob, 2) for prob in top3_prob3]
            other_prob3 = round(other_prob3, 2)

            # 构建 DataFrame
            prediction = {
                # EfficientNet-B0
                'Model': 'EfficientNet-B0',
                'Class1': top3_classes1[0],
                'Prob1': top3_prob1[0],
                'Class2': top3_classes1[1],
                'Prob2': top3_prob1[1],
                'Class3': top3_classes1[2],
                'Prob3': top3_prob1[2],
                'Other_Prob': other_prob1
            }
            df1 = pd.DataFrame([prediction])

            prediction = {
                # MobileNetV3-Large
                'Model': 'MobileNetV3-Large',
                'Class1': top3_classes2[0],
                'Prob1': top3_prob2[0],
                'Class2': top3_classes2[1],
                'Prob2': top3_prob2[1],
                'Class3': top3_classes2[2],
                'Prob3': top3_prob2[2],
                'Other_Prob': other_prob2
            }
            df2 = pd.DataFrame([prediction])

            prediction = {
                # ResNet-50
                'Model': 'ResNet-50',
                'Class1': top3_classes3[0],
                'Prob1': top3_prob3[0],
                'Class2': top3_classes3[1],
                'Prob2': top3_prob3[1],
                'Class3': top3_classes3[2],
                'Prob3': top3_prob3[2],
                'Other_Prob': other_prob3
            }
            df3 = pd.DataFrame([prediction])

            # 合并三个 DataFrame，分别为三行
            df = pd.concat([df1, df2, df3], ignore_index=True)

            # 保存为 CSV 文件
            csv_path = os.path.join(output_dir, f"{base_name}.csv")
            df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()
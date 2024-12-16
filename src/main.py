# src/main.py

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Image Classification Project")
    parser.add_argument('--train_efficientnet_b0', action='store_true', help='训练 EfficientNet-B0 模型')
    parser.add_argument('--train_mobilenet_v3_large', action='store_true', help='训练 MobileNetV3-Large 模型')
    parser.add_argument('--train_resnet50', action='store_true', help='训练 ResNet-50 模型')
    parser.add_argument('--evaluate_efficientnet_b0', action='store_true', help='评估 EfficientNet-B0 模型')
    parser.add_argument('--evaluate_mobilenet_v3_large', action='store_true', help='评估 MobileNetV3-Large 模型')
    parser.add_argument('--evaluate_resnet50', action='store_true', help='评估 ResNet-50 模型')
    parser.add_argument('--predict', action='store_true', help='进行预测')
    args = parser.parse_args()

    if args.train_efficientnet_b0:
        subprocess.run(['python', 'src/training/train_efficientnet_b0.py'])
    if args.train_mobilenet_v3_large:
        subprocess.run(['python', 'src/training/train_mobilenet_v3_large.py'])
    if args.train_resnet50:
        subprocess.run(['python', 'src/training/train_resnet50.py'])
    if args.evaluate_efficientnet_b0:
        subprocess.run(['python', 'src/evaluation/evaluate_efficientnet_b0.py'])
    if args.evaluate_mobilenet_v3_large:
        subprocess.run(['python', 'src/evaluation/evaluate_mobilenet_v3_large.py'])
    if args.evaluate_resnet50:
        subprocess.run(['python', 'src/evaluation/evaluate_resnet50.py'])
    if args.predict:
        subprocess.run(['python', 'src/predict.py'])

if __name__ == '__main__':
    main()
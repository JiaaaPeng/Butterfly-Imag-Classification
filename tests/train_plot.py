import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# 定义三个CSV文件的路径
csv_files = {
    'efficientnet_b0': '../experiments/efficientnet_b0/results/metrics.csv',  # 替换为Model A的CSV文件路径
    'mobilenet_v3_large': '../experiments/mobilenet_v3_large/results/metrics.csv',  # 替换为Model B的CSV文件路径
    'resnet50': '../experiments/resnet50/results/metrics.csv'   # 替换为Model C的CSV文件路径
}

# 初始化TensorBoard的SummaryWriter
log_dir = 'logs/training_results'  # 定义日志目录
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # 如果日志目录不存在，则创建它

# 创建一个字典来存储每个模型的SummaryWriter
writers = {}
for model_name in csv_files.keys():
    model_log_dir = os.path.join(log_dir, model_name)  # 每个模型有自己的子目录
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)  # 创建模型的子目录
    writers[model_name] = SummaryWriter(log_dir=model_log_dir)  # 创建SummaryWriter

# 定义要绘制的指标
metrics = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']

# 遍历每个模型并记录指标
for model_name, csv_path in csv_files.items():
    writer = writers.get(model_name)
    if writer is None:
        print(f"警告: 没有找到模型 {model_name} 的writer。")
        continue

    # 读取CSV文件到pandas的DataFrame
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 文件 {csv_path} 未找到。")
        continue
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 {csv_path} 是空的。")
        continue
    except pd.errors.ParserError:
        print(f"错误: 文件 {csv_path} 解析失败。")
        continue

    # 确保CSV文件包含所需的列
    required_columns = ['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc']
    if not all(column in df.columns for column in required_columns):
        print(f"错误: 文件 {csv_path} 不包含所有必需的列。")
        continue

    # 遍历DataFrame中的每一行并记录指标
    for _, row in df.iterrows():
        epoch = int(row['epoch'])  # 获取当前epoch
        for metric in metrics:
            value = row[metric]  # 获取当前指标的值
            tag = metric  # 标签仅为指标名
            writer.add_scalar(tag, value, epoch)  # 将标量添加到TensorBoard

# 关闭所有SummaryWriter
for writer in writers.values():
    writer.close()

print(f"日志记录完成。")
# src/utils/visualization.py

import pandas as pd

def save_metrics_to_csv(metrics, csv_path):
    """
    将训练和验证的指标保存到 CSV 文件
    """
    df = pd.DataFrame(metrics)
    df.to_csv(csv_path, index=False)
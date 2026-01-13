"""
数据加载模块

提供统一的数据集加载接口。
"""

from src.data.dataset_loader import (
    # 数据类
    DataLoader,
    Sample,
    extract_boxed_answer,
    # Countdown
    generate_countdown_data,
    # AIME
    load_aime_dataset,
    # DAPO-Math-17k
    load_dapo_math_dataset,
    # GSM8K
    load_gsm8k,
    # MATH
    load_math_dataset,
    # OpenR1 (SFT)
    load_openr1_dataset,
)

__all__ = [
    # 数据类
    "DataLoader",
    "Sample",
    # 加载函数
    "load_gsm8k",
    "load_math_dataset",
    "load_dapo_math_dataset",
    "load_aime_dataset",
    "load_openr1_dataset",
    "generate_countdown_data",
    # 工具函数
    "extract_boxed_answer",
]

"""
训练配置类

统一的配置类定义，用于 SFT 和 RL 训练。

Author: Guanghan Ning
Date: 2025-01-10
Updated: 2026-01-13 (从训练脚本提取)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    """基础配置类 - 共享字段"""

    # 实验设置
    experiment_name: str = "experiment"
    scale: str = "small"  # quick, small, medium, large, full
    seed: int = 42

    # 模型设置
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 128

    # 输出设置
    output_dir: str = "outputs"

    # 日志设置
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50


@dataclass
class SFTConfig(BaseConfig):
    """
    Cold Start SFT 配置

    用于 SFT 训练，"唤醒" 模型生成 <think>...</think> 格式的能力。
    """

    # 实验设置
    experiment_name: str = "coldstart_sft"
    scale: str = "small"

    # 训练设置
    num_steps: int = 1000
    batch_size: int = 4
    gradient_accumulation: int = 4  # 有效 batch = batch_size * gradient_accumulation

    # 优化器设置
    learning_rate: float = 2e-5
    warmup_steps: int = 100

    # 数据设置
    max_seq_length: int = 8192  # 包含 prompt + response
    max_samples: int | None = None
    dataset_config: str = "default"  # default, extended, all

    # 评估设置
    eval_samples: int = 50

    # 输出设置
    output_dir: str = "outputs/coldstart_sft"

    def __post_init__(self):
        """根据 scale 调整参数"""
        scale_configs = {
            "quick": {
                "num_steps": 10,
                "batch_size": 2,
                "gradient_accumulation": 1,
                "warmup_steps": 2,
                "log_interval": 2,
                "save_interval": 10,
                "eval_interval": 5,
                "eval_samples": 10,
                "max_samples": 100,
            },
            "small": {
                "num_steps": 800,
                "batch_size": 8,
                "gradient_accumulation": 2,
                "warmup_steps": 50,
                "log_interval": 10,
                "save_interval": 100,
                "eval_interval": 50,
                "eval_samples": 30,
                "max_samples": 10000,
            },
            "medium": {
                "num_steps": 2000,
                "batch_size": 8,
                "gradient_accumulation": 4,
                "warmup_steps": 100,
                "log_interval": 20,
                "save_interval": 200,
                "eval_interval": 100,
                "eval_samples": 50,
                "max_samples": 50000,
            },
            "large": {
                "num_steps": 5000,
                "batch_size": 4,
                "gradient_accumulation": 8,
                "warmup_steps": 200,
                "log_interval": 50,
                "save_interval": 500,
                "eval_interval": 200,
                "eval_samples": 100,
                "max_samples": None,
            },
        }

        if self.scale in scale_configs:
            for key, value in scale_configs[self.scale].items():
                setattr(self, key, value)


@dataclass
class RLConfig(BaseConfig):
    """
    RL (GRPO/JustRL) 训练配置

    用于 Reasoning Model 的 RL 训练。
    """

    # 实验设置
    experiment_name: str = "justrl_reasoning"
    scale: str = "medium"

    # Reasoning 模式设置
    reasoning_mode: bool = True
    thinking_budget: str = "medium"  # thinking token 预算: low, medium, high
    format_reward_weight: float = 0.1  # 格式奖励权重
    redundancy_weight: float = 0.3  # 冗余度惩罚权重
    redundancy_threshold: float = 0.3  # 冗余度阈值

    # 训练设置
    num_steps: int = 200
    batch_size: int = 16
    rollout_n: int = 8  # 每个问题采样的响应数

    # JustRL 核心参数
    learning_rate: float = 1e-6
    clip_ratio_low: float = 0.8
    clip_ratio_high: float = 1.28
    kl_coef: float = 0.0  # JustRL: 无 KL 惩罚
    temperature: float = 1.0

    # 早停设置
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.05

    # 生成设置
    max_prompt_length: int = 1024
    max_response_length: int = 15360  # JustRL 使用 15k

    # 评估设置
    eval_samples: int = 50

    # 数据集设置
    train_dataset: str = "dapo-math-17k"
    eval_datasets: list[str] = field(default_factory=lambda: ["math", "aime-2024"])
    math_subjects: list[str] | None = None

    # 输出设置
    output_dir: str = "outputs/justrl_reasoning"

    def __post_init__(self):
        """根据 scale 调整参数"""
        scale_configs = {
            "quick": {
                "num_steps": 2,
                "batch_size": 2,
                "eval_interval": 2,
                "save_interval": 2,
                "eval_samples": 10,
            },
            "medium": {
                "num_steps": 800,
                "batch_size": 32,
                "eval_interval": 10,
                "save_interval": 10,
                "eval_samples": 200,
            },
            "full": {
                "num_steps": 2000,
                "batch_size": 32,
                "eval_interval": 100,
                "save_interval": 200,
                "eval_samples": 200,
            },
        }

        if self.scale in scale_configs:
            for key, value in scale_configs[self.scale].items():
                setattr(self, key, value)

        # Reasoning 模式需要更长的输出
        if self.reasoning_mode:
            thinking_budgets = {
                "low": 8192,
                "medium": 15360,
                "high": 20480,
            }
            self.max_response_length = thinking_budgets.get(
                self.thinking_budget, 15360
            )

        # 设置默认评估数据集
        if not self.eval_datasets:
            self.eval_datasets = ["math", "aime-2024"]


# 别名，保持向后兼容
ReasoningConfig = RLConfig
JustRLConfig = RLConfig

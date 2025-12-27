"""
数据集加载和处理工具

支持的数据集:
- GSM8K: 小学数学应用题
- MATH: 竞赛数学问题
- Countdown: 数字组合游戏（用于入门实验）

Author: Guanghan Ning
Date: 2025-12-23
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    """数据样本"""
    question: str
    answer: str
    solution: Optional[str] = None
    metadata: Optional[Dict] = None


def load_gsm8k(
    split: str = "train",
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Sample]:
    """
    加载GSM8K数据集

    GSM8K包含约8500个小学数学应用题，每个问题有详细的解题步骤。

    Args:
        split: "train" 或 "test"
        num_samples: 限制加载的样本数量
        cache_dir: 缓存目录

    Returns:
        Sample列表

    Example:
        >>> samples = load_gsm8k("train", num_samples=100)
        >>> print(samples[0].question)
        "Janet's ducks lay 16 eggs per day..."
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装datasets库: pip install datasets")

    dataset = load_dataset(
        "gsm8k",
        "main",
        split=split,
        cache_dir=cache_dir
    )

    samples = []
    for item in dataset:
        # GSM8K的答案格式: 解题步骤 + "#### " + 最终答案
        answer_text = item["answer"]
        parts = answer_text.split("####")

        final_answer = parts[-1].strip() if len(parts) > 1 else answer_text
        solution = parts[0].strip() if len(parts) > 1 else None

        # 清理答案（移除逗号等）
        final_answer = final_answer.replace(",", "")

        samples.append(Sample(
            question=item["question"],
            answer=final_answer,
            solution=solution,
            metadata={"source": "gsm8k", "split": split}
        ))

        if num_samples and len(samples) >= num_samples:
            break

    return samples


def load_math(
    split: str = "train",
    num_samples: Optional[int] = None,
    difficulty: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> List[Sample]:
    """
    加载MATH数据集

    MATH包含12500个竞赛数学问题，难度从1到5，涵盖7个数学领域。

    Args:
        split: "train" 或 "test"
        num_samples: 限制加载的样本数量
        difficulty: 可选，筛选特定难度 ("Level 1" 到 "Level 5")
        cache_dir: 缓存目录

    Returns:
        Sample列表
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装datasets库: pip install datasets")

    dataset = load_dataset(
        "hendrycks/competition_math",
        split=split,
        cache_dir=cache_dir
    )

    samples = []
    for item in dataset:
        # 筛选难度
        if difficulty and item.get("level") != difficulty:
            continue

        # MATH的答案在\boxed{}中
        solution = item["solution"]
        answer = extract_boxed_answer(solution)

        samples.append(Sample(
            question=item["problem"],
            answer=answer or "",
            solution=solution,
            metadata={
                "source": "math",
                "level": item.get("level"),
                "type": item.get("type"),
            }
        ))

        if num_samples and len(samples) >= num_samples:
            break

    return samples


def extract_boxed_answer(solution: str) -> Optional[str]:
    """
    从MATH数据集的solution中提取\\boxed{}中的答案

    Example:
        >>> extract_boxed_answer("So the answer is \\boxed{42}.")
        "42"
    """
    # 匹配 \boxed{...}
    pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern, solution)
    if match:
        return match.group(1)

    # 尝试匹配嵌套的boxed
    pattern_nested = r"\\boxed\{(.+?)\}(?=[^{]*$)"
    match = re.search(pattern_nested, solution)
    if match:
        return match.group(1)

    return None


def generate_countdown_data(
    num_samples: int = 1000,
    num_numbers: int = 4,
    target_range: Tuple[int, int] = (10, 100),
    number_range: Tuple[int, int] = (1, 10),
    seed: int = 42
) -> List[Sample]:
    """
    生成Countdown游戏数据

    Countdown是一个数字组合游戏：给定几个数字和一个目标，
    用四则运算组合这些数字得到目标值。

    这是Mini-R1教程中使用的任务，适合RLVR入门实验。

    Args:
        num_samples: 生成样本数量
        num_numbers: 每个问题中的数字个数
        target_range: 目标值范围
        number_range: 数字范围
        seed: 随机种子

    Returns:
        Sample列表

    Example:
        >>> samples = generate_countdown_data(num_samples=10)
        >>> print(samples[0].question)
        "Using numbers [2, 5, 3, 7], reach the target 24 using +, -, *, /"
    """
    import random
    random.seed(seed)

    samples = []

    for _ in range(num_samples):
        # 随机生成数字和目标
        numbers = [random.randint(*number_range) for _ in range(num_numbers)]
        target = random.randint(*target_range)

        question = (
            f"Using the numbers {numbers}, reach the target {target} "
            f"using addition (+), subtraction (-), multiplication (*), "
            f"and division (/). Each number can only be used once. "
            f"Show your calculation steps."
        )

        samples.append(Sample(
            question=question,
            answer=str(target),
            metadata={
                "source": "countdown",
                "numbers": numbers,
                "target": target,
            }
        ))

    return samples


def format_prompt_for_math(sample: Sample, style: str = "cot") -> str:
    """
    将数学问题格式化为prompt

    Args:
        sample: 数据样本
        style: prompt风格
            - "cot": Chain-of-Thought，要求展示推理步骤
            - "direct": 直接回答
            - "r1": DeepSeek-R1风格，包含<think>标签

    Returns:
        格式化后的prompt
    """
    if style == "cot":
        return f"""Solve the following math problem step by step.
Show your reasoning clearly, then provide your final answer after "The answer is: ".

Problem: {sample.question}

Solution:"""

    elif style == "direct":
        return f"""Problem: {sample.question}

What is the answer?"""

    elif style == "r1":
        return f"""<|begin_of_text|>Problem: {sample.question}

<think>
Let me solve this step by step.
"""

    else:
        raise ValueError(f"Unknown prompt style: {style}")


def format_prompt_for_countdown(sample: Sample) -> str:
    """
    将Countdown游戏格式化为prompt

    这个格式参考了Mini-R1教程
    """
    meta = sample.metadata or {}
    numbers = meta.get("numbers", [])
    target = meta.get("target", sample.answer)

    return f"""You are playing the Countdown numbers game.
Given the numbers {numbers}, find a way to reach the target {target}.
You can use each number at most once.
You can use addition (+), subtraction (-), multiplication (*), and division (/).
Show your calculation step by step, then state the final answer.

Think carefully and show your work:"""


class DataLoader:
    """
    统一的数据加载器

    Example:
        >>> loader = DataLoader("gsm8k")
        >>> train_data = loader.load("train", num_samples=1000)
        >>> eval_data = loader.load("test", num_samples=200)
    """

    SUPPORTED_DATASETS = ["gsm8k", "math", "countdown"]

    def __init__(self, dataset_name: str, cache_dir: Optional[str] = None):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

    def load(
        self,
        split: str = "train",
        num_samples: Optional[int] = None,
        **kwargs
    ) -> List[Sample]:
        """加载数据集"""
        if self.dataset_name == "gsm8k":
            return load_gsm8k(split, num_samples, self.cache_dir)
        elif self.dataset_name == "math":
            return load_math(split, num_samples, cache_dir=self.cache_dir, **kwargs)
        elif self.dataset_name == "countdown":
            return generate_countdown_data(num_samples or 1000, **kwargs)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def format_prompt(self, sample: Sample, **kwargs) -> str:
        """格式化prompt"""
        if self.dataset_name in ["gsm8k", "math"]:
            return format_prompt_for_math(sample, **kwargs)
        elif self.dataset_name == "countdown":
            return format_prompt_for_countdown(sample)
        else:
            return sample.question


if __name__ == "__main__":
    # 测试数据加载
    print("测试GSM8K加载...")
    try:
        samples = load_gsm8k("train", num_samples=5)
        print(f"  加载了{len(samples)}个样本")
        print(f"  样例问题: {samples[0].question[:100]}...")
        print(f"  样例答案: {samples[0].answer}")
    except Exception as e:
        print(f"  GSM8K加载失败: {e}")

    print("\n测试Countdown生成...")
    samples = generate_countdown_data(num_samples=3)
    for i, s in enumerate(samples):
        print(f"  样本{i+1}: {s.question[:80]}...")
        print(f"    目标: {s.answer}")

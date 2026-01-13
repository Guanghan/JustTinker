#!/usr/bin/env python3
"""
JustRL数学推理训练 - Reasoning Model版本

使用MATH数据集训练reasoning model，支持thinking mode。

与justrl_math.py的区别：
- 使用MATH数据集（竞赛数学，更难）
- 支持thinking mode（<think>...</think>格式）
- 更适合训练reasoning model

使用方法:
    # 设置API Key
    export TINKER_API_KEY=your_api_key

    # 快速验证（非reasoning模式）
    python scripts/tinker/justrl_math_reasoning.py --scale quick

    # 启用reasoning mode
    python scripts/tinker/justrl_math_reasoning.py --scale quick --reasoning

    # 中等规模训练
    python scripts/tinker/justrl_math_reasoning.py --scale medium --reasoning

Author: Guanghan Ning
Date: 2025-01-10
Reference: https://arxiv.org/abs/2512.16649
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Tinker imports
try:
    import tinker
    from tinker import SamplingParams
except ImportError:
    print("Warning: tinker package not found. Run: pip install tinker")
    tinker = None
    SamplingParams = None

# Tinker cookbook imports (仅使用 tokenizer，不使用 renderer)
try:
    from tinker_cookbook import tokenizer_utils as tinker_tokenizer_utils
    HAS_TINKER_COOKBOOK = True
except ImportError:
    print("Warning: tinker_cookbook not found. Run: pip install tinker-cookbook")
    tinker_tokenizer_utils = None
    HAS_TINKER_COOKBOOK = False

# 添加项目根目录到path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 从 src 模块导入公共组件
# 注意：这些导入是可选的，脚本中保留了内联实现以确保兼容性
# 未来可以逐步迁移到使用 src 模块的实现
# ============================================================
try:
    from src.configs import RLConfig as _RLConfig
    from src.data import (
        extract_boxed_answer as _extract_boxed_answer,
    )
    from src.data import (
        load_aime_dataset as _load_aime_dataset,
    )
    from src.data import (
        load_dapo_math_dataset as _load_dapo_math_dataset,
    )
    from src.data import (
        load_math_dataset as _load_math_dataset,
    )
    from src.evaluation import MathVerifier as _MathVerifier
    from src.prompts import (
        format_prompt_manual_template as _format_prompt_manual_template,
    )
    from src.prompts import (
        format_prompt_with_tokenizer as _format_prompt_with_tokenizer,
    )
    from src.prompts import (
        is_base_model as _is_base_model,
    )
    HAS_SRC_MODULES = True
except ImportError:
    HAS_SRC_MODULES = False


# ============================================================
# 配置
# ============================================================

@dataclass
class ReasoningConfig:
    """Reasoning Model训练配置"""

    # 实验设置
    experiment_name: str = "justrl_math_reasoning"
    scale: str = "quick"  # quick, medium, full
    seed: int = 42

    # 模型设置
    # Qwen模型无需HuggingFace授权，Llama需要授权
    # Qwen3-4B: $0.22/M | Llama-3.2-1B: $0.09/M (需授权)
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 128  # 必须与 SFT 训练时一致！coldstart_sft.py 使用 128

    # Reasoning模式设置
    reasoning_mode: bool = False  # 是否启用thinking mode
    thinking_budget: str = "medium"  # thinking token预算: low, medium, high
    format_reward_weight: float = 0.1  # 格式奖励权重：没有thinking时的惩罚
    redundancy_weight: float = 0.3  # 冗余度惩罚权重：高重复内容时的惩罚
    redundancy_threshold: float = 0.3  # 冗余度阈值：超过此值才惩罚

    # 训练设置（根据scale调整）
    num_steps: int = 200
    batch_size: int = 16  # MATH更难，减小batch size
    rollout_n: int = 8

    # JustRL核心参数
    learning_rate: float = 1e-6
    clip_ratio_low: float = 0.8
    clip_ratio_high: float = 1.28
    kl_coef: float = 0.0  # TODO: Tinker PPO 不支持 kl_coef，需要手动实现 KL 惩罚
    temperature: float = 1.0

    # 早停设置
    early_stopping: bool = True
    early_stopping_patience: int = 3  # 连续 N 次 eval 下降后停止
    early_stopping_threshold: float = 0.05  # 下降超过此比例触发计数

    # 生成设置
    max_prompt_length: int = 1024  # MATH问题更长
    max_response_length: int = 15360  # JustRL 使用 15k，对齐论文设置

    # 评估设置
    eval_interval: int = 50
    eval_samples: int = 50  # MATH评估更慢，减少样本数
    save_interval: int = 100

    # 数据集设置
    train_dataset: str = "dapo-math-17k"  # 训练数据集: "math" 或 "dapo-math-17k"
    eval_datasets: list[str] = None  # 评估数据集列表，默认 ["math", "aime-2024"]
    math_subjects: list[str] | None = None  # None表示所有科目 (仅 math 数据集)

    # 输出设置
    output_dir: str = "outputs/justrl_reasoning"

    def __post_init__(self):
        """根据scale调整参数"""
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
                "eval_interval": 10,  # 更频繁评估，便于早停
                "save_interval": 10,
                "eval_samples": 200,
            },
            "full": {
                "num_steps": 2000,
                "batch_size": 32,
                "eval_interval": 100,
                "save_interval": 200,
                "eval_samples": 200,   # 总共 5000 测试题目
            },
        }

        if self.scale in scale_configs:
            for key, value in scale_configs[self.scale].items():
                setattr(self, key, value)

        # Reasoning模式需要更长的输出 (对齐 JustRL: 15k)
        if self.reasoning_mode:
            thinking_budgets = {
                "low": 8192,
                "medium": 15360,   # JustRL 默认 15k
                "high": 20480,     # 更长的推理
            }
            self.max_response_length = thinking_budgets.get(
                self.thinking_budget, 15360
            )

        # 设置默认评估数据集
        if self.eval_datasets is None:
            self.eval_datasets = ["math", "aime-2024"]


# ============================================================
# 数据加载
# ============================================================

def load_math_dataset(
    split: str = "train",
    max_samples: int | None = None,
    subjects: list[str] | None = None,
    stratified: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    加载MATH数据集 (EleutherAI/hendrycks_math)

    MATH数据集包含7个科目（测试集大小）：
    - algebra (代数) - 1187
    - counting_and_probability (组合概率) - 474
    - geometry (几何) - 479
    - intermediate_algebra (中级代数) - 903
    - number_theory (数论) - 540
    - prealgebra (预备代数) - 871
    - precalculus (预备微积分) - 546
    总计: 5000

    Args:
        split: 数据集分割 ("train" 或 "test")
        max_samples: 最大样本数，None 表示全部
        subjects: 指定科目列表，None 表示全部科目
        stratified: 是否按科目比例分层采样（保持原始分布）
        seed: 随机种子
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 请安装datasets库: pip install datasets")
        sys.exit(1)

    print(f"加载MATH {split}集...")

    # 所有可用的科目
    all_subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    # 使用指定的科目或全部科目
    target_subjects = subjects if subjects else all_subjects

    # 加载各科目数据
    subject_data = {}
    total_count = 0
    for subject in target_subjects:
        try:
            ds = load_dataset(
                "EleutherAI/hendrycks_math",
                subject,
                split=split,
            )
            subject_data[subject] = list(ds)
            total_count += len(ds)
            print(f"  加载 {subject}: {len(ds)} 个样本")
        except Exception as e:
            print(f"  加载 {subject} 失败: {e}")

    if not subject_data:
        print("所有科目加载失败，使用模拟数据...")
        return _get_mock_math_data(max_samples or 100)

    print(f"  总计: {total_count} 个样本")

    # 设置随机种子
    random.seed(seed)

    # 分层采样或简单采样
    samples = []

    if stratified and max_samples and max_samples < total_count:
        # 分层采样：按各科目比例采样
        print(f"  分层采样 {max_samples} 个样本（保持科目比例）...")

        for subject, data in subject_data.items():
            # 计算该科目应采样的数量（按比例）
            subject_ratio = len(data) / total_count
            subject_sample_count = max(1, round(max_samples * subject_ratio))

            # 打乱该科目数据
            random.shuffle(data)

            # 采样
            sampled = data[:subject_sample_count]
            for item in sampled:
                samples.append({
                    "problem": item.get("problem", ""),
                    "solution": item.get("solution", ""),
                    "answer": _extract_boxed_answer(item.get("solution", "")),
                    "subject": subject,
                    "level": item.get("level", "unknown"),
                })
    else:
        # 简单采样：合并后打乱
        all_items = []
        for subject, data in subject_data.items():
            for item in data:
                all_items.append({
                    "problem": item.get("problem", ""),
                    "solution": item.get("solution", ""),
                    "answer": _extract_boxed_answer(item.get("solution", "")),
                    "subject": subject,
                    "level": item.get("level", "unknown"),
                })

        random.shuffle(all_items)

        if max_samples:
            samples = all_items[:max_samples]
        else:
            samples = all_items

    # 过滤无效样本（空 answer 或无效 answer）
    original_count = len(samples)
    samples = [s for s in samples if s.get("answer") and s["answer"].strip()]
    filtered_count = original_count - len(samples)
    if filtered_count > 0:
        print(f"  [WARNING] 过滤了 {filtered_count} 个无效样本（空 answer）")

    # 最终打乱（分层采样后各科目是连续的，需要混合）
    random.shuffle(samples)

    print(f"  使用 {len(samples)} 个样本")

    # 打印科目分布
    subject_counts = defaultdict(int)
    for s in samples:
        subject_counts[s["subject"]] += 1

    # 计算并显示比例
    print("  科目分布:")
    for subject in target_subjects:
        count = subject_counts.get(subject, 0)
        original_count = len(subject_data.get(subject, []))
        original_ratio = original_count / total_count * 100 if total_count > 0 else 0
        sample_ratio = count / len(samples) * 100 if samples else 0
        print(f"    {subject}: {count} ({sample_ratio:.1f}%) [原始: {original_ratio:.1f}%]")

    return samples


def _extract_boxed_answer(solution: str) -> str:
    """
    从solution中提取\\boxed{}答案

    支持两种格式:
    1. \\boxed{答案} - 标准格式，支持任意层级嵌套
    2. \\boxed 答案 - 无括号格式（如 \\boxed 2）

    例如:
    - \\boxed{\\dfrac{\\sqrt{6}}{6}} -> \\dfrac{\\sqrt{6}}{6}
    - \\boxed 2 -> 2
    """
    import re

    # 方法1: 找 \boxed{...} 格式（取最后一个）
    prefix = "\\boxed{"
    idx = solution.rfind(prefix)

    if idx != -1:
        # 从 \boxed{ 之后开始，使用括号匹配找到对应的 }
        start = idx + len(prefix)
        depth = 1
        i = start

        while i < len(solution) and depth > 0:
            if solution[i] == '{':
                depth += 1
            elif solution[i] == '}':
                depth -= 1
            i += 1

        if depth == 0:
            return solution[start:i-1]  # 不包含最后的 }

    # 方法2: 找 \boxed X 格式（无括号，如 \boxed 2）
    # 匹配 \boxed 后跟空格和内容（到行尾、句号或$符号为止）
    match = re.search(r"\\boxed\s+([^\s\$\.\,\)]+)", solution)
    if match:
        return match.group(1).rstrip(".")

    return ""


def load_dapo_math_dataset(
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    加载 DAPO-Math-17k 数据集 (BytedTsinghua-SIA/DAPO-Math-17k)

    DAPO-Math-17k 是 DAPO 论文使用的训练数据集，包含约 17k 高质量数学题目。
    数据来源于多个数学数据集的筛选和去重。

    数据格式:
    {
        "prompt": [{"content": "问题内容...", "role": "user"}],
        "reward_model": {"ground_truth": "答案", "style": "rule-lighteval/MATH_v2"},
        ...
    }

    Args:
        max_samples: 最大样本数，None 表示全部 (~17k)
        seed: 随机种子
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 请安装datasets库: pip install datasets")
        sys.exit(1)

    print("加载 DAPO-Math-17k 数据集...")

    try:
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        print(f"  原始数据量: {len(ds)} 条")
    except Exception as e:
        print(f"  加载失败: {e}")
        print("  尝试使用处理过的版本...")
        try:
            ds = load_dataset("open-r1/DAPO-Math-17k-Processed", split="train")
            print(f"  使用 open-r1/DAPO-Math-17k-Processed: {len(ds)} 条")
        except Exception as e2:
            print(f"  备用数据集也加载失败: {e2}")
            return _get_mock_math_data(max_samples or 100)

    # 转换数据格式
    samples = []
    for item in ds:
        try:
            # 提取问题内容
            prompt_list = item.get("prompt", [])
            if prompt_list and len(prompt_list) > 0:
                problem = prompt_list[0].get("content", "")
            else:
                continue

            # 提取答案
            reward_model = item.get("reward_model", {})
            answer = reward_model.get("ground_truth", "")

            if problem and answer:
                samples.append({
                    "problem": problem,
                    "answer": str(answer).strip(),
                    "source": "dapo-math-17k",
                    "data_source": item.get("data_source", "unknown"),
                })
        except Exception:
            continue

    print(f"  有效样本: {len(samples)} 条")

    # 去重（基于问题内容）
    seen_problems = set()
    unique_samples = []
    for s in samples:
        problem_hash = hash(s["problem"][:200])  # 用前200字符作为hash
        if problem_hash not in seen_problems:
            seen_problems.add(problem_hash)
            unique_samples.append(s)

    print(f"  去重后: {len(unique_samples)} 条")
    samples = unique_samples

    # 随机打乱并采样
    random.seed(seed)
    random.shuffle(samples)

    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]
        print(f"  采样: {len(samples)} 条")

    return samples


def load_aime_dataset(
    year: str = "2024",
    seed: int = 42,
) -> list[dict]:
    """
    加载 AIME 数据集 (HuggingFaceH4/aime_2024)

    AIME (American Invitational Mathematics Examination) 是美国数学邀请赛，
    难度高于 AMC，是选拔 IMO 美国队的第二轮考试。

    2024 年 AIME 共 30 道题目（AIME I + AIME II 各 15 题）。
    答案均为 0-999 的整数。

    数据格式:
    {
        "id": 60,
        "problem": "问题内容...",
        "solution": "解答过程...",
        "answer": "204",
        "url": "https://...",
        "year": "2024"
    }

    Args:
        year: 年份，目前支持 "2024"
        seed: 随机种子
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 请安装datasets库: pip install datasets")
        sys.exit(1)

    print(f"加载 AIME {year} 数据集...")

    dataset_name = "HuggingFaceH4/aime_2024" if year == "2024" else f"HuggingFaceH4/aime_{year}"

    try:
        ds = load_dataset(dataset_name, split="train")
        print(f"  数据量: {len(ds)} 条")
    except Exception as e:
        print(f"  加载失败: {e}")
        # 尝试备用数据集
        try:
            ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
            print(f"  使用备用数据集 Maxwell-Jia/AIME_2024: {len(ds)} 条")
        except Exception as e2:
            print(f"  备用数据集也加载失败: {e2}")
            return []

    # 转换数据格式
    samples = []
    for item in ds:
        problem = item.get("problem", "")
        answer = item.get("answer", "")

        if problem and answer:
            samples.append({
                "problem": problem,
                "answer": str(answer).strip(),
                "source": f"aime-{year}",
                "solution": item.get("solution", ""),
                "url": item.get("url", ""),
            })

    print(f"  有效样本: {len(samples)} 条")

    return samples


def _get_mock_math_data(n: int) -> list[dict]:
    """生成模拟的MATH数据用于测试"""
    mock_problems = [
        {
            "problem": "Solve for x: 2x + 5 = 13",
            "answer": "4",
            "subject": "algebra",
            "level": "Level 1",
        },
        {
            "problem": "What is the sum of the first 10 positive integers?",
            "answer": "55",
            "subject": "algebra",
            "level": "Level 1",
        },
        {
            "problem": "If f(x) = x^2 + 3x + 2, what is f(2)?",
            "answer": "12",
            "subject": "algebra",
            "level": "Level 2",
        },
        {
            "problem": "How many ways can you arrange the letters in MATH?",
            "answer": "24",
            "subject": "counting_and_probability",
            "level": "Level 2",
        },
        {
            "problem": "What is the area of a circle with radius 3? Express in terms of pi.",
            "answer": "9\\pi",
            "subject": "geometry",
            "level": "Level 1",
        },
    ]

    # 循环扩展到n个样本
    samples = []
    for i in range(n):
        sample = mock_problems[i % len(mock_problems)].copy()
        sample["solution"] = f"Solution for problem {i+1}"
        samples.append(sample)

    return samples


# ============================================================
# Prompt格式化
# ============================================================

def is_base_model(model_name: str) -> bool:
    """判断是否是base model（非instruct/chat）"""
    model_lower = model_name.lower()
    # 包含这些关键词的是instruct模型
    instruct_keywords = ["instruct", "chat", "it", "rlhf"]
    for kw in instruct_keywords:
        if kw in model_lower:
            return False
    # 显式标注为Base的
    if "base" in model_lower:
        return True
    # Llama-3.2-1B, Llama-3.2-3B 等没有后缀的是base model
    return bool("llama" in model_lower and not any(kw in model_lower for kw in instruct_keywords))


def format_prompt_for_base_model(problem: str, reasoning_mode: bool = False) -> str:
    """
    为Base Model设计的prompt格式

    Base models需要few-shot示例来学习输出格式，
    而不是依赖instruction following能力。
    """
    if reasoning_mode:
        # Few-shot format for reasoning
        return f"""Problem: What is 2 + 3?
Solution: Let me think step by step.
2 + 3 = 5
The answer is \\boxed{{5}}.

Problem: If x + 5 = 12, what is x?
Solution: Let me think step by step.
x + 5 = 12
x = 12 - 5
x = 7
The answer is \\boxed{{7}}.

Problem: {problem}
Solution: Let me think step by step.
"""
    else:
        # Simple completion format
        return f"""Problem: What is 2 + 3?
Solution: 2 + 3 = 5
The answer is \\boxed{{5}}.

Problem: If x + 5 = 12, what is x?
Solution: x = 12 - 5 = 7
The answer is \\boxed{{7}}.

Problem: {problem}
Solution:"""


def format_prompt_with_tokenizer(
    problem: str,
    tokenizer,
    reasoning_mode: bool = False,
    model_name: str = "",
) -> str:
    """
    使用tokenizer的chat template格式化prompt

    Args:
        problem: 数学问题
        tokenizer: 模型的tokenizer
        reasoning_mode: 是否启用thinking mode
        model_name: 模型名称
    """
    # Base model使用不同的prompt格式（few-shot completion）
    if is_base_model(model_name):
        return format_prompt_for_base_model(problem, reasoning_mode)

    if tokenizer is None:
        # fallback到简单格式
        return format_prompt_simple(problem, reasoning_mode, model_name)

    # 构建消息（Instruct models）
    is_qwen3 = "Qwen3" in model_name or "qwen3" in model_name.lower()

    if reasoning_mode:
        if is_qwen3:
            # Qwen3 thinking mode:
            # - 不要在system prompt中提及<think>格式，让模型自然生成
            # - 使用 /think 后缀触发thinking mode
            # - Qwen3会自动生成 <think>...</think> 格式
            system_msg = "You are a helpful mathematical assistant. Solve problems step by step and put your final answer in \\boxed{}."
            user_msg = f"""Solve the following math problem.

Problem: {problem} /think"""
        else:
            # 非Qwen3模型：明确要求使用 <think> 格式
            system_msg = """You are a mathematical reasoning assistant. You MUST think step-by-step before answering.

IMPORTANT: Always wrap your thinking process in <think>...</think> tags, then provide your final answer.

Format:
<think>
[Your detailed reasoning here]
</think>

[Final answer in \\boxed{}]"""
            user_msg = f"""Solve the following math problem.

Problem: {problem}"""
    else:
        system_msg = "You are a helpful mathematical assistant."
        user_msg = f"""Solve the following math problem. Put your final answer in \\boxed{{}}.

Problem: {problem}"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        # Qwen3 thinking mode需要特殊处理
        if is_qwen3 and reasoning_mode:
            # 尝试使用enable_thinking参数
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,  # Qwen3 thinking mode
                )
            except TypeError:
                # 如果tokenizer不支持enable_thinking参数
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # 对于非Qwen3模型，在reasoning模式下手动添加<think>前缀
        if reasoning_mode and not is_qwen3:
            text += "<think>\n"

        return text
    except Exception:
        return format_prompt_simple(problem, reasoning_mode, model_name)


def format_prompt_simple(
    problem: str,
    reasoning_mode: bool = False,
    model_name: str = "",
) -> str:
    """
    简单的prompt格式（不使用chat template）
    """
    if reasoning_mode:
        return f"""<|im_start|>system
You are a mathematical reasoning assistant. Think deeply about problems before answering.
Use <think>...</think> tags for your reasoning process, then provide your final answer in \\boxed{{}}.
<|im_end|>
<|im_start|>user
Solve the following math problem.

Problem: {problem}
<|im_end|>
<|im_start|>assistant
<think>
"""
    else:
        return f"""Solve the following math problem step by step. Show your work clearly.
Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""


# 保持向后兼容
def format_prompt(
    problem: str,
    reasoning_mode: bool = False,
    model_name: str = "",
) -> str:
    """简单格式（向后兼容）"""
    return format_prompt_simple(problem, reasoning_mode, model_name)


# ============================================================
# 验证器
# ============================================================

class MathReasoningVerifier:
    """
    MATH数据集验证器

    支持多种答案格式：
    - \\boxed{answer}
    - The answer is: X
    - 数值答案
    - 符号答案（如 9\\pi）

    支持格式奖励：
    - 如果使用了 <think>...</think> 格式，给予额外奖励
    - 如果没有使用格式，轻微惩罚

    支持冗余度惩罚：
    - 检测响应中的重复/冗余内容
    - 对 reward hacking 行为（长重复内容）进行惩罚
    """

    # Thinking token IDs (Qwen3)
    THINK_END_TOKEN_ID = 151668  # </think>

    def __init__(
        self,
        format_reward_weight: float = 0.1,
        redundancy_weight: float = 0.3,
        redundancy_threshold: float = 0.3,
    ):
        """
        Args:
            format_reward_weight: 格式奖励/惩罚的权重
                - 正确答案 + 有thinking: reward = 1.0
                - 正确答案 + 无thinking: reward = 1.0 - format_reward_weight
                - 错误答案: reward = 0.0
            redundancy_weight: 冗余度惩罚的最大权重 (默认 0.3)
            redundancy_threshold: 冗余度阈值，超过此值才惩罚 (默认 0.3)
        """
        import re
        import zlib
        self.re = re
        self.zlib = zlib
        self.format_reward_weight = format_reward_weight
        self.redundancy_weight = redundancy_weight
        self.redundancy_threshold = redundancy_threshold

    def has_thinking_format(self, tokens: list[int] = None, text: str = None) -> bool:
        """
        检查是否使用了 thinking 格式

        Args:
            tokens: 生成的 token 列表（推荐，更准确）
            text: 生成的文本（备用）

        Returns:
            是否包含 thinking 格式
        """
        # 优先检查 token
        if tokens is not None:
            return self.THINK_END_TOKEN_ID in tokens

        # 备用：检查文本
        if text is not None:
            return "</think>" in text

        return False

    def _compute_compression_redundancy(self, text: str) -> float:
        """
        使用压缩率计算文本冗余度

        原理：高重复内容压缩后体积小 → 低压缩比 → 高冗余分数

        Args:
            text: 输入文本

        Returns:
            冗余度分数 (0-1)，越高表示越冗余
        """
        if len(text) < 100:
            return 0.0

        text_bytes = text.encode('utf-8')
        compressed = self.zlib.compress(text_bytes, level=9)

        # 压缩比 = compressed_size / original_size
        # 典型范围: 0.1 (高重复) - 0.7 (低重复)
        compression_ratio = len(compressed) / len(text_bytes)

        # 归一化到 0-1，反转使高值=高冗余
        # 压缩比 0.1 → 冗余度 1.0
        # 压缩比 0.7 → 冗余度 0.0
        redundancy = max(0, min(1, (0.7 - compression_ratio) / 0.6))

        return redundancy

    def _compute_ngram_redundancy(self, text: str, n: int = 5) -> float:
        """
        计算 n-gram 重复率

        Args:
            text: 输入文本
            n: n-gram 大小

        Returns:
            重复率 (0-1)，越高表示越多重复
        """
        words = text.split()
        if len(words) < n * 2:
            return 0.0

        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0

        unique_ratio = len(set(ngrams)) / len(ngrams)
        return 1.0 - unique_ratio

    def _compute_chunk_similarity(self, text: str, chunk_size: int = 500, shingle_k: int = 5) -> float:
        """
        计算相邻 chunk 之间的平均相似度（轻量级版本）

        原理：将文本切分成 chunks，用 k-shingles 计算相邻 chunk 的 Jaccard 相似度
        高相似度意味着不同位置的内容高度重复

        Args:
            text: 输入文本
            chunk_size: 每个 chunk 的字符数
            shingle_k: k-shingle 的大小

        Returns:
            平均 chunk 相似度 (0-1)，越高表示越多近似重复
        """
        if len(text) < chunk_size * 2:
            return 0.0

        # 切分成 chunks（50% overlap 以捕捉边界情况）
        step = chunk_size // 2
        chunks = [text[i:i+chunk_size] for i in range(0, len(text) - chunk_size + 1, step)]

        if len(chunks) < 2:
            return 0.0

        def get_shingles(s: str) -> set:
            """获取 k-character shingles"""
            if len(s) < shingle_k:
                return set()
            return set(s[i:i+shingle_k] for i in range(len(s) - shingle_k + 1))

        # 计算相邻 chunk 的 Jaccard 相似度
        similarities = []
        for i in range(len(chunks) - 1):
            shingles_a = get_shingles(chunks[i])
            shingles_b = get_shingles(chunks[i + 1])

            if not shingles_a or not shingles_b:
                continue

            intersection = len(shingles_a & shingles_b)
            union = len(shingles_a | shingles_b)

            if union > 0:
                similarities.append(intersection / union)

        if not similarities:
            return 0.0

        return sum(similarities) / len(similarities)

    def compute_redundancy(self, text: str) -> dict[str, float]:
        """
        综合计算文本冗余度

        使用两种方法计算惩罚：
        1. 压缩率（权重 0.6）：捕捉字符级重复
        2. N-gram 重复率（权重 0.4）：捕捉词级重复

        额外监控指标（不参与惩罚计算）：
        3. Chunk similarity：检测近似重复，用于监控潜在的"狡猾"模式

        Args:
            text: 输入文本

        Returns:
            包含各项指标的字典：
            - compression_score: 压缩率冗余分数
            - ngram_score: N-gram 重复率
            - chunk_similarity: 相邻 chunk 相似度（仅监控）
            - combined_score: 综合冗余分数（用于惩罚）
            - penalty: 实际惩罚值（考虑阈值）
        """
        compression_score = self._compute_compression_redundancy(text)
        ngram_score = self._compute_ngram_redundancy(text, n=5)
        chunk_similarity = self._compute_chunk_similarity(text)

        # 加权平均（压缩率更可靠）
        # 注意：chunk_similarity 仅作为监控指标，不参与惩罚计算
        combined_score = 0.6 * compression_score + 0.4 * ngram_score

        # 只有超过阈值才惩罚
        if combined_score < self.redundancy_threshold:
            penalty = 0.0
        else:
            # 超过阈值的部分线性惩罚
            excess = combined_score - self.redundancy_threshold
            max_excess = 1.0 - self.redundancy_threshold
            penalty = (excess / max_excess) * self.redundancy_weight

        return {
            "compression_score": compression_score,
            "ngram_score": ngram_score,
            "chunk_similarity": chunk_similarity,  # 监控指标
            "combined_score": combined_score,
            "penalty": penalty,
        }

    def _normalize_answer(self, answer: str) -> str:
        """标准化答案字符串"""
        if not answer:
            return ""

        answer = answer.strip()

        # 1. 处理 \text{...}, \textbf{...}, \mathrm{...} 等 - 保留内容，移除包装
        answer = self.re.sub(r"\\text\s*\{([^}]*)\}", r"\1", answer)
        answer = self.re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", answer)
        answer = self.re.sub(r"\\textbf\s*\{([^}]*)\}", r"\1", answer)
        answer = self.re.sub(r"\\textit\s*\{([^}]*)\}", r"\1", answer)
        answer = self.re.sub(r"\\mathbf\s*\{([^}]*)\}", r"\1", answer)

        # 2. 处理货币符号
        answer = answer.replace("\\$", "")  # LaTeX 转义的美元符号
        answer = answer.replace("$", "")     # 普通美元符号

        # 3. 处理百分号
        answer = answer.replace("\\%", "%")  # 统一为普通百分号

        # 4. 处理度数符号
        answer = answer.replace("^\\circ", "°")
        answer = answer.replace("^{\\circ}", "°")
        answer = answer.replace("\\circ", "°")
        answer = answer.replace("degrees", "°")

        # 5. 移除逗号和空格（保留必要的结构）
        answer = answer.replace(", ", ",")  # 先统一逗号后的空格
        answer = answer.replace(" ", "")

        # 6. 统一分数格式
        answer = answer.replace("\\frac", "frac")
        answer = answer.replace("\\dfrac", "frac")
        answer = answer.replace("\\tfrac", "frac")

        # 7. 统一其他 LaTeX 符号
        answer = answer.replace("\\pi", "π")
        # sqrt: \sqrt{2} -> √2, \sqrt2 -> √2
        answer = self.re.sub(r"\\sqrt\{([^}]+)\}", r"√\1", answer)
        answer = self.re.sub(r"\\sqrt(\d)", r"√\1", answer)
        answer = answer.replace("\\cdot", "*")
        answer = answer.replace("\\times", "*")
        answer = answer.replace("\\div", "/")
        answer = answer.replace("\\left", "")
        answer = answer.replace("\\right", "")
        answer = answer.replace("\\infty", "∞")
        answer = answer.replace("\\pm", "±")
        answer = answer.replace("\\mp", "∓")
        answer = answer.replace("\\leq", "≤")
        answer = answer.replace("\\le", "≤")
        answer = answer.replace("\\geq", "≥")
        answer = answer.replace("\\ge", "≥")
        answer = answer.replace("\\neq", "≠")
        answer = answer.replace("\\ne", "≠")
        answer = answer.replace("\\ldots", "...")
        answer = answer.replace("\\cdots", "...")
        answer = answer.replace("\\dots", "...")

        # 8. 统一指数格式: x^2 和 x^{2} 统一
        answer = self.re.sub(r"\^{(\d+)}", r"^\1", answer)  # ^{2} -> ^2
        answer = self.re.sub(r"\^{([a-z])}", r"^\1", answer)  # ^{n} -> ^n

        answer = answer.lower()

        # 9. 处理选择题格式: (a), (b), (c), (d), (e) -> a, b, c, d, e
        choice_match = self.re.match(r"^\(([a-e])\)$", answer)
        if choice_match:
            answer = choice_match.group(1)

        # 10. 移除多余逗号（在数字中间的逗号）
        answer = self.re.sub(r"(\d),(\d)", r"\1\2", answer)

        # 11. 移除常见单位 (JustRL 风格)
        units = [
            "centimeter", "centimeters", "cm",
            "millimeter", "millimeters", "mm",
            "meter", "meters", "m",
            "kilometer", "kilometers", "km",
            "inch", "inches", "ft", "feet", "foot", "yard", "yards", "mile", "miles",
            "kilogram", "kilograms", "kg", "gram", "grams", "g", "mg", "lb", "lbs", "oz", "ounce", "ounces",
            "liter", "liters", "ml", "gallon", "gallons",
            "second", "seconds", "sec", "minute", "minutes", "min", "hour", "hours", "hr", "day", "days",
            "dollar", "dollars", "cent", "cents", "euro", "euros",
            "square", "cubic", "sq", "cu",
        ]
        for unit in units:
            # 移除数字后面的单位 (如 "5cm" -> "5")
            answer = self.re.sub(rf"(\d)\s*{unit}s?\b", r"\1", answer, flags=self.re.IGNORECASE)

        # 12. 处理文字数字 (JustRL 风格)
        text_numbers = {
            "million": "000000",
            "billion": "000000000",
            "trillion": "000000000000",
            "thousand": "000",
            "hundred": "00",
        }
        for word, zeros in text_numbers.items():
            # "5 million" -> "5000000"
            match = self.re.search(rf"(\d+\.?\d*)\s*{word}", answer, flags=self.re.IGNORECASE)
            if match:
                num = match.group(1)
                if "." in num:
                    # 处理小数: "1.5 million" -> "1500000"
                    parts = num.split(".")
                    int_part = parts[0]
                    dec_part = parts[1] if len(parts) > 1 else ""
                    zeros_to_add = len(zeros) - len(dec_part)
                    replacement = int_part + dec_part + "0" * zeros_to_add
                else:
                    replacement = num + zeros
                answer = self.re.sub(rf"{num}\s*{word}", replacement, answer, flags=self.re.IGNORECASE)

        return answer

    def extract_answer(self, text: str) -> str | None:
        """从response中提取答案"""
        # 优先匹配 \boxed{...} - 使用递归方法处理嵌套大括号
        def find_boxed_content(s: str) -> str | None:
            """递归提取 \boxed{} 内容，正确处理嵌套大括号"""
            # 找所有 \boxed{ 的位置
            starts = []
            idx = 0
            while True:
                pos = s.find("\\boxed{", idx)
                if pos == -1:
                    break
                starts.append(pos)
                idx = pos + 1

            if not starts:
                return None

            # 取最后一个 \boxed{
            start = starts[-1]
            brace_start = start + len("\\boxed{")

            # 匹配平衡的大括号
            depth = 1
            i = brace_start
            while i < len(s) and depth > 0:
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                i += 1

            if depth == 0:
                return s[brace_start:i-1]
            return None

        boxed = find_boxed_content(text)
        if boxed is not None:
            return boxed

        # 其他模式
        patterns = [
            r"[Tt]he (?:final )?answer is:?\s*([^\n\.]+)",
            r"####\s*([^\n]+)",
            r"[Aa]nswer:?\s*([^\n\.]+)",
        ]

        for pattern in patterns:
            match = self.re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def _numeric_equal(self, a: str, b: str) -> bool:
        """尝试数值比较，处理分数、π、√ 等"""
        import math

        def try_eval(s: str) -> float | None:
            """尝试将字符串转为数值"""
            if not s:
                return None
            try:
                # 直接尝试 float
                return float(s)
            except ValueError:
                pass

            # 处理简单分数: frac{a}{b} 或 a/b
            frac_match = self.re.match(r"^frac\{(-?\d+)\}\{(\d+)\}$", s)
            if frac_match:
                num, den = int(frac_match.group(1)), int(frac_match.group(2))
                if den != 0:
                    return num / den

            frac_match2 = self.re.match(r"^(-?\d+)/(\d+)$", s)
            if frac_match2:
                num, den = int(frac_match2.group(1)), int(frac_match2.group(2))
                if den != 0:
                    return num / den

            # 处理 π (支持 2π, -3π, π, -π 等格式)
            pi_match = self.re.match(r"^(-?\d*\.?\d*)π$", s)
            if pi_match:
                coef = pi_match.group(1)
                if coef == '' or coef == '+':
                    coef = 1.0
                elif coef == '-':
                    coef = -1.0
                else:
                    coef = float(coef)
                return coef * math.pi

            # 处理 √n 格式
            sqrt_match = self.re.match(r"^√(\d+)$", s)
            if sqrt_match:
                return math.sqrt(int(sqrt_match.group(1)))

            # 处理 a√b 格式 (如 2√3)
            sqrt_match2 = self.re.match(r"^(-?\d+)√(\d+)$", s)
            if sqrt_match2:
                coef = int(sqrt_match2.group(1))
                val = int(sqrt_match2.group(2))
                return coef * math.sqrt(val)

            return None

        val_a = try_eval(a)
        val_b = try_eval(b)

        if val_a is not None and val_b is not None:
            return abs(val_a - val_b) < 1e-6

        return False

    def _sympy_equal(self, a: str, b: str) -> bool:
        """使用 SymPy 进行符号比较（作为 fallback）"""
        try:
            import warnings

            from sympy import N, Symbol, simplify, sympify
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )
        except ImportError:
            return False

        def try_parse(s: str):
            """尝试解析表达式"""
            # 清理字符串
            # √2 -> sqrt(2), √{2} -> sqrt(2)
            s = self.re.sub(r"√\{?(\d+)\}?", r"sqrt(\1)", s)
            s = s.replace("π", "pi")
            s = s.replace("∞", "oo")  # SymPy 的无穷
            s = s.replace("^", "**")  # SymPy 用 ** 表示乘方
            # frac{a}{b} -> (a)/(b)
            s = self.re.sub(r"frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
            # 处理隐式乘法: 2x -> 2*x
            s = self.re.sub(r"(\d)([a-z])", r"\1*\2", s)

            transformations = standard_transformations + (implicit_multiplication_application,)

            try:
                return parse_expr(s, transformations=transformations)
            except Exception:
                pass

            try:
                return sympify(s)
            except Exception:
                pass

            return None

        # 使用 context manager 抑制 SymPy 解析时的 SyntaxWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)

            expr_a = try_parse(a)
            expr_b = try_parse(b)

            if expr_a is None or expr_b is None:
                return False

            try:
                # 尝试化简差值
                diff = simplify(expr_a - expr_b)
                if diff == 0:
                    return True

                # 数值比较
                val_diff = abs(complex(N(diff)))
                return val_diff < 1e-6
            except Exception:
                return False

    def verify(
        self,
        response: str,
        gold: str,
        tokens: list[int] = None,
        check_format: bool = True,
        check_redundancy: bool = True,
    ) -> dict:
        """
        验证答案并计算奖励

        Args:
            response: 模型的回答文本
            gold: 标准答案
            tokens: 生成的 token 列表（用于检查 thinking 格式）
            check_format: 是否检查格式并应用格式奖励
            check_redundancy: 是否检查冗余度并应用冗余惩罚

        Returns:
            包含 is_correct, reward, extracted, has_thinking, redundancy 等字段的字典
        """
        extracted = self.extract_answer(response)

        # 检查 thinking 格式
        has_thinking = False
        if check_format and self.format_reward_weight > 0:
            has_thinking = self.has_thinking_format(tokens=tokens, text=response)

        # 计算冗余度
        redundancy_info = {"combined_score": 0.0, "penalty": 0.0}
        if check_redundancy and self.redundancy_weight > 0:
            redundancy_info = self.compute_redundancy(response)

        if extracted is None:
            return {
                "is_correct": False,
                "reward": 0.0,
                "extracted": None,
                "gold_normalized": self._normalize_answer(gold),
                "has_thinking": has_thinking,
                "format_penalty": 0.0,
                "redundancy_score": redundancy_info["combined_score"],
                "redundancy_penalty": redundancy_info["penalty"],
                "chunk_similarity": redundancy_info.get("chunk_similarity", 0.0),
            }

        # 标准化比较
        extracted_norm = self._normalize_answer(extracted)
        gold_norm = self._normalize_answer(gold)

        is_correct = False

        # 三层验证：字符串 → 数值 → SymPy
        if extracted_norm == gold_norm:
            # 1. 直接字符串比较
            is_correct = True
        elif self._numeric_equal(extracted_norm, gold_norm):
            # 2. 数值比较 (分数、π、√ 等)
            is_correct = True
        elif self._sympy_equal(extracted_norm, gold_norm):
            # 3. SymPy 符号比较 (fallback)
            is_correct = True
        else:
            is_correct = False

        # 计算奖励
        format_penalty = 0.0
        redundancy_penalty = redundancy_info["penalty"]

        if is_correct:
            if has_thinking:
                # 正确 + 有 thinking: 基础满分
                reward = 1.0
            else:
                # 正确 + 无 thinking: 格式惩罚
                format_penalty = self.format_reward_weight if check_format else 0.0
                reward = 1.0 - format_penalty

            # 应用冗余度惩罚（对正确答案也惩罚重复内容）
            reward = max(0.0, reward - redundancy_penalty)
        else:
            # 错误答案: 0 分（不需要额外惩罚）
            reward = 0.0
            format_penalty = 0.0
            redundancy_penalty = 0.0  # 错误答案不记录冗余惩罚

        return {
            "is_correct": is_correct,
            "reward": reward,
            "extracted": extracted,
            "gold_normalized": gold_norm,
            "has_thinking": has_thinking,
            "format_penalty": format_penalty,
            "redundancy_score": redundancy_info["combined_score"],
            "redundancy_penalty": redundancy_penalty,
            "chunk_similarity": redundancy_info.get("chunk_similarity", 0.0),
        }


# ============================================================
# GRPO训练器
# ============================================================

class ReasoningTrainer:
    """
    Reasoning Model 训练器

    基于 JustRL，增加对 thinking tokens 的支持
    使用 tokenizer.apply_chat_template（与 SFT 训练一致，不使用 renderer）
    """

    def __init__(
        self,
        config: ReasoningConfig,
        training_client: Any,
        verifier: MathReasoningVerifier,
    ):
        self.config = config
        self.training_client = training_client
        self.verifier = verifier

        self.global_step = 0
        self.history = defaultdict(list)

        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        """
        初始化 tokenizer（不使用 renderer！）

        重要：SFT 训练时使用 tokenizer.apply_chat_template，prompt 以 'assistant\\n' 结尾
        模型学会了自己输出 <think> 作为第一个 response token
        如果使用 renderer，会在 prompt 末尾添加 <think>，与 SFT 训练格式不一致
        """
        model_name = self.config.model_name

        # 优先使用 tinker_cookbook 的 tokenizer（但不使用 renderer）
        if HAS_TINKER_COOKBOOK:
            try:
                print(f"加载 tokenizer: {model_name}")
                self.tokenizer = tinker_tokenizer_utils.get_tokenizer(model_name)
                print("Tokenizer 加载完成（使用 apply_chat_template，与 SFT 一致）")
                return
            except Exception as e:
                print(f"Warning: Tinker Cookbook tokenizer 加载失败: {e}")
                print("回退到 HuggingFace tokenizer...")

        # 回退到 HuggingFace tokenizer
        try:
            from transformers import AutoTokenizer
            print(f"使用 HuggingFace 加载 tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            print("Tokenizer 加载完成（使用 apply_chat_template，与 SFT 一致）")
        except Exception as e:
            error_msg = str(e)
            print(f"\nError: 无法加载 tokenizer: {e}")

            if "gated repo" in error_msg or "401" in error_msg or "restricted" in error_msg:
                print("\n" + "=" * 60)
                print("这是一个需要授权的模型 (gated model)")
                print("解决方案:")
                print("  1. 访问模型页面并申请访问权限:")
                print(f"     https://huggingface.co/{model_name}")
                print("  2. 登录 HuggingFace:")
                print("     huggingface-cli login")
                print("  3. 或者使用无需授权的模型 (如 Qwen):")
                print("     --model Qwen/Qwen3-4B-Instruct-2507")
                print("=" * 60)

            sys.exit(1)

    def format_prompt(self, problem: str) -> tinker.ModelInput:
        """
        格式化prompt，返回ModelInput

        重要：使用与 SFT 训练完全相同的手动模板！
        原因：
        - SFT 训练时使用手动构建的模板字符串（见 coldstart_sft.py 第 523-528 行）
        - tokenizer.apply_chat_template 可能产生微妙差异（空白、换行等）
        - 模型学会了在特定格式后输出 <think> 作为第一个 response token
        - 任何 prompt 格式差异都会导致模型无法正确输出 <think>
        """
        # 与 SFT 训练完全相同的系统消息和用户消息
        system_msg = "You are a helpful mathematical assistant. Think step by step before answering."
        user_msg = f"Solve the following math problem.\n\nProblem: {problem}"

        # 使用与 SFT 训练完全相同的手动模板（来自 coldstart_sft.py）
        # 注意：assistant 后面只有一个换行，没有额外空格！
        prompt_text = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return tinker.ModelInput.from_ints(tokens)

    def get_prompt_tokens(self, problem: str) -> list[int]:
        """获取prompt的token列表"""
        model_input = self.format_prompt(problem)
        return model_input.to_ints()

    def compute_advantages(self, rewards: list[float]) -> list[float]:
        """计算组内归一化的advantages"""
        import numpy as np
        rewards_arr = np.array(rewards)
        mean = np.mean(rewards_arr)
        advantages = rewards_arr - mean
        return advantages.tolist()

    def get_stop_sequences(self) -> list[int]:
        """
        获取停止序列（token IDs）

        对于 Qwen3 系列模型，使用 <|im_end|> 作为停止符
        """
        # 获取 <|im_end|> 的 token ID
        im_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if im_end_token:
            return [im_end_token[0]]  # 返回第一个 token（通常就是 <|im_end|>）

        # 回退: base model 需要手动指定字符串
        if is_base_model(self.config.model_name):
            return [
                "\n\nProblem:",
                "\nProblem:",
                "\n\n\n",
            ]
        return []

    def parse_response(self, tokens: list[int]) -> dict[str, Any]:
        """
        解析模型响应

        直接 decode tokens，thinking 格式检测由 MathReasoningVerifier 处理
        """
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)

        # 简单提取 thinking 内容（如果有）
        thinking = None
        content = text
        if "<think>" in text and "</think>" in text:
            import re
            match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                content = text[match.end():].strip()

        return {
            "content": content,
            "thinking": thinking,
            "full_text": text,
            "parse_success": True,
        }

    def train_step(
        self,
        problems: list[str],
        gold_answers: list[str],
        sampling_client: Any,
    ) -> dict[str, float]:
        """执行一个训练步骤"""
        self.global_step += 1
        step_start = time.time()

        # 构建采样参数
        stop_seqs = self.get_stop_sequences()
        sampling_params = SamplingParams(
            max_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            stop=stop_seqs if stop_seqs else None,
        )

        all_samples = []
        import tinker

        # 并发发送所有采样请求
        futures = []
        prompt_data = []  # 存储 (prompt_tokens, prompt_length)
        for problem in problems:
            # 使用新的format_prompt方法获取ModelInput
            model_input = self.format_prompt(problem)
            prompt_tokens = model_input.to_ints()
            prompt_length = len(prompt_tokens)
            prompt_data.append((prompt_tokens, prompt_length))

            future = sampling_client.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=self.config.rollout_n,
            )
            futures.append(future)

        # 统一等待并处理结果
        for future, (prompt_tokens, prompt_length) in zip(futures, prompt_data, strict=False):
            result = future.result()

            samples = []
            for seq in result.sequences:
                response_tokens = list(seq.tokens) if hasattr(seq.tokens, '__iter__') else seq.tokens
                response_logprobs = list(seq.logprobs) if seq.logprobs and hasattr(seq.logprobs, '__iter__') else [0.0] * len(response_tokens)

                # 完整序列 = prompt + response（用于训练）
                full_tokens = list(prompt_tokens) + list(response_tokens)
                full_logprobs = [0.0] * prompt_length + list(response_logprobs)

                # 使用parse_response解析输出（只解析response部分）
                parsed = self.parse_response(response_tokens)

                samples.append({
                    "text": parsed["full_text"],  # 用于验证
                    "content": parsed["content"],
                    "thinking": parsed["thinking"],
                    "tokens": full_tokens,  # 完整序列：prompt + response
                    "logprobs": full_logprobs,
                    "prompt_length": prompt_length,
                })
            all_samples.append(samples)

        # 计算奖励
        all_rewards = []
        correct_count = 0
        total_count = 0

        total_redundancy_score = 0.0
        total_redundancy_penalty = 0.0
        total_chunk_similarity = 0.0

        for samples, gold in zip(all_samples, gold_answers, strict=False):
            rewards = []
            for sample in samples:
                # 传递 tokens 以检查 thinking 格式和冗余度
                result = self.verifier.verify(
                    sample["text"],
                    gold,
                    tokens=sample.get("tokens"),
                    check_format=self.config.reasoning_mode,
                    check_redundancy=self.config.reasoning_mode,
                )
                sample["reward"] = result["reward"]
                sample["has_thinking"] = result.get("has_thinking", False)
                sample["redundancy_score"] = result.get("redundancy_score", 0.0)
                sample["redundancy_penalty"] = result.get("redundancy_penalty", 0.0)
                sample["chunk_similarity"] = result.get("chunk_similarity", 0.0)
                rewards.append(result["reward"])

                total_redundancy_score += result.get("redundancy_score", 0.0)
                total_redundancy_penalty += result.get("redundancy_penalty", 0.0)
                total_chunk_similarity += result.get("chunk_similarity", 0.0)

                if result["is_correct"]:
                    correct_count += 1
                total_count += 1
            all_rewards.append(rewards)

        # 计算advantages
        for samples, rewards in zip(all_samples, all_rewards, strict=False):
            advantages = self.compute_advantages(rewards)
            for sample, adv in zip(samples, advantages, strict=False):
                sample["advantage"] = adv

        # 收集positive advantage样本
        train_samples = []
        for samples in all_samples:
            for sample in samples:
                if sample["advantage"] > 0:
                    train_samples.append(sample)

        # 执行梯度更新
        if train_samples:
            import tinker
            import torch

            data = []
            for sample in train_samples:
                tokens = sample["tokens"]
                logprobs = sample["logprobs"]
                prompt_len = sample["prompt_length"]
                advantage = float(sample["advantage"])

                seq_len = len(tokens)
                if len(logprobs) != seq_len:
                    logprobs = logprobs[:seq_len] + [0.0] * (seq_len - len(logprobs))

                input_tokens = tokens[:-1]
                ob_len = prompt_len - 1

                target_tokens = [0] * ob_len + tokens[ob_len:]
                target_tokens = target_tokens[:len(input_tokens)]

                padded_logprobs = [0.0] * ob_len + logprobs[ob_len:]
                padded_logprobs = padded_logprobs[:len(input_tokens)]

                padded_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)

                model_input = tinker.ModelInput.from_ints(input_tokens)

                datum = tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_torch(
                            torch.tensor(target_tokens, dtype=torch.long)
                        ),
                        "logprobs": tinker.TensorData.from_torch(
                            torch.tensor(padded_logprobs, dtype=torch.float32)
                        ),
                        "advantages": tinker.TensorData.from_torch(
                            torch.tensor(padded_advantages, dtype=torch.float32)
                        ),
                    }
                )
                data.append(datum)

            # 注意：Tinker PPO loss 只支持 clip_low_threshold 和 clip_high_threshold
            # kl_coef 不被支持，KL 惩罚需要在 advantage 计算时手动实现
            fwd_bwd_future = self.training_client.forward_backward(
                data=data,
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": self.config.clip_ratio_low,
                    "clip_high_threshold": self.config.clip_ratio_high,
                }
            )
            fwd_bwd_future.result()

            self.training_client.optim_step(
                tinker.AdamParams(learning_rate=self.config.learning_rate)
            )

        # 统计
        import numpy as np
        flat_rewards = [r for rewards in all_rewards for r in rewards]

        # 统计 thinking 格式使用率和长度
        thinking_count = 0
        total_response_tokens = 0
        total_thinking_tokens = 0

        # Thinking token IDs (Qwen3)
        think_start_id = 151667  # <think>
        think_end_id = 151668    # </think>

        for samples in all_samples:
            for sample in samples:
                tokens = sample.get("tokens", [])
                total_response_tokens += len(tokens)

                if sample.get("has_thinking", False):
                    thinking_count += 1
                    # 计算 thinking 部分的 token 数量
                    # 找到 <think> 和 </think> 之间的 tokens
                    try:
                        if think_start_id in tokens and think_end_id in tokens:
                            start_idx = tokens.index(think_start_id)
                            end_idx = tokens.index(think_end_id)
                            thinking_tokens = end_idx - start_idx + 1  # 包含 <think> 和 </think>
                            total_thinking_tokens += thinking_tokens
                    except (ValueError, IndexError):
                        pass

        stats = {
            "step": self.global_step,
            "mean_reward": np.mean(flat_rewards),
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "thinking_rate": thinking_count / total_count if total_count > 0 else 0,
            "num_train_samples": len(train_samples),
            "total_samples": total_count,
            "step_time": time.time() - step_start,
            "avg_response_length": total_response_tokens / total_count if total_count > 0 else 0,
            "avg_thinking_length": total_thinking_tokens / thinking_count if thinking_count > 0 else 0,
            "avg_redundancy_score": total_redundancy_score / total_count if total_count > 0 else 0,
            "avg_redundancy_penalty": total_redundancy_penalty / total_count if total_count > 0 else 0,
            "avg_chunk_similarity": total_chunk_similarity / total_count if total_count > 0 else 0,
        }

        for key, value in stats.items():
            self.history[key].append(value)

        return stats

    def evaluate(
        self,
        problems: list[str],
        gold_answers: list[str],
        sampling_client: Any,
    ) -> dict[str, Any]:
        """评估模型"""
        import tinker

        # 评估时也使用stop sequences
        # 注意：使用 temperature=0.7（与 SFT 评估一致），而非 0.0
        # 原因：temperature=0.0 (greedy) 可能导致 LoRA 权重不足以克服基座模型的先验
        stop_seqs = self.get_stop_sequences()
        eval_sampling_params = SamplingParams(
            max_tokens=self.config.max_response_length,
            temperature=0.7,  # 与 SFT 评估保持一致！
            stop=stop_seqs if stop_seqs else None,
        )

        total = len(problems)

        # 并发发送请求
        futures = []
        for problem in problems:
            model_input = self.format_prompt(problem)

            future = sampling_client.sample(
                prompt=model_input,
                sampling_params=eval_sampling_params,
                num_samples=1,
            )
            futures.append(future)

        # 收集结果
        correct = 0
        eval_samples = []

        for i, (future, gold) in enumerate(zip(futures, gold_answers, strict=False)):
            sample_result = future.result()

            tokens = None
            if sample_result.sequences:
                seq = sample_result.sequences[0]
                tokens = list(seq.tokens) if hasattr(seq.tokens, '__iter__') else seq.tokens
                # 使用parse_response解析
                parsed = self.parse_response(tokens)
                response_text = parsed["full_text"]
                thinking = parsed.get("thinking")
            else:
                response_text = ""
                thinking = None

            # 评估时也检查格式（用于统计）
            result = self.verifier.verify(
                response_text,
                gold,
                tokens=tokens,
                check_format=self.config.reasoning_mode,
            )
            is_correct = result["is_correct"]
            has_thinking = result.get("has_thinking", False)
            if is_correct:
                correct += 1

            eval_samples.append({
                "index": i,
                "problem": problems[i],
                "gold_answer": gold,
                "response": response_text,
                "thinking": thinking,  # 提取的thinking块
                "has_thinking": has_thinking,  # 是否有</think> token
                "extracted_answer": result.get("extracted"),
                "is_correct": is_correct,
            })

        # 统计 thinking 使用率
        thinking_count = sum(1 for s in eval_samples if s.get("has_thinking", False))

        return {
            "eval_accuracy": correct / total if total > 0 else 0,
            "eval_correct": correct,
            "eval_total": total,
            "thinking_rate": thinking_count / total if total > 0 else 0,
            "thinking_count": thinking_count,
            "samples": eval_samples,
        }


# ============================================================
# 样本展示辅助函数
# ============================================================

def print_eval_samples(
    samples: list[dict],
    num_correct: int = 1,
    num_incorrect: int = 2,
    max_response_len: int = 800,
):
    """打印评估样本供人工检查"""
    correct_samples = [s for s in samples if s["is_correct"]]
    incorrect_samples = [s for s in samples if not s["is_correct"]]

    print("\n" + "-" * 60)
    print("样本质量检查")
    print("-" * 60)

    def print_sample(sample, idx, label):
        print(f"\n  --- {label} #{idx+1} ---")
        if sample.get("problem"):
            p = sample["problem"][:150] + "..." if len(sample["problem"]) > 150 else sample["problem"]
            print(f"  问题: {p}")
        print(f"  标准答案: {sample['gold_answer']}")
        print(f"  提取答案: {sample['extracted_answer']} {'(未提取到)' if sample['extracted_answer'] is None else ''}")

        # 显示thinking内容（如果有）
        thinking = sample.get("thinking")
        if thinking:
            thinking_preview = thinking[:300] + "..." if len(thinking) > 300 else thinking
            print(f"  [Thinking]:\n    {thinking_preview.replace(chr(10), chr(10) + '    ')}")

        resp = sample["response"]
        if len(resp) > max_response_len:
            resp = resp[:max_response_len] + f"... [截断，共{len(sample['response'])}字符]"
        print(f"  回答:\n    {resp.replace(chr(10), chr(10) + '    ')}")

    if correct_samples and num_correct > 0:
        print(f"\n[正确样本] ({len(correct_samples)}/{len(samples)} total)")
        for i, sample in enumerate(correct_samples[:num_correct]):
            print_sample(sample, i, "正确样本")

    if incorrect_samples and num_incorrect > 0:
        print(f"\n[错误样本] ({len(incorrect_samples)}/{len(samples)} total)")
        for i, sample in enumerate(incorrect_samples[:num_incorrect]):
            print_sample(sample, i, "错误样本")

    print("-" * 60 + "\n")


def save_eval_samples(samples: list[dict], filepath: Path, step: int):
    """保存评估样本到JSON文件"""
    data = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "samples": samples,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 主训练循环
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="JustRL Math Reasoning Training")
    parser.add_argument("--scale", type=str, default="quick",
                        choices=["quick", "medium", "full"],
                        help="实验规模")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="模型名称 (Qwen无需授权; Llama需HuggingFace授权)")
    parser.add_argument("--reasoning", action="store_true",
                        help="启用reasoning/thinking mode")
    parser.add_argument("--thinking-budget", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Thinking token预算")
    parser.add_argument("--format-reward", type=float, default=0.1,
                        help="格式奖励权重：正确但无thinking时的惩罚 (默认0.1)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="从指定的 checkpoint 继续训练 (如 coldstart_sft_final)")
    parser.add_argument("--eval-only", action="store_true",
                        help="只进行评估，不训练 (需要配合 --checkpoint 使用)")
    parser.add_argument("--eval-temperature", type=float, default=0.7,
                        help="评估时的采样温度 (默认 0.7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/justrl_reasoning")
    parser.add_argument("--dry-run", action="store_true",
                        help="干运行模式")
    args = parser.parse_args()

    # 检查API Key
    if not args.dry_run and not os.environ.get("TINKER_API_KEY"):
        print("Error: 请设置TINKER_API_KEY环境变量")
        sys.exit(1)

    # 创建配置
    config = ReasoningConfig(
        scale=args.scale,
        model_name=args.model,
        reasoning_mode=args.reasoning,
        thinking_budget=args.thinking_budget,
        format_reward_weight=args.format_reward,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    random.seed(config.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "reasoning" if config.reasoning_mode else "standard"
    run_dir = Path(config.output_dir) / f"{config.experiment_name}_{mode_suffix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print("=" * 60)
    print("JustRL Math Reasoning Training")
    print("=" * 60)
    print(f"Scale: {config.scale}")
    print(f"Model: {config.model_name}")
    model_type = "Base Model (few-shot)" if is_base_model(config.model_name) else "Instruct Model (chat)"
    print(f"Model Type: {model_type}")
    print(f"Reasoning Mode: {'ON' if config.reasoning_mode else 'OFF'}")
    if config.reasoning_mode:
        print(f"Thinking Budget: {config.thinking_budget}")
        print(f"Format Reward Weight: {config.format_reward_weight} (penalty for missing </think>)")
        print(f"Redundancy Penalty: weight={config.redundancy_weight}, threshold={config.redundancy_threshold}")
    print(f"Max Response Length: {config.max_response_length}")
    print(f"Steps: {config.num_steps}")
    print(f"Batch size: {config.batch_size}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint} (继续训练)")
    print(f"Output: {run_dir}")
    print("=" * 60)

    # 估算成本（根据模型自动选择定价）
    # MoE模型按激活参数定价，Dense模型按总参数定价
    model_pricing = {
        # Dense models
        "meta-llama/Llama-3.2-1B": 0.09,
        "meta-llama/Llama-3.2-3B": 0.18,
        "Qwen/Qwen3-4B-Instruct-2507": 0.22,
        "meta-llama/Llama-3.1-8B": 0.40,
        "meta-llama/Llama-3.1-8B-Instruct": 0.40,
        "Qwen/Qwen3-8B": 0.40,
        "Qwen/Qwen3-8B-Base": 0.40,
        "Qwen/Qwen3-32B": 1.20,
        "meta-llama/Llama-3.1-70B": 2.80,
        "meta-llama/Llama-3.3-70B-Instruct": 2.80,
        # MoE models (按激活参数定价，更便宜)
        "Qwen/Qwen3-30B-A3B": 0.18,  # 激活3B
        "Qwen/Qwen3-30B-A3B-Base": 0.18,
        "Qwen/Qwen3-30B-A3B-Instruct-2507": 0.18,
        "Qwen/Qwen3-235B-A22B-Instruct-2507": 0.90,  # 激活22B
        "deepseek-ai/DeepSeek-V3.1": 0.90,
        "deepseek-ai/DeepSeek-V3.1-Base": 0.90,
    }
    price_per_m_tokens = model_pricing.get(config.model_name, 0.25)  # 默认$0.25/M

    # Reasoning模式的response更长，需要更多tokens
    avg_response_len = config.max_response_length // 2 if not config.reasoning_mode else config.max_response_length * 0.7
    tokens_per_step = config.batch_size * config.rollout_n * (config.max_prompt_length + avg_response_len)
    total_tokens = config.num_steps * tokens_per_step / 1e6  # 百万
    estimated_cost = total_tokens * price_per_m_tokens * 2  # Sample + Train

    print(f"\n预估Token消耗: {total_tokens:.1f}M tokens")
    print(f"模型定价: ${price_per_m_tokens}/M tokens")
    print(f"预估成本: ~${estimated_cost:.0f}")
    if config.reasoning_mode:
        print("  (Reasoning模式输出更长，成本较高)")
    print("=" * 60)

    # 加载训练数据
    print(f"\n训练数据集: {config.train_dataset}")
    if config.train_dataset == "dapo-math-17k":
        train_data = load_dapo_math_dataset()
    else:
        train_data = load_math_dataset("train")

    # 加载评估数据集（支持多个）
    print(f"\n评估数据集: {config.eval_datasets}")
    eval_data_dict = {}
    for eval_ds in config.eval_datasets:
        if eval_ds == "math":
            eval_data_dict["math"] = load_math_dataset("test", max_samples=config.eval_samples)
        elif eval_ds == "aime-2024":
            eval_data_dict["aime-2024"] = load_aime_dataset(year="2024")
        else:
            print(f"  [WARNING] 未知的评估数据集: {eval_ds}")

    # 兼容旧代码：主评估数据集
    eval_data = eval_data_dict.get("math", list(eval_data_dict.values())[0] if eval_data_dict else [])

    # 初始化验证器（使用配置的格式奖励和冗余度惩罚权重）
    verifier = MathReasoningVerifier(
        format_reward_weight=config.format_reward_weight,
        redundancy_weight=config.redundancy_weight,
        redundancy_threshold=config.redundancy_threshold,
    )

    if args.dry_run:
        print("\n[Dry Run Mode] 跳过 Tinker API 调用")

        # 加载 tokenizer
        tokenizer = None
        if HAS_TINKER_COOKBOOK:
            try:
                print(f"\n加载 tokenizer: {config.model_name}")
                tokenizer = tinker_tokenizer_utils.get_tokenizer(config.model_name)
                print("Tokenizer 加载完成")
            except Exception as e:
                print(f"Warning: Tinker Cookbook tokenizer 加载失败: {e}")

        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
                print(f"\n使用 HuggingFace 加载 tokenizer: {config.model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_name,
                    trust_remote_code=True,
                )
                print("Tokenizer 加载完成")
            except Exception as e:
                print(f"Tokenizer 加载失败: {e}")
                tokenizer = None

        # 测试 prompt 格式
        sample_problem = train_data[0]["problem"] if train_data else "What is 2 + 3?"
        print("\n" + "=" * 60)
        print("示例 Prompt 格式（与 SFT 训练一致）")
        print("=" * 60)
        print(f"问题: {sample_problem[:100]}...")
        print("-" * 60)

        # 与 SFT 训练完全相同的系统消息和用户消息
        system_msg = "You are a helpful mathematical assistant. Think step by step before answering."
        user_msg = f"Solve the following math problem.\n\nProblem: {sample_problem}"

        # 使用与 SFT 训练完全相同的手动模板（不使用 apply_chat_template！）
        if tokenizer:
            prompt_text = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
            print(prompt_text)
            print("-" * 60)
            tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            print(f"\nPrompt token 数: {len(tokens)}")
            print(f"Prompt 末尾: {repr(prompt_text[-50:])}")

        print("\n注意:")
        print("  - 使用与 SFT 训练完全相同的手动模板（见 coldstart_sft.py 523-528 行）")
        print("  - 不使用 tokenizer.apply_chat_template（可能有微妙差异）")
        print("  - Prompt 以 'assistant\\n' 结尾，不包含 <think>")
        print("  - 模型将根据 SFT 训练，自己输出 <think> 作为第一个 token")
        print("=" * 60)
        return

    # 初始化Tinker
    print("\n正在连接Tinker服务...")
    service_client = tinker.ServiceClient()

    print(f"正在加载模型: {config.model_name}")
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
        train_unembed=True,  # 必须与 SFT 训练时一致！
    )
    print("模型加载完成")

    # 从 checkpoint 恢复（如果指定）
    start_step = 0  # 默认从0开始（即第一个step是1）
    if args.checkpoint:
        print(f"\n从 checkpoint 恢复: {args.checkpoint}")
        try:
            training_client.load_state(args.checkpoint)
            print(f"Checkpoint 加载成功: {args.checkpoint}")

            # 从 checkpoint 名称解析 step 数
            # 支持格式: checkpoint_step_50, checkpoint_step_50/weights, tinker://xxx/checkpoint_step_50
            import re
            step_match = re.search(r'checkpoint_step_(\d+)', args.checkpoint)
            if step_match:
                start_step = int(step_match.group(1))
                print(f"从 step {start_step} 继续训练 (下一步为 step {start_step + 1})")
            else:
                print("Warning: 无法从 checkpoint 名称解析 step 数，从 step 1 开始")
        except Exception as e:
            print(f"Warning: 无法加载 checkpoint: {e}")
            print("继续使用基座模型...")

    # 创建训练器
    trainer = ReasoningTrainer(config, training_client, verifier)
    trainer.global_step = start_step  # 设置起始 step

    # ============================================================
    # Eval-only 模式
    # ============================================================
    if args.eval_only:
        if not args.checkpoint:
            print("Error: --eval-only 需要配合 --checkpoint 使用")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Eval-Only 模式")
        print("=" * 60)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Temperature: {args.eval_temperature}")

        # 临时修改 config 的 temperature
        original_temp = config.temperature
        config.temperature = args.eval_temperature

        # 获取 sampling client
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name="eval_only_temp"
        )

        # 评估 MATH
        eval_problems = [item["problem"] for item in eval_data]
        eval_answers = [item["answer"] for item in eval_data]

        print(f"\n评估 MATH ({len(eval_problems)} 样本)...")
        # 需要临时调整 trainer 的 eval temperature
        eval_stats = trainer.evaluate(eval_problems, eval_answers, sampling_client)
        print(f"  MATH 准确率: {eval_stats['eval_accuracy']:.2%} ({eval_stats['eval_correct']}/{eval_stats['eval_total']})")
        print(f"  Thinking Rate: {eval_stats['thinking_rate']:.2%}")

        # 评估 AIME (如果有)
        if "aime-2024" in eval_data_dict and eval_data_dict["aime-2024"]:
            aime_data = eval_data_dict["aime-2024"]
            aime_problems = [item["problem"] for item in aime_data]
            aime_answers = [item["answer"] for item in aime_data]

            print(f"\n评估 AIME 2024 ({len(aime_problems)} 样本)...")
            aime_stats = trainer.evaluate(aime_problems, aime_answers, sampling_client)
            print(f"  AIME 准确率: {aime_stats['eval_accuracy']:.2%} ({aime_stats['eval_correct']}/{aime_stats['eval_total']})")
            print(f"  Thinking Rate: {aime_stats['thinking_rate']:.2%}")

        print("\n" + "=" * 60)
        print("Eval-Only 完成")
        print("=" * 60)

        # 恢复 config
        config.temperature = original_temp
        sys.exit(0)

    # 训练循环
    print("\n开始训练...")
    print("-" * 60)

    # 早停相关变量
    best_eval_accuracy = 0.0
    early_stop_counter = 0
    should_stop = False

    for step in range(start_step + 1, config.num_steps + 1):
        batch_indices = random.sample(range(len(train_data)), config.batch_size)
        batch = [train_data[i] for i in batch_indices]

        # 直接传递problems，train_step内部会格式化
        problems = [item["problem"] for item in batch]
        gold_answers = [item["answer"] for item in batch]

        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"step_{step}"
        )

        stats = trainer.train_step(problems, gold_answers, sampling_client)

        # 构建输出信息
        output_parts = [
            f"Step {step}/{config.num_steps}",
            f"Reward: {stats['mean_reward']:.3f}",
            f"Acc: {stats['accuracy']:.2%}",
        ]
        # 在 reasoning 模式下显示 thinking rate 和长度
        if config.reasoning_mode:
            output_parts.append(f"Think: {stats['thinking_rate']:.0%}")
            output_parts.append(f"Len: {stats['avg_thinking_length']:.0f}/{stats['avg_response_length']:.0f}")
        output_parts.extend([
            f"Train: {stats['num_train_samples']}/{stats['total_samples']}",
            f"Time: {stats['step_time']:.1f}s",
        ])
        print(" | ".join(output_parts))

        # 训练健康监控
        if config.reasoning_mode:
            avg_resp_len = stats.get('avg_response_length', 0)
            thinking_rate = stats.get('thinking_rate', 1.0)
            avg_redundancy = stats.get('avg_redundancy_score', 0)
            avg_chunk_sim = stats.get('avg_chunk_similarity', 0)

            # 响应长度爆炸警告
            if avg_resp_len > 5000:
                print(f"  [WARNING] Response length explosion: {avg_resp_len:.0f} tokens (threshold: 5000)")

            # Thinking rate 下降警告
            if thinking_rate < 0.6:
                print(f"  [WARNING] Low thinking rate: {thinking_rate:.0%} (threshold: 60%)")

            # 冗余度过高警告
            if avg_redundancy > 0.4:
                print(f"  [WARNING] High redundancy score: {avg_redundancy:.1%} (threshold: 40%)")

            # Chunk similarity 异常警告（检测潜在的"狡猾"模式）
            # 如果 chunk_sim 高但 redundancy 低，可能是近似重复绕过了压缩率检测
            if avg_chunk_sim > 0.5 and avg_redundancy < 0.3:
                print(f"  [WARNING] Suspicious pattern: high chunk_sim ({avg_chunk_sim:.1%}) but low redundancy ({avg_redundancy:.1%})")
            elif avg_chunk_sim > 0.6:
                print(f"  [WARNING] High chunk similarity: {avg_chunk_sim:.1%} (threshold: 60%)")

        # 评估（支持多数据集）
        if step % config.eval_interval == 0:
            # 主评估数据集（MATH）
            eval_problems = [item["problem"] for item in eval_data]
            eval_answers = [item["answer"] for item in eval_data]

            eval_stats = trainer.evaluate(
                eval_problems, eval_answers, sampling_client
            )
            eval_msg = (f"  [Eval MATH] Accuracy: {eval_stats['eval_accuracy']:.2%} "
                        f"({eval_stats['eval_correct']}/{eval_stats['eval_total']})")
            if config.reasoning_mode:
                eval_msg += f" | Think: {eval_stats['thinking_rate']:.0%}"
            print(eval_msg)

            # 记录 eval 指标到 history（用于绘图）
            trainer.history["eval_step"].append(step)
            trainer.history["eval_accuracy"].append(eval_stats["eval_accuracy"])
            trainer.history["eval_thinking_rate"].append(eval_stats.get("thinking_rate", 0))

            print_eval_samples(
                eval_stats["samples"],
                num_correct=1,
                num_incorrect=2,
                max_response_len=config.max_response_length
            )

            samples_file = run_dir / f"eval_samples_step_{step}.json"
            save_eval_samples(eval_stats["samples"], samples_file, step)

            # AIME 2024 评估（如果配置了）
            if "aime-2024" in eval_data_dict and eval_data_dict["aime-2024"]:
                aime_data = eval_data_dict["aime-2024"]
                aime_problems = [item["problem"] for item in aime_data]
                aime_answers = [item["answer"] for item in aime_data]

                aime_stats = trainer.evaluate(
                    aime_problems, aime_answers, sampling_client
                )
                aime_msg = (f"  [Eval AIME] Accuracy: {aime_stats['eval_accuracy']:.2%} "
                            f"({aime_stats['eval_correct']}/{aime_stats['eval_total']})")
                if config.reasoning_mode:
                    aime_msg += f" | Think: {aime_stats['thinking_rate']:.0%}"
                print(aime_msg)

                # 记录 AIME 指标
                if "eval_aime_accuracy" not in trainer.history:
                    trainer.history["eval_aime_accuracy"] = []
                trainer.history["eval_aime_accuracy"].append(aime_stats["eval_accuracy"])

                if "eval_aime_thinking_rate" not in trainer.history:
                    trainer.history["eval_aime_thinking_rate"] = []
                trainer.history["eval_aime_thinking_rate"].append(aime_stats.get("thinking_rate", 0))

                # 保存 AIME 样本
                aime_samples_file = run_dir / f"eval_aime_samples_step_{step}.json"
                save_eval_samples(aime_stats["samples"], aime_samples_file, step)

            # 早停检查
            if config.early_stopping:
                current_accuracy = eval_stats["eval_accuracy"]

                if current_accuracy > best_eval_accuracy:
                    # 新的最佳结果
                    best_eval_accuracy = current_accuracy
                    early_stop_counter = 0
                elif current_accuracy < best_eval_accuracy - config.early_stopping_threshold:
                    # 显著下降
                    early_stop_counter += 1
                    print(f"  [Early Stop] Accuracy dropped: {current_accuracy:.2%} < {best_eval_accuracy:.2%} - {config.early_stopping_threshold:.0%}")
                    print(f"               Counter: {early_stop_counter}/{config.early_stopping_patience}")

                    if early_stop_counter >= config.early_stopping_patience:
                        print(f"\n{'='*60}")
                        print(f"早停触发！连续 {early_stop_counter} 次评估准确率下降")
                        print(f"最佳 Eval Accuracy: {best_eval_accuracy:.2%}")
                        print(f"当前 Eval Accuracy: {current_accuracy:.2%}")
                        print(f"{'='*60}")
                        should_stop = True
                else:
                    # 小幅波动，不计入
                    pass

        # 检查是否需要早停
        if should_stop:
            # 保存当前状态后退出
            checkpoint_name = f"checkpoint_step_{step}_early_stop"
            training_client.save_state(checkpoint_name)
            print(f"  [Save] Early stop checkpoint: {checkpoint_name}")
            with open(run_dir / "history.json", "w") as f:
                json.dump(dict(trainer.history), f, indent=2)
            break

        # 保存检查点
        if step % config.save_interval == 0:
            checkpoint_name = f"checkpoint_step_{step}"
            training_client.save_state(checkpoint_name)
            print(f"  [Save] Checkpoint saved: {checkpoint_name}")

            with open(run_dir / "history.json", "w") as f:
                json.dump(dict(trainer.history), f, indent=2)

    # 最终保存
    print("\n" + "=" * 60)
    if should_stop:
        print(f"训练提前停止 (早停机制触发于 Step {step})")
        print(f"最佳 Eval Accuracy: {best_eval_accuracy:.2%}")
    else:
        print("训练完成!")
    training_client.save_state("final_model")

    # 最终评估
    sampling_client = training_client.save_weights_and_get_sampling_client(name="final")

    # MATH 最终评估
    eval_problems = [item["problem"] for item in eval_data]
    eval_answers = [item["answer"] for item in eval_data]
    final_eval = trainer.evaluate(
        eval_problems, eval_answers, sampling_client
    )

    print("\n--- MATH 最终评估 ---")
    final_msg = f"MATH 准确率: {final_eval['eval_accuracy']:.2%}"
    if config.reasoning_mode:
        final_msg += f" | Thinking率: {final_eval['thinking_rate']:.0%}"
    print(final_msg)
    print_eval_samples(final_eval["samples"], num_correct=2, num_incorrect=2,
                       max_response_len=config.max_response_length)
    save_eval_samples(final_eval["samples"], run_dir / "eval_samples_final.json", config.num_steps)

    # AIME 2024 最终评估
    final_aime_eval = None
    if "aime-2024" in eval_data_dict and eval_data_dict["aime-2024"]:
        aime_data = eval_data_dict["aime-2024"]
        aime_problems = [item["problem"] for item in aime_data]
        aime_answers = [item["answer"] for item in aime_data]
        final_aime_eval = trainer.evaluate(
            aime_problems, aime_answers, sampling_client
        )

        print("\n--- AIME 2024 最终评估 ---")
        aime_msg = f"AIME 准确率: {final_aime_eval['eval_accuracy']:.2%} ({final_aime_eval['eval_correct']}/30)"
        if config.reasoning_mode:
            aime_msg += f" | Thinking率: {final_aime_eval['thinking_rate']:.0%}"
        print(aime_msg)
        print_eval_samples(final_aime_eval["samples"], num_correct=1, num_incorrect=2,
                           max_response_len=config.max_response_length)
        save_eval_samples(final_aime_eval["samples"], run_dir / "eval_aime_samples_final.json", config.num_steps)

    print(f"\n输出目录: {run_dir}")
    print("=" * 60)

    # 保存最终结果
    results = {
        "config": asdict(config),
        "final_eval": {
            "math_accuracy": final_eval["eval_accuracy"],
            "math_correct": final_eval["eval_correct"],
            "math_total": final_eval["eval_total"],
        },
        "training_summary": {
            "total_steps": trainer.global_step,
            "final_train_accuracy": trainer.history["accuracy"][-1] if trainer.history["accuracy"] else 0,
        },
    }

    # 添加 AIME 结果
    if final_aime_eval:
        results["final_eval"]["aime_accuracy"] = final_aime_eval["eval_accuracy"]
        results["final_eval"]["aime_correct"] = final_aime_eval["eval_correct"]
        results["final_eval"]["aime_total"] = final_aime_eval["eval_total"]

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

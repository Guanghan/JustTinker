"""
数据集加载和处理工具

支持的数据集:
- GSM8K: 小学数学应用题
- MATH: 竞赛数学问题 (EleutherAI/hendrycks_math)
- DAPO-Math-17k: DAPO论文训练数据集
- AIME 2024: 美国数学邀请赛
- OpenR1-Math-220k: DeepSeek-R1生成的数学推理数据（SFT用）
- Countdown: 数字组合游戏（用于入门实验）

Author: Guanghan Ning
Date: 2025-12-23
Updated: 2026-01-13 (合并所有数据加载函数)
"""

import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Sample:
    """数据样本"""
    question: str
    answer: str
    solution: str | None = None
    metadata: dict | None = None


# ============================================================
# 答案提取辅助函数
# ============================================================

def extract_boxed_answer(solution: str) -> str:
    """
    从solution中提取\\boxed{}答案

    支持两种格式:
    1. \\boxed{答案} - 标准格式，支持任意层级嵌套
    2. \\boxed 答案 - 无括号格式（如 \\boxed 2）

    例如:
    - \\boxed{\\dfrac{\\sqrt{6}}{6}} -> \\dfrac{\\sqrt{6}}{6}
    - \\boxed 2 -> 2

    Args:
        solution: 包含答案的解题文本

    Returns:
        提取的答案字符串，如果找不到则返回空字符串
    """
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


# ============================================================
# GSM8K 数据集
# ============================================================

def load_gsm8k(
    split: str = "train",
    max_samples: int | None = None,
    cache_dir: str | None = None
) -> list[dict]:
    """
    加载GSM8K数据集

    GSM8K包含约8500个小学数学应用题，每个问题有详细的解题步骤。
    答案格式: 解题步骤 + "#### " + 最终答案

    Args:
        split: "train" 或 "test"
        max_samples: 限制加载的样本数量
        cache_dir: 缓存目录

    Returns:
        样本字典列表，每个包含 question, answer, solution 字段

    Example:
        >>> samples = load_gsm8k("train", max_samples=100)
        >>> print(samples[0]["question"])
        "Janet's ducks lay 16 eggs per day..."
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 请安装datasets库: pip install datasets")
        sys.exit(1)

    print(f"加载 GSM8K {split}集...")
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    samples = []
    for item in dataset:
        # GSM8K的答案格式: 解题步骤 + "#### " + 最终答案
        answer_text = item["answer"]
        parts = answer_text.split("####")

        final_answer = parts[-1].strip().replace(",", "") if len(parts) > 1 else answer_text
        solution = parts[0].strip() if len(parts) > 1 else None

        samples.append({
            "question": item["question"],
            "answer": final_answer,
            "solution": solution,
            "source": "gsm8k",
        })

        if max_samples and len(samples) >= max_samples:
            break

    print(f"  加载了 {len(samples)} 个样本")
    return samples


# ============================================================
# MATH 数据集 (EleutherAI/hendrycks_math)
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

    Returns:
        样本字典列表
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 请安装datasets库: pip install datasets")
        sys.exit(1)

    print(f"加载 MATH {split}集...")

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
                    "answer": extract_boxed_answer(item.get("solution", "")),
                    "subject": subject,
                    "level": item.get("level", "unknown"),
                    "source": "math",
                })
    else:
        # 简单采样：合并后打乱
        all_items = []
        for subject, data in subject_data.items():
            for item in data:
                all_items.append({
                    "problem": item.get("problem", ""),
                    "solution": item.get("solution", ""),
                    "answer": extract_boxed_answer(item.get("solution", "")),
                    "subject": subject,
                    "level": item.get("level", "unknown"),
                    "source": "math",
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
        original_count_subj = len(subject_data.get(subject, []))
        original_ratio = original_count_subj / total_count * 100 if total_count > 0 else 0
        sample_ratio = count / len(samples) * 100 if samples else 0
        print(f"    {subject}: {count} ({sample_ratio:.1f}%) [原始: {original_ratio:.1f}%]")

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
        sample["source"] = "mock"
        samples.append(sample)

    return samples


# ============================================================
# DAPO-Math-17k 数据集
# ============================================================

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

    Returns:
        样本字典列表
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


# ============================================================
# AIME 数据集
# ============================================================

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

    Returns:
        样本字典列表
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


# ============================================================
# OpenR1-Math-220k 数据集 (SFT 训练用)
# ============================================================

def load_openr1_dataset(
    config: str = "default",
    max_samples: int | None = None,
    seed: int = 42,
    tokenizer: Any = None,
    max_seq_length: int = 8192,
) -> tuple[list[dict], list[dict]]:
    """
    加载 OpenR1-Math-220k 数据集 (用于 SFT 训练)

    该数据集包含 DeepSeek-R1 生成的数学推理数据，使用 <think>...</think> 格式。

    Args:
        config: 数据集配置 (default, extended, all)
        max_samples: 最大样本数
        seed: 随机种子
        tokenizer: 用于计算 token 长度的 tokenizer（可选）
        max_seq_length: 最大序列长度，超过此长度的样本将被过滤

    Returns:
        (train_data, eval_data) 元组
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 请安装 datasets 库: pip install datasets")
        sys.exit(1)

    print(f"加载 OpenR1-Math-220k ({config})...")

    try:
        dataset = load_dataset("open-r1/OpenR1-Math-220k", config, split="train")
    except Exception as e:
        print(f"Error: 加载数据集失败: {e}")
        print("尝试使用模拟数据...")
        return _get_mock_openr1_data(max_samples or 100, seed)

    print(f"  原始数据: {len(dataset)} 个问题")

    # 如果提供了 tokenizer，用于过滤超长样本
    system_msg = "You are a helpful mathematical assistant. Think step by step before answering."
    filter_by_length = tokenizer is not None

    if filter_by_length:
        print(f"  启用长度过滤: max_seq_length={max_seq_length}")

    # 处理数据：每个问题可能有多个 generations
    samples = []
    skipped_no_think = 0
    skipped_too_long = 0

    for item in dataset:
        problem = item.get("problem", "")
        generations = item.get("generations", [])
        correctness = item.get("correctness_math_verify", [])

        if not problem or not generations:
            skipped_no_think += 1
            continue

        # 选择一个正确的 generation
        selected_response = None
        for i, gen in enumerate(generations):
            is_correct = correctness[i] if i < len(correctness) else False
            # 检查是否包含 <think> 格式
            if is_correct and gen and "<think>" in gen and "</think>" in gen:
                selected_response = gen
                break

        # 如果没有正确的，选择第一个有 thinking 格式的
        if selected_response is None:
            for gen in generations:
                if gen and "<think>" in gen and "</think>" in gen:
                    selected_response = gen
                    break

        if selected_response is None:
            skipped_no_think += 1
            continue

        # 长度过滤
        if filter_by_length:
            user_msg = f"Solve the following math problem.\n\nProblem: {problem}"
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                response_tokens = tokenizer.encode(selected_response, add_special_tokens=False)
                total_length = len(prompt_tokens) + len(response_tokens)

                if total_length > max_seq_length:
                    skipped_too_long += 1
                    continue
            except Exception:
                # tokenizer 失败时不过滤
                pass

        samples.append({
            "problem": problem,
            "response": selected_response,
            "source": item.get("source", "unknown"),
            "problem_type": item.get("problem_type", "unknown"),
        })

        if max_samples and len(samples) >= max_samples:
            break

    print(f"  有效样本: {len(samples)}")
    print(f"  跳过（无 thinking 格式）: {skipped_no_think}")
    if filter_by_length:
        print(f"  跳过（超过 {max_seq_length} tokens）: {skipped_too_long}")

    # 打乱并分割
    random.seed(seed)
    random.shuffle(samples)

    # 90% train, 10% eval
    split_idx = int(len(samples) * 0.9)
    train_data = samples[:split_idx]
    eval_data = samples[split_idx:]

    print(f"  训练集: {len(train_data)}, 评估集: {len(eval_data)}")

    # 统计 source 分布
    source_counts = defaultdict(int)
    for s in train_data:
        source_counts[s["source"]] += 1
    print(f"  数据来源分布: {dict(source_counts)}")

    return train_data, eval_data


def _get_mock_openr1_data(n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """生成模拟数据用于测试"""
    random.seed(seed)

    mock_samples = [
        {
            "problem": "What is 123 + 456?",
            "response": """<think>
I need to add 123 and 456.
Let me break this down:
- 123 + 456
- First, 3 + 6 = 9
- Then, 20 + 50 = 70
- Finally, 100 + 400 = 500
- Total: 500 + 70 + 9 = 579
</think>

The answer is $\\boxed{579}$.""",
            "source": "mock",
            "problem_type": "Algebra",
        },
        {
            "problem": "Solve for x: 2x + 5 = 13",
            "response": """<think>
I need to solve the equation 2x + 5 = 13.
Step 1: Subtract 5 from both sides.
2x + 5 - 5 = 13 - 5
2x = 8
Step 2: Divide both sides by 2.
2x / 2 = 8 / 2
x = 4
Let me verify: 2(4) + 5 = 8 + 5 = 13. Correct!
</think>

The solution is $x = \\boxed{4}$.""",
            "source": "mock",
            "problem_type": "Algebra",
        },
        {
            "problem": "What is the area of a circle with radius 5?",
            "response": """<think>
I need to find the area of a circle with radius 5.
The formula for the area of a circle is A = πr².
Given r = 5:
A = π × 5²
A = π × 25
A = 25π
</think>

The area is $\\boxed{25\\pi}$ square units.""",
            "source": "mock",
            "problem_type": "Geometry",
        },
    ]

    # 扩展到 n 个样本
    samples = []
    for i in range(n):
        sample = mock_samples[i % len(mock_samples)].copy()
        samples.append(sample)

    # 分割
    split_idx = int(len(samples) * 0.9)
    return samples[:split_idx], samples[split_idx:]


# ============================================================
# Countdown 游戏数据
# ============================================================

def generate_countdown_data(
    num_samples: int = 1000,
    num_numbers: int = 4,
    target_range: tuple[int, int] = (10, 100),
    number_range: tuple[int, int] = (1, 10),
    seed: int = 42
) -> list[dict]:
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
        样本字典列表

    Example:
        >>> samples = generate_countdown_data(num_samples=10)
        >>> print(samples[0]["question"])
        "Using numbers [2, 5, 3, 7], reach the target 24 using +, -, *, /"
    """
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

        samples.append({
            "question": question,
            "answer": str(target),
            "source": "countdown",
            "numbers": numbers,
            "target": target,
        })

    return samples


# ============================================================
# 统一数据加载器
# ============================================================

class DataLoader:
    """
    统一的数据加载器

    Example:
        >>> loader = DataLoader("gsm8k")
        >>> train_data = loader.load("train", max_samples=1000)
        >>> eval_data = loader.load("test", max_samples=200)
    """

    SUPPORTED_DATASETS = ["gsm8k", "math", "dapo-math-17k", "aime-2024", "openr1", "countdown"]

    def __init__(self, dataset_name: str, cache_dir: str | None = None):
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
        max_samples: int | None = None,
        **kwargs
    ) -> list[dict]:
        """加载数据集"""
        if self.dataset_name == "gsm8k":
            return load_gsm8k(split, max_samples, self.cache_dir)
        elif self.dataset_name == "math":
            return load_math_dataset(split, max_samples, **kwargs)
        elif self.dataset_name == "dapo-math-17k":
            return load_dapo_math_dataset(max_samples, **kwargs)
        elif self.dataset_name == "aime-2024":
            return load_aime_dataset(**kwargs)
        elif self.dataset_name == "openr1":
            train, _ = load_openr1_dataset(max_samples=max_samples, **kwargs)
            return train
        elif self.dataset_name == "countdown":
            return generate_countdown_data(max_samples or 1000, **kwargs)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据加载器测试")
    print("=" * 60)

    # 测试 GSM8K
    print("\n测试 GSM8K 加载...")
    try:
        samples = load_gsm8k("train", max_samples=5)
        print(f"  加载了 {len(samples)} 个样本")
        print(f"  样例问题: {samples[0]['question'][:100]}...")
        print(f"  样例答案: {samples[0]['answer']}")
    except Exception as e:
        print(f"  GSM8K 加载失败: {e}")

    # 测试 Countdown
    print("\n测试 Countdown 生成...")
    samples = generate_countdown_data(num_samples=3)
    for i, s in enumerate(samples):
        print(f"  样本 {i+1}: {s['question'][:80]}...")
        print(f"    目标: {s['answer']}")

    # 测试 boxed 提取
    print("\n测试 boxed 答案提取...")
    test_cases = [
        ("So the answer is $\\boxed{42}$.", "42"),
        ("Therefore, \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("The result is \\boxed 7.", "7"),
    ]
    for solution, expected in test_cases:
        result = extract_boxed_answer(solution)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] {solution[:40]}... -> {result}")

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

import os
import sys
import json
import random
import argparse
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from collections import defaultdict

# Tinker imports
try:
    import tinker
    from tinker import SamplingParams
except ImportError:
    print("Warning: tinker package not found. Run: pip install tinker")
    tinker = None
    SamplingParams = None

# Tinker cookbook imports (for renderer system)
try:
    from tinker_cookbook import renderers as tinker_renderers
    from tinker_cookbook import tokenizer_utils as tinker_tokenizer_utils
    HAS_TINKER_COOKBOOK = True
except ImportError:
    print("Warning: tinker_cookbook not found. Run: pip install tinker-cookbook")
    tinker_renderers = None
    tinker_tokenizer_utils = None
    HAS_TINKER_COOKBOOK = False

# 添加项目根目录到path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
    lora_rank: int = 64

    # Reasoning模式设置
    reasoning_mode: bool = False  # 是否启用thinking mode
    thinking_budget: str = "medium"  # thinking token预算: low, medium, high
    format_reward_weight: float = 0.1  # 格式奖励权重：没有thinking时的惩罚

    # 训练设置（根据scale调整）
    num_steps: int = 200
    batch_size: int = 16  # MATH更难，减小batch size
    rollout_n: int = 8

    # JustRL核心参数
    learning_rate: float = 1e-6
    clip_ratio_low: float = 0.8
    clip_ratio_high: float = 1.28
    kl_coef: float = 0.0
    temperature: float = 1.0

    # 生成设置
    max_prompt_length: int = 1024  # MATH问题更长
    max_response_length: int = 8192  # reasoning需要更长输出

    # 评估设置
    eval_interval: int = 50
    eval_samples: int = 50  # MATH评估更慢，减少样本数
    save_interval: int = 100

    # 数据集设置
    math_subjects: Optional[List[str]] = None  # None表示所有科目

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
                "num_steps": 500,
                "batch_size": 32,
                "eval_interval": 50,
                "save_interval": 100,
                "eval_samples": 100,
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

        # Reasoning模式需要更长的输出
        if self.reasoning_mode:
            thinking_budgets = {
                "low": 4096,
                "medium": 8192,
                "high": 16384,
            }
            self.max_response_length = thinking_budgets.get(
                self.thinking_budget, 8192
            )


# ============================================================
# 数据加载
# ============================================================

def load_math_dataset(
    split: str = "train",
    max_samples: Optional[int] = None,
    subjects: Optional[List[str]] = None,
) -> List[Dict]:
    """
    加载MATH数据集 (EleutherAI/hendrycks_math)

    MATH数据集包含7个科目：
    - algebra (代数)
    - counting_and_probability (组合概率)
    - geometry (几何)
    - intermediate_algebra (中级代数)
    - number_theory (数论)
    - prealgebra (预备代数)
    - precalculus (预备微积分)
    """
    try:
        from datasets import load_dataset, concatenate_datasets
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

    # 加载各科目数据并合并
    all_datasets = []
    for subject in target_subjects:
        try:
            ds = load_dataset(
                "EleutherAI/hendrycks_math",
                subject,
                split=split,
            )
            # 添加subject字段
            ds = ds.map(lambda x: {"subject": subject, **x})
            all_datasets.append(ds)
            print(f"  加载 {subject}: {len(ds)} 个样本")
        except Exception as e:
            print(f"  加载 {subject} 失败: {e}")

    if not all_datasets:
        print("所有科目加载失败，使用模拟数据...")
        return _get_mock_math_data(max_samples or 100)

    # 合并所有科目
    dataset = concatenate_datasets(all_datasets)
    print(f"  总计: {len(dataset)} 个样本")

    # 打乱数据
    dataset = dataset.shuffle(seed=42)

    samples = []
    for item in dataset:
        problem = item.get("problem", "")
        solution = item.get("solution", "")
        level = item.get("level", "unknown")
        subject = item.get("subject", "unknown")

        # 从solution中提取答案
        answer = _extract_boxed_answer(solution)

        samples.append({
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "subject": subject,
            "level": level,
        })

        if max_samples and len(samples) >= max_samples:
            break

    print(f"  使用 {len(samples)} 个样本")

    # 打印科目分布
    subject_counts = defaultdict(int)
    for s in samples:
        subject_counts[s["subject"]] += 1
    print("  科目分布:", dict(subject_counts))

    return samples


def _extract_boxed_answer(solution: str) -> str:
    """从solution中提取\\boxed{}答案"""
    import re
    # 匹配 \boxed{...} 格式
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if match:
        return match.group(1)
    return ""


def _get_mock_math_data(n: int) -> List[Dict]:
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
    if "llama" in model_lower and not any(kw in model_lower for kw in instruct_keywords):
        return True
    return False


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
    """

    # Thinking token IDs (Qwen3)
    THINK_END_TOKEN_ID = 151668  # </think>

    def __init__(self, format_reward_weight: float = 0.1):
        """
        Args:
            format_reward_weight: 格式奖励/惩罚的权重
                - 正确答案 + 有thinking: reward = 1.0
                - 正确答案 + 无thinking: reward = 1.0 - format_reward_weight
                - 错误答案: reward = 0.0
        """
        import re
        self.re = re
        self.format_reward_weight = format_reward_weight

    def has_thinking_format(self, tokens: List[int] = None, text: str = None) -> bool:
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

    def _normalize_answer(self, answer: str) -> str:
        """标准化答案字符串"""
        if not answer:
            return ""

        answer = answer.strip()
        # 移除常见的包装
        answer = answer.replace("$", "")
        answer = answer.replace(",", "")
        answer = answer.replace(" ", "")
        # 统一分数格式
        answer = answer.replace("\\frac", "frac")
        answer = answer.replace("\\dfrac", "frac")
        # 统一其他LaTeX
        answer = answer.replace("\\pi", "pi")
        answer = answer.replace("\\sqrt", "sqrt")
        answer = answer.lower()

        return answer

    def extract_answer(self, text: str) -> Optional[str]:
        """从response中提取答案"""
        # 优先匹配 \boxed{...}
        boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = list(self.re.finditer(boxed_pattern, text))
        if matches:
            # 取最后一个boxed答案（通常是最终答案）
            return matches[-1].group(1)

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

    def verify(
        self,
        response: str,
        gold: str,
        tokens: List[int] = None,
        check_format: bool = True,
    ) -> Dict:
        """
        验证答案并计算奖励

        Args:
            response: 模型的回答文本
            gold: 标准答案
            tokens: 生成的 token 列表（用于检查 thinking 格式）
            check_format: 是否检查格式并应用格式奖励

        Returns:
            包含 is_correct, reward, extracted, has_thinking 等字段的字典
        """
        extracted = self.extract_answer(response)

        # 检查 thinking 格式
        has_thinking = False
        if check_format and self.format_reward_weight > 0:
            has_thinking = self.has_thinking_format(tokens=tokens, text=response)

        if extracted is None:
            return {
                "is_correct": False,
                "reward": 0.0,
                "extracted": None,
                "gold_normalized": self._normalize_answer(gold),
                "has_thinking": has_thinking,
                "format_penalty": 0.0,
            }

        # 标准化比较
        extracted_norm = self._normalize_answer(extracted)
        gold_norm = self._normalize_answer(gold)

        is_correct = False

        # 直接字符串比较
        if extracted_norm == gold_norm:
            is_correct = True
        else:
            # 尝试数值比较
            try:
                ext_val = float(extracted_norm.replace("pi", "").replace("sqrt", ""))
                gold_val = float(gold_norm.replace("pi", "").replace("sqrt", ""))
                if abs(ext_val - gold_val) < 1e-6:
                    is_correct = True
            except (ValueError, TypeError):
                pass

        # 计算奖励
        if is_correct:
            if has_thinking:
                # 正确 + 有 thinking: 满分
                reward = 1.0
                format_penalty = 0.0
            else:
                # 正确 + 无 thinking: 轻微惩罚
                format_penalty = self.format_reward_weight if check_format else 0.0
                reward = 1.0 - format_penalty
        else:
            # 错误答案: 0 分
            reward = 0.0
            format_penalty = 0.0

        return {
            "is_correct": is_correct,
            "reward": reward,
            "extracted": extracted,
            "gold_normalized": gold_norm,
            "has_thinking": has_thinking,
            "format_penalty": format_penalty,
        }


# ============================================================
# GRPO训练器
# ============================================================

class ReasoningTrainer:
    """
    Reasoning Model训练器

    基于JustRL，增加对thinking tokens的支持
    使用Tinker的renderer系统正确处理thinking mode
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
        self.renderer = None
        self._init_tokenizer_and_renderer()

    def _init_tokenizer_and_renderer(self):
        """初始化tokenizer和renderer"""
        model_name = self.config.model_name
        is_qwen3 = "qwen3" in model_name.lower()

        # 优先使用tinker_cookbook的tokenizer和renderer
        if HAS_TINKER_COOKBOOK and is_qwen3:
            try:
                print(f"使用Tinker Cookbook加载tokenizer: {model_name}")
                self.tokenizer = tinker_tokenizer_utils.get_tokenizer(model_name)

                # 选择renderer: qwen3 默认启用thinking mode
                if self.config.reasoning_mode:
                    renderer_name = 'qwen3'  # enable_thinking=True by default
                    print("使用 qwen3 renderer (thinking mode enabled)")
                else:
                    renderer_name = 'qwen3_disable_thinking'
                    print("使用 qwen3_disable_thinking renderer")

                self.renderer = tinker_renderers.get_renderer(renderer_name, self.tokenizer)
                print("Tokenizer和Renderer加载完成")
                return
            except Exception as e:
                print(f"Warning: Tinker Cookbook加载失败: {e}")
                print("回退到HuggingFace tokenizer...")

        # 回退到HuggingFace tokenizer
        try:
            from transformers import AutoTokenizer
            print(f"使用HuggingFace加载tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self.renderer = None  # 没有renderer
            print("Tokenizer加载完成 (无renderer)")
        except Exception as e:
            error_msg = str(e)
            print(f"\nError: 无法加载tokenizer: {e}")

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

        如果有renderer，使用renderer.build_generation_prompt
        否则回退到tokenizer.apply_chat_template
        """
        # 构建消息
        system_msg = "You are a helpful mathematical assistant. Solve problems step by step and put your final answer in \\boxed{}."
        user_msg = f"Solve the following math problem.\n\nProblem: {problem}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        if self.renderer is not None:
            # 使用Tinker renderer (推荐)
            model_input = self.renderer.build_generation_prompt(messages)
            return model_input
        else:
            # 回退到旧方式
            text = format_prompt_with_tokenizer(
                problem,
                self.tokenizer,
                self.config.reasoning_mode,
                self.config.model_name,
            )
            tokens = self.tokenizer.encode(text)
            return tinker.ModelInput.from_ints(tokens)

    def get_prompt_tokens(self, problem: str) -> List[int]:
        """获取prompt的token列表"""
        model_input = self.format_prompt(problem)
        return model_input.to_ints()

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """计算组内归一化的advantages"""
        import numpy as np
        rewards_arr = np.array(rewards)
        mean = np.mean(rewards_arr)
        advantages = rewards_arr - mean
        return advantages.tolist()

    def get_stop_sequences(self) -> List[int]:
        """
        获取停止序列

        如果有renderer，使用renderer.get_stop_sequences()
        否则回退到手动指定
        """
        if self.renderer is not None:
            # 使用renderer的stop sequences (返回token IDs)
            return self.renderer.get_stop_sequences()

        # 回退: base model需要手动指定
        if is_base_model(self.config.model_name):
            # 返回字符串，让SamplingParams处理
            return [
                "\n\nProblem:",
                "\nProblem:",
                "\n\n\n",
                "<|im_end|>",
                "<|im_start|>",
                "<|endoftext|>",
            ]
        return []

    def parse_response(self, tokens: List[int]) -> Dict[str, Any]:
        """
        解析模型响应

        如果有renderer，使用renderer.parse_response来正确提取thinking块
        否则直接decode
        """
        if self.renderer is not None:
            # 使用renderer解析 (会正确处理<think>块)
            parsed_message, success = self.renderer.parse_response(tokens)
            if success and parsed_message:
                content = parsed_message.get('content', '')
                # 检查是否有thinking字段
                thinking = parsed_message.get('thinking', None)
                return {
                    "content": content,
                    "thinking": thinking,
                    "full_text": f"<think>{thinking}</think>\n{content}" if thinking else content,
                    "parse_success": success,
                }
            else:
                # 解析失败，回退到直接decode
                text = self.tokenizer.decode(tokens, skip_special_tokens=False)
                return {
                    "content": text,
                    "thinking": None,
                    "full_text": text,
                    "parse_success": False,
                }
        else:
            # 无renderer，直接decode
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return {
                "content": text,
                "thinking": None,
                "full_text": text,
                "parse_success": True,
            }

    def train_step(
        self,
        problems: List[str],
        gold_answers: List[str],
        sampling_client: Any,
    ) -> Dict[str, float]:
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
        prompt_lengths = []
        for problem in problems:
            # 使用新的format_prompt方法获取ModelInput
            model_input = self.format_prompt(problem)
            prompt_tokens = model_input.to_ints()
            prompt_length = len(prompt_tokens)
            prompt_lengths.append(prompt_length)

            future = sampling_client.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=self.config.rollout_n,
            )
            futures.append(future)

        # 统一等待并处理结果
        for future, prompt_length in zip(futures, prompt_lengths):
            result = future.result()

            samples = []
            for seq in result.sequences:
                tokens = list(seq.tokens) if hasattr(seq.tokens, '__iter__') else seq.tokens
                logprobs = list(seq.logprobs) if seq.logprobs and hasattr(seq.logprobs, '__iter__') else [0.0] * len(tokens)

                # 使用parse_response解析输出
                parsed = self.parse_response(tokens)

                samples.append({
                    "text": parsed["full_text"],  # 用于验证
                    "content": parsed["content"],
                    "thinking": parsed["thinking"],
                    "tokens": tokens,
                    "logprobs": logprobs,
                    "prompt_length": prompt_length,
                })
            all_samples.append(samples)

        # 计算奖励
        all_rewards = []
        correct_count = 0
        total_count = 0

        for samples, gold in zip(all_samples, gold_answers):
            rewards = []
            for sample in samples:
                # 传递 tokens 以检查 thinking 格式
                result = self.verifier.verify(
                    sample["text"],
                    gold,
                    tokens=sample.get("tokens"),
                    check_format=self.config.reasoning_mode,
                )
                sample["reward"] = result["reward"]
                sample["has_thinking"] = result.get("has_thinking", False)
                rewards.append(result["reward"])
                if result["is_correct"]:
                    correct_count += 1
                total_count += 1
            all_rewards.append(rewards)

        # 计算advantages
        for samples, rewards in zip(all_samples, all_rewards):
            advantages = self.compute_advantages(rewards)
            for sample, adv in zip(samples, advantages):
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

            fwd_bwd_future = self.training_client.forward_backward(
                data=data,
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": self.config.clip_ratio_low,
                    "clip_high_threshold": self.config.clip_ratio_high,
                }
            )
            fwd_bwd_result = fwd_bwd_future.result()

            self.training_client.optim_step(
                tinker.AdamParams(learning_rate=self.config.learning_rate)
            )

        # 统计
        import numpy as np
        flat_rewards = [r for rewards in all_rewards for r in rewards]

        # 统计 thinking 格式使用率
        thinking_count = 0
        for samples in all_samples:
            for sample in samples:
                if sample.get("has_thinking", False):
                    thinking_count += 1

        stats = {
            "step": self.global_step,
            "mean_reward": np.mean(flat_rewards),
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "thinking_rate": thinking_count / total_count if total_count > 0 else 0,
            "num_train_samples": len(train_samples),
            "step_time": time.time() - step_start,
        }

        for key, value in stats.items():
            self.history[key].append(value)

        return stats

    def evaluate(
        self,
        problems: List[str],
        gold_answers: List[str],
        sampling_client: Any,
    ) -> Dict[str, Any]:
        """评估模型"""
        import tinker

        # 评估时也使用stop sequences
        stop_seqs = self.get_stop_sequences()
        eval_sampling_params = SamplingParams(
            max_tokens=self.config.max_response_length,
            temperature=0.0,
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

        for i, (future, gold) in enumerate(zip(futures, gold_answers)):
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
    samples: List[Dict],
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


def save_eval_samples(samples: List[Dict], filepath: Path, step: int):
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
        print(f"  (Reasoning模式输出更长，成本较高)")
    print("=" * 60)

    # 加载数据
    train_data = load_math_dataset("train")
    eval_data = load_math_dataset("test", max_samples=config.eval_samples)

    # 初始化验证器（使用配置的格式奖励权重）
    verifier = MathReasoningVerifier(format_reward_weight=config.format_reward_weight)

    if args.dry_run:
        print("\n[Dry Run Mode] 跳过Tinker API调用")

        is_qwen3 = "qwen3" in config.model_name.lower()
        tokenizer = None
        renderer = None

        # 优先使用tinker_cookbook的renderer
        if HAS_TINKER_COOKBOOK and is_qwen3:
            try:
                print(f"\n使用Tinker Cookbook加载: {config.model_name}")
                tokenizer = tinker_tokenizer_utils.get_tokenizer(config.model_name)

                if config.reasoning_mode:
                    renderer_name = 'qwen3'  # enable_thinking=True
                    print("使用 qwen3 renderer (thinking mode enabled)")
                else:
                    renderer_name = 'qwen3_disable_thinking'
                    print("使用 qwen3_disable_thinking renderer")

                renderer = tinker_renderers.get_renderer(renderer_name, tokenizer)
                print("Tokenizer和Renderer加载完成")
            except Exception as e:
                print(f"Warning: Tinker Cookbook加载失败: {e}")
                print("回退到HuggingFace tokenizer...")

        # 回退到HuggingFace
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
                print(f"\n使用HuggingFace加载tokenizer: {config.model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_name,
                    trust_remote_code=True,
                )
                print("Tokenizer加载完成 (无renderer)")
            except Exception as e:
                print(f"Tokenizer加载失败: {e}")
                tokenizer = None

        # 测试prompt格式
        sample_problem = train_data[0]["problem"] if train_data else "What is 2 + 3?"
        print("\n" + "=" * 60)
        print("示例Prompt格式")
        print("=" * 60)
        print(f"问题: {sample_problem[:100]}...")
        print("-" * 60)

        # 构建消息
        system_msg = "You are a helpful mathematical assistant. Solve problems step by step and put your final answer in \\boxed{}."
        user_msg = f"Solve the following math problem.\n\nProblem: {sample_problem}"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        if renderer is not None:
            # 使用renderer构建prompt
            model_input = renderer.build_generation_prompt(messages)
            tokens = model_input.to_ints()
            prompt_text = tokenizer.decode(tokens)
            print(prompt_text)
            print("-" * 60)
            print(f"\nPrompt token数: {len(tokens)}")

            # 显示stop sequences
            stop_seqs = renderer.get_stop_sequences()
            print(f"Stop sequences (token IDs): {stop_seqs}")
        else:
            # 旧方式
            prompt = format_prompt_with_tokenizer(
                sample_problem,
                tokenizer,
                config.reasoning_mode,
                config.model_name,
            )
            print(prompt)
            print("-" * 60)
            if tokenizer:
                tokens = tokenizer.encode(prompt)
                print(f"\nPrompt token数: {len(tokens)}")

        print("\n注意:")
        if renderer is not None:
            print("  - 使用 Tinker Cookbook Renderer")
            if config.reasoning_mode:
                print("  - Renderer: qwen3 (enable_thinking=True)")
                print("  - 模型将自动生成 <think>...</think> 格式")
                print("  - 使用 renderer.parse_response() 解析输出")
            else:
                print("  - Renderer: qwen3_disable_thinking")
        elif is_base_model(config.model_name):
            print("  - 使用 Few-shot 格式 (Base Model)")
        else:
            print("  - 使用 HuggingFace Chat Template 格式")
        print("=" * 60)
        return

    # 初始化Tinker
    print("\n正在连接Tinker服务...")
    service_client = tinker.ServiceClient()

    print(f"正在加载模型: {config.model_name}")
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )
    print("模型加载完成")

    # 从 checkpoint 恢复（如果指定）
    if args.checkpoint:
        print(f"\n从 checkpoint 恢复: {args.checkpoint}")
        try:
            training_client.load_state(args.checkpoint)
            print(f"Checkpoint 加载成功: {args.checkpoint}")
        except Exception as e:
            print(f"Warning: 无法加载 checkpoint: {e}")
            print("继续使用基座模型...")

    # 创建训练器
    trainer = ReasoningTrainer(config, training_client, verifier)

    # 训练循环
    print("\n开始训练...")
    print("-" * 60)

    for step in range(1, config.num_steps + 1):
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
        # 在 reasoning 模式下显示 thinking rate
        if config.reasoning_mode:
            output_parts.append(f"Think: {stats['thinking_rate']:.0%}")
        output_parts.extend([
            f"Train: {stats['num_train_samples']}",
            f"Time: {stats['step_time']:.1f}s",
        ])
        print(" | ".join(output_parts))

        # 评估
        if step % config.eval_interval == 0:
            # 直接传递problems，evaluate内部会格式化
            eval_problems = [item["problem"] for item in eval_data]
            eval_answers = [item["answer"] for item in eval_data]

            eval_stats = trainer.evaluate(
                eval_problems, eval_answers, sampling_client
            )
            eval_msg = (f"  [Eval] Accuracy: {eval_stats['eval_accuracy']:.2%} "
                        f"({eval_stats['eval_correct']}/{eval_stats['eval_total']})")
            if config.reasoning_mode:
                eval_msg += f" | Think: {eval_stats['thinking_rate']:.0%}"
            print(eval_msg)

            print_eval_samples(
                eval_stats["samples"],
                num_correct=1,
                num_incorrect=2,
                max_response_len=config.max_response_length
            )

            samples_file = run_dir / f"eval_samples_step_{step}.json"
            save_eval_samples(eval_stats["samples"], samples_file, step)

        # 保存检查点
        if step % config.save_interval == 0:
            checkpoint_name = f"checkpoint_step_{step}"
            training_client.save_state(checkpoint_name)
            print(f"  [Save] Checkpoint saved: {checkpoint_name}")

            with open(run_dir / "history.json", "w") as f:
                json.dump(dict(trainer.history), f, indent=2)

    # 最终保存
    print("\n" + "=" * 60)
    print("训练完成!")
    training_client.save_state("final_model")

    # 最终评估
    sampling_client = training_client.save_weights_and_get_sampling_client(name="final")
    eval_problems = [item["problem"] for item in eval_data]
    eval_answers = [item["answer"] for item in eval_data]
    final_eval = trainer.evaluate(
        eval_problems, eval_answers, sampling_client
    )

    final_msg = f"最终评估准确率: {final_eval['eval_accuracy']:.2%}"
    if config.reasoning_mode:
        final_msg += f" | Thinking率: {final_eval['thinking_rate']:.0%}"
    print(final_msg)
    print_eval_samples(final_eval["samples"], num_correct=3, num_incorrect=3,
                       max_response_len=config.max_response_length)

    save_eval_samples(final_eval["samples"], run_dir / "eval_samples_final.json", config.num_steps)
    print(f"输出目录: {run_dir}")
    print("=" * 60)

    # 保存最终结果
    results = {
        "config": asdict(config),
        "final_eval": {
            "accuracy": final_eval["eval_accuracy"],
            "correct": final_eval["eval_correct"],
            "total": final_eval["eval_total"],
        },
        "training_summary": {
            "total_steps": trainer.global_step,
            "final_train_accuracy": trainer.history["accuracy"][-1] if trainer.history["accuracy"] else 0,
        },
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

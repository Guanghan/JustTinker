"""
Prompt 格式化工具

提供统一的 prompt 格式化函数，支持多种模型和模式。

Author: Guanghan Ning
Date: 2025-01-10
Updated: 2026-01-13 (从训练脚本提取)
"""

from typing import Any, Optional


def is_base_model(model_name: str) -> bool:
    """
    判断是否是 base model（非 instruct/chat）

    Args:
        model_name: 模型名称

    Returns:
        是否是 base model
    """
    model_lower = model_name.lower()
    # 包含这些关键词的是 instruct 模型
    instruct_keywords = ["instruct", "chat", "it", "rlhf"]
    for kw in instruct_keywords:
        if kw in model_lower:
            return False
    # 显式标注为 Base 的
    if "base" in model_lower:
        return True
    # Llama-3.2-1B, Llama-3.2-3B 等没有后缀的是 base model
    return bool("llama" in model_lower and not any(kw in model_lower for kw in instruct_keywords))


def format_prompt_for_base_model(problem: str, reasoning_mode: bool = False) -> str:
    """
    为 Base Model 设计的 prompt 格式

    Base models 需要 few-shot 示例来学习输出格式，
    而不是依赖 instruction following 能力。

    Args:
        problem: 数学问题
        reasoning_mode: 是否启用 reasoning 模式

    Returns:
        格式化后的 prompt
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


def format_prompt_simple(
    problem: str,
    reasoning_mode: bool = False,
    model_name: str = "",
) -> str:
    """
    简单的 prompt 格式（不使用 chat template）

    Args:
        problem: 数学问题
        reasoning_mode: 是否启用 reasoning 模式
        model_name: 模型名称（目前未使用）

    Returns:
        格式化后的 prompt
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


def format_prompt_with_tokenizer(
    problem: str,
    tokenizer: Any,
    reasoning_mode: bool = False,
    model_name: str = "",
) -> str:
    """
    使用 tokenizer 的 chat template 格式化 prompt

    Args:
        problem: 数学问题
        tokenizer: 模型的 tokenizer
        reasoning_mode: 是否启用 thinking mode
        model_name: 模型名称

    Returns:
        格式化后的 prompt
    """
    # Base model 使用不同的 prompt 格式（few-shot completion）
    if is_base_model(model_name):
        return format_prompt_for_base_model(problem, reasoning_mode)

    if tokenizer is None:
        # fallback 到简单格式
        return format_prompt_simple(problem, reasoning_mode, model_name)

    # 构建消息（Instruct models）
    is_qwen3 = "Qwen3" in model_name or "qwen3" in model_name.lower()

    if reasoning_mode:
        if is_qwen3:
            # Qwen3 thinking mode:
            # - 不要在 system prompt 中提及 <think> 格式，让模型自然生成
            # - 使用 /think 后缀触发 thinking mode
            # - Qwen3 会自动生成 <think>...</think> 格式
            system_msg = "You are a helpful mathematical assistant. Solve problems step by step and put your final answer in \\boxed{}."
            user_msg = f"""Solve the following math problem.

Problem: {problem} /think"""
        else:
            # 非 Qwen3 模型：明确要求使用 <think> 格式
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
        # Qwen3 thinking mode 需要特殊处理
        if is_qwen3 and reasoning_mode:
            # 尝试使用 enable_thinking 参数
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,  # Qwen3 thinking mode
                )
            except TypeError:
                # 如果 tokenizer 不支持 enable_thinking 参数
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

        # 对于非 Qwen3 模型，在 reasoning 模式下手动添加 <think> 前缀
        if reasoning_mode and not is_qwen3:
            text += "<think>\n"

        return text
    except Exception:
        return format_prompt_simple(problem, reasoning_mode, model_name)


def format_prompt_manual_template(
    problem: str,
    system_msg: str | None = None,
) -> str:
    """
    手动构建 chat template（与 SFT 训练完全一致）

    重要：此函数用于 RL 训练，确保与 SFT 训练时的 prompt 格式完全一致。
    这是避免 thinking token 无法正确触发的关键。

    Args:
        problem: 数学问题
        system_msg: 系统消息（可选）

    Returns:
        格式化后的 prompt
    """
    if system_msg is None:
        system_msg = "You are a helpful mathematical assistant. Think step by step before answering."

    user_msg = f"Solve the following math problem.\n\nProblem: {problem}"

    # 使用与 SFT 训练完全相同的手动模板
    # 注意：assistant 后面只有一个换行，没有额外空格！
    prompt_text = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
    return prompt_text


# 保持向后兼容
def format_prompt(
    problem: str,
    reasoning_mode: bool = False,
    model_name: str = "",
) -> str:
    """简单格式（向后兼容）"""
    return format_prompt_simple(problem, reasoning_mode, model_name)

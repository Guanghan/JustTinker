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
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

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
# ============================================================
from src.configs import ReasoningConfig
from src.data import load_aime_dataset, load_dapo_math_dataset, load_math_dataset
from src.evaluation import MathVerifier
from src.prompts import is_base_model

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
        verifier: MathVerifier,
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

        直接 decode tokens，thinking 格式检测由 MathVerifier 处理
        """
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)

        # 简单提取 thinking 内容（如果有）
        thinking = None
        content = text
        if "<think>" in text and "</think>" in text:
            match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                content = text[match.end() :].strip()

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
                response_tokens = list(seq.tokens) if hasattr(seq.tokens, "__iter__") else seq.tokens
                response_logprobs = (
                    list(seq.logprobs)
                    if seq.logprobs and hasattr(seq.logprobs, "__iter__")
                    else [0.0] * len(response_tokens)
                )

                # 完整序列 = prompt + response（用于训练）
                full_tokens = list(prompt_tokens) + list(response_tokens)
                full_logprobs = [0.0] * prompt_length + list(response_logprobs)

                # 使用parse_response解析输出（只解析response部分）
                parsed = self.parse_response(response_tokens)

                samples.append(
                    {
                        "text": parsed["full_text"],  # 用于验证
                        "content": parsed["content"],
                        "thinking": parsed["thinking"],
                        "tokens": full_tokens,  # 完整序列：prompt + response
                        "logprobs": full_logprobs,
                        "prompt_length": prompt_length,
                    }
                )
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
                target_tokens = target_tokens[: len(input_tokens)]

                padded_logprobs = [0.0] * ob_len + logprobs[ob_len:]
                padded_logprobs = padded_logprobs[: len(input_tokens)]

                padded_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)

                model_input = tinker.ModelInput.from_ints(input_tokens)

                datum = tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                        "logprobs": tinker.TensorData.from_torch(torch.tensor(padded_logprobs, dtype=torch.float32)),
                        "advantages": tinker.TensorData.from_torch(
                            torch.tensor(padded_advantages, dtype=torch.float32)
                        ),
                    },
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
                },
            )
            fwd_bwd_future.result()

            self.training_client.optim_step(tinker.AdamParams(learning_rate=self.config.learning_rate))

        # 统计
        import numpy as np

        flat_rewards = [r for rewards in all_rewards for r in rewards]

        # 统计 thinking 格式使用率和长度
        thinking_count = 0
        total_response_tokens = 0
        total_thinking_tokens = 0

        # Thinking token IDs (Qwen3)
        think_start_id = 151667  # <think>
        think_end_id = 151668  # </think>

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
                tokens = list(seq.tokens) if hasattr(seq.tokens, "__iter__") else seq.tokens
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

            eval_samples.append(
                {
                    "index": i,
                    "problem": problems[i],
                    "gold_answer": gold,
                    "response": response_text,
                    "thinking": thinking,  # 提取的thinking块
                    "has_thinking": has_thinking,  # 是否有</think> token
                    "extracted_answer": result.get("extracted"),
                    "is_correct": is_correct,
                }
            )

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
        print(f"\n  --- {label} #{idx + 1} ---")
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
    parser.add_argument("--scale", type=str, default="quick", choices=["quick", "medium", "full"], help="实验规模")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="模型名称 (Qwen无需授权; Llama需HuggingFace授权)",
    )
    parser.add_argument("--reasoning", action="store_true", help="启用reasoning/thinking mode")
    parser.add_argument(
        "--thinking-budget", type=str, default="medium", choices=["low", "medium", "high"], help="Thinking token预算"
    )
    parser.add_argument(
        "--format-reward", type=float, default=0.1, help="格式奖励权重：正确但无thinking时的惩罚 (默认0.1)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="从指定的 checkpoint 继续训练 (如 coldstart_sft_final)"
    )
    parser.add_argument("--eval-only", action="store_true", help="只进行评估，不训练 (需要配合 --checkpoint 使用)")
    parser.add_argument("--eval-temperature", type=float, default=0.7, help="评估时的采样温度 (默认 0.7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/justrl_reasoning")
    parser.add_argument("--dry-run", action="store_true", help="干运行模式")
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
    avg_response_len = (
        config.max_response_length // 2 if not config.reasoning_mode else config.max_response_length * 0.7
    )
    tokens_per_step = config.batch_size * config.rollout_n * (config.max_prompt_length + avg_response_len)
    total_tokens = config.num_steps * tokens_per_step / 1e6  # 百万
    estimated_cost = total_tokens * price_per_m_tokens * 2  # Sample + Train

    print(f"\n预估Token消耗: {total_tokens:.1f}M tokens")
    print(f"模型定价: ${price_per_m_tokens}/M tokens")
    print(f"预估成本: ~${estimated_cost:.0f}")
    if config.reasoning_mode:
        print("  (Reasoning模式输出更长，成本较高)")
    print("=" * 60)

    # 加载训练数据（使用 src.data 模块）
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

    # 初始化验证器（使用 src.evaluation.MathVerifier）
    verifier = MathVerifier(
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
            step_match = re.search(r"checkpoint_step_(\d+)", args.checkpoint)
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
        sampling_client = training_client.save_weights_and_get_sampling_client(name="eval_only_temp")

        # 评估 MATH
        eval_problems = [item["problem"] for item in eval_data]
        eval_answers = [item["answer"] for item in eval_data]

        print(f"\n评估 MATH ({len(eval_problems)} 样本)...")
        # 需要临时调整 trainer 的 eval temperature
        eval_stats = trainer.evaluate(eval_problems, eval_answers, sampling_client)
        print(
            f"  MATH 准确率: {eval_stats['eval_accuracy']:.2%} ({eval_stats['eval_correct']}/{eval_stats['eval_total']})"
        )
        print(f"  Thinking Rate: {eval_stats['thinking_rate']:.2%}")

        # 评估 AIME (如果有)
        if "aime-2024" in eval_data_dict and eval_data_dict["aime-2024"]:
            aime_data = eval_data_dict["aime-2024"]
            aime_problems = [item["problem"] for item in aime_data]
            aime_answers = [item["answer"] for item in aime_data]

            print(f"\n评估 AIME 2024 ({len(aime_problems)} 样本)...")
            aime_stats = trainer.evaluate(aime_problems, aime_answers, sampling_client)
            print(
                f"  AIME 准确率: {aime_stats['eval_accuracy']:.2%} ({aime_stats['eval_correct']}/{aime_stats['eval_total']})"
            )
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

        sampling_client = training_client.save_weights_and_get_sampling_client(name=f"step_{step}")

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
        output_parts.extend(
            [
                f"Train: {stats['num_train_samples']}/{stats['total_samples']}",
                f"Time: {stats['step_time']:.1f}s",
            ]
        )
        print(" | ".join(output_parts))

        # 训练健康监控
        if config.reasoning_mode:
            avg_resp_len = stats.get("avg_response_length", 0)
            thinking_rate = stats.get("thinking_rate", 1.0)
            avg_redundancy = stats.get("avg_redundancy_score", 0)
            avg_chunk_sim = stats.get("avg_chunk_similarity", 0)

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
                print(
                    f"  [WARNING] Suspicious pattern: high chunk_sim ({avg_chunk_sim:.1%}) but low redundancy ({avg_redundancy:.1%})"
                )
            elif avg_chunk_sim > 0.6:
                print(f"  [WARNING] High chunk similarity: {avg_chunk_sim:.1%} (threshold: 60%)")

        # 评估（支持多数据集）
        if step % config.eval_interval == 0:
            # 主评估数据集（MATH）
            eval_problems = [item["problem"] for item in eval_data]
            eval_answers = [item["answer"] for item in eval_data]

            eval_stats = trainer.evaluate(eval_problems, eval_answers, sampling_client)
            eval_msg = (
                f"  [Eval MATH] Accuracy: {eval_stats['eval_accuracy']:.2%} "
                f"({eval_stats['eval_correct']}/{eval_stats['eval_total']})"
            )
            if config.reasoning_mode:
                eval_msg += f" | Think: {eval_stats['thinking_rate']:.0%}"
            print(eval_msg)

            # 记录 eval 指标到 history（用于绘图）
            trainer.history["eval_step"].append(step)
            trainer.history["eval_accuracy"].append(eval_stats["eval_accuracy"])
            trainer.history["eval_thinking_rate"].append(eval_stats.get("thinking_rate", 0))

            print_eval_samples(
                eval_stats["samples"], num_correct=1, num_incorrect=2, max_response_len=config.max_response_length
            )

            samples_file = run_dir / f"eval_samples_step_{step}.json"
            save_eval_samples(eval_stats["samples"], samples_file, step)

            # AIME 2024 评估（如果配置了）
            if "aime-2024" in eval_data_dict and eval_data_dict["aime-2024"]:
                aime_data = eval_data_dict["aime-2024"]
                aime_problems = [item["problem"] for item in aime_data]
                aime_answers = [item["answer"] for item in aime_data]

                aime_stats = trainer.evaluate(aime_problems, aime_answers, sampling_client)
                aime_msg = (
                    f"  [Eval AIME] Accuracy: {aime_stats['eval_accuracy']:.2%} "
                    f"({aime_stats['eval_correct']}/{aime_stats['eval_total']})"
                )
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
                    print(
                        f"  [Early Stop] Accuracy dropped: {current_accuracy:.2%} < {best_eval_accuracy:.2%} - {config.early_stopping_threshold:.0%}"
                    )
                    print(f"               Counter: {early_stop_counter}/{config.early_stopping_patience}")

                    if early_stop_counter >= config.early_stopping_patience:
                        print(f"\n{'=' * 60}")
                        print(f"早停触发！连续 {early_stop_counter} 次评估准确率下降")
                        print(f"最佳 Eval Accuracy: {best_eval_accuracy:.2%}")
                        print(f"当前 Eval Accuracy: {current_accuracy:.2%}")
                        print(f"{'=' * 60}")
                        should_stop = True

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
    final_eval = trainer.evaluate(eval_problems, eval_answers, sampling_client)

    print("\n--- MATH 最终评估 ---")
    final_msg = f"MATH 准确率: {final_eval['eval_accuracy']:.2%}"
    if config.reasoning_mode:
        final_msg += f" | Thinking率: {final_eval['thinking_rate']:.0%}"
    print(final_msg)
    print_eval_samples(
        final_eval["samples"], num_correct=2, num_incorrect=2, max_response_len=config.max_response_length
    )
    save_eval_samples(final_eval["samples"], run_dir / "eval_samples_final.json", config.num_steps)

    # AIME 2024 最终评估
    final_aime_eval = None
    if "aime-2024" in eval_data_dict and eval_data_dict["aime-2024"]:
        aime_data = eval_data_dict["aime-2024"]
        aime_problems = [item["problem"] for item in aime_data]
        aime_answers = [item["answer"] for item in aime_data]
        final_aime_eval = trainer.evaluate(aime_problems, aime_answers, sampling_client)

        print("\n--- AIME 2024 最终评估 ---")
        aime_msg = f"AIME 准确率: {final_aime_eval['eval_accuracy']:.2%} ({final_aime_eval['eval_correct']}/30)"
        if config.reasoning_mode:
            aime_msg += f" | Thinking率: {final_aime_eval['thinking_rate']:.0%}"
        print(aime_msg)
        print_eval_samples(
            final_aime_eval["samples"], num_correct=1, num_incorrect=2, max_response_len=config.max_response_length
        )
        save_eval_samples(final_aime_eval["samples"], run_dir / "eval_aime_samples_final.json", config.num_steps)

    print(f"\n输出目录: {run_dir}")
    print("=" * 60)

    # 保存最终结果
    results = {
        "config": asdict(config),
        "final_eval": {
            "math_accuracy": final_eval["eval_accuracy"],
            "math_thinking_rate": final_eval.get("thinking_rate", 0),
        },
        "training_summary": {
            "total_steps": trainer.global_step,
            "early_stopped": should_stop,
            "best_eval_accuracy": best_eval_accuracy,
        },
    }

    if final_aime_eval:
        results["final_eval"]["aime_accuracy"] = final_aime_eval["eval_accuracy"]
        results["final_eval"]["aime_thinking_rate"] = final_aime_eval.get("thinking_rate", 0)

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(run_dir / "history.json", "w") as f:
        json.dump(dict(trainer.history), f, indent=2)


if __name__ == "__main__":
    main()

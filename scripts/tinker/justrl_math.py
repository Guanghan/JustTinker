#!/usr/bin/env python3
"""
JustRL数学推理训练 - Tinker平台

基于论文 "JustRL: Simplicity at Scale" 的简化RLVR训练实现
使用Tinker API进行LoRA微调

核心原则（来自JustRL）：
- 单阶段训练，固定超参数
- 无KL惩罚，无长度惩罚
- clip-higher机制稳定训练
- 二元奖励（正确=1，错误=0）

使用方法:
    # 设置API Key
    export TINKER_API_KEY=your_api_key

    # 快速验证（~$50-80）
    python scripts/tinker/justrl_math.py --scale quick

    # 学习实验（~$800）
    python scripts/tinker/justrl_math.py --scale medium

    # 深入实验（~$1600）
    python scripts/tinker/justrl_math.py --scale full

Author: Guanghan Ning
Date: 2025-12-24
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

# 添加项目根目录到path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 配置
# ============================================================

@dataclass
class JustRLConfig:
    """JustRL训练配置"""

    # 实验设置
    experiment_name: str = "justrl_gsm8k"
    scale: str = "quick"  # quick, medium, full
    seed: int = 42

    # 模型设置
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"  # 推荐：指令模型，初始正确率高
    lora_rank: int = 64

    # 训练设置（根据scale调整）
    num_steps: int = 200
    batch_size: int = 32
    rollout_n: int = 8

    # JustRL核心参数（固定，不需要调整）
    learning_rate: float = 1e-6
    clip_ratio_low: float = 0.8
    clip_ratio_high: float = 1.28
    kl_coef: float = 0.0  # 无KL惩罚
    temperature: float = 1.0

    # 生成设置
    max_prompt_length: int = 512
    max_response_length: int = 4096  # Qwen3支持16K，但4K对数学任务足够

    # 评估设置
    eval_interval: int = 50
    eval_samples: int = 100
    save_interval: int = 100

    # 输出设置
    output_dir: str = "outputs/justrl"

    def __post_init__(self):
        """根据scale调整参数"""
        scale_configs = {
            "quick": {
                "num_steps": 2, #200,
                "batch_size": 2, #32,
                "eval_interval": 2, #50,
                "save_interval": 2, #100,
            },
            "medium": {
                "num_steps": 1000,
                "batch_size": 64,
                "eval_interval": 100,
                "save_interval": 200,
            },
            "full": {
                "num_steps": 2000,
                "batch_size": 64,
                "eval_interval": 200,
                "save_interval": 500,
            },
        }

        if self.scale in scale_configs:
            for key, value in scale_configs[self.scale].items():
                setattr(self, key, value)


# ============================================================
# 数据加载
# ============================================================

def load_gsm8k(split: str = "train", max_samples: Optional[int] = None) -> List[Dict]:
    """加载GSM8K数据集"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 请安装datasets库: pip install datasets")
        sys.exit(1)

    print(f"加载GSM8K {split}集...")
    dataset = load_dataset("gsm8k", "main", split=split)

    samples = []
    for item in dataset:
        # 提取答案（####后面的数字）
        answer_text = item["answer"]
        parts = answer_text.split("####")
        final_answer = parts[-1].strip().replace(",", "") if len(parts) > 1 else answer_text

        samples.append({
            "question": item["question"],
            "answer": final_answer,
            "solution": parts[0].strip() if len(parts) > 1 else None,
        })

        if max_samples and len(samples) >= max_samples:
            break

    print(f"  加载了 {len(samples)} 个样本")
    return samples


def format_prompt(question: str) -> str:
    """格式化数学问题的prompt"""
    return f"""Solve the following math problem step by step. Show your reasoning clearly.
At the end, provide your final answer after "The answer is: ".

Problem: {question}

Solution:"""


# ============================================================
# 验证器
# ============================================================

class MathVerifier:
    """简化的数学答案验证器"""

    def __init__(self):
        import re
        self.re = re

        self.patterns = [
            r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"\\boxed\{([^}]+)\}",
            r"=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
        ]

    def extract_answer(self, text: str) -> Optional[str]:
        """提取答案"""
        for pattern in self.patterns:
            match = self.re.search(pattern, text, self.re.MULTILINE)
            if match:
                answer = match.group(1).replace(",", "").strip()
                return answer
        return None

    def verify(self, response: str, gold: str) -> Dict:
        """验证答案"""
        extracted = self.extract_answer(response)
        gold = gold.replace(",", "").strip()

        if extracted is None:
            return {"is_correct": False, "reward": 0.0, "extracted": None}

        # 数值比较
        try:
            is_correct = abs(float(extracted) - float(gold)) < 1e-6
        except ValueError:
            is_correct = extracted == gold

        return {
            "is_correct": is_correct,
            "reward": 1.0 if is_correct else 0.0,
            "extracted": extracted,
        }


# ============================================================
# GRPO训练器
# ============================================================

class JustRLTrainer:
    """
    JustRL训练器

    实现简化的GRPO算法，适配Tinker平台
    """

    def __init__(
        self,
        config: JustRLConfig,
        training_client: Any,
        verifier: MathVerifier,
    ):
        self.config = config
        self.training_client = training_client
        self.verifier = verifier

        self.global_step = 0
        self.history = defaultdict(list)

        # 初始化tokenizer用于批量采样
        # 批量采样需要将字符串转为ModelInput对象
        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        """初始化tokenizer用于批量采样"""
        try:
            from transformers import AutoTokenizer
            print(f"加载tokenizer: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            print("Tokenizer加载完成")
        except Exception as e:
            print(f"Warning: 无法加载tokenizer: {e}")
            print("批量采样将被禁用，使用循环方式")
            self.tokenizer = None

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        计算组内归一化的advantages

        GRPO核心: advantage = reward - group_mean
        注意：JustRL论文发现std normalization有害，只使用mean-centering
        """
        import numpy as np

        rewards_arr = np.array(rewards)
        mean = np.mean(rewards_arr)

        # JustRL: 只减去均值，不除以std
        advantages = rewards_arr - mean
        return advantages.tolist()

    def train_step(
        self,
        prompts: List[str],
        gold_answers: List[str],
        sampling_client: Any,
    ) -> Dict[str, float]:
        """
        执行一个训练步骤

        1. 为每个prompt采样rollout_n个response
        2. 计算奖励
        3. 计算advantages
        4. 用positive advantage样本更新模型
        """
        self.global_step += 1
        step_start = time.time()

        # 1. 采样responses（GRPO需要保存tokens和logprobs）
        # 创建采样参数
        sampling_params = SamplingParams(
            max_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
        )

        # 存储采样结果（每个prompt的多个rollouts）
        # 每个sample包含: text, tokens, logprobs, prompt_length
        all_samples = []  # List[List[Dict]]

        # 并发采样（优化：先发送所有请求，再统一等待）
        import tinker

        # 第一步：并发发送所有采样请求
        futures = []
        prompt_lengths = []
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(prompt)
            prompt_length = len(prompt_tokens)
            prompt_lengths.append(prompt_length)
            model_input = tinker.ModelInput.from_ints(prompt_tokens)

            future = sampling_client.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=self.config.rollout_n,
            )
            futures.append(future)

        # 第二步：统一等待并处理结果
        for future, prompt_length in zip(futures, prompt_lengths):
            result = future.result()

            # 保存每个rollout的完整信息
            samples = []
            for seq in result.sequences:
                # seq包含：tokens（完整序列），logprobs（每个token的logprob）
                # 转换为Python list以确保兼容性
                tokens = list(seq.tokens) if hasattr(seq.tokens, '__iter__') else seq.tokens
                logprobs = list(seq.logprobs) if seq.logprobs and hasattr(seq.logprobs, '__iter__') else [0.0] * len(tokens)

                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                samples.append({
                    "text": text,
                    "tokens": tokens,  # Python list
                    "logprobs": logprobs,  # Python list
                    "prompt_length": prompt_length,
                })
            all_samples.append(samples)

        # 2. 计算奖励（对每个group）
        all_rewards = []
        correct_count = 0
        total_count = 0

        for samples, gold in zip(all_samples, gold_answers):
            rewards = []
            for sample in samples:
                result = self.verifier.verify(sample["text"], gold)
                sample["reward"] = result["reward"]  # 保存reward到sample中
                rewards.append(result["reward"])
                if result["is_correct"]:
                    correct_count += 1
                total_count += 1
            all_rewards.append(rewards)

        # 3. 计算group-relative advantages
        for samples, rewards in zip(all_samples, all_rewards):
            advantages = self.compute_advantages(rewards)
            # 将advantage保存到每个sample中
            for sample, adv in zip(samples, advantages):
                sample["advantage"] = adv

        # 4. 收集positive advantage样本（JustRL核心：只训练positive samples）
        train_samples = []
        for samples in all_samples:
            for sample in samples:
                if sample["advantage"] > 0:
                    train_samples.append(sample)

        # 5. 执行梯度更新（GRPO/PPO）
        if train_samples:
            import tinker
            import torch

            # 构建训练数据（按Tinker cookbook的正确格式）
            # PPO loss只接受3个字段：target_tokens, logprobs, advantages（不支持mask）
            data = []
            for sample in train_samples:
                # 获取完整序列的tokens和logprobs（已包含prompt）
                tokens = sample["tokens"]
                logprobs = sample["logprobs"]
                prompt_len = sample["prompt_length"]
                advantage = float(sample["advantage"])  # 确保是Python float

                # 确保tokens和logprobs长度一致
                seq_len = len(tokens)
                if len(logprobs) != seq_len:
                    # 如果logprobs长度不对，用0填充
                    logprobs = logprobs[:seq_len] + [0.0] * (seq_len - len(logprobs))

                # 按cookbook格式构建数据：
                # input = tokens[:-1]（用于前向传播）
                # target = tokens（用于计算loss，prompt部分用0填充）
                input_tokens = tokens[:-1]
                ob_len = prompt_len - 1  # observation length (prompt部分在input中的长度)

                # 按cookbook格式：用0填充prompt部分，保持对齐
                # target_tokens: [0]*ob_len + tokens[ob_len:]
                target_tokens = [0] * ob_len + tokens[ob_len:]
                # 确保长度与input_tokens一致
                target_tokens = target_tokens[:len(input_tokens)]

                # logprobs: [0.0]*ob_len + logprobs[ob_len:]
                padded_logprobs = [0.0] * ob_len + logprobs[ob_len:]
                padded_logprobs = padded_logprobs[:len(input_tokens)]

                # advantages: [0.0]*ob_len + [advantage]*(len-ob_len)
                padded_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)

                # 创建ModelInput
                model_input = tinker.ModelInput.from_ints(input_tokens)

                # 创建Datum对象（只有3个字段，不包含mask）
                # 注意：必须指定正确的数据类型
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

            # 执行forward_backward
            # GRPO使用PPO的clipped objective
            fwd_bwd_future = self.training_client.forward_backward(
                data=data,
                loss_fn="ppo",  # GRPO使用PPO loss
                loss_fn_config={
                    "clip_low_threshold": self.config.clip_ratio_low,   # 0.8
                    "clip_high_threshold": self.config.clip_ratio_high, # 1.28 (JustRL clip-higher)
                }
            )

            # 等待结果
            fwd_bwd_result = fwd_bwd_future.result()

            # 执行优化步骤
            self.training_client.optim_step(
                tinker.AdamParams(learning_rate=self.config.learning_rate)
            )

        # 统计
        import numpy as np
        flat_rewards = [r for rewards in all_rewards for r in rewards]

        stats = {
            "step": self.global_step,
            "mean_reward": np.mean(flat_rewards),
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "num_train_samples": len(train_samples),
            "step_time": time.time() - step_start,
        }

        # 保存历史
        for key, value in stats.items():
            self.history[key].append(value)

        return stats

    def evaluate(
        self,
        prompts: List[str],
        gold_answers: List[str],
        sampling_client: Any,
    ) -> Dict[str, float]:
        """评估模型（greedy decoding，并发采样）"""
        import tinker

        # 评估时使用greedy decoding
        eval_sampling_params = SamplingParams(
            max_tokens=self.config.max_response_length,
            temperature=0.0,  # Greedy
        )

        total = len(prompts)

        # 第一步：并发发送所有采样请求（非阻塞）
        futures = []
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(prompt)
            model_input = tinker.ModelInput.from_ints(prompt_tokens)

            future = sampling_client.sample(
                prompt=model_input,
                sampling_params=eval_sampling_params,
                num_samples=1,
            )
            futures.append(future)

        # 第二步：统一等待所有结果
        correct = 0
        for future, gold in zip(futures, gold_answers):
            sample_result = future.result()  # 此时可能已经完成

            # 解码response文本
            if sample_result.sequences:
                seq = sample_result.sequences[0]
                tokens = list(seq.tokens) if hasattr(seq.tokens, '__iter__') else seq.tokens
                response_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                response_text = ""

            result = self.verifier.verify(response_text, gold)
            if result["is_correct"]:
                correct += 1

        return {
            "eval_accuracy": correct / total if total > 0 else 0,
            "eval_correct": correct,
            "eval_total": total,
        }


# ============================================================
# 主训练循环
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="JustRL Math Training on Tinker")
    parser.add_argument("--scale", type=str, default="quick",
                        choices=["quick", "medium", "full"],
                        help="实验规模: quick(~$100), medium(~$1000), full(~$2000)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="模型名称（推荐Qwen3-4B指令模型）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="outputs/justrl",
                        help="输出目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="干运行模式，不实际调用Tinker API")
    args = parser.parse_args()

    # 检查API Key
    if not args.dry_run and not os.environ.get("TINKER_API_KEY"):
        print("Error: 请设置TINKER_API_KEY环境变量")
        print("  export TINKER_API_KEY=your_api_key")
        sys.exit(1)

    # 创建配置
    config = JustRLConfig(
        scale=args.scale,
        model_name=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # 设置随机种子
    random.seed(config.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{config.experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print("=" * 60)
    print("JustRL Math Training on Tinker")
    print("=" * 60)
    print(f"Scale: {config.scale}")
    print(f"Model: {config.model_name}")
    print(f"Steps: {config.num_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Rollout N: {config.rollout_n}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    # 估算成本（根据模型自动选择定价）
    model_pricing = {
        "meta-llama/Llama-3.2-1B": 0.09,
        "meta-llama/Llama-3.2-3B": 0.18,
        "Qwen/Qwen3-4B-Instruct-2507": 0.22,
        "meta-llama/Llama-3.1-8B": 0.40,
        "meta-llama/Llama-3.1-8B-Instruct": 0.40,
    }
    price_per_m_tokens = model_pricing.get(config.model_name, 0.20)  # 默认$0.20/M

    tokens_per_step = config.batch_size * config.rollout_n * (config.max_prompt_length + config.max_response_length // 2)
    total_tokens = config.num_steps * tokens_per_step / 1e6  # 百万
    estimated_cost = total_tokens * price_per_m_tokens * 2  # Sample + Train

    print(f"\n预估Token消耗: {total_tokens:.1f}M tokens")
    print(f"模型定价: ${price_per_m_tokens}/M tokens")
    print(f"预估成本: ~${estimated_cost:.0f}")
    print("=" * 60)

    # 加载数据
    train_data = load_gsm8k("train")
    eval_data = load_gsm8k("test", max_samples=config.eval_samples)

    # 初始化组件
    verifier = MathVerifier()

    if args.dry_run:
        print("\n[Dry Run Mode] 跳过Tinker API调用")
        print("配置验证通过，可以正式运行")
        return

    # 初始化Tinker
    try:
        import tinker
    except ImportError:
        print("Error: 请安装tinker: pip install tinker")
        sys.exit(1)

    print("\n正在连接Tinker服务...")
    service_client = tinker.ServiceClient()

    print(f"正在加载模型: {config.model_name}")
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )
    print("模型加载完成")

    # 创建训练器
    trainer = JustRLTrainer(config, training_client, verifier)

    # 训练循环
    print("\n开始训练...")
    print("-" * 60)

    for step in range(1, config.num_steps + 1):
        # 采样batch
        batch_indices = random.sample(range(len(train_data)), config.batch_size)
        batch = [train_data[i] for i in batch_indices]

        prompts = [format_prompt(item["question"]) for item in batch]
        gold_answers = [item["answer"] for item in batch]

        # 获取采样客户端
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"step_{step}"
        )

        # 训练步骤
        stats = trainer.train_step(prompts, gold_answers, sampling_client)

        # 打印进度
        print(f"Step {step}/{config.num_steps} | "
              f"Reward: {stats['mean_reward']:.3f} | "
              f"Acc: {stats['accuracy']:.2%} | "
              f"Train samples: {stats['num_train_samples']} | "
              f"Time: {stats['step_time']:.1f}s")

        # 评估
        if step % config.eval_interval == 0:
            eval_prompts = [format_prompt(item["question"]) for item in eval_data]
            eval_answers = [item["answer"] for item in eval_data]

            eval_stats = trainer.evaluate(eval_prompts, eval_answers, sampling_client)
            print(f"  [Eval] Accuracy: {eval_stats['eval_accuracy']:.2%} "
                  f"({eval_stats['eval_correct']}/{eval_stats['eval_total']})")

        # 保存检查点
        if step % config.save_interval == 0:
            checkpoint_name = f"checkpoint_step_{step}"
            training_client.save_state(checkpoint_name)
            print(f"  [Save] Checkpoint saved: {checkpoint_name}")

            # 保存训练历史
            with open(run_dir / "history.json", "w") as f:
                json.dump(dict(trainer.history), f, indent=2)

    # 最终保存
    print("\n" + "=" * 60)
    print("训练完成!")
    training_client.save_state("final_model")

    # 最终评估
    sampling_client = training_client.save_weights_and_get_sampling_client(name="final")
    eval_prompts = [format_prompt(item["question"]) for item in eval_data]
    eval_answers = [item["answer"] for item in eval_data]
    final_eval = trainer.evaluate(eval_prompts, eval_answers, sampling_client)

    print(f"最终评估准确率: {final_eval['eval_accuracy']:.2%}")
    print(f"输出目录: {run_dir}")
    print("=" * 60)

    # 保存最终结果
    results = {
        "config": asdict(config),
        "final_eval": final_eval,
        "training_summary": {
            "total_steps": trainer.global_step,
            "final_train_accuracy": trainer.history["accuracy"][-1] if trainer.history["accuracy"] else 0,
        },
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

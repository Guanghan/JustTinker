#!/usr/bin/env python3
"""
Cold Start SFT: 使用 OpenR1-Math-220k 数据集训练 Thinking Mode

目的：
    在 Qwen3-4B-Instruct 基础上进行 SFT，"唤醒" 模型生成 <think>...</think> 格式的能力。
    训练完成后保存 checkpoint，用于后续 RLVR 训练。

数据集：
    open-r1/OpenR1-Math-220k
    - 包含 DeepSeek-R1 生成的数学推理数据
    - 使用 <think>...</think> 格式
    - 包含多个 generations 和 correctness 验证

预算估算 ($50):
    - Qwen3-4B: ~$0.22/M tokens
    - SFT 训练约 2x token 消耗（forward + backward）
    - $50 ≈ 100M 有效 tokens ≈ 25k-50k 样本

使用方法:
    # 设置 API Key
    export TINKER_API_KEY=your_api_key

    # 快速验证
    python scripts/tinker/coldstart_sft.py --scale quick

    # 标准训练 (~$10)
    python scripts/tinker/coldstart_sft.py --scale small

    # 完整训练 (~$50)
    python scripts/tinker/coldstart_sft.py --scale medium

    # Dry run（不调用 API）
    python scripts/tinker/coldstart_sft.py --dry-run

Author: Guanghan Ning
Date: 2025-01-10
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

# Tinker cookbook imports (for renderer system)
try:
    from tinker_cookbook import renderers as tinker_renderers
    from tinker_cookbook import tokenizer_utils as tinker_tokenizer_utils
    from tinker_cookbook.supervised.data import TrainOnWhat, conversation_to_datum

    # 尝试导入正确的函数名
    try:
        from tinker_cookbook.supervised.common import datum_from_model_input_weights

        datum_from_tokens_weights = datum_from_model_input_weights  # 别名兼容
    except ImportError:
        try:
            from tinker_cookbook.supervised.common import datum_from_tokens_weights
        except ImportError:
            datum_from_tokens_weights = None
            print("Warning: datum_from_tokens_weights/datum_from_model_input_weights not found")
    HAS_TINKER_COOKBOOK = True
except ImportError as e:
    print(f"Warning: tinker_cookbook import failed: {e}")
    print("Run: pip install tinker-cookbook")
    tinker_renderers = None
    tinker_tokenizer_utils = None
    datum_from_tokens_weights = None
    conversation_to_datum = None
    TrainOnWhat = None
    HAS_TINKER_COOKBOOK = False

# 添加项目根目录到path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 从 src 模块导入公共组件
# ============================================================
from src.configs import SFTConfig
from src.data import load_openr1_dataset

# ============================================================
# SFT Trainer
# ============================================================


class SFTTrainer:
    """
    Cold Start SFT Trainer

    使用 Tinker API 进行监督微调，训练模型生成 <think>...</think> 格式
    """

    def __init__(
        self,
        config: SFTConfig,
        training_client: Any,
        run_dir: Path,
    ):
        self.config = config
        self.training_client = training_client
        self.run_dir = run_dir

        self.global_step = 0
        self.accumulated_loss = 0.0
        self.accumulation_count = 0

        # 训练历史 (每个 optimizer step 记录)
        self.history = defaultdict(list)
        # 评估历史 (每次 eval 记录)
        self.eval_history = defaultdict(list)

        self.tokenizer = None
        self.renderer = None
        self._init_tokenizer_and_renderer()

    def _init_tokenizer_and_renderer(self):
        """初始化 tokenizer 和 renderer"""
        model_name = self.config.model_name

        if HAS_TINKER_COOKBOOK:
            try:
                print(f"使用 Tinker Cookbook 加载 tokenizer: {model_name}")
                self.tokenizer = tinker_tokenizer_utils.get_tokenizer(model_name)
                # SFT 时使用 qwen3 renderer (enable_thinking=True)
                # 这样生成的 prompt 末尾会有 <think>
                self.renderer = tinker_renderers.get_renderer("qwen3", self.tokenizer)
                print("Tokenizer 和 Renderer 加载完成 (thinking mode)")
                return
            except Exception as e:
                print(f"Warning: Tinker Cookbook 加载失败: {e}")

        # 回退到 HuggingFace
        try:
            from transformers import AutoTokenizer

            print(f"使用 HuggingFace 加载 tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self.renderer = None
            print("Tokenizer 加载完成 (无 renderer)")
        except Exception as e:
            print(f"Error: 无法加载 tokenizer: {e}")
            sys.exit(1)

    def create_sft_datum(
        self,
        problem: str,
        response: str,
    ) -> tinker.Datum | None:
        """
        使用 tinker_cookbook 创建 SFT Datum

        Args:
            problem: 数学问题
            response: 包含 <think>...</think> 的回答

        Returns:
            Tinker Datum 对象，或 None（如果样本无效）
        """
        # 构建完整对话（包含 assistant response）
        system_msg = "You are a helpful mathematical assistant. Think step by step before answering."
        user_msg = f"Solve the following math problem.\n\nProblem: {problem}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},  # 包含完整的 thinking 回答
        ]

        # 注意：不使用 conversation_to_datum！
        # 因为 Qwen3 renderer 会自动分离 <think>...</think>，导致 thinking 部分不参与训练。
        # 我们需要手动创建 Datum，确保完整的 thinking 内容参与训练。
        return self._create_datum_manual(messages)

    def _create_datum_manual(
        self,
        messages: list[dict],
    ) -> tinker.Datum | None:
        """
        手动创建 SFT Datum

        关键设计：
        - 不使用 Qwen3 renderer（它会自动添加 <think> 到 prompt 末尾）
        - 让 <think> 作为 response 的第一个 token，这样模型能学会主动输出 <think>

        根据 Tinker 文档，cross_entropy loss 需要：
        - model_input: 右移的输入 tokens
        - loss_fn_inputs["weights"]: 权重张量（0 表示不计算 loss，1 表示计算 loss）
        """
        import torch

        # 分离 prompt 和 response
        response_content = messages[-1]["content"]  # assistant，包含完整的 <think>...</think>

        # 确保 response 以 <think> 开头
        response_clean = response_content.strip()
        if not response_clean.startswith("<think>"):
            # 调试：显示 response 实际开头
            if not hasattr(self, "_debug_skip_count"):
                self._debug_skip_count = 0
            if self._debug_skip_count < 3:
                print(f"  [Debug] Response 不以 <think> 开头: {repr(response_clean[:50])}")
                self._debug_skip_count += 1
            return None

        # 使用标准 chat template 构建 prompt（不使用 renderer，避免自动添加 <think>）
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]

        # 构建 prompt：system + user + assistant 开头（不包含 <think>）
        prompt_text = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # Response 保持完整格式：<think>...</think>\n\nanswer<|im_end|>
        # 这样 <think> 作为 response 第一个 token 参与训练
        response_text = response_clean + "<|im_end|>"
        response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)

        # 验证 response 的第一个 token 是 <think>
        think_token_id = self.tokenizer.encode("<think>", add_special_tokens=False)[0]
        if response_tokens[0] != think_token_id:
            # 调试：显示实际的第一个 token
            if not hasattr(self, "_debug_token_count"):
                self._debug_token_count = 0
            if self._debug_token_count < 3:
                first_token = response_tokens[0]
                first_token_text = self.tokenizer.decode([first_token])
                print(f"  [Debug] Response 第一个 token 不是 <think>: ID={first_token}, text={repr(first_token_text)}")
                print(f"  [Debug] 期望的 <think> token ID: {think_token_id}")
                self._debug_token_count += 1
            return None

        # 组合完整序列
        full_tokens = prompt_tokens + response_tokens
        prompt_length = len(prompt_tokens)

        # 截断
        if len(full_tokens) > self.config.max_seq_length:
            full_tokens = full_tokens[: self.config.max_seq_length]
            if len(full_tokens) <= prompt_length:
                return None

        # 创建 weights：prompt 部分为 0，response 部分为 1
        weights = [0.0] * prompt_length + [1.0] * (len(full_tokens) - prompt_length)

        # 使用 datum_from_tokens_weights（传入 tokens tensor，不是 ModelInput）
        if HAS_TINKER_COOKBOOK and datum_from_tokens_weights is not None:
            try:
                tokens_tensor = torch.tensor(full_tokens, dtype=torch.long)
                weights_tensor = torch.tensor(weights, dtype=torch.float32)
                datum = datum_from_tokens_weights(
                    tokens_tensor,
                    weights_tensor,
                    self.config.max_seq_length,
                )
                return datum
            except Exception as e:
                if not hasattr(self, "_datum_error_shown"):
                    print(f"Warning: datum_from_tokens_weights failed: {e}, using fallback")
                    self._datum_error_shown = True

        # Fallback: 手动创建 Datum
        input_tokens = full_tokens[:-1]
        shifted_weights = weights[1:]

        model_input = tinker.ModelInput.from_ints(input_tokens)
        datum = tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "weights": tinker.TensorData.from_torch(torch.tensor(shifted_weights, dtype=torch.float32)),
            },
        )
        return datum

    def train_step(
        self,
        batch: list[dict],
    ) -> dict[str, float] | None:
        """
        执行一个训练步骤（可能包含梯度累积）

        Args:
            batch: 训练样本列表

        Returns:
            如果执行了 optimizer step，返回统计信息；否则返回 None
        """
        # 准备数据
        data = []
        total_tokens = 0
        skipped_count = 0

        for sample in batch:
            datum = self.create_sft_datum(
                sample["problem"],
                sample["response"],
            )
            if datum is None:
                skipped_count += 1
                continue

            data.append(datum)
            total_tokens += datum.model_input.length

        if skipped_count > 0 and self.global_step < 5:
            print(f"  [Debug] Skipped {skipped_count}/{len(batch)} samples (response 不以 <think> 开头)")

        if not data:
            return None

        # Forward-backward（不需要 loss_fn_config）
        fwd_bwd_result = self.training_client.forward_backward(
            data=data,
            loss_fn="cross_entropy",
        ).result()

        # 累积 loss
        # Tinker 的 loss_fn_outputs 是一个列表，每个元素包含 'elementwise_loss'
        batch_loss = 0.0
        if hasattr(fwd_bwd_result, "loss_fn_outputs") and fwd_bwd_result.loss_fn_outputs:
            outputs = fwd_bwd_result.loss_fn_outputs
            if isinstance(outputs, list):
                # 计算每个样本的平均 loss，然后取 batch 平均
                sample_losses = []
                for sample_output in outputs:
                    if isinstance(sample_output, dict) and "elementwise_loss" in sample_output:
                        elem_loss = sample_output["elementwise_loss"]
                        # elem_loss 可能是 TensorData，需要提取数据
                        if hasattr(elem_loss, "data"):
                            loss_values = elem_loss.data
                        else:
                            loss_values = elem_loss
                        # 计算非零 loss 的平均值（只有 response 部分有值）
                        non_zero_losses = [v for v in loss_values if v > 0]
                        if non_zero_losses:
                            sample_losses.append(sum(non_zero_losses) / len(non_zero_losses))
                if sample_losses:
                    batch_loss = sum(sample_losses) / len(sample_losses)

        self.accumulated_loss += batch_loss
        self.accumulation_count += 1

        # 检查是否需要执行 optimizer step
        if self.accumulation_count >= self.config.gradient_accumulation:
            self.global_step += 1

            # 计算学习率（带 warmup）
            if self.global_step <= self.config.warmup_steps:
                lr = self.config.learning_rate * (self.global_step / self.config.warmup_steps)
            else:
                lr = self.config.learning_rate

            # Optimizer step
            self.training_client.optim_step(tinker.AdamParams(learning_rate=lr))

            # 记录统计
            avg_loss = self.accumulated_loss / self.accumulation_count
            stats = {
                "step": self.global_step,
                "loss": avg_loss,
                "lr": lr,
                "tokens": total_tokens * self.accumulation_count,
            }

            for key, value in stats.items():
                self.history[key].append(value)

            # 重置累积
            self.accumulated_loss = 0.0
            self.accumulation_count = 0

            return stats

        return None

    def save_history(self):
        """保存训练历史到文件"""
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(dict(self.history), f, indent=2)

    def save_eval_history(self):
        """保存评估历史到文件"""
        with open(self.run_dir / "eval_history.json", "w") as f:
            json.dump(dict(self.eval_history), f, indent=2)

    def record_eval(self, eval_stats: dict[str, Any]):
        """记录评估结果到历史"""
        self.eval_history["step"].append(self.global_step)
        self.eval_history["thinking_rate"].append(eval_stats["thinking_rate"])
        self.eval_history["boxed_rate"].append(eval_stats["boxed_rate"])
        self.eval_history["total"].append(eval_stats["total"])
        self.eval_history["avg_response_length"].append(eval_stats.get("avg_response_length", 0))
        self.eval_history["avg_thinking_length"].append(eval_stats.get("avg_thinking_length", 0))

    def evaluate(
        self,
        eval_data: list[dict],
        sampling_client: Any,
    ) -> dict[str, Any]:
        """
        评估模型：检查是否能正确生成 thinking 格式

        Args:
            eval_data: 评估数据
            sampling_client: Tinker sampling client

        Returns:
            评估统计
        """
        # 采样参数
        stop_seqs = self.renderer.get_stop_sequences() if self.renderer else []
        sampling_params = SamplingParams(
            max_tokens=7000,
            temperature=0.7,
            stop=stop_seqs if stop_seqs else None,
        )

        results = []
        thinking_count = 0
        boxed_count = 0
        total_response_length = 0
        total_thinking_length = 0

        # 并发发送请求
        futures = []
        for sample in eval_data:
            system_msg = "You are a helpful mathematical assistant. Think step by step before answering."
            user_msg = f"Solve the following math problem.\n\nProblem: {sample['problem']}"

            # 使用标准 chat template（不包含 <think>），测试模型是否学会自己生成 <think>
            prompt_text = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            model_input = tinker.ModelInput.from_ints(prompt_tokens)

            future = sampling_client.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=1,
            )
            futures.append((future, sample))

        # 收集结果
        for future, sample in futures:
            result = future.result()

            if result.sequences:
                seq = result.sequences[0]
                tokens = list(seq.tokens)

                # 解析响应 - 检查模型是否生成了完整的 <think>...</think> 格式
                response_text = self.tokenizer.decode(tokens, skip_special_tokens=False)

                # 检查是否有完整的 thinking 格式：必须以 <think> 开头，且包含 </think>
                has_think_start = response_text.strip().startswith("<think>")
                has_think_end = "</think>" in response_text
                has_thinking = has_think_start and has_think_end

                has_boxed = "\\boxed" in response_text

                if has_thinking:
                    thinking_count += 1
                if has_boxed:
                    boxed_count += 1

                # 计算响应长度（字符数）
                response_length = len(response_text)
                total_response_length += response_length

                # 计算 thinking 内容长度
                thinking_length = 0
                if has_thinking:
                    # 提取 <think>...</think> 之间的内容
                    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
                    if think_match:
                        thinking_length = len(think_match.group(1))
                total_thinking_length += thinking_length

                results.append(
                    {
                        "problem": sample["problem"][:100],
                        "response": response_text[:500],
                        "has_thinking": has_thinking,
                        "has_boxed": has_boxed,
                        "response_length": response_length,
                        "thinking_length": thinking_length,
                    }
                )

        total = len(results)
        return {
            "thinking_rate": thinking_count / total if total > 0 else 0,
            "boxed_rate": boxed_count / total if total > 0 else 0,
            "avg_response_length": total_response_length / total if total > 0 else 0,
            "avg_thinking_length": total_thinking_length / total if total > 0 else 0,
            "total": total,
            "samples": results[:5],  # 保存前 5 个样本供检查
        }


# ============================================================
# 主训练循环
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Cold Start SFT for Thinking Mode")
    parser.add_argument(
        "--scale", type=str, default="small", choices=["quick", "small", "medium", "large"], help="训练规模"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="基座模型")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="default",
        choices=["default", "extended", "all"],
        help="OpenR1-Math-220k 配置",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/coldstart_sft")
    parser.add_argument("--dry-run", action="store_true", help="干运行模式")
    args = parser.parse_args()

    # 检查 API Key
    if not args.dry_run and not os.environ.get("TINKER_API_KEY"):
        print("Error: 请设置 TINKER_API_KEY 环境变量")
        sys.exit(1)

    # 创建配置
    config = SFTConfig(
        scale=args.scale,
        model_name=args.model,
        dataset_config=args.dataset_config,
        learning_rate=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    random.seed(config.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{config.experiment_name}_{config.scale}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print("=" * 60)
    print("Cold Start SFT - Thinking Mode Training")
    print("=" * 60)
    print(f"Scale: {config.scale}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: OpenR1-Math-220k ({config.dataset_config})")
    print(f"Max samples: {config.max_samples}")
    print(f"Max seq length: {config.max_seq_length} (超长样本将被过滤)")
    print(f"Steps: {config.num_steps}")
    print(
        f"Batch size: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}"
    )
    print(f"Learning rate: {config.learning_rate}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    # 预估成本
    model_pricing = {
        "Qwen/Qwen3-4B-Instruct-2507": 0.22,
    }
    price_per_m = model_pricing.get(config.model_name, 0.25)
    effective_batch = config.batch_size * config.gradient_accumulation
    tokens_per_step = effective_batch * config.max_seq_length
    total_tokens = config.num_steps * tokens_per_step / 1e6
    estimated_cost = total_tokens * price_per_m * 2  # forward + backward

    print(f"\n预估 Token 消耗: {total_tokens:.1f}M tokens")
    print(f"预估成本: ~${estimated_cost:.0f}")
    print("=" * 60)

    # 预加载 tokenizer 用于数据过滤
    print("\n加载 tokenizer 用于数据过滤...")
    try:
        from transformers import AutoTokenizer

        filter_tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        print(f"  Tokenizer 加载成功: {config.model_name}")
    except Exception as e:
        print(f"  Warning: Tokenizer 加载失败: {e}")
        print("  将跳过长度过滤")
        filter_tokenizer = None

    # 加载数据（使用 src.data 模块）
    train_data, eval_data = load_openr1_dataset(
        config=config.dataset_config,
        max_samples=config.max_samples,
        seed=config.seed,
        tokenizer=filter_tokenizer,
        max_seq_length=config.max_seq_length,
    )

    if args.dry_run:
        print("\n[Dry Run Mode] 跳过 Tinker API 调用")

        # 测试数据格式化
        if HAS_TINKER_COOKBOOK:
            tokenizer = tinker_tokenizer_utils.get_tokenizer(config.model_name)
            renderer = tinker_renderers.get_renderer("qwen3", tokenizer)

            sample = train_data[0]
            print("\n" + "=" * 60)
            print("示例数据格式")
            print("=" * 60)
            print(f"问题: {sample['problem'][:200]}...")
            print("-" * 60)
            print(f"回答: {sample['response'][:500]}...")
            print("-" * 60)

            # 构建 prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful mathematical assistant. Think step by step before answering.",
                },
                {"role": "user", "content": f"Solve the following math problem.\n\nProblem: {sample['problem']}"},
            ]
            model_input = renderer.build_generation_prompt(messages)
            prompt_tokens = model_input.to_ints()
            prompt_text = tokenizer.decode(prompt_tokens)

            print(f"\nPrompt (末尾):\n...{prompt_text[-200:]}")
            print(f"\nPrompt tokens: {len(prompt_tokens)}")
            print("=" * 60)
        return

    # 初始化 Tinker
    print("\n正在连接 Tinker 服务...")
    service_client = tinker.ServiceClient()

    print(f"正在加载模型: {config.model_name}")
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
        train_unembed=True,
    )
    print("模型加载完成")

    # 创建训练器
    trainer = SFTTrainer(config, training_client, run_dir)

    # 验证训练数据格式
    print("\n验证训练数据格式...")
    valid_count = 0
    invalid_count = 0
    for sample in train_data[: min(100, len(train_data))]:  # 检查前100个样本
        response = sample["response"].strip()
        if response.startswith("<think>"):
            valid_count += 1
        else:
            invalid_count += 1
            if invalid_count <= 3:
                print(f"  [Warning] 样本 response 不以 <think> 开头: {repr(response[:60])}")

    print(f"  检查前 {valid_count + invalid_count} 个样本: {valid_count} 个有效, {invalid_count} 个无效")
    if invalid_count > 0:
        print("  [Warning] 部分样本将被跳过")

    # 训练循环
    print("\n开始训练...")
    print("-" * 60)

    data_index = 0
    step_times = []

    while trainer.global_step < config.num_steps:
        step_start = time.time()

        # 采样 batch
        batch = []
        for _ in range(config.batch_size):
            batch.append(train_data[data_index % len(train_data)])
            data_index += 1

        # 训练步骤
        stats = trainer.train_step(batch)

        if stats is not None:
            step_time = time.time() - step_start
            step_times.append(step_time)
            stats["step_time"] = step_time
            # 记录 step_time 到历史
            trainer.history["step_time"].append(step_time)

            # 日志
            if trainer.global_step % config.log_interval == 0:
                avg_time = sum(step_times[-10:]) / len(step_times[-10:])
                print(
                    f"Step {stats['step']}/{config.num_steps} | "
                    f"Loss: {stats['loss']:.4f} | "
                    f"LR: {stats['lr']:.2e} | "
                    f"Time: {avg_time:.1f}s"
                )
                # 保存训练历史（每次 log 时保存）
                trainer.save_history()

            # 评估
            if trainer.global_step % config.eval_interval == 0:
                sampling_client = training_client.save_weights_and_get_sampling_client(
                    name=f"step_{trainer.global_step}"
                )
                eval_stats = trainer.evaluate(eval_data[: config.eval_samples], sampling_client)

                print(
                    f"  [Eval] Thinking Rate: {eval_stats['thinking_rate']:.1%} | "
                    f"Boxed Rate: {eval_stats['boxed_rate']:.1%}"
                )

                # 显示样本
                if eval_stats["samples"]:
                    sample = eval_stats["samples"][0]
                    print(f"  [Sample] has_thinking={sample['has_thinking']}, has_boxed={sample['has_boxed']}")
                    print(f"    Response: {sample['response'][:200]}...")

                # 记录评估结果到历史
                trainer.record_eval(eval_stats)
                trainer.save_eval_history()

                # 保存评估结果详情
                eval_file = run_dir / f"eval_step_{trainer.global_step}.json"
                with open(eval_file, "w", encoding="utf-8") as f:
                    json.dump(eval_stats, f, ensure_ascii=False, indent=2)

            # 保存 checkpoint
            if trainer.global_step % config.save_interval == 0:
                checkpoint_name = f"checkpoint_step_{trainer.global_step}"
                training_client.save_state(checkpoint_name)
                print(f"  [Save] Checkpoint: {checkpoint_name}")

    # 最终保存
    print("\n" + "=" * 60)
    print("训练完成!")

    final_checkpoint = "coldstart_sft_final"
    training_client.save_state(final_checkpoint)
    print(f"最终模型保存为: {final_checkpoint}")

    # 最终评估
    sampling_client = training_client.save_weights_and_get_sampling_client(name="final")
    final_eval = trainer.evaluate(eval_data[: config.eval_samples], sampling_client)

    print("\n最终评估:")
    print(f"  Thinking Rate: {final_eval['thinking_rate']:.1%}")
    print(f"  Boxed Rate: {final_eval['boxed_rate']:.1%}")

    # 记录最终评估
    trainer.record_eval(final_eval)

    # 显示几个样本
    print("\n样本检查:")
    for i, sample in enumerate(final_eval["samples"][:3]):
        print(f"\n  样本 {i + 1}:")
        print(f"    问题: {sample['problem'][:80]}...")
        print(f"    has_thinking: {sample['has_thinking']}, has_boxed: {sample['has_boxed']}")
        print(f"    回答: {sample['response'][:300]}...")

    # 保存所有历史
    trainer.save_history()
    trainer.save_eval_history()

    # 保存最终结果
    results = {
        "config": asdict(config),
        "final_eval": {
            "thinking_rate": final_eval["thinking_rate"],
            "boxed_rate": final_eval["boxed_rate"],
            "total": final_eval["total"],
        },
        "training_summary": {
            "total_steps": trainer.global_step,
            "final_loss": trainer.history["loss"][-1] if trainer.history["loss"] else 0,
            "total_time": sum(trainer.history["step_time"]) if trainer.history["step_time"] else 0,
        },
        "checkpoint_name": final_checkpoint,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 保存最终评估详情
    with open(run_dir / "eval_final.json", "w", encoding="utf-8") as f:
        json.dump(final_eval, f, ensure_ascii=False, indent=2)

    print(f"\n输出目录: {run_dir}")
    print(f"后续 RLVR 训练请使用 checkpoint: {final_checkpoint}")
    print("=" * 60)


if __name__ == "__main__":
    main()

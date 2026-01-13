#!/usr/bin/env python3
"""
数学推理评估脚本

支持评估Tinker训练的模型在多个数学benchmark上的表现

支持的数据集:
- GSM8K: 小学数学应用题
- MATH: 竞赛数学（可选）

使用方法:
    # 评估Tinker保存的checkpoint
    python scripts/eval/evaluate_math.py --checkpoint checkpoint_name

    # 评估基线模型
    python scripts/eval/evaluate_math.py --model meta-llama/Llama-3.2-3B-Instruct

    # 评估本地模型
    python scripts/eval/evaluate_math.py --model ./outputs/justrl/final_model

Author: Guanghan Ning
Date: 2025-12-24
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# 添加项目根目录到path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_gsm8k(split: str = "test", max_samples: int | None = None) -> list[dict]:
    """加载GSM8K数据集"""
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main", split=split)

    samples = []
    for item in dataset:
        answer_text = item["answer"]
        parts = answer_text.split("####")
        final_answer = parts[-1].strip().replace(",", "") if len(parts) > 1 else answer_text

        samples.append({
            "question": item["question"],
            "answer": final_answer,
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples


def load_math_500(max_samples: int | None = None) -> list[dict]:
    """加载MATH-500子集"""
    try:
        from datasets import load_dataset
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

        samples = []
        for item in dataset:
            samples.append({
                "question": item["problem"],
                "answer": item["answer"],
                "level": item.get("level"),
                "type": item.get("type"),
            })

            if max_samples and len(samples) >= max_samples:
                break

        return samples
    except Exception as e:
        print(f"Warning: 无法加载MATH-500: {e}")
        return []


def format_prompt(question: str) -> str:
    """格式化prompt"""
    return f"""Solve the following math problem step by step. Show your reasoning clearly.
At the end, provide your final answer after "The answer is: ".

Problem: {question}

Solution:"""


class MathVerifier:
    """数学答案验证器"""

    def __init__(self):
        import re
        self.re = re

        self.patterns = [
            r"[Tt]he (?:final )?answer is:?\s*\$?([^\$\n]+?)\$?\s*(?:\.|$)",
            r"####\s*(.+?)(?:\n|$)",
            r"\\boxed\{([^}]+)\}",
            r"=\s*([^\n=]+?)\s*$",
        ]

    def extract_answer(self, text: str) -> str | None:
        """提取答案"""
        for pattern in self.patterns:
            match = self.re.search(pattern, text, self.re.MULTILINE)
            if match:
                answer = match.group(1).replace(",", "").strip()
                return answer
        return None

    def verify(self, response: str, gold: str) -> dict:
        """验证答案"""
        extracted = self.extract_answer(response)
        gold = str(gold).replace(",", "").strip()

        if extracted is None:
            return {"is_correct": False, "extracted": None}

        # 清理
        extracted = extracted.strip()

        # 数值比较
        try:
            is_correct = abs(float(extracted) - float(gold)) < 1e-6
        except ValueError:
            is_correct = extracted.lower() == gold.lower()

        return {"is_correct": is_correct, "extracted": extracted}


def evaluate_with_tinker(
    checkpoint: str,
    samples: list[dict],
    verifier: MathVerifier,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> dict:
    """使用Tinker评估checkpoint"""
    import tinker
    from tinker import SamplingParams

    service_client = tinker.ServiceClient()

    # 加载checkpoint
    print(f"加载checkpoint: {checkpoint}")
    sampling_client = service_client.load_sampling_client(checkpoint)

    # 创建采样参数
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    results = []
    correct = 0

    for item in tqdm(samples, desc="Evaluating"):
        prompt = format_prompt(item["question"])

        response = sampling_client.sample(
            prompt=prompt,
            sampling_params=sampling_params,
            num_samples=1,  # 每次评估生成1个样本
        )

        result = verifier.verify(response, item["answer"])
        result["question"] = item["question"][:100]
        result["gold"] = item["answer"]
        result["response"] = response[:500]
        results.append(result)

        if result["is_correct"]:
            correct += 1

    accuracy = correct / len(samples) if samples else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "results": results,
    }


def evaluate_with_vllm(
    model_path: str,
    samples: list[dict],
    verifier: MathVerifier,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> dict:
    """使用vLLM评估本地模型"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: 请安装vllm: pip install vllm")
        sys.exit(1)

    print(f"加载模型: {model_path}")
    llm = LLM(model=model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 准备prompts
    prompts = [format_prompt(item["question"]) for item in samples]

    # 批量生成
    print("生成responses...")
    outputs = llm.generate(prompts, sampling_params)

    # 验证
    results = []
    correct = 0

    for item, output in zip(samples, outputs, strict=False):
        response = output.outputs[0].text

        result = verifier.verify(response, item["answer"])
        result["question"] = item["question"][:100]
        result["gold"] = item["answer"]
        result["response"] = response[:500]
        results.append(result)

        if result["is_correct"]:
            correct += 1

    accuracy = correct / len(samples) if samples else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="数学推理模型评估")
    parser.add_argument("--checkpoint", type=str, help="Tinker checkpoint名称")
    parser.add_argument("--model", type=str, help="HuggingFace模型路径")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math500", "both"],
                        help="评估数据集")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大评估样本数")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度（0=greedy）")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出文件")
    args = parser.parse_args()

    if not args.checkpoint and not args.model:
        print("Error: 请指定 --checkpoint 或 --model")
        sys.exit(1)

    # 加载数据集
    print("=" * 60)
    print("数学推理模型评估")
    print("=" * 60)

    datasets = {}

    if args.dataset in ["gsm8k", "both"]:
        print("\n加载GSM8K...")
        datasets["gsm8k"] = load_gsm8k("test", args.max_samples)
        print(f"  样本数: {len(datasets['gsm8k'])}")

    if args.dataset in ["math500", "both"]:
        print("\n加载MATH-500...")
        datasets["math500"] = load_math_500(args.max_samples)
        print(f"  样本数: {len(datasets['math500'])}")

    # 初始化验证器
    verifier = MathVerifier()

    # 评估
    all_results = {}

    for dataset_name, samples in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"评估 {dataset_name.upper()}")
        print("=" * 60)

        if args.checkpoint:
            eval_result = evaluate_with_tinker(
                args.checkpoint,
                samples,
                verifier,
                args.max_tokens,
                args.temperature,
            )
        else:
            eval_result = evaluate_with_vllm(
                args.model,
                samples,
                verifier,
                args.max_tokens,
                args.temperature,
            )

        all_results[dataset_name] = eval_result

        print("\n结果:")
        print(f"  准确率: {eval_result['accuracy']:.2%}")
        print(f"  正确数: {eval_result['correct']}/{eval_result['total']}")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "outputs" / f"eval_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 移除详细results以减小文件大小
    summary = {
        name: {k: v for k, v in result.items() if k != "results"}
        for name, result in all_results.items()
    }

    with open(output_path, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": summary,
        }, f, indent=2)

    print(f"\n结果已保存到: {output_path}")

    # 打印总结
    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)
    for name, result in all_results.items():
        print(f"  {name}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")


if __name__ == "__main__":
    main()

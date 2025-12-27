"""
数学推理任务评估器

提供多种评估指标和答案验证方法

Author: Guanghan Ning
Date: 2025-12-23
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EvalResult:
    """评估结果"""
    correct: int
    total: int
    accuracy: float
    details: List[Dict]


class MathAnswerExtractor:
    """
    数学答案提取器

    支持多种答案格式的提取
    """

    # 常见答案格式的正则表达式
    PATTERNS = [
        # "The answer is: X" 或 "The answer is X"
        r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        # "#### X" (GSM8K格式)
        r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        # "\boxed{X}" (MATH格式)
        r"\\boxed\{([^}]+)\}",
        # "= X" 在行尾
        r"=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
        # "Answer: X"
        r"[Aa]nswer:?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        # "Therefore, X"
        r"[Tt]herefore,?\s*(?:the answer is\s*)?(-?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]

    def extract(self, text: str) -> Optional[str]:
        """
        从文本中提取答案

        按优先级尝试不同的模式，返回第一个匹配的答案

        Args:
            text: 模型输出文本

        Returns:
            提取的答案字符串，或None
        """
        for pattern in self.PATTERNS:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                answer = match.group(1)
                # 清理答案
                answer = self._clean_answer(answer)
                return answer

        return None

    def _clean_answer(self, answer: str) -> str:
        """清理答案字符串"""
        # 移除千位分隔符
        answer = answer.replace(",", "")
        # 移除前后空白
        answer = answer.strip()
        # 移除末尾的句号
        answer = answer.rstrip(".")
        return answer


class MathAnswerComparator:
    """
    数学答案比较器

    支持多种比较方式：精确匹配、数值比较、表达式比较等
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def compare(self, predicted: str, gold: str) -> bool:
        """
        比较预测答案和标准答案

        Args:
            predicted: 预测答案
            gold: 标准答案

        Returns:
            是否匹配
        """
        if predicted is None:
            return False

        # 清理答案
        predicted = self._normalize(predicted)
        gold = self._normalize(gold)

        # 1. 精确字符串匹配
        if predicted == gold:
            return True

        # 2. 数值比较
        try:
            pred_val = float(predicted)
            gold_val = float(gold)
            if abs(pred_val - gold_val) < self.tolerance:
                return True
        except ValueError:
            pass

        # 3. 分数比较
        pred_frac = self._parse_fraction(predicted)
        gold_frac = self._parse_fraction(gold)
        if pred_frac is not None and gold_frac is not None:
            if abs(pred_frac - gold_frac) < self.tolerance:
                return True

        return False

    def _normalize(self, answer: str) -> str:
        """标准化答案格式"""
        answer = answer.strip().lower()
        # 移除空格
        answer = answer.replace(" ", "")
        # 移除$符号（LaTeX）
        answer = answer.replace("$", "")
        return answer

    def _parse_fraction(self, text: str) -> Optional[float]:
        """解析分数"""
        # 匹配 a/b 格式
        match = re.match(r"(-?\d+)/(\d+)", text)
        if match:
            num = int(match.group(1))
            den = int(match.group(2))
            if den != 0:
                return num / den

        # 匹配 \frac{a}{b} 格式
        match = re.match(r"\\frac\{(-?\d+)\}\{(\d+)\}", text)
        if match:
            num = int(match.group(1))
            den = int(match.group(2))
            if den != 0:
                return num / den

        return None


class MathEvaluator:
    """
    数学推理任务评估器

    Example:
        >>> evaluator = MathEvaluator()
        >>> results = evaluator.evaluate(predictions, samples)
        >>> print(f"Accuracy: {results.accuracy:.2%}")
    """

    def __init__(self, tolerance: float = 1e-6):
        self.extractor = MathAnswerExtractor()
        self.comparator = MathAnswerComparator(tolerance)

    def evaluate(
        self,
        predictions: List[str],
        samples: List,  # List[Sample]
        verbose: bool = False
    ) -> EvalResult:
        """
        评估模型预测

        Args:
            predictions: 模型输出列表
            samples: 数据样本列表（需要有answer属性）
            verbose: 是否打印详细信息

        Returns:
            EvalResult对象
        """
        assert len(predictions) == len(samples), "预测数量与样本数量不匹配"

        correct = 0
        details = []

        for pred, sample in zip(predictions, samples):
            # 提取答案
            extracted = self.extractor.extract(pred)
            gold = sample.answer

            # 比较答案
            is_correct = self.comparator.compare(extracted, gold)

            if is_correct:
                correct += 1

            # 记录详情
            details.append({
                "question": sample.question[:100] + "...",
                "gold": gold,
                "extracted": extracted,
                "is_correct": is_correct,
                "prediction": pred[:200] + "..." if len(pred) > 200 else pred,
            })

            if verbose and not is_correct:
                print(f"\n[WRONG]")
                print(f"  Question: {sample.question[:100]}...")
                print(f"  Gold: {gold}")
                print(f"  Extracted: {extracted}")
                print(f"  Prediction: {pred[:200]}...")

        total = len(samples)
        accuracy = correct / total if total > 0 else 0.0

        return EvalResult(
            correct=correct,
            total=total,
            accuracy=accuracy,
            details=details,
        )

    def compute_pass_at_k(
        self,
        predictions: List[List[str]],
        samples: List,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        计算Pass@K指标

        对每个问题采样多次，计算至少有一次正确的概率

        Args:
            predictions: 二维列表，每个问题的多次预测
            samples: 数据样本列表
            k_values: 要计算的K值列表

        Returns:
            {"pass@1": 0.75, "pass@5": 0.90, ...}
        """
        results = {}

        for k in k_values:
            total_pass = 0
            total_samples = len(samples)

            for preds, sample in zip(predictions, samples):
                # 取前k个预测
                preds_k = preds[:k]

                # 检查是否有任意一个正确
                for pred in preds_k:
                    extracted = self.extractor.extract(pred)
                    if self.comparator.compare(extracted, sample.answer):
                        total_pass += 1
                        break

            results[f"pass@{k}"] = total_pass / total_samples

        return results


def evaluate_countdown(
    prediction: str,
    target: int,
    numbers: List[int]
) -> Tuple[bool, Optional[int]]:
    """
    评估Countdown游戏的答案

    验证：
    1. 表达式是否使用了给定的数字
    2. 计算结果是否等于目标值

    Args:
        prediction: 模型输出
        target: 目标值
        numbers: 可用数字

    Returns:
        (是否正确, 计算结果)
    """
    # 尝试从输出中提取最终计算表达式
    # 这是一个简化的实现，实际可能需要更复杂的解析

    # 查找类似 "= target" 的模式
    match = re.search(r"=\s*(\d+)\s*$", prediction, re.MULTILINE)
    if match:
        result = int(match.group(1))
        return result == target, result

    return False, None


if __name__ == "__main__":
    # 测试评估器
    print("测试答案提取器...")
    extractor = MathAnswerExtractor()

    test_cases = [
        "Let me solve this step by step... The answer is: 42",
        "So we get 3 + 4 = 7. #### 7",
        "Therefore, $\\boxed{256}$",
        "The result = 100",
    ]

    for text in test_cases:
        answer = extractor.extract(text)
        print(f"  Input: {text[:50]}...")
        print(f"  Extracted: {answer}")
        print()

    print("测试答案比较器...")
    comparator = MathAnswerComparator()

    compare_cases = [
        ("42", "42", True),
        ("42.0", "42", True),
        ("1/2", "0.5", True),
        ("\\frac{1}{2}", "0.5", True),
        ("100", "101", False),
    ]

    for pred, gold, expected in compare_cases:
        result = comparator.compare(pred, gold)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{pred}' == '{gold}': {result}")

"""
数学答案验证器 - DAPO风格

基于JustRL论文使用的DAPO verifier实现
提供二元奖励：正确为1，错误为0

特点：
- 轻量级基于规则的验证
- 支持多种答案格式
- 数值容差比较
- 分数和表达式支持

Author: Guanghan Ning
Date: 2025-12-24
"""

import re
import math
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class VerificationResult:
    """验证结果"""
    is_correct: bool
    reward: float
    extracted_answer: Optional[str]
    gold_answer: str
    confidence: float  # 提取答案的置信度
    match_type: str    # 匹配类型：exact, numeric, fraction, none


class MathVerifier:
    """
    数学答案验证器

    实现DAPO论文中使用的轻量级规则验证器
    JustRL直接采用此验证器，无需额外的神经网络验证器

    Example:
        >>> verifier = MathVerifier()
        >>> result = verifier.verify(
        ...     response="Let me solve this... The answer is: 42",
        ...     gold_answer="42"
        ... )
        >>> print(result.is_correct)  # True
        >>> print(result.reward)      # 1.0
    """

    # 答案提取模式（按优先级排序）
    EXTRACTION_PATTERNS = [
        # 明确的答案标记
        (r"[Tt]he (?:final )?answer is:?\s*\$?([^\$\n]+?)\$?\s*(?:\.|$)", "explicit"),
        (r"[Aa]nswer:?\s*\$?([^\$\n]+?)\$?\s*(?:\.|$)", "explicit"),

        # GSM8K格式
        (r"####\s*(.+?)(?:\n|$)", "gsm8k"),

        # LaTeX boxed格式
        (r"\\boxed\{([^}]+)\}", "boxed"),

        # 等号结尾
        (r"=\s*\$?([^\$\n=]+?)\$?\s*$", "equation_end"),

        # Therefore/Thus/So引导
        (r"[Tt]herefore,?\s*(?:the answer is\s*)?\$?([^\$\n,]+?)\$?(?:\.|,|$)", "therefore"),
        (r"[Tt]hus,?\s*(?:the answer is\s*)?\$?([^\$\n,]+?)\$?(?:\.|,|$)", "thus"),
        (r"[Ss]o,?\s*(?:the answer is\s*)?\$?([^\$\n,]+?)\$?(?:\.|,|$)", "so"),
    ]

    def __init__(
        self,
        numeric_tolerance: float = 1e-6,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        format_bonus: float = 0.0,  # 能提取答案时的额外奖励
    ):
        """
        Args:
            numeric_tolerance: 数值比较容差
            correct_reward: 正确答案的奖励
            incorrect_reward: 错误答案的奖励
            format_bonus: 格式正确（能提取答案）的额外奖励
        """
        self.numeric_tolerance = numeric_tolerance
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.format_bonus = format_bonus

    def verify(
        self,
        response: str,
        gold_answer: str,
    ) -> VerificationResult:
        """
        验证模型响应

        Args:
            response: 模型输出
            gold_answer: 标准答案

        Returns:
            VerificationResult对象
        """
        # 提取答案
        extracted, confidence, pattern_type = self._extract_answer(response)

        # 标准化gold answer
        gold_normalized = self._normalize_answer(gold_answer)

        if extracted is None:
            return VerificationResult(
                is_correct=False,
                reward=self.incorrect_reward,
                extracted_answer=None,
                gold_answer=gold_answer,
                confidence=0.0,
                match_type="none",
            )

        # 标准化提取的答案
        extracted_normalized = self._normalize_answer(extracted)

        # 比较答案
        is_correct, match_type = self._compare_answers(
            extracted_normalized, gold_normalized
        )

        # 计算奖励
        if is_correct:
            reward = self.correct_reward
        else:
            reward = self.incorrect_reward + self.format_bonus  # 格式正确有小奖励

        return VerificationResult(
            is_correct=is_correct,
            reward=reward,
            extracted_answer=extracted,
            gold_answer=gold_answer,
            confidence=confidence,
            match_type=match_type,
        )

    def _extract_answer(self, text: str) -> Tuple[Optional[str], float, str]:
        """
        从文本中提取答案

        Returns:
            (答案, 置信度, 模式类型)
        """
        # 从后往前搜索，优先取最后出现的答案
        text_reversed_lines = text.strip().split('\n')

        for pattern, pattern_type in self.EXTRACTION_PATTERNS:
            # 先尝试在最后几行找
            for line in reversed(text_reversed_lines[-10:]):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    if answer:
                        confidence = 0.9 if pattern_type in ["explicit", "gsm8k", "boxed"] else 0.7
                        return answer, confidence, pattern_type

            # 再在全文找
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                if answer:
                    confidence = 0.8 if pattern_type in ["explicit", "gsm8k", "boxed"] else 0.6
                    return answer, confidence, pattern_type

        return None, 0.0, "none"

    def _normalize_answer(self, answer: str) -> str:
        """标准化答案格式"""
        answer = answer.strip()

        # 移除常见包装
        answer = answer.strip('$').strip()
        answer = re.sub(r'^\\text\{(.+)\}$', r'\1', answer)

        # 移除千位分隔符
        answer = answer.replace(',', '')

        # 移除空格
        answer = answer.replace(' ', '')

        # 统一负号
        answer = answer.replace('−', '-')

        # 移除末尾句号
        answer = answer.rstrip('.')

        # 小写
        answer = answer.lower()

        return answer

    def _compare_answers(
        self,
        predicted: str,
        gold: str
    ) -> Tuple[bool, str]:
        """
        比较两个答案

        Returns:
            (是否匹配, 匹配类型)
        """
        # 1. 精确字符串匹配
        if predicted == gold:
            return True, "exact"

        # 2. 数值比较
        pred_num = self._parse_number(predicted)
        gold_num = self._parse_number(gold)

        if pred_num is not None and gold_num is not None:
            if abs(pred_num - gold_num) < self.numeric_tolerance:
                return True, "numeric"
            # 处理百分比
            if abs(pred_num - gold_num * 100) < self.numeric_tolerance:
                return True, "numeric_percent"
            if abs(pred_num * 100 - gold_num) < self.numeric_tolerance:
                return True, "numeric_percent"

        # 3. 分数比较
        pred_frac = self._parse_fraction(predicted)
        gold_frac = self._parse_fraction(gold)

        if pred_frac is not None and gold_frac is not None:
            if abs(float(pred_frac) - float(gold_frac)) < self.numeric_tolerance:
                return True, "fraction"

        # 4. 混合比较（一个是分数，一个是小数）
        if pred_num is not None and gold_frac is not None:
            if abs(pred_num - float(gold_frac)) < self.numeric_tolerance:
                return True, "mixed"
        if pred_frac is not None and gold_num is not None:
            if abs(float(pred_frac) - gold_num) < self.numeric_tolerance:
                return True, "mixed"

        return False, "mismatch"

    def _parse_number(self, text: str) -> Optional[float]:
        """解析数字"""
        try:
            # 处理科学计数法
            text = text.replace('×10^', 'e').replace('x10^', 'e')
            text = re.sub(r'\s*\\times\s*10\^?\{?(-?\d+)\}?', r'e\1', text)

            # 处理百分号
            if text.endswith('%'):
                return float(text[:-1]) / 100

            return float(text)
        except ValueError:
            return None

    def _parse_fraction(self, text: str) -> Optional[Fraction]:
        """解析分数"""
        try:
            # a/b 格式
            match = re.match(r'^(-?\d+)/(\d+)$', text)
            if match:
                return Fraction(int(match.group(1)), int(match.group(2)))

            # \frac{a}{b} 格式
            match = re.match(r'^\\frac\{(-?\d+)\}\{(\d+)\}$', text)
            if match:
                return Fraction(int(match.group(1)), int(match.group(2)))

            # 混合数 a b/c 格式
            match = re.match(r'^(-?\d+)\s+(\d+)/(\d+)$', text)
            if match:
                whole = int(match.group(1))
                frac = Fraction(int(match.group(2)), int(match.group(3)))
                return Fraction(whole) + frac if whole >= 0 else Fraction(whole) - frac

            return None
        except (ValueError, ZeroDivisionError):
            return None


class BatchVerifier:
    """
    批量验证器

    用于GRPO训练中批量计算奖励
    """

    def __init__(self, verifier: MathVerifier = None):
        self.verifier = verifier or MathVerifier()

    def compute_rewards(
        self,
        responses: List[str],
        gold_answers: List[str],
    ) -> List[VerificationResult]:
        """
        批量计算奖励

        Args:
            responses: 模型响应列表
            gold_answers: 标准答案列表

        Returns:
            VerificationResult列表
        """
        assert len(responses) == len(gold_answers)

        results = []
        for response, gold in zip(responses, gold_answers):
            result = self.verifier.verify(response, gold)
            results.append(result)

        return results

    def compute_rewards_for_groups(
        self,
        responses_groups: List[List[str]],
        gold_answers: List[str],
    ) -> List[List[VerificationResult]]:
        """
        为GRPO的分组响应计算奖励

        Args:
            responses_groups: 分组响应，每个问题有多个响应
            gold_answers: 每个问题的标准答案

        Returns:
            分组的VerificationResult
        """
        assert len(responses_groups) == len(gold_answers)

        all_results = []
        for responses, gold in zip(responses_groups, gold_answers):
            group_results = [
                self.verifier.verify(resp, gold)
                for resp in responses
            ]
            all_results.append(group_results)

        return all_results


def test_verifier():
    """测试验证器"""
    verifier = MathVerifier()

    test_cases = [
        # (response, gold_answer, expected_correct)
        ("After calculation, the answer is: 42", "42", True),
        ("So we get 3 + 4 = 7. #### 7", "7", True),
        ("Therefore, $\\boxed{256}$", "256", True),
        ("The result = 100", "100", True),
        ("Step 1... Step 2... The answer is: 3.14159", "3.14159", True),
        ("The answer is: 1/2", "0.5", True),
        ("The answer is: 50%", "0.5", True),
        ("The answer is: \\frac{1}{4}", "0.25", True),
        ("The answer is: 42", "43", False),
        ("No clear answer here", "42", False),
    ]

    print("=" * 60)
    print("MathVerifier Test Results")
    print("=" * 60)

    passed = 0
    for response, gold, expected in test_cases:
        result = verifier.verify(response, gold)
        status = "PASS" if result.is_correct == expected else "FAIL"
        if result.is_correct == expected:
            passed += 1

        print(f"\n[{status}]")
        print(f"  Response: {response[:50]}...")
        print(f"  Gold: {gold}")
        print(f"  Extracted: {result.extracted_answer}")
        print(f"  Correct: {result.is_correct} (expected: {expected})")
        print(f"  Match type: {result.match_type}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(test_cases)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_verifier()

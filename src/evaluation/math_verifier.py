"""
数学答案验证器

特点：
- 支持多种答案格式：\\boxed{}, "The answer is:", 数值等
- 格式奖励：检测 <think>...</think> 格式
- 冗余度惩罚：检测重复内容（防止 reward hacking）
- 三层验证：字符串 → 数值 → SymPy

Author: Guanghan Ning
Date: 2025-01-10
Updated: 2026-01-13 (从 justrl_math_reasoning.py 提取)
"""

import re
import zlib
from typing import Optional


class MathVerifier:
    """
    MATH 数据集验证器

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

        text_bytes = text.encode("utf-8")
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

        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
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
        chunks = [text[i : i + chunk_size] for i in range(0, len(text) - chunk_size + 1, step)]

        if len(chunks) < 2:
            return 0.0

        def get_shingles(s: str) -> set:
            """获取 k-character shingles"""
            if len(s) < shingle_k:
                return set()
            return set(s[i : i + shingle_k] for i in range(len(s) - shingle_k + 1))

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
        answer = answer.replace("$", "")  # 普通美元符号

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
            "centimeter",
            "centimeters",
            "cm",
            "millimeter",
            "millimeters",
            "mm",
            "meter",
            "meters",
            "m",
            "kilometer",
            "kilometers",
            "km",
            "inch",
            "inches",
            "ft",
            "feet",
            "foot",
            "yard",
            "yards",
            "mile",
            "miles",
            "kilogram",
            "kilograms",
            "kg",
            "gram",
            "grams",
            "g",
            "mg",
            "lb",
            "lbs",
            "oz",
            "ounce",
            "ounces",
            "liter",
            "liters",
            "ml",
            "gallon",
            "gallons",
            "second",
            "seconds",
            "sec",
            "minute",
            "minutes",
            "min",
            "hour",
            "hours",
            "hr",
            "day",
            "days",
            "dollar",
            "dollars",
            "cent",
            "cents",
            "euro",
            "euros",
            "square",
            "cubic",
            "sq",
            "cu",
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
            """递归提取 \\boxed{} 内容，正确处理嵌套大括号"""
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
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                i += 1

            if depth == 0:
                return s[brace_start : i - 1]
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
                if coef == "" or coef == "+":
                    coef = 1.0
                elif coef == "-":
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

            from sympy import N, simplify
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
                from sympy import sympify

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


class BatchVerifier:
    """
    批量验证器

    用于 GRPO 训练中批量计算奖励
    """

    def __init__(self, verifier: MathVerifier = None):
        self.verifier = verifier or MathVerifier()

    def compute_rewards(
        self,
        responses: list[str],
        gold_answers: list[str],
        tokens_list: list[list[int]] = None,
    ) -> list[dict]:
        """
        批量计算奖励

        Args:
            responses: 模型响应列表
            gold_answers: 标准答案列表
            tokens_list: 每个响应的 token 列表（可选）

        Returns:
            验证结果列表
        """
        assert len(responses) == len(gold_answers)

        results = []
        for i, (response, gold) in enumerate(zip(responses, gold_answers, strict=False)):
            tokens = tokens_list[i] if tokens_list else None
            result = self.verifier.verify(response, gold, tokens=tokens)
            results.append(result)

        return results

    def compute_rewards_for_groups(
        self,
        responses_groups: list[list[str]],
        gold_answers: list[str],
        tokens_groups: list[list[list[int]]] = None,
    ) -> list[list[dict]]:
        """
        为 GRPO 的分组响应计算奖励

        Args:
            responses_groups: 分组响应，每个问题有多个响应
            gold_answers: 每个问题的标准答案
            tokens_groups: 分组的 token 列表（可选）

        Returns:
            分组的验证结果
        """
        assert len(responses_groups) == len(gold_answers)

        all_results = []
        for i, (responses, gold) in enumerate(zip(responses_groups, gold_answers, strict=False)):
            tokens_list = tokens_groups[i] if tokens_groups else None
            group_results = []
            for j, resp in enumerate(responses):
                tokens = tokens_list[j] if tokens_list else None
                result = self.verifier.verify(resp, gold, tokens=tokens)
                group_results.append(result)
            all_results.append(group_results)

        return all_results


# ============================================================
# 测试
# ============================================================


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
        result = verifier.verify(response, gold, check_format=False, check_redundancy=False)
        status = "PASS" if result["is_correct"] == expected else "FAIL"
        if result["is_correct"] == expected:
            passed += 1

        print(f"\n[{status}]")
        print(f"  Response: {response[:50]}...")
        print(f"  Gold: {gold}")
        print(f"  Extracted: {result['extracted']}")
        print(f"  Correct: {result['is_correct']} (expected: {expected})")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(test_cases)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_verifier()

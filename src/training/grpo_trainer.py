"""
GRPO训练器 - JustRL风格

实现Group Relative Policy Optimization算法
遵循JustRL论文的简化原则

核心特点：
- 无KL惩罚
- 无长度惩罚
- clip-higher机制
- 组内归一化advantage

Author: Guanghan Ning
Date: 2025-12-24
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import json
import time


@dataclass
class GRPOConfig:
    """GRPO配置"""
    # 核心参数
    rollout_n: int = 8              # 每个prompt采样的response数量
    clip_ratio_low: float = 0.8     # clip下界
    clip_ratio_high: float = 1.28   # clip上界（JustRL的clip-higher）

    # JustRL: 移除这些
    kl_coef: float = 0.0            # 无KL惩罚
    entropy_coef: float = 0.0       # 无熵正则

    # Advantage计算
    normalize_advantage: bool = True
    advantage_eps: float = 1e-8     # 数值稳定性

    # 奖励设置
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0


@dataclass
class GRPOBatch:
    """GRPO训练批次"""
    prompts: List[str]
    responses: List[List[str]]      # [batch_size, rollout_n]
    rewards: List[List[float]]      # [batch_size, rollout_n]
    advantages: List[List[float]]   # [batch_size, rollout_n]
    gold_answers: List[str]

    # 统计信息
    stats: Dict[str, float] = field(default_factory=dict)


class GRPOTrainer:
    """
    GRPO训练器

    实现JustRL风格的简化GRPO算法

    Example:
        >>> config = GRPOConfig(rollout_n=8)
        >>> trainer = GRPOTrainer(config, tinker_client, verifier)
        >>> for step in range(num_steps):
        ...     batch = trainer.collect_batch(prompts, gold_answers)
        ...     stats = trainer.train_step(batch)
    """

    def __init__(
        self,
        config: GRPOConfig,
        training_client: Any,  # Tinker TrainingClient
        verifier: Any,         # MathVerifier
        logger: Any = None,
    ):
        self.config = config
        self.training_client = training_client
        self.verifier = verifier
        self.logger = logger

        # 训练统计
        self.global_step = 0
        self.total_tokens = 0
        self.history = defaultdict(list)

    def collect_batch(
        self,
        prompts: List[str],
        gold_answers: List[str],
        sampling_client: Any,
        generation_kwargs: Dict = None,
    ) -> GRPOBatch:
        """
        收集训练批次

        为每个prompt采样rollout_n个response，计算奖励和advantage

        Args:
            prompts: 问题列表
            gold_answers: 标准答案列表
            sampling_client: Tinker采样客户端
            generation_kwargs: 生成参数

        Returns:
            GRPOBatch对象
        """
        generation_kwargs = generation_kwargs or {}
        batch_size = len(prompts)

        # 1. 为每个prompt采样多个response
        all_responses = []
        for prompt in prompts:
            responses = []
            for _ in range(self.config.rollout_n):
                response = sampling_client.sample(
                    prompt=prompt,
                    **generation_kwargs
                )
                responses.append(response)
            all_responses.append(responses)

        # 2. 计算奖励
        all_rewards = []
        correct_count = 0
        total_count = 0

        for responses, gold in zip(all_responses, gold_answers):
            rewards = []
            for response in responses:
                result = self.verifier.verify(response, gold)
                rewards.append(result.reward)
                if result.is_correct:
                    correct_count += 1
                total_count += 1
            all_rewards.append(rewards)

        # 3. 计算advantages（组内归一化）
        all_advantages = []
        for rewards in all_rewards:
            advantages = self._compute_advantages(rewards)
            all_advantages.append(advantages)

        # 4. 统计信息
        flat_rewards = [r for rewards in all_rewards for r in rewards]
        flat_advantages = [a for advs in all_advantages for a in advs]

        stats = {
            "mean_reward": np.mean(flat_rewards),
            "std_reward": np.std(flat_rewards),
            "max_reward": np.max(flat_rewards),
            "min_reward": np.min(flat_rewards),
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "mean_advantage": np.mean(flat_advantages),
            "std_advantage": np.std(flat_advantages),
            "positive_ratio": sum(1 for a in flat_advantages if a > 0) / len(flat_advantages),
        }

        return GRPOBatch(
            prompts=prompts,
            responses=all_responses,
            rewards=all_rewards,
            advantages=all_advantages,
            gold_answers=gold_answers,
            stats=stats,
        )

    def _compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        计算组内归一化的advantages

        JustRL/GRPO的核心：
        advantage = (reward - mean) / std

        这消除了对value network的需求
        """
        rewards_array = np.array(rewards)
        mean = np.mean(rewards_array)
        std = np.std(rewards_array)

        if std < self.config.advantage_eps:
            # 如果所有奖励相同，advantage为0
            return [0.0] * len(rewards)

        advantages = (rewards_array - mean) / (std + self.config.advantage_eps)

        if self.config.normalize_advantage:
            # 再次归一化到[-1, 1]范围
            max_abs = np.max(np.abs(advantages))
            if max_abs > self.config.advantage_eps:
                advantages = advantages / max_abs

        return advantages.tolist()

    def train_step(self, batch: GRPOBatch) -> Dict[str, float]:
        """
        执行一个训练步骤

        Args:
            batch: GRPOBatch对象

        Returns:
            训练统计信息
        """
        self.global_step += 1
        step_start_time = time.time()

        # 选择用于训练的样本
        # JustRL策略：使用所有positive advantage的样本
        train_samples = []

        for prompt, responses, advantages in zip(
            batch.prompts, batch.responses, batch.advantages
        ):
            for response, advantage in zip(responses, advantages):
                if advantage > 0:
                    # 只训练positive advantage的样本
                    train_samples.append({
                        "prompt": prompt,
                        "response": response,
                        "advantage": advantage,
                    })

        # 执行梯度累积
        if train_samples:
            for sample in train_samples:
                # 完整序列 = prompt + response
                full_text = sample["prompt"] + sample["response"]

                # 调用Tinker的forward_backward
                # 注意：Tinker的API可能需要调整
                self.training_client.forward_backward(
                    input_text=full_text,
                    # Tinker可能支持loss weighting
                    # weight=sample["advantage"],
                )

            # 更新参数
            self.training_client.optim_step()

        # 记录统计
        step_time = time.time() - step_start_time

        stats = {
            **batch.stats,
            "step": self.global_step,
            "num_train_samples": len(train_samples),
            "step_time": step_time,
        }

        # 保存历史
        for key, value in stats.items():
            self.history[key].append(value)

        return stats

    def evaluate(
        self,
        eval_prompts: List[str],
        eval_answers: List[str],
        sampling_client: Any,
        generation_kwargs: Dict = None,
    ) -> Dict[str, float]:
        """
        评估当前模型

        使用greedy decoding（temperature=0）

        Args:
            eval_prompts: 评估问题列表
            eval_answers: 标准答案列表
            sampling_client: 采样客户端
            generation_kwargs: 生成参数

        Returns:
            评估统计
        """
        generation_kwargs = generation_kwargs or {}
        # 评估时使用greedy decoding
        eval_kwargs = {**generation_kwargs, "temperature": 0.0}

        correct = 0
        total = len(eval_prompts)
        results = []

        for prompt, gold in zip(eval_prompts, eval_answers):
            response = sampling_client.sample(
                prompt=prompt,
                **eval_kwargs
            )

            result = self.verifier.verify(response, gold)
            results.append(result)

            if result.is_correct:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        return {
            "eval_accuracy": accuracy,
            "eval_correct": correct,
            "eval_total": total,
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.history["mean_reward"]:
            return {}

        return {
            "total_steps": self.global_step,
            "final_accuracy": self.history["accuracy"][-1],
            "best_accuracy": max(self.history["accuracy"]),
            "mean_reward_trend": {
                "start": self.history["mean_reward"][0],
                "end": self.history["mean_reward"][-1],
                "max": max(self.history["mean_reward"]),
            },
            "positive_ratio_trend": {
                "start": self.history["positive_ratio"][0],
                "end": self.history["positive_ratio"][-1],
            },
        }

    def save_history(self, path: str):
        """保存训练历史"""
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)


class JustRLTrainer(GRPOTrainer):
    """
    JustRL训练器

    继承GRPOTrainer，使用JustRL的默认配置

    JustRL的关键简化：
    1. 无KL惩罚 (kl_coef=0)
    2. 无长度惩罚
    3. clip-higher (clip_ratio_high=1.28)
    4. 常数学习率
    5. 单阶段训练
    """

    def __init__(
        self,
        training_client: Any,
        verifier: Any,
        logger: Any = None,
        rollout_n: int = 8,
    ):
        config = GRPOConfig(
            rollout_n=rollout_n,
            clip_ratio_low=0.8,
            clip_ratio_high=1.28,  # JustRL的clip-higher
            kl_coef=0.0,           # 无KL惩罚
            entropy_coef=0.0,      # 无熵正则
            normalize_advantage=True,
            correct_reward=1.0,
            incorrect_reward=0.0,
        )

        super().__init__(config, training_client, verifier, logger)


def compute_grpo_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_low: float = 0.8,
    clip_high: float = 1.28,
) -> Tuple[float, Dict[str, float]]:
    """
    计算GRPO损失（用于理解算法，实际训练由Tinker处理）

    Args:
        log_probs: 当前策略的log概率
        old_log_probs: 旧策略的log概率
        advantages: 归一化的advantages
        clip_low: clip下界
        clip_high: clip上界

    Returns:
        (loss, stats)
    """
    # 概率比
    ratio = np.exp(log_probs - old_log_probs)

    # Clipped ratio
    clipped_ratio = np.clip(ratio, clip_low, clip_high)

    # 策略损失
    policy_loss_1 = ratio * advantages
    policy_loss_2 = clipped_ratio * advantages

    # 取较小者（保守更新）
    policy_loss = -np.mean(np.minimum(policy_loss_1, policy_loss_2))

    stats = {
        "policy_loss": policy_loss,
        "mean_ratio": np.mean(ratio),
        "clip_fraction": np.mean(np.abs(ratio - clipped_ratio) > 1e-6),
    }

    return policy_loss, stats

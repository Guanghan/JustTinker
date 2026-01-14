#!/usr/bin/env python3
"""
训练曲线绘图脚本

用法:
    # 绘制最新的训练结果
    python scripts/plot_training.py

    # 指定具体的history.json文件
    python scripts/plot_training.py outputs/justrl/justrl_gsm8k_20260110_123456/history.json

    # 只生成简洁的accuracy曲线
    python scripts/plot_training.py --simple

    # 指定输出目录
    python scripts/plot_training.py --output-dir outputs/plots

输出内容:

    完整版（默认）：生成 8 个子图的 PNG + PDF (4x2 布局)
    - Train Accuracy 曲线 (训练数据集 DAPO-MATH-17K)
    - Eval MATH Accuracy 曲线
    - Eval AIME Accuracy 曲线
    - Mean Reward 曲线
    - Redundancy Metrics 曲线 (Redundancy Score + Chunk Similarity)
    - Step Time 曲线
    - Thinking Rate 曲线 (Train + Eval MATH + Eval AIME)
    - Response Length 曲线（总长度 + Thinking 长度）

    简洁版（--simple）：只有 Accuracy 曲线

Author: Guanghan Ning
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def find_latest_history() -> Path:
    """查找最新的history.json文件"""
    # output_dir = Path("outputs/justrl")
    output_dir = Path("outputs/justrl_reasoning")
    if not output_dir.exists():
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")

    # 查找所有history.json
    history_files = list(output_dir.glob("*/history.json"))
    if not history_files:
        raise FileNotFoundError("未找到history.json文件，请先运行训练")

    # 按修改时间排序，返回最新的
    latest = max(history_files, key=lambda p: p.stat().st_mtime)
    return latest


def load_history(path: Path) -> dict:
    """加载训练历史"""
    with open(path) as f:
        return json.load(f)


def plot_training_curves(history: dict, output_path: Path, title_prefix: str = ""):
    """绘制训练曲线"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")  # 无头模式，适合服务器
    except ImportError:
        print("Error: 请安装matplotlib: pip install matplotlib")
        return

    # 设置中文字体（如果可用）
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 获取数据
    steps = history.get("step", [])
    accuracy = history.get("accuracy", [])
    mean_reward = history.get("mean_reward", [])
    num_train_samples = history.get("num_train_samples", [])
    step_time = history.get("step_time", [])
    thinking_rate = history.get("thinking_rate", [])
    avg_response_length = history.get("avg_response_length", [])
    avg_thinking_length = history.get("avg_thinking_length", [])
    # Redundancy 数据 (新增)
    avg_redundancy_score = history.get("avg_redundancy_score", [])
    avg_chunk_similarity = history.get("avg_chunk_similarity", [])
    # Eval 数据
    eval_steps = history.get("eval_step", [])
    eval_accuracy = history.get("eval_accuracy", [])
    eval_thinking_rate = history.get("eval_thinking_rate", [])
    eval_aime_accuracy = history.get("eval_aime_accuracy", [])
    eval_aime_thinking_rate = history.get("eval_aime_thinking_rate", [])

    if not steps:
        print("Warning: 历史数据为空")
        return

    # 创建图表 (4x2 = 8 个子图)
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(f"{title_prefix}JustRL Training Progress", fontsize=14, fontweight="bold")

    # 1. Train Accuracy曲线 (DAPO-MATH-17K)
    ax1 = axes[0, 0]
    if accuracy:
        ax1.plot(steps, accuracy, "b-", linewidth=1.5, alpha=0.7, label="Train Accuracy")
        ax1.fill_between(steps, accuracy, alpha=0.2, color="blue")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Train Accuracy (DAPO-MATH-17K)")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        if len(accuracy) > 0:
            ax1.annotate(
                f"{accuracy[-1]:.2%}",
                xy=(steps[-1], accuracy[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="blue",
            )

    # 2. Eval MATH Accuracy 曲线
    ax2 = axes[0, 1]
    if eval_steps and eval_accuracy:
        ax2.plot(eval_steps, eval_accuracy, "r-o", linewidth=2, markersize=6, label="Eval MATH")
        ax2.fill_between(eval_steps, eval_accuracy, alpha=0.2, color="red")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Eval MATH Accuracy (200 samples, pass@1, temp=0.7)")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        # 标注起始值和最终值
        if len(eval_accuracy) > 0:
            ax2.annotate(
                f"SFT: {eval_accuracy[0]:.2%}",
                xy=(eval_steps[0], eval_accuracy[0]),
                xytext=(10, -15),
                textcoords="offset points",
                fontsize=10,
                color="red",
            )
            ax2.annotate(
                f"Final: {eval_accuracy[-1]:.2%}",
                xy=(eval_steps[-1], eval_accuracy[-1]),
                xytext=(-60, 10),
                textcoords="offset points",
                fontsize=10,
                color="red",
            )
    else:
        ax2.text(0.5, 0.5, "No MATH eval data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Eval MATH Accuracy")

    # 3. Eval AIME Accuracy 曲线
    ax3 = axes[1, 0]
    if eval_steps and eval_aime_accuracy and len(eval_aime_accuracy) > 0:
        aime_eval_steps = eval_steps[-len(eval_aime_accuracy) :]
        ax3.plot(aime_eval_steps, eval_aime_accuracy, "g-s", linewidth=2, markersize=6, label="Eval AIME")
        ax3.fill_between(aime_eval_steps, eval_aime_accuracy, alpha=0.2, color="green")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Eval AIME 2024 Accuracy (30 samples, pass@1, temp=0.7)")
        # 动态调整y轴范围，让变化更明显
        min_acc = min(eval_aime_accuracy)
        max_acc = max(eval_aime_accuracy)
        margin = (max_acc - min_acc) * 0.3 if max_acc > min_acc else 0.05
        ax3.set_ylim(max(0, min_acc - margin - 0.05), min(1, max_acc + margin + 0.05))
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        if len(eval_aime_accuracy) > 0:
            ax3.annotate(
                f"SFT: {eval_aime_accuracy[0]:.2%}",
                xy=(aime_eval_steps[0], eval_aime_accuracy[0]),
                xytext=(10, -15),
                textcoords="offset points",
                fontsize=10,
                color="green",
            )
            ax3.annotate(
                f"Final: {eval_aime_accuracy[-1]:.2%}",
                xy=(aime_eval_steps[-1], eval_aime_accuracy[-1]),
                xytext=(-60, 10),
                textcoords="offset points",
                fontsize=10,
                color="green",
            )
            # 标注最高值（如果有多个相同最大值，取最后一个）
            max_aime_acc = max(eval_aime_accuracy)
            # 找到所有最大值的索引，取最后一个
            max_aime_idx = len(eval_aime_accuracy) - 1 - eval_aime_accuracy[::-1].index(max_aime_acc)
            max_aime_step = aime_eval_steps[max_aime_idx]
            ax3.annotate(
                f"Best: {max_aime_acc:.2%}",
                xy=(max_aime_step, max_aime_acc),
                xytext=(0, 15),
                textcoords="offset points",
                fontsize=10,
                color="darkgreen",
                fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
            )
    else:
        ax3.text(0.5, 0.5, "No AIME eval data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Eval AIME Accuracy")

    # 4. Mean Reward曲线
    ax4 = axes[1, 1]
    if mean_reward:
        ax4.plot(steps, mean_reward, "g-", linewidth=1.5, label="Mean Reward")
        ax4.fill_between(steps, mean_reward, alpha=0.3, color="green")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Mean Reward")
        ax4.set_title("Mean Reward per Step")
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        if len(mean_reward) > 0:
            ax4.annotate(
                f"{mean_reward[-1]:.3f}",
                xy=(steps[-1], mean_reward[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="green",
            )

    # 5. Redundancy Metrics
    ax5 = axes[2, 0]
    if avg_redundancy_score:
        ax5.plot(
            steps[: len(avg_redundancy_score)], avg_redundancy_score, "red", linewidth=1.5, label="Redundancy Score"
        )
        ax5.axhline(y=0.3, color="orange", linestyle="--", alpha=0.7, label="Threshold (0.3)")
        if avg_chunk_similarity:
            ax5.plot(
                steps[: len(avg_chunk_similarity)],
                avg_chunk_similarity,
                "purple",
                linewidth=1.5,
                alpha=0.7,
                label="Chunk Similarity",
            )
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Score")
        ax5.set_title("Redundancy Metrics")
        ax5.set_ylim(0, 0.6)
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        avg_red = sum(avg_redundancy_score) / len(avg_redundancy_score)
        ax5.axhline(y=avg_red, color="red", linestyle=":", alpha=0.5)
        ax5.annotate(
            f"Avg: {avg_red:.1%}",
            xy=(steps[len(avg_redundancy_score) - 1], avg_red),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="red",
        )
    elif num_train_samples:
        ax5.bar(steps, num_train_samples, alpha=0.7, color="orange", label="Positive Samples")
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Count")
        ax5.set_title("Positive Advantage Samples per Step")
        ax5.grid(True, alpha=0.3, axis="y")
        ax5.legend()
    else:
        ax5.text(
            0.5, 0.5, "No redundancy data available", ha="center", va="center", transform=ax5.transAxes, fontsize=12
        )
        ax5.set_title("Redundancy Metrics")

    # 6. Step Time
    ax6 = axes[2, 1]
    if step_time:
        ax6.plot(steps, step_time, "r-", linewidth=1, alpha=0.7)
        ax6.set_xlabel("Step")
        ax6.set_ylabel("Time (seconds)")
        ax6.set_title("Time per Step")
        ax6.grid(True, alpha=0.3)
        avg_time = sum(step_time) / len(step_time)
        ax6.axhline(y=avg_time, color="r", linestyle="--", alpha=0.5, label=f"Avg: {avg_time:.1f}s")
        ax6.legend()

    # 7. Thinking Rate (Train + Eval MATH + Eval AIME)
    ax7 = axes[3, 0]
    if thinking_rate:
        ax7.plot(steps, thinking_rate, "purple", linewidth=1.5, alpha=0.7, label="Train (DAPO)")
        ax7.fill_between(steps, thinking_rate, alpha=0.2, color="purple")
        # Eval MATH thinking rate
        if eval_steps and eval_thinking_rate:
            ax7.plot(eval_steps, eval_thinking_rate, "orange", linewidth=2, marker="o", markersize=4, label="Eval MATH")
            if len(eval_thinking_rate) > 0:
                ax7.annotate(
                    f"{eval_thinking_rate[-1]:.0%}",
                    xy=(eval_steps[-1], eval_thinking_rate[-1]),
                    xytext=(5, -15),
                    textcoords="offset points",
                    fontsize=10,
                    color="orange",
                )
        # Eval AIME thinking rate
        if eval_steps and eval_aime_thinking_rate and len(eval_aime_thinking_rate) > 0:
            aime_eval_steps = eval_steps[-len(eval_aime_thinking_rate) :]
            ax7.plot(
                aime_eval_steps,
                eval_aime_thinking_rate,
                "green",
                linewidth=2,
                marker="s",
                markersize=4,
                label="Eval AIME",
            )
            ax7.annotate(
                f"{eval_aime_thinking_rate[-1]:.0%}",
                xy=(aime_eval_steps[-1], eval_aime_thinking_rate[-1]),
                xytext=(5, 10),
                textcoords="offset points",
                fontsize=10,
                color="green",
            )
        ax7.set_xlabel("Step")
        ax7.set_ylabel("Rate")
        ax7.set_title("Thinking Token Usage Rate")
        ax7.set_ylim(0, 1)
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        if len(thinking_rate) > 0:
            ax7.annotate(
                f"{thinking_rate[-1]:.0%}",
                xy=(steps[-1], thinking_rate[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="purple",
            )

    # 8. Response Length (avg total + avg thinking)
    ax8 = axes[3, 1]
    if avg_response_length:
        ax8.plot(steps, avg_response_length, "steelblue", linewidth=1.5, label="Avg Response Length")
        if avg_thinking_length:
            ax8.plot(steps, avg_thinking_length, "coral", linewidth=1.5, label="Avg Thinking Length")
        ax8.set_xlabel("Step")
        ax8.set_ylabel("Tokens")
        ax8.set_title("Average Response & Thinking Length")
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        if len(avg_response_length) > 0:
            ax8.annotate(
                f"{avg_response_length[-1]:.0f}",
                xy=(steps[-1], avg_response_length[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color="steelblue",
            )
        if avg_thinking_length and len(avg_thinking_length) > 0:
            ax8.annotate(
                f"{avg_thinking_length[-1]:.0f}",
                xy=(steps[-1], avg_thinking_length[-1]),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=9,
                color="coral",
            )

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {output_path}")

    # 同时保存PDF版本（高质量）
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF版本: {pdf_path}")

    plt.close()


def plot_accuracy_only(history: dict, output_path: Path):
    """只绘制Accuracy曲线（简洁版）"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
    except ImportError:
        print("Error: 请安装matplotlib: pip install matplotlib")
        return

    steps = history.get("step", [])
    accuracy = history.get("accuracy", [])

    if not steps or not accuracy:
        print("Warning: 数据为空")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, accuracy, "b-", linewidth=2, marker="o", markersize=3)
    ax.fill_between(steps, accuracy, alpha=0.2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("JustRL Training Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 标注起始和结束值
    ax.annotate(
        f"Start: {accuracy[0]:.2%}",
        xy=(steps[0], accuracy[0]),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.annotate(
        f"Final: {accuracy[-1]:.2%}",
        xy=(steps[-1], accuracy[-1]),
        xytext=(-60, 20),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"简洁版图表已保存: {output_path}")
    plt.close()


def print_summary(history: dict):
    """打印训练摘要"""
    steps = history.get("step", [])
    accuracy = history.get("accuracy", [])
    mean_reward = history.get("mean_reward", [])
    step_time = history.get("step_time", [])
    thinking_rate = history.get("thinking_rate", [])
    avg_response_length = history.get("avg_response_length", [])
    avg_thinking_length = history.get("avg_thinking_length", [])
    # Redundancy 数据
    avg_redundancy_score = history.get("avg_redundancy_score", [])
    avg_chunk_similarity = history.get("avg_chunk_similarity", [])
    # Eval 数据
    eval_steps = history.get("eval_step", [])
    eval_accuracy = history.get("eval_accuracy", [])
    eval_thinking_rate = history.get("eval_thinking_rate", [])
    eval_aime_accuracy = history.get("eval_aime_accuracy", [])
    eval_aime_thinking_rate = history.get("eval_aime_thinking_rate", [])

    print("\n" + "=" * 50)
    print("训练摘要")
    print("=" * 50)

    if steps:
        print(f"总步数: {len(steps)}")

    if accuracy:
        print(f"初始 Train 准确率: {accuracy[0]:.2%}")
        print(f"最终 Train 准确率: {accuracy[-1]:.2%}")
        print(f"最高 Train 准确率: {max(accuracy):.2%} (Step {steps[accuracy.index(max(accuracy))]})")
        print(f"Train 准确率提升: {accuracy[-1] - accuracy[0]:+.2%}")

    if eval_accuracy:
        print("\n--- Eval MATH 指标 ---")
        print(f"Eval 次数: {len(eval_accuracy)}")
        print(f"初始 Eval 准确率: {eval_accuracy[0]:.2%} (Step {eval_steps[0]})")
        print(f"最终 Eval 准确率: {eval_accuracy[-1]:.2%} (Step {eval_steps[-1]})")
        print(
            f"最高 Eval 准确率: {max(eval_accuracy):.2%} (Step {eval_steps[eval_accuracy.index(max(eval_accuracy))]})"
        )

    if eval_aime_accuracy:
        print("\n--- Eval AIME 指标 ---")
        aime_eval_steps = eval_steps[-len(eval_aime_accuracy) :]
        print(f"AIME Eval 次数: {len(eval_aime_accuracy)}")
        print(f"初始 AIME 准确率: {eval_aime_accuracy[0]:.2%} (Step {aime_eval_steps[0]})")
        print(f"最终 AIME 准确率: {eval_aime_accuracy[-1]:.2%} (Step {aime_eval_steps[-1]})")
        print(f"最高 AIME 准确率: {max(eval_aime_accuracy):.2%}")

    if mean_reward:
        print(f"\n平均奖励: {sum(mean_reward) / len(mean_reward):.3f}")

    print("\n--- Thinking Rate ---")
    if thinking_rate:
        print(f"Train (DAPO): {thinking_rate[0]:.0%} → {thinking_rate[-1]:.0%}")
    if eval_thinking_rate:
        print(f"Eval MATH: {eval_thinking_rate[0]:.0%} → {eval_thinking_rate[-1]:.0%}")
    if eval_aime_thinking_rate:
        print(f"Eval AIME: {eval_aime_thinking_rate[0]:.0%} → {eval_aime_thinking_rate[-1]:.0%}")

    if avg_response_length:
        print(f"平均响应长度: {sum(avg_response_length) / len(avg_response_length):.0f} tokens")

    if avg_thinking_length:
        valid_lengths = [length for length in avg_thinking_length if length > 0]
        if valid_lengths:
            print(f"平均 Thinking 长度: {sum(valid_lengths) / len(valid_lengths):.0f} tokens")

    if avg_redundancy_score:
        print("\n--- Redundancy 指标 ---")
        print(f"平均 Redundancy Score: {sum(avg_redundancy_score) / len(avg_redundancy_score):.2%}")
        print(f"最终 Redundancy Score: {avg_redundancy_score[-1]:.2%}")
        if avg_chunk_similarity:
            print(f"平均 Chunk Similarity: {sum(avg_chunk_similarity) / len(avg_chunk_similarity):.2%}")

    if step_time:
        total_time = sum(step_time)
        print(f"\n总训练时间: {total_time / 60:.1f} 分钟")
        print(f"平均每步时间: {total_time / len(step_time):.1f} 秒")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="绘制JustRL训练曲线")
    parser.add_argument("history_file", nargs="?", default=None, help="history.json文件路径（默认使用最新的）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认与history.json同目录）")
    parser.add_argument("--simple", action="store_true", help="只生成简洁的accuracy曲线")
    args = parser.parse_args()

    # 找到history文件
    if args.history_file:
        history_path = Path(args.history_file)
        if not history_path.exists():
            print(f"Error: 文件不存在: {history_path}")
            return
    else:
        try:
            history_path = find_latest_history()
            print(f"使用最新的训练结果: {history_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    # 加载数据
    history = load_history(history_path)

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = history_path.parent

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 打印摘要
    print_summary(history)

    # 绘图
    if args.simple:
        output_path = output_dir / f"accuracy_{timestamp}.png"
        plot_accuracy_only(history, output_path)
    else:
        output_path = output_dir / f"training_curves_{timestamp}.png"
        plot_training_curves(history, output_path)


if __name__ == "__main__":
    main()

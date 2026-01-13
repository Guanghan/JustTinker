#!/usr/bin/env python3
"""
Cold Start SFT 训练曲线绘图脚本

用法:
    # 绘制最新的 SFT 训练结果
    python scripts/plot_sft_training.py

    # 指定具体的 history.json 文件
    python scripts/plot_sft_training.py outputs/coldstart_sft/coldstart_sft_small_20260110_123456/history.json

    # 只生成简洁的 loss 曲线
    python scripts/plot_sft_training.py --simple

    # 指定输出目录
    python scripts/plot_sft_training.py --output-dir outputs/plots

输出内容:

    完整版（默认）：生成 2x2 子图的 PNG + PDF
    - Training Loss 曲线
    - Learning Rate 曲线（含 warmup）
    - Thinking Rate（评估时）
    - Boxed Rate（评估时）

    简洁版（--simple）：只有 Loss + Thinking Rate

SFT vs RLVR 的区别：
    - SFT 使用 Loss（越低越好），RLVR 使用 Reward（越高越好）
    - SFT 的核心指标是 Thinking Rate（格式学习）
    - SFT 有 Learning Rate Warmup

Author: Guanghan Ning
Date: 2025-01-10
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def find_latest_history(output_dir: str = "outputs/coldstart_sft") -> tuple[Path, Path | None]:
    """
    查找最新的 history.json 和 eval_history.json 文件

    Returns:
        (history_path, eval_history_path) - eval_history_path 可能为 None
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"输出目录不存在: {output_path}")

    # 查找所有 history.json
    history_files = list(output_path.glob("*/history.json"))
    if not history_files:
        raise FileNotFoundError("未找到 history.json 文件，请先运行 SFT 训练")

    # 按修改时间排序，返回最新的
    latest = max(history_files, key=lambda p: p.stat().st_mtime)

    # 查找对应的 eval_history.json
    eval_history = latest.parent / "eval_history.json"
    if eval_history.exists():
        return latest, eval_history
    return latest, None


def load_json(path: Path) -> dict:
    """加载 JSON 文件"""
    with open(path) as f:
        return json.load(f)


def plot_sft_curves(
    history: dict,
    eval_history: dict | None,
    output_path: Path,
    title_prefix: str = "",
):
    """
    绘制 SFT 训练曲线（完整版 2x2 布局）

    Args:
        history: 训练历史 (step, loss, lr, tokens, step_time)
        eval_history: 评估历史 (step, thinking_rate, boxed_rate)
        output_path: 输出文件路径
        title_prefix: 标题前缀
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")  # 无头模式，适合服务器
    except ImportError:
        print("Error: 请安装 matplotlib: pip install matplotlib")
        return

    # 设置字体
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 获取训练数据
    steps = history.get("step", [])
    loss = history.get("loss", [])
    lr = history.get("lr", [])
    step_time = history.get("step_time", [])

    if not steps:
        print("Warning: 训练历史数据为空")
        return

    # 获取评估数据
    eval_steps = []
    thinking_rate = []
    boxed_rate = []
    avg_response_length = []
    avg_thinking_length = []
    if eval_history:
        eval_steps = eval_history.get("step", [])
        thinking_rate = eval_history.get("thinking_rate", [])
        boxed_rate = eval_history.get("boxed_rate", [])
        avg_response_length = eval_history.get("avg_response_length", [])
        avg_thinking_length = eval_history.get("avg_thinking_length", [])

    # 创建图表 (3x2 布局以包含新指标)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f"{title_prefix}Cold Start SFT Training Progress", fontsize=14, fontweight="bold")

    # 1. Loss 曲线
    ax1 = axes[0, 0]
    if loss:
        ax1.plot(steps, loss, "b-", linewidth=1.5, label="Training Loss")
        ax1.fill_between(steps, loss, alpha=0.2)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss (Cross-Entropy)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 添加最终值标注
        if len(loss) > 0:
            ax1.annotate(
                f"{loss[-1]:.4f}",
                xy=(steps[-1], loss[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="blue",
            )

    # 2. Learning Rate 曲线（显示 warmup）
    ax2 = axes[0, 1]
    if lr:
        ax2.plot(steps, lr, "orange", linewidth=1.5, label="Learning Rate")
        ax2.fill_between(steps, lr, alpha=0.2, color="orange")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate (with Warmup)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # 标注 warmup 结束点
        max_lr = max(lr)
        warmup_end_idx = next((i for i, lr_val in enumerate(lr) if lr_val >= max_lr * 0.99), len(lr) - 1)
        if warmup_end_idx > 0 and warmup_end_idx < len(steps) - 1:
            ax2.axvline(x=steps[warmup_end_idx], color="red", linestyle="--", alpha=0.5, label="Warmup End")
            ax2.legend()

    # 3. Thinking Rate 曲线
    ax3 = axes[1, 0]
    if thinking_rate:
        ax3.plot(eval_steps, thinking_rate, "g-", linewidth=2, marker="o", markersize=6, label="Thinking Rate")
        ax3.fill_between(eval_steps, thinking_rate, alpha=0.2, color="green")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Thinking Rate")
        ax3.set_title("Thinking Format Rate (Eval)")
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 添加最终值标注
        if len(thinking_rate) > 0:
            ax3.annotate(
                f"{thinking_rate[-1]:.1%}",
                xy=(eval_steps[-1], thinking_rate[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="green",
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "No eval data available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
            color="gray",
        )
        ax3.set_title("Thinking Format Rate (Eval)")

    # 4. Boxed Rate 或 Step Time
    ax4 = axes[1, 1]
    if boxed_rate:
        ax4.plot(eval_steps, boxed_rate, "purple", linewidth=2, marker="s", markersize=6, label="Boxed Rate")
        ax4.fill_between(eval_steps, boxed_rate, alpha=0.2, color="purple")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Boxed Rate")
        ax4.set_title("Answer Format Rate (\\boxed{})")
        ax4.set_ylim(-0.05, 1.05)
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        if len(boxed_rate) > 0:
            ax4.annotate(
                f"{boxed_rate[-1]:.1%}",
                xy=(eval_steps[-1], boxed_rate[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="purple",
            )
    elif step_time:
        # 如果没有 boxed_rate，显示 step time
        ax4.plot(steps, step_time, "r-", linewidth=1, alpha=0.7)
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Time (seconds)")
        ax4.set_title("Time per Step")
        ax4.grid(True, alpha=0.3)

        # 显示平均时间
        avg_time = sum(step_time) / len(step_time)
        ax4.axhline(y=avg_time, color="r", linestyle="--", alpha=0.5, label=f"Avg: {avg_time:.1f}s")
        ax4.legend()
    else:
        ax4.text(
            0.5, 0.5, "No data available", ha="center", va="center", transform=ax4.transAxes, fontsize=12, color="gray"
        )
        ax4.set_title("Boxed Rate / Step Time")

    # 5. Average Response Length 曲线
    ax5 = axes[2, 0]
    if avg_response_length:
        ax5.plot(
            eval_steps,
            avg_response_length,
            "tab:cyan",
            linewidth=2,
            marker="o",
            markersize=6,
            label="Avg Response Length",
        )
        ax5.fill_between(eval_steps, avg_response_length, alpha=0.2, color="tab:cyan")
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Length (chars)")
        ax5.set_title("Average Response Length (Eval)")
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        if len(avg_response_length) > 0:
            ax5.annotate(
                f"{avg_response_length[-1]:.0f}",
                xy=(eval_steps[-1], avg_response_length[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="tab:cyan",
            )
    else:
        ax5.text(
            0.5,
            0.5,
            "No eval data available",
            ha="center",
            va="center",
            transform=ax5.transAxes,
            fontsize=12,
            color="gray",
        )
        ax5.set_title("Average Response Length (Eval)")

    # 6. Average Thinking Length 曲线
    ax6 = axes[2, 1]
    if avg_thinking_length:
        ax6.plot(
            eval_steps,
            avg_thinking_length,
            "tab:brown",
            linewidth=2,
            marker="s",
            markersize=6,
            label="Avg Thinking Length",
        )
        ax6.fill_between(eval_steps, avg_thinking_length, alpha=0.2, color="tab:brown")
        ax6.set_xlabel("Step")
        ax6.set_ylabel("Length (chars)")
        ax6.set_title("Average Thinking Content Length (Eval)")
        ax6.grid(True, alpha=0.3)
        ax6.legend()

        if len(avg_thinking_length) > 0:
            ax6.annotate(
                f"{avg_thinking_length[-1]:.0f}",
                xy=(eval_steps[-1], avg_thinking_length[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="tab:brown",
            )
    else:
        ax6.text(
            0.5,
            0.5,
            "No eval data available",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=12,
            color="gray",
        )
        ax6.set_title("Average Thinking Content Length (Eval)")

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {output_path}")

    # 同时保存 PDF 版本（高质量）
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF 版本: {pdf_path}")

    plt.close()


def plot_simple(
    history: dict,
    eval_history: dict | None,
    output_path: Path,
):
    """
    绘制简洁版曲线（Loss + Thinking Rate）
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
    except ImportError:
        print("Error: 请安装 matplotlib: pip install matplotlib")
        return

    steps = history.get("step", [])
    loss = history.get("loss", [])

    eval_steps = []
    thinking_rate = []
    if eval_history:
        eval_steps = eval_history.get("step", [])
        thinking_rate = eval_history.get("thinking_rate", [])

    if not steps or not loss:
        print("Warning: 数据为空")
        return

    # 创建双 Y 轴图表
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左 Y 轴：Loss
    color1 = "tab:blue"
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Loss", color=color1, fontsize=12)
    line1 = ax1.plot(steps, loss, color=color1, linewidth=2, label="Training Loss")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.fill_between(steps, loss, alpha=0.1, color=color1)

    # 右 Y 轴：Thinking Rate
    if thinking_rate:
        ax2 = ax1.twinx()
        color2 = "tab:green"
        ax2.set_ylabel("Thinking Rate", color=color2, fontsize=12)
        line2 = ax2.plot(
            eval_steps, thinking_rate, color=color2, linewidth=2, marker="o", markersize=6, label="Thinking Rate"
        )
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(-0.05, 1.05)

        # 合并图例
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    ax1.set_title("Cold Start SFT: Loss and Thinking Rate", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 标注起始和结束值
    ax1.annotate(
        f"Start: {loss[0]:.4f}",
        xy=(steps[0], loss[0]),
        xytext=(10, 20),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax1.annotate(
        f"Final: {loss[-1]:.4f}",
        xy=(steps[-1], loss[-1]),
        xytext=(-60, -20),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"简洁版图表已保存: {output_path}")
    plt.close()


def print_summary(history: dict, eval_history: dict | None):
    """打印训练摘要"""
    steps = history.get("step", [])
    loss = history.get("loss", [])
    lr = history.get("lr", [])
    step_time = history.get("step_time", [])

    print("\n" + "=" * 50)
    print("Cold Start SFT 训练摘要")
    print("=" * 50)

    if steps:
        print(f"总步数: {len(steps)}")

    if loss:
        print(f"初始 Loss: {loss[0]:.4f}")
        print(f"最终 Loss: {loss[-1]:.4f}")
        print(f"最低 Loss: {min(loss):.4f} (Step {steps[loss.index(min(loss))]})")
        print(f"Loss 变化: {loss[-1] - loss[0]:+.4f}")

    if lr:
        print(f"最终 LR: {lr[-1]:.2e}")

    if step_time:
        total_time = sum(step_time)
        print(f"总训练时间: {total_time / 60:.1f} 分钟")
        print(f"平均每步时间: {total_time / len(step_time):.1f} 秒")

    if eval_history:
        thinking_rate = eval_history.get("thinking_rate", [])
        boxed_rate = eval_history.get("boxed_rate", [])
        avg_response_length = eval_history.get("avg_response_length", [])
        avg_thinking_length = eval_history.get("avg_thinking_length", [])

        print("-" * 50)
        print("评估指标:")

        if thinking_rate:
            print(f"初始 Thinking Rate: {thinking_rate[0]:.1%}")
            print(f"最终 Thinking Rate: {thinking_rate[-1]:.1%}")
            print(f"最高 Thinking Rate: {max(thinking_rate):.1%}")

        if boxed_rate:
            print(f"最终 Boxed Rate: {boxed_rate[-1]:.1%}")

        if avg_response_length:
            print(f"平均响应长度: {avg_response_length[-1]:.0f} 字符")

        if avg_thinking_length:
            print(f"平均思考长度: {avg_thinking_length[-1]:.0f} 字符")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="绘制 Cold Start SFT 训练曲线")
    parser.add_argument("history_file", nargs="?", default=None, help="history.json 文件路径（默认使用最新的）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认与 history.json 同目录）")
    parser.add_argument("--simple", action="store_true", help="只生成简洁的 Loss + Thinking Rate 曲线")
    args = parser.parse_args()

    # 找到 history 文件
    if args.history_file:
        history_path = Path(args.history_file)
        if not history_path.exists():
            print(f"Error: 文件不存在: {history_path}")
            return

        # 查找对应的 eval_history.json
        eval_history_path = history_path.parent / "eval_history.json"
        if not eval_history_path.exists():
            eval_history_path = None
    else:
        try:
            history_path, eval_history_path = find_latest_history()
            print(f"使用最新的 SFT 训练结果: {history_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    # 加载数据
    history = load_json(history_path)
    eval_history = None
    if eval_history_path:
        eval_history = load_json(eval_history_path)
        print(f"加载评估历史: {eval_history_path}")

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = history_path.parent

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 打印摘要
    print_summary(history, eval_history)

    # 绘图
    if args.simple:
        output_path = output_dir / f"sft_simple_{timestamp}.png"
        plot_simple(history, eval_history, output_path)
    else:
        output_path = output_dir / f"sft_training_curves_{timestamp}.png"
        plot_sft_curves(history, eval_history, output_path)


if __name__ == "__main__":
    main()

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

    完整版（默认）：生成 4 个子图的 PNG + PDF
    - Training Accuracy 曲线
    - Mean Reward 曲线
    - Positive Samples 柱状图
    - Step Time 曲线

    简洁版（--simple）：只有 Accuracy 曲线

Author: Guanghan Ning
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def find_latest_history() -> Path:
    """查找最新的history.json文件"""
    output_dir = Path("outputs/justrl")
    if not output_dir.exists():
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")

    # 查找所有history.json
    history_files = list(output_dir.glob("*/history.json"))
    if not history_files:
        raise FileNotFoundError(f"未找到history.json文件，请先运行训练")

    # 按修改时间排序，返回最新的
    latest = max(history_files, key=lambda p: p.stat().st_mtime)
    return latest


def load_history(path: Path) -> dict:
    """加载训练历史"""
    with open(path, "r") as f:
        return json.load(f)


def plot_training_curves(history: dict, output_path: Path, title_prefix: str = ""):
    """绘制训练曲线"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 无头模式，适合服务器
    except ImportError:
        print("Error: 请安装matplotlib: pip install matplotlib")
        return

    # 设置中文字体（如果可用）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取数据
    steps = history.get("step", [])
    accuracy = history.get("accuracy", [])
    mean_reward = history.get("mean_reward", [])
    num_train_samples = history.get("num_train_samples", [])
    step_time = history.get("step_time", [])

    if not steps:
        print("Warning: 历史数据为空")
        return

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix}JustRL Training Progress", fontsize=14, fontweight='bold')

    # 1. Accuracy曲线
    ax1 = axes[0, 0]
    if accuracy:
        ax1.plot(steps, accuracy, 'b-', linewidth=1.5, label='Train Accuracy')
        ax1.fill_between(steps, accuracy, alpha=0.3)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 添加最终值标注
        if len(accuracy) > 0:
            ax1.annotate(f'{accuracy[-1]:.2%}',
                        xy=(steps[-1], accuracy[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, color='blue')

    # 2. Mean Reward曲线
    ax2 = axes[0, 1]
    if mean_reward:
        ax2.plot(steps, mean_reward, 'g-', linewidth=1.5, label='Mean Reward')
        ax2.fill_between(steps, mean_reward, alpha=0.3, color='green')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Mean Reward')
        ax2.set_title('Mean Reward per Step')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        if len(mean_reward) > 0:
            ax2.annotate(f'{mean_reward[-1]:.3f}',
                        xy=(steps[-1], mean_reward[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, color='green')

    # 3. 训练样本数
    ax3 = axes[1, 0]
    if num_train_samples:
        ax3.bar(steps, num_train_samples, alpha=0.7, color='orange', label='Positive Samples')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Count')
        ax3.set_title('Positive Advantage Samples per Step')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend()

    # 4. Step Time
    ax4 = axes[1, 1]
    if step_time:
        ax4.plot(steps, step_time, 'r-', linewidth=1, alpha=0.7)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Time per Step')
        ax4.grid(True, alpha=0.3)

        # 显示平均时间
        avg_time = sum(step_time) / len(step_time)
        ax4.axhline(y=avg_time, color='r', linestyle='--', alpha=0.5, label=f'Avg: {avg_time:.1f}s')
        ax4.legend()

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")

    # 同时保存PDF版本（高质量）
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF版本: {pdf_path}")

    plt.close()


def plot_accuracy_only(history: dict, output_path: Path):
    """只绘制Accuracy曲线（简洁版）"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Error: 请安装matplotlib: pip install matplotlib")
        return

    steps = history.get("step", [])
    accuracy = history.get("accuracy", [])

    if not steps or not accuracy:
        print("Warning: 数据为空")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, accuracy, 'b-', linewidth=2, marker='o', markersize=3)
    ax.fill_between(steps, accuracy, alpha=0.2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('JustRL Training Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 标注起始和结束值
    ax.annotate(f'Start: {accuracy[0]:.2%}',
                xy=(steps[0], accuracy[0]),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'Final: {accuracy[-1]:.2%}',
                xy=(steps[-1], accuracy[-1]),
                xytext=(-60, 20), textcoords='offset points',
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"简洁版图表已保存: {output_path}")
    plt.close()


def print_summary(history: dict):
    """打印训练摘要"""
    steps = history.get("step", [])
    accuracy = history.get("accuracy", [])
    mean_reward = history.get("mean_reward", [])
    step_time = history.get("step_time", [])

    print("\n" + "=" * 50)
    print("训练摘要")
    print("=" * 50)

    if steps:
        print(f"总步数: {len(steps)}")

    if accuracy:
        print(f"初始准确率: {accuracy[0]:.2%}")
        print(f"最终准确率: {accuracy[-1]:.2%}")
        print(f"最高准确率: {max(accuracy):.2%} (Step {steps[accuracy.index(max(accuracy))]})")
        print(f"准确率提升: {accuracy[-1] - accuracy[0]:+.2%}")

    if mean_reward:
        print(f"平均奖励: {sum(mean_reward)/len(mean_reward):.3f}")

    if step_time:
        total_time = sum(step_time)
        print(f"总训练时间: {total_time/60:.1f} 分钟")
        print(f"平均每步时间: {total_time/len(step_time):.1f} 秒")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="绘制JustRL训练曲线")
    parser.add_argument("history_file", nargs="?", default=None,
                        help="history.json文件路径（默认使用最新的）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认与history.json同目录）")
    parser.add_argument("--simple", action="store_true",
                        help="只生成简洁的accuracy曲线")
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

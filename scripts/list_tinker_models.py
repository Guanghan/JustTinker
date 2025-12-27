#!/usr/bin/env python3
"""
查询Tinker支持的模型列表

使用方法:
    python scripts/list_tinker_models.py
"""

import os
import sys

def main():
    # 检查API Key（只是提示，不强制）
    api_key = os.getenv("TINKER_API_KEY")
    has_api_key = bool(api_key)

    if not has_api_key:
        print("⚠️  TINKER_API_KEY 未设置（仅显示已知模型列表）")
        print()

    print("=" * 60)
    print("Tinker支持的模型")
    print("=" * 60)
    print()

    # 根据文档已知的模型列表
    print("小型模型（适合RLVR数学推理）:")
    print()

    models = [
        ("meta-llama/Llama-3.2-1B", "1B", "基座模型", "极低成本验证"),
        ("meta-llama/Llama-3.2-3B", "3B", "基座模型", "需先SFT再RLVR"),
        ("Qwen/Qwen3-4B-Instruct-2507", "4B", "指令模型", "⭐ 推荐：直接RLVR"),
        ("meta-llama/Llama-3.1-8B", "8B", "基座模型", "需先SFT"),
        ("meta-llama/Llama-3.1-8B-Instruct", "8B", "指令模型", "更好效果但成本高"),
    ]

    print(f"{'模型ID':<40} {'参数':<8} {'类型':<12} {'说明'}")
    print("-" * 80)
    for model_id, params, model_type, desc in models:
        print(f"{model_id:<40} {params:<8} {model_type:<12} {desc}")

    print()
    print("=" * 60)
    print("推荐配置")
    print("=" * 60)
    print()
    print("⭐ 标准配置（推荐）:")
    print("  模型: Qwen/Qwen3-4B-Instruct-2507")
    print("  规模: quick (200 steps, ~$100)")
    print("  优势: 指令模型，初始正确率高，直接RLVR")
    print()
    print("低成本验证:")
    print("  模型: meta-llama/Llama-3.2-1B")
    print("  规模: quick (200 steps, ~$40)")
    print("  注意: 基座模型，效果较差")
    print()
    print("高性能方案:")
    print("  模型: meta-llama/Llama-3.1-8B-Instruct")
    print("  规模: medium (1000 steps, ~$1,800)")
    print("  优势: 更好效果，但成本高")
    print()

    print("=" * 60)
    print("重要提示")
    print("=" * 60)
    print()
    print("✅ 推荐Instruct模型直接做RLVR:")
    print("    ⭐ Qwen/Qwen3-4B-Instruct-2507 (推荐)")
    print("    ✓ meta-llama/Llama-3.1-8B-Instruct (更强但贵)")
    print()
    print("⚠️  Base模型需要先SFT再RLVR:")
    print("    • meta-llama/Llama-3.2-3B (需要SFT)")
    print("    • meta-llama/Llama-3.1-8B (需要SFT)")
    print("    • 总成本: SFT($100-200) + RLVR($800) = $900-1000")
    print()
    print("⚠️  当前仅支持LoRA微调，不支持全量微调")
    print()

    # 尝试通过Tinker API验证（仅在有API key时）
    if has_api_key:
        try:
            import tinker
            print("正在连接Tinker服务...")
            client = tinker.ServiceClient()
            print("✓ Tinker连接成功")
            print()
            print("注意: 实际可用模型可能因账户权限而异")
            print("      建议在训练前使用 --dry-run 模式测试")
        except ImportError:
            print("提示: 安装tinker包后可以验证连接")
            print("      pip install tinker")
        except Exception as e:
            print(f"Tinker连接失败: {e}")
    else:
        print("提示: 设置TINKER_API_KEY后可以验证连接")
        print("      export TINKER_API_KEY=your_key")

    print()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
环境检查脚本

验证RLVR项目所需的依赖是否正确安装

Usage:
    python scripts/check_setup.py
"""

import sys


def check_package(name: str, import_name: str = None) -> tuple[bool, str]:
    """检查包是否安装"""
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "not installed"


def main():
    print("=" * 50)
    print("RLVR Environment Check")
    print("=" * 50)
    print()

    # 必需的包
    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ]

    # 可选但推荐的包
    optional_packages = [
        ("tinker", "tinker"),
        ("vllm", "vllm"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("wandb", "wandb"),
    ]

    print("Required Packages:")
    print("-" * 40)
    all_required_ok = True
    for name, import_name in required_packages:
        ok, version = check_package(name, import_name)
        status = f"✓ {version}" if ok else "✗ NOT INSTALLED"
        print(f"  {name:20} {status}")
        if not ok:
            all_required_ok = False

    print()
    print("Optional Packages:")
    print("-" * 40)
    for name, import_name in optional_packages:
        ok, version = check_package(name, import_name)
        status = f"✓ {version}" if ok else "○ not installed"
        print(f"  {name:20} {status}")

    print()
    print("Environment Variables:")
    print("-" * 40)
    import os

    tinker_key = os.environ.get("TINKER_API_KEY")
    if tinker_key:
        print(f"  TINKER_API_KEY      ✓ set ({tinker_key[:8]}...)")
    else:
        print("  TINKER_API_KEY      ○ not set")

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        print("  WANDB_API_KEY       ✓ set")
    else:
        print("  WANDB_API_KEY       ○ not set")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("  HF_TOKEN            ✓ set")
    else:
        print("  HF_TOKEN            ○ not set")

    print()
    print("GPU Information:")
    print("-" * 40)
    try:
        import torch

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print("  CUDA Available      ✓")
            print(f"  Number of GPUs      {num_gpus}")
            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    GPU {i}: {name} ({memory:.1f} GB)")
        else:
            print("  CUDA Available      ✗ No GPU detected")
    except Exception as e:
        print(f"  Error checking GPU: {e}")

    print()
    print("=" * 50)
    if all_required_ok:
        print("✓ All required packages installed!")
        print("  You're ready to start training.")
    else:
        print("✗ Some required packages are missing.")
        print("  Please run: pip install -r requirements.txt")
    print("=" * 50)

    return 0 if all_required_ok else 1


if __name__ == "__main__":
    sys.exit(main())

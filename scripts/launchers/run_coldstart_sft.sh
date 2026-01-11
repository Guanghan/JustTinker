#!/bin/bash
###############################################################################
# Cold Start SFT Training Runner - MacBook Sleep-Safe Version
#
# 功能:
#   - 使用 OpenR1-Math-220k 数据集训练 thinking mode
#   - 训练模型生成 <think>...</think> 格式
#   - 防止MacBook睡眠 (caffeinate)
#   - 后台运行 (nohup)
#   - 日志记录
#
# 使用方法:
#   # 前台运行 (快速测试)
#   ./scripts/launchers/run_coldstart_sft.sh quick --foreground
#
#   # 后台运行 - 不同规模
#   ./scripts/launchers/run_coldstart_sft.sh quick      # 10 steps, ~$0.5
#   ./scripts/launchers/run_coldstart_sft.sh small      # 500 steps, ~$10
#   ./scripts/launchers/run_coldstart_sft.sh medium     # 2000 steps, ~$50
#   ./scripts/launchers/run_coldstart_sft.sh large      # 5000 steps, ~$100+
#
#   # 停止训练
#   ./scripts/launchers/stop_coldstart_sft.sh
#
#   # 查看日志
#   tail -f logs/coldstart_sft_*.log
#
###############################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 项目根目录 (从 scripts/launchers/ 往上两级)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$PROJECT_ROOT"

# 默认参数
SCALE="small"
MODE="background"
MODEL=""
LR=""
DATASET_CONFIG=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        quick|small|medium|large)
            SCALE="$1"
            shift
            ;;
        --foreground)
            MODE="foreground"
            shift
            ;;
        --model)
            MODEL="--model $2"
            shift 2
            ;;
        --lr)
            LR="--lr $2"
            shift 2
            ;;
        --dataset-config)
            DATASET_CONFIG="--dataset-config $2"
            shift 2
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用方法: $0 <scale> [--foreground] [--model <model_name>] [--lr <learning_rate>]"
            echo ""
            echo "Scale 选项:"
            echo "  quick   - 10 steps, 快速验证 (~\$0.5)"
            echo "  small   - 500 steps, 标准训练 (~\$10)"
            echo "  medium  - 2000 steps, 完整训练 (~\$50)"
            echo "  large   - 5000 steps, 大规模训练 (~\$100+)"
            exit 1
            ;;
    esac
done

# 创建logs目录
mkdir -p logs

# 生成日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/coldstart_sft_${SCALE}_${TIMESTAMP}.log"
PID_FILE="logs/coldstart_sft.pid"

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}Cold Start SFT Training Runner${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Scale: $SCALE"
echo "Mode: $MODE"
echo "Log file: $LOG_FILE"
echo ""

# 显示预估信息
case $SCALE in
    quick)
        echo "预估: 10 steps, ~\$0.5, ~2分钟"
        ;;
    small)
        echo "预估: 500 steps, ~\$10, ~1小时"
        ;;
    medium)
        echo "预估: 2000 steps, ~\$50, ~4小时"
        ;;
    large)
        echo "预估: 5000 steps, ~\$100+, ~10小时"
        ;;
esac
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${RED}错误: 虚拟环境 .venv 不存在${NC}"
    echo "请先运行: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 激活虚拟环境
echo -e "${YELLOW}激活虚拟环境...${NC}"
source .venv/bin/activate

# 检查API Key
if [ -z "$TINKER_API_KEY" ]; then
    echo -e "${RED}错误: TINKER_API_KEY 环境变量未设置${NC}"
    echo "请运行: export TINKER_API_KEY=your_api_key"
    exit 1
fi

# 检查训练脚本
if [ ! -f "scripts/tinker/coldstart_sft.py" ]; then
    echo -e "${RED}错误: 训练脚本不存在${NC}"
    exit 1
fi

# 构建命令
# 使用 -u 禁用Python输出缓冲，确保日志实时写入
TRAIN_CMD="python -u scripts/tinker/coldstart_sft.py --scale $SCALE $MODEL $LR $DATASET_CONFIG"

echo -e "${YELLOW}执行命令: $TRAIN_CMD${NC}"
echo ""

if [ "$MODE" == "foreground" ]; then
    # 前台运行
    echo -e "${GREEN}开始前台训练...${NC}"
    echo "按 Ctrl+C 可以中止训练"
    echo ""

    # 使用caffeinate防止睡眠，直接运行
    caffeinate -i $TRAIN_CMD 2>&1 | tee "$LOG_FILE"

else
    # 后台运行
    echo -e "${GREEN}开始后台训练...${NC}"
    echo ""

    # 检查是否已有进程在运行
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}警告: 检测到已有 SFT 训练进程运行 (PID: $OLD_PID)${NC}"
            echo "如需停止，请运行: ./scripts/stop_coldstart_sft.sh"
            echo "如需继续启动新进程，请先停止旧进程"
            exit 1
        else
            echo -e "${YELLOW}清理旧的PID文件${NC}"
            rm -f "$PID_FILE"
        fi
    fi

    # 使用nohup + caffeinate后台运行
    nohup caffeinate -i $TRAIN_CMD > "$LOG_FILE" 2>&1 &
    PID=$!

    # 保存PID
    echo $PID > "$PID_FILE"

    echo -e "${GREEN}✓ SFT 训练已在后台启动${NC}"
    echo "  PID: $PID"
    echo "  日志: $LOG_FILE"
    echo ""
    echo "查看实时日志: tail -f $LOG_FILE"
    echo "停止训练: ./scripts/stop_coldstart_sft.sh"
    echo "检查进程: ps -p $PID"
    echo ""

    # 等待几秒确认进程启动成功
    sleep 3
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 进程运行正常${NC}"
    else
        echo -e "${RED}✗ 进程启动失败，请检查日志${NC}"
        cat "$LOG_FILE"
        exit 1
    fi
fi

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo "训练完成后，可以使用以下命令绘制训练曲线:"
echo "  python scripts/plot_sft_training.py"
echo ""
echo "后续 RLVR 训练请使用:"
echo "  ./scripts/run_justrl_reasoning.sh medium --reasoning --checkpoint coldstart_sft_final"
echo -e "${CYAN}==============================================================${NC}"

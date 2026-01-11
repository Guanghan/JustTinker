#!/bin/bash
###############################################################################
# JustRL Training Runner - MacBook Sleep-Safe Version
#
# 功能:
#   - 防止MacBook睡眠 (caffeinate)
#   - 后台运行 (nohup)
#   - 日志记录
#   - 进程管理
#
# 使用方法:
#   # 前台运行 (简单，适合快速测试)
#   ./scripts/launchers/run_justrl.sh quick --foreground
#
#   # 后台运行 (推荐，适合长时间训练)
#   ./scripts/launchers/run_justrl.sh quick
#   ./scripts/launchers/run_justrl.sh medium
#   ./scripts/launchers/run_justrl.sh full
#
#   # 停止后台任务
#   ./scripts/launchers/stop_justrl.sh
#
#   # 查看日志
#   tail -f logs/justrl_quick_*.log
#
###############################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录 (从 scripts/launchers/ 往上两级)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$PROJECT_ROOT"

# 解析参数
SCALE=${1:-quick}
MODE=${2:-background}  # background 或 foreground

# 验证scale参数
if [[ ! "$SCALE" =~ ^(quick|medium|full)$ ]]; then
    echo -e "${RED}错误: scale必须是 quick, medium, 或 full${NC}"
    echo "使用方法: $0 <scale> [--foreground]"
    exit 1
fi

# 解析mode
if [[ "$MODE" == "--foreground" ]]; then
    MODE="foreground"
fi

# 创建logs目录
mkdir -p logs

# 生成日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/justrl_${SCALE}_${TIMESTAMP}.log"
PID_FILE="logs/justrl.pid"

echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}JustRL Training Runner${NC}"
echo -e "${GREEN}==============================================================${NC}"
echo "Scale: $SCALE"
echo "Mode: $MODE"
echo "Log file: $LOG_FILE"
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
if [ ! -f "scripts/tinker/justrl_math.py" ]; then
    echo -e "${RED}错误: 训练脚本不存在${NC}"
    exit 1
fi

# 构建命令
# 使用 -u 禁用Python输出缓冲，确保日志实时写入
TRAIN_CMD="python -u scripts/tinker/justrl_math.py --scale $SCALE"

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
            echo -e "${YELLOW}警告: 检测到已有训练进程运行 (PID: $OLD_PID)${NC}"
            echo "如需停止，请运行: ./scripts/stop_justrl.sh"
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

    echo -e "${GREEN}✓ 训练已在后台启动${NC}"
    echo "  PID: $PID"
    echo "  日志: $LOG_FILE"
    echo ""
    echo "查看实时日志: tail -f $LOG_FILE"
    echo "停止训练: ./scripts/stop_justrl.sh"
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
echo -e "${GREEN}==============================================================${NC}"

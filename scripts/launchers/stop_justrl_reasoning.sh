#!/bin/bash
###############################################################################
# 停止JustRL Reasoning后台训练
#
# 使用方法:
#   ./scripts/launchers/stop_justrl_reasoning.sh
#
###############################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 项目根目录 (从 scripts/launchers/ 往上两级)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$PROJECT_ROOT"

PID_FILE="logs/justrl_reasoning.pid"

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}停止JustRL Reasoning训练${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo ""

# 检查PID文件
if [ ! -f "$PID_FILE" ]; then
    echo -e "${RED}未找到运行中的训练进程${NC}"
    echo "PID文件不存在: $PID_FILE"
    exit 1
fi

# 读取PID
PID=$(cat "$PID_FILE")

echo "检测到进程 PID: $PID"

# 检查进程是否存在
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}进程已经停止${NC}"
    rm -f "$PID_FILE"
    exit 0
fi

# 显示进程信息
echo "进程信息:"
ps -p "$PID" -o pid,ppid,command

echo ""
echo -e "${YELLOW}发送SIGTERM信号...${NC}"
kill "$PID"

# 等待进程优雅退出
echo "等待进程退出..."
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 进程已成功停止${NC}"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo -e "${YELLOW}进程未响应SIGTERM，发送SIGKILL...${NC}"
kill -9 "$PID"

sleep 1

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 进程已强制停止${NC}"
    rm -f "$PID_FILE"
else
    echo -e "${RED}✗ 无法停止进程${NC}"
    exit 1
fi

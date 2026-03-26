#!/bin/bash
# run.sh — 一键启动实验流水线
# 用法:
#   bash run.sh          # 启动并实时查看日志
#   bash run.sh --stage 2  # 从指定阶段启动
#   bash run.sh --list     # 查看各阶段状态

set -e

# ── 配置 ──────────────────────────────────────────────────────
export RAW_CAN_PATH="${RAW_CAN_PATH:-data/raw/car_hacking_rt.csv}"
export RAW_CIC_PATH="${RAW_CIC_PATH:-data/raw/cicids2017.csv}"
export CACHE_DIR="${CACHE_DIR:-/root/cache}"
LOG_FILE="$CACHE_DIR/logs/pipeline.log"
STAGE_ARGS="${@}"

# ── 初始化目录 ────────────────────────────────────────────────
mkdir -p "$CACHE_DIR"/{data,models,results,logs}
chmod -R 777 "$CACHE_DIR"

# ── 如果只是查状态，直接执行不走后台 ─────────────────────────
if [[ "$*" == *"--list"* ]]; then
    python pipeline.py --list
    exit 0
fi

# ── 停掉旧进程 ────────────────────────────────────────────────
OLD_PID=$(pgrep -f "pipeline.py" || true)
if [ -n "$OLD_PID" ]; then
    echo "停止旧进程 PID=$OLD_PID ..."
    kill $OLD_PID 2>/dev/null || true
    sleep 2
fi

# ── 后台启动 ──────────────────────────────────────────────────
echo "======================================================"
echo "  启动实验流水线"
echo "  CAN  : $RAW_CAN_PATH"
echo "  CICIDS: $RAW_CIC_PATH"
echo "  CACHE : $CACHE_DIR"
echo "  日志  : $LOG_FILE"
echo "======================================================"

nohup python pipeline.py $STAGE_ARGS \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "后台进程 PID=$PID 已启动"
echo ""
echo "实时查看进度（Ctrl+C 退出查看，进程继续运行）:"
echo "------------------------------------------------------"

# 等日志文件出现再 tail
sleep 2
tail -f "$LOG_FILE"
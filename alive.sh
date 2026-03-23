#!/bin/bash

# 设置日志文件路径
LOG_FILE="./live.log"

# 创建日志文件（如果不存在）
touch "$LOG_FILE"

# 设置最大日志行数（避免文件过大）
MAX_LINES=1000000
LINE_COUNT=0

while true; do
    # 带详细时间戳的输出
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Service is alive" >> "$LOG_FILE"
    
    # 增加行数计数
    ((LINE_COUNT++))
    
    # 如果超过最大行数，清空文件重新开始
    if [ $LINE_COUNT -ge $MAX_LINES ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file rotated" > "$LOG_FILE"
        LINE_COUNT=1
    fi
    
    sleep 60
done
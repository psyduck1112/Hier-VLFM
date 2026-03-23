#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
# Modifications Copyright (c) 2026 Yikang.

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
#export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export SAM_PORT=${SAM_PORT:-12183}
export YOLOV7_PORT=${YOLOV7_PORT:-12184}
#export YOLOWORLD_ITM_PORT=${YOLOWORLD_ITM_PORT:-12186}  # 已废弃
export ULTRALYTICS_YOLOWORLD_ITM_PORT=${ULTRALYTICS_YOLOWORLD_ITM_PORT:-12187}  # 新的YOLOWorld端口
export ULTRALYTICS_YOLOWORLD_MODEL=${ULTRALYTICS_YOLOWORLD_MODEL:-yolov8x-worldv2.pt}
export ULTRALYTICS_YOLOWORLD_DEVICE=${ULTRALYTICS_YOLOWORLD_DEVICE:-cuda}

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session with optimized layout
tmux new-session -d -s ${session_name}

# 创建清晰的2x3网格布局
# 第一行: GroundingDINO | SAM | YOLOv7
# 第二行: YOLO-World | 状态监控 | 日志监控

# 分割成上下两个区域
tmux split-window -v -t ${session_name}:0

# 上半部分分成3个pane (GroundingDINO, SAM, YOLOv7)
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.1

# 下半部分分成3个pane (UltralyticsYOLO, 状态监控, 日志)
tmux split-window -h -t ${session_name}:0.3
tmux split-window -h -t ${session_name}:0.4

# 调整pane大小为均匀分布
tmux select-layout -t ${session_name} tiled

# 为每个pane设置标题并运行命令
# 第一行: 核心VLM服务
tmux send-keys -t ${session_name}:0.0 "echo '🎯 GroundingDINO Server (Port ${GROUNDING_DINO_PORT})' && ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "echo '✂️  SAM Server (Port ${SAM_PORT})' && ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m  
tmux send-keys -t ${session_name}:0.2 "echo '📦 YOLOv7 Server (Port ${YOLOV7_PORT})' && ${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m
#tmux send-keys -t ${session_name}:0.3 "echo '🤖 BLIP2ITM Server (Port ${BLIP2ITM_PORT})' && ${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m

# 第二行: YOLOv7 + YOLO-World + 监控
tmux send-keys -t ${session_name}:0.3 "echo '🌍 Ultralytics YOLO-World Server (Port ${ULTRALYTICS_YOLOWORLD_ITM_PORT})' && ${VLFM_PYTHON} -m vlfm.vlm.ultralytics_yoloworld --port ${ULTRALYTICS_YOLOWORLD_ITM_PORT} --model ${ULTRALYTICS_YOLOWORLD_MODEL} --device ${ULTRALYTICS_YOLOWORLD_DEVICE} --conf 0.0001" C-m
tmux send-keys -t ${session_name}:0.4 "echo '📊 VLM服务器状态监控' && watch -n 2 'echo \"=== VLM服务器状态 ===\" && netstat -tlnp 2>/dev/null | grep -E \"(12181|12183|12184|12187)\" | sort && echo && echo \"=== GPU使用情况 ===\" && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo \"CPU模式\"'" C-m
tmux send-keys -t ${session_name}:0.5 "echo '📋 系统日志监控' && tail -f /var/log/syslog 2>/dev/null || (echo '无系统日志权限，显示进程状态:' && watch -n 3 'ps aux | grep -E \"(grounding_dino|sam|yolov7|yoloworld)\" | grep -v grep | head -10')" C-m

# 设置pane标题
tmux set-option -t ${session_name} set-titles on
tmux set-option -t ${session_name} set-titles-string "VLM服务器集群"

# Attach to the tmux session to view the windows
echo "✅ 创建了优化的tmux会话 '${session_name}'"
echo "📊 布局: 2x3网格 - 上排GroundingDINO/SAM/YOLOv7，下排YOLO-World/监控/日志"
echo "⏳ 请等待最多90秒让模型权重加载完成"
echo ""
echo "🔗 连接到会话: tmux attach-session -t ${session_name}"
echo "🔑 常用快捷键:"
echo "  Ctrl+B 然后 方向键 - 切换pane"
echo "  Ctrl+B 然后 d      - 分离会话(后台运行)"
echo "  Ctrl+B 然后 z      - 放大/缩小当前pane"
echo ""
echo "💡 停止所有服务器: tmux kill-session -t ${session_name}"

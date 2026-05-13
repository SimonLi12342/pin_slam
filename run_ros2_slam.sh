#!/bin/bash
# PIN-SLAM ROS2 一键启动脚本

set -e  # 遇到错误立即退出

# 默认参数
BAG_PATH="${1:-/mnt/d/data/ros2_data/Exit_ros2/Exit_ros2.db3}"
CONFIG_PATH="${2:-./config/lidar_slam/run.yaml}"
TOPIC="${3:-/velodyne_points}"
RATE="${4:-1.0}"

echo "========================================="
echo "PIN-SLAM ROS2 一键启动"
echo "========================================="
echo "Bag 路径: $BAG_PATH"
echo "配置文件: $CONFIG_PATH"
echo "Topic: $TOPIC"
echo "播放速率: $RATE"
echo "========================================="

# 检查文件是否存在
if [ ! -f "$BAG_PATH" ]; then
    echo "错误: Bag 文件不存在: $BAG_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 检查 ROS2 环境
if ! command -v ros2 &> /dev/null; then
    echo "错误: ROS2 未安装或未 source"
    echo "请运行: source /opt/ros/humble/setup.bash"
    exit 1
fi

# 检查 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "pin" ]; then
    echo "警告: 未激活 pin conda 环境"
    echo "尝试激活..."
    if [ -f "/mnt/d/anaconda3/etc/profile.d/conda.sh" ]; then
        source /mnt/d/anaconda3/etc/profile.d/conda.sh
        conda activate pin
    else
        echo "错误: 无法找到 conda，请手动运行: conda activate pin"
        exit 1
    fi
fi

# 创建临时目录存储 PID
TMP_DIR="/tmp/pin_slam_$$"
mkdir -p "$TMP_DIR"

# 清理函数
cleanup() {
    echo ""
    echo "========================================="
    echo "正在停止所有进程..."
    echo "========================================="
    
    if [ -f "$TMP_DIR/bag_player.pid" ]; then
        BAG_PID=$(cat "$TMP_DIR/bag_player.pid")
        if ps -p $BAG_PID > /dev/null 2>&1; then
            echo "停止 bag player (PID: $BAG_PID)"
            kill $BAG_PID 2>/dev/null || true
        fi
    fi
    
    if [ -f "$TMP_DIR/pin_slam.pid" ]; then
        SLAM_PID=$(cat "$TMP_DIR/pin_slam.pid")
        if ps -p $SLAM_PID > /dev/null 2>&1; then
            echo "停止 PIN-SLAM (PID: $SLAM_PID)"
            kill $SLAM_PID 2>/dev/null || true
        fi
    fi
    
    rm -rf "$TMP_DIR"
    echo "清理完成"
}

# 注册清理函数
trap cleanup EXIT INT TERM

echo ""
echo "========================================="
echo "启动 ROS2 bag player..."
echo "========================================="

# 启动 bag player（后台运行）
ros2 bag play "$BAG_PATH" -r "$RATE" --clock > /tmp/pin_slam_bag_$$.log 2>&1 &
BAG_PID=$!
echo $BAG_PID > "$TMP_DIR/bag_player.pid"
echo "Bag player 已启动 (PID: $BAG_PID)"

# 等待 bag player 初始化
sleep 2

echo ""
echo "========================================="
echo "启动 PIN-SLAM 节点..."
echo "========================================="

# 启动 PIN-SLAM（前台运行，可以看到输出）
python3 pin_slam_ros2.py "$CONFIG_PATH" "$TOPIC" &
SLAM_PID=$!
echo $SLAM_PID > "$TMP_DIR/pin_slam.pid"
echo "PIN-SLAM 已启动 (PID: $SLAM_PID)"

echo ""
echo "========================================="
echo "所有进程已启动！"
echo "========================================="
echo "按 Ctrl+C 停止所有进程"
echo ""

# 等待 PIN-SLAM 进程结束
wait $SLAM_PID

echo ""
echo "========================================="
echo "PIN-SLAM 处理完成！"
echo "========================================="

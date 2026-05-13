#!/usr/bin/env python3
"""
PIN-SLAM 一键启动脚本（最简单版本）
支持离线和实时两种模式
"""

import os
import sys
import subprocess
import argparse
import signal
import time
from pathlib import Path


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")


def check_file_exists(path, name):
    """检查文件是否存在"""
    if not Path(path).exists():
        print_error(f"{name} 不存在: {path}")
        return False
    print_success(f"{name} 找到: {path}")
    return True


def run_offline_mode(bag_path, config_path, topic):
    """离线模式：直接处理 bag 文件（推荐）"""
    print_header("离线模式：处理 ROS2 bag")
    
    # 检查文件
    if not check_file_exists(bag_path, "Bag 文件"):
        return False
    if not check_file_exists(config_path, "配置文件"):
        return False
    
    print(f"\n{Colors.BOLD}参数:{Colors.END}")
    print(f"  Bag: {bag_path}")
    print(f"  Config: {config_path}")
    print(f"  Topic: {topic}")
    print(f"  模式: 离线处理（无需 ROS2）")
    
    # 构建命令
    cmd = [
        "python3", "pin_slam.py",
        config_path,
        "rosbag", topic,
        "-i", str(Path(bag_path).parent),
        "-vsmd"
    ]
    
    print(f"\n{Colors.BOLD}执行命令:{Colors.END}")
    print(f"  {' '.join(cmd)}\n")
    
    try:
        print_header("开始处理...")
        subprocess.run(cmd, check=True)
        print_success("处理完成！")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"处理失败: {e}")
        return False
    except KeyboardInterrupt:
        print_warning("\n用户中断")
        return False


def run_realtime_mode(bag_path, config_path, topic, rate):
    """实时模式：播放 bag 并实时处理"""
    print_header("实时模式：ROS2 实时处理")
    
    # 检查文件
    if not check_file_exists(bag_path, "Bag 文件"):
        return False
    if not check_file_exists(config_path, "配置文件"):
        return False
    
    # 检查 ROS2
    try:
        subprocess.run(["ros2", "--version"], capture_output=True, check=True)
        print_success("ROS2 环境检测成功")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("ROS2 未安装或未 source")
        print_warning("请运行: source /opt/ros/humble/setup.bash")
        return False
    
    print(f"\n{Colors.BOLD}参数:{Colors.END}")
    print(f"  Bag: {bag_path}")
    print(f"  Config: {config_path}")
    print(f"  Topic: {topic}")
    print(f"  播放速率: {rate}x")
    print(f"  模式: 实时处理（需要 ROS2）")
    
    processes = []
    
    def cleanup():
        print_warning("\n正在停止所有进程...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
                p.kill()
        print_success("清理完成")
    
    # 注册信号处理
    signal.signal(signal.SIGINT, lambda s, f: cleanup() or sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: cleanup() or sys.exit(0))
    
    try:
        # 启动 bag player
        print_header("启动 ROS2 bag player...")
        bag_cmd = ["ros2", "bag", "play", bag_path, "-r", str(rate), "--clock"]
        print(f"命令: {' '.join(bag_cmd)}")
        bag_proc = subprocess.Popen(bag_cmd)
        processes.append(bag_proc)
        print_success(f"Bag player 已启动 (PID: {bag_proc.pid})")
        
        # 等待初始化
        time.sleep(2)
        
        # 启动 PIN-SLAM
        print_header("启动 PIN-SLAM 节点...")
        slam_cmd = ["python3", "pin_slam_ros2.py", config_path, topic]
        print(f"命令: {' '.join(slam_cmd)}")
        slam_proc = subprocess.Popen(slam_cmd)
        processes.append(slam_proc)
        print_success(f"PIN-SLAM 已启动 (PID: {slam_proc.pid})")
        
        print_header("所有进程已启动！")
        print(f"{Colors.YELLOW}按 Ctrl+C 停止{Colors.END}\n")
        
        # 等待 SLAM 进程结束
        slam_proc.wait()
        
        print_success("PIN-SLAM 处理完成！")
        cleanup()
        return True
        
    except Exception as e:
        print_error(f"启动失败: {e}")
        cleanup()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="PIN-SLAM 一键启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 离线模式（推荐，无需 ROS2）
  python3 run_slam_easy.py --offline
  
  # 实时模式（需要 ROS2）
  python3 run_slam_easy.py --realtime
  
  # 自定义参数
  python3 run_slam_easy.py --offline --bag /path/to/bag.db3 --topic /points
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['offline', 'realtime'],
        default='offline',
        help='运行模式 (默认: offline)'
    )
    parser.add_argument(
        '--offline',
        action='store_const',
        const='offline',
        dest='mode',
        help='离线模式（推荐）'
    )
    parser.add_argument(
        '--realtime',
        action='store_const',
        const='realtime',
        dest='mode',
        help='实时模式'
    )
    parser.add_argument(
        '--bag',
        default='/mnt/d/data/ros2_data/Exit_ros2/Exit_ros2.db3',
        help='ROS2 bag 文件路径'
    )
    parser.add_argument(
        '--config',
        default='./config/lidar_slam/run.yaml',
        help='PIN-SLAM 配置文件路径'
    )
    parser.add_argument(
        '--topic',
        default='/velodyne_points',
        help='点云 topic 名称'
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=1.0,
        help='Bag 播放速率（仅实时模式）'
    )
    
    args = parser.parse_args()
    
    print_header("PIN-SLAM 一键启动脚本")
    
    if args.mode == 'offline':
        success = run_offline_mode(args.bag, args.config, args.topic)
    else:
        success = run_realtime_mode(args.bag, args.config, args.topic, args.rate)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

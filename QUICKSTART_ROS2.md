# PIN-SLAM ROS2 快速开始

## 🚀 一键运行（推荐）

### 最简单的方式

```bash
# 1. 激活环境
conda activate pin

# 2. 一键运行（离线模式，无需 ROS2）
python3 run_slam_easy.py --offline
```

就这么简单！脚本会自动：
- ✅ 检查所有文件
- ✅ 处理你的 ROS2 bag
- ✅ 保存地图和 mesh
- ✅ 显示处理进度

---

## 📋 三种启动方式对比

| 方式 | 命令 | 需要 ROS2 | 推荐度 |
|------|------|-----------|--------|
| **Python 脚本** | `python3 run_slam_easy.py --offline` | ❌ | ⭐⭐⭐⭐⭐ |
| **Bash 脚本** | `./run_ros2_slam.sh` | ✅ | ⭐⭐⭐⭐ |
| **ROS2 Launch** | `ros2 launch launch_ros2.py` | ✅ | ⭐⭐⭐ |

---

## 🎯 使用示例

### 示例 1：使用默认参数

```bash
python3 run_slam_easy.py --offline
```

默认处理：
- Bag: `/mnt/d/data/ros2_data/Exit_ros2/Exit_ros2.db3`
- Topic: `/velodyne_points`
- Config: `./config/lidar_slam/run.yaml`

### 示例 2：自定义 bag 路径

```bash
python3 run_slam_easy.py --offline --bag /path/to/your/bag.db3
```

### 示例 3：自定义 topic

```bash
python3 run_slam_easy.py --offline --topic /your/point/cloud/topic
```

### 示例 4：实时模式（需要 ROS2）

```bash
# 终端 1：source ROS2
source /opt/ros/humble/setup.bash

# 终端 2：运行
python3 run_slam_easy.py --realtime
```

---

## 📊 处理过程

运行后你会看到：

```
============================================================
PIN-SLAM 一键启动脚本
============================================================

============================================================
离线模式：处理 ROS2 bag
============================================================

✓ Bag 文件找到: /mnt/d/data/ros2_data/Exit_ros2/Exit_ros2.db3
✓ 配置文件找到: ./config/lidar_slam/run.yaml

参数:
  Bag: /mnt/d/data/ros2_data/Exit_ros2/Exit_ros2.db3
  Config: ./config/lidar_slam/run.yaml
  Topic: /velodyne_points
  模式: 离线处理（无需 ROS2）

============================================================
开始处理...
============================================================

  0%|          | 0/2338 [00:00<?, ?it/s]
  5%|▌         | 120/2338 [00:20<06:05, 6.07it/s]
 10%|█         | 240/2338 [00:40<05:45, 6.08it/s]
 ...
```

---

## 📁 输出结果

处理完成后，结果保存在 `./experiments/` 目录：

```
experiments/
└── run_YYYYMMDD_HHMMSS/
    ├── map/
    │   └── neural_points.ply      # Neural point 地图
    ├── mesh/
    │   └── mesh_*.ply              # 重建的 mesh
    ├── log/
    │   ├── traj_est.txt            # 估计轨迹
    │   └── timing.txt              # 时间统计
    └── config.yaml                 # 使用的配置
```

---

## 🔧 高级选项

### 查看所有选项

```bash
python3 run_slam_easy.py --help
```

### 调整播放速率（实时模式）

```bash
# 0.5x 慢速播放
python3 run_slam_easy.py --realtime --rate 0.5

# 2.0x 快速播放
python3 run_slam_easy.py --realtime --rate 2.0
```

### 使用不同配置文件

```bash
python3 run_slam_easy.py --offline --config ./config/lidar_slam/custom.yaml
```

---

## ❓ 常见问题

### Q1: 提示 "Bag 文件不存在"

**A**: 检查路径是否正确：
```bash
ls -lh /mnt/d/data/ros2_data/Exit_ros2/Exit_ros2.db3
```

### Q2: 处理速度很慢

**A**: 这是正常的，第一帧初始化需要较长时间（~20秒），之后会加速到 5-6 fps。

### Q3: 想要可视化

**A**: 使用 `-v` 参数：
```bash
python3 pin_slam.py ./config/lidar_slam/run.yaml rosbag /velodyne_points \
    -i /mnt/d/data/ros2_data/Exit_ros2 -vsmd
```

### Q4: 只想保存地图，不要 mesh

**A**: 去掉 `-m` 参数：
```bash
python3 pin_slam.py ./config/lidar_slam/run.yaml rosbag /velodyne_points \
    -i /mnt/d/data/ros2_data/Exit_ros2 -vsd
```

---

## 📚 更多信息

- 完整文档：[ROS2_USAGE.md](./ROS2_USAGE.md)
- 原始 README：[README.md](./README.md)
- 问题反馈：[GitHub Issues](https://github.com/PRBonn/PIN_SLAM/issues)

---

## 🎉 快速测试

想快速测试是否工作？运行这个：

```bash
conda activate pin
python3 run_slam_easy.py --offline
```

等待几分钟，检查 `./experiments/` 目录是否有输出。如果有，说明一切正常！

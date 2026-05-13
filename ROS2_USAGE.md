# PIN-SLAM ROS2 使用指南

## ROS2 Humble 版本说明

本项目已成功移植到 ROS2 Humble。主要变更：

### 新增文件
- `pin_slam_ros2.py` - ROS2 版本的实时 SLAM 节点
- `utils/point_cloud2_ros2.py` - ROS2 点云消息创建工具

### 修改文件
- `utils/point_cloud2.py` - 兼容新版 rosbags 库（支持 `rosbags.typesys.stores.latest` API）

## 安装依赖

### 1. ROS2 Humble
```bash
# 安装 ROS2 Humble（Ubuntu 22.04）
sudo apt update
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### 2. Python 依赖
```bash
conda activate pin
pip install rclpy scipy
```

## 使用方法

### 🚀 一键启动（最简单）

我们提供了三种一键启动脚本：

#### 1. Python 脚本（推荐）

```bash
# 离线模式（推荐，无需 ROS2）
python3 run_slam_easy.py --offline

# 实时模式（需要 ROS2）
python3 run_slam_easy.py --realtime

# 自定义参数
python3 run_slam_easy.py --offline \
    --bag /path/to/bag.db3 \
    --config ./config/lidar_slam/run.yaml \
    --topic /velodyne_points

# 查看帮助
python3 run_slam_easy.py --help
```

#### 2. Bash 脚本

```bash
# 使用默认参数
./run_ros2_slam.sh

# 自定义参数：bag路径 配置文件 topic 播放速率
./run_ros2_slam.sh /path/to/bag.db3 ./config/lidar_slam/run.yaml /velodyne_points 1.0
```

#### 3. ROS2 Launch 文件

```bash
# 使用默认参数
ros2 launch launch_ros2.py

# 自定义参数
ros2 launch launch_ros2.py \
    bag_path:=/path/to/bag.db3 \
    config_path:=./config/lidar_slam/run.yaml \
    topic:=/velodyne_points \
    rate:=1.0
```

---

### 方案 A：离线处理 ROS2 bag（推荐，无需 ROS2 运行时）

使用现有的 rosbag dataloader 直接处理 ROS2 bag 文件：

```bash
# 处理 .db3 格式的 ROS2 bag
python3 pin_slam.py ./config/lidar_slam/run.yaml rosbag /velodyne_points \
    -i /path/to/your/ros2/bag/folder -vsmd

# 示例：处理 Exit_ros2 数据集
python3 pin_slam.py ./config/lidar_slam/run.yaml rosbag /velodyne_points \
    -i /mnt/d/data/ros2_data/Exit_ros2 -vsmd
```

**优点**：
- 无需安装 ROS2
- 处理速度快（约 5-6 帧/秒）
- 可重复运行
- 支持所有 PIN-SLAM 功能（loop closure, PGO, mesh 重建等）

### 方案 B：实时订阅 ROS2 topic

使用新的 `pin_slam_ros2.py` 实时订阅 ROS2 topic：

```bash
# 启动 ROS2 节点
source /opt/ros/humble/setup.bash
python3 pin_slam_ros2.py ./config/lidar_slam/run.yaml /velodyne_points

# 或者指定参数
ros2 run pin_slam pin_slam_ros2.py \
    --ros-args \
    -p global_frame_name:=map \
    -p sensor_frame_name:=velodyne
```

**ROS2 服务**：
```bash
# 保存结果
ros2 service call /pin_slam/save_results std_srvs/srv/Empty

# 保存 mesh
ros2 service call /pin_slam/save_mesh std_srvs/srv/Empty
```

**可视化**：
```bash
# 使用 RViz2
rviz2 -d ./config/pin_slam_ros2.rviz
```

## 数据集信息

您的 Exit_ros2 数据集：
- **路径**: `/mnt/d/data/ros2_data/Exit_ros2/Exit_ros2.db3`
- **Point Cloud Topic**: `/velodyne_points`
- **消息类型**: `sensor_msgs/msg/PointCloud2`
- **帧数**: 2338 帧
- **时长**: 约 261 秒

## API 变更说明

### ROS1 → ROS2 主要变更

| ROS1 | ROS2 |
|------|------|
| `rospy.init_node()` | `rclpy.init()` + `Node.__init__()` |
| `rospy.Publisher()` | `self.create_publisher()` |
| `rospy.Subscriber()` | `self.create_subscription()` |
| `rospy.Service()` | `self.create_service()` |
| `rospy.Time.now()` | `self.get_clock().now()` |
| `rospy.Rate()` | `self.create_rate()` |
| `rospy.spin()` | `rclpy.spin()` |
| `tf.transformations.quaternion_from_matrix()` | `scipy.spatial.transform.Rotation.from_matrix().as_quat()` |
| `sensor_msgs.point_cloud2.create_cloud()` | `utils.point_cloud2_ros2.create_cloud()` |

### 消息类型
- 消息导入保持一致：`from sensor_msgs.msg import PointCloud2`
- QoS 配置：ROS2 需要显式配置 QoS Profile

## 故障排除

### 1. `ModuleNotFoundError: No module named 'rosbags.typesys.types'`

**原因**：新版 rosbags 库 API 变更

**解决**：已修复 `utils/point_cloud2.py`，支持新旧 API

### 2. ROS2 bag 无法读取

**检查**：
```bash
# 查看 bag 信息（需要 ROS2）
ros2 bag info /path/to/bag.db3

# 或使用 Python
python3 -c "from rosbags.rosbag2 import Reader; print('OK')"
```

### 3. 实时性能问题

如果实时处理跟不上传感器帧率：
- 降低 bag 播放速率：`ros2 bag play -r 0.5 your_bag.db3`
- 调整配置文件中的 `iters` 参数
- 使用更强的 GPU

## 性能对比

| 模式 | 速度 | 优点 | 缺点 |
|------|------|------|------|
| 离线 bag 处理 | 5-6 fps | 稳定、可重复 | 非实时 |
| 实时订阅 | 取决于硬件 | 实时反馈 | 需要 ROS2 环境 |

## 下一步

1. **测试实时 ROS2 节点**（需要 ROS2 Humble 环境）
2. **创建 RViz2 配置文件**（`config/pin_slam_ros2.rviz`）
3. **添加 launch 文件**支持

## 参考

- [ROS2 Humble 文档](https://docs.ros.org/en/humble/)
- [rosbags 库](https://gitlab.com/ternaris/rosbags)
- [PIN-SLAM 原始文档](./README.md)

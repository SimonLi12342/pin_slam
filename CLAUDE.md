# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PIN-SLAM is a full-fledged implicit neural LiDAR SLAM system that includes odometry, loop closure detection, and globally consistent mapping. It uses a point-based implicit neural representation for building elastic and compact maps.

**Key Features:**
- Point-based implicit neural map representation
- Correspondence-free point-to-implicit registration
- Loop closure detection using neural point features
- Support for LiDAR and RGB-D sensors
- Real-time operation on moderate GPU

## Development Setup

**Environment:**
- Python 3.10 with conda
- PyTorch 2.5.1 with CUDA 11.8 (check your CUDA version with `nvcc --version`)
- GPU recommended (>4GB VRAM), CPU-only mode available but slower

**Installation:**
```bash
conda create --name pin python=3.10
conda activate pin
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -r requirements.txt
```

## Common Commands

**Sanity Test:**
```bash
# Download example data (KITTI seq 00, first 100 frames)
sh ./scripts/download_kitti_example.sh

# Run with visualization, save map and mesh
python3 pin_slam.py ./config/lidar_slam/run_demo.yaml -vsm

# CPU-only mode (no GPU)
python3 pin_slam.py ./config/lidar_slam/run_demo.yaml -vsmc

# Without visualization (for servers without X)
python3 pin_slam.py ./config/lidar_slam/run_demo.yaml -sm
```

**Run on Custom Data:**
```bash
# Generic point cloud folder (*.ply, *.pcd, *.las, *.bin)
python3 pin_slam.py -i /path/to/point/cloud/folder -vsm

# With specific config file
python3 pin_slam.py path/to/config.yaml -i /path/to/data -vsm

# Using specific dataloaders (-d flag required)
# Available: apollo, boreas, generic, helipr, kitti, kitti360, kitti_mot, kitti_raw, 
#            mcap, mulran, ncd, nclt, neuralrgbd, nuscenes, ouster, replica, rosbag, tum
python3 pin_slam.py ./config/lidar_slam/run_kitti.yaml kitti 00 -i /path/to/kitti -vsmd
python3 pin_slam.py ./config/rgbd_slam/run_replica.yaml replica room0 -i /path/to/replica -vsmd

# ROS bag processing
python3 pin_slam.py ./config/lidar_slam/run.yaml rosbag point_cloud_topic -i /path/to/bag -vsmd
```

**Command Flags:**
- `-v`: Enable visualizer GUI
- `-s`: Save PIN map after SLAM
- `-m`: Save reconstructed mesh
- `-p`: Save merged point cloud
- `-d`: Use specific dataloader
- `-c`: CPU-only mode
- `-l`: Enable log printing
- `-w`: Enable Weights & Bias logging
- `--deskew`: Deskew LiDAR scans

**Post-Processing:**
```bash
# Reconstruct mesh from PIN map with custom resolution
python3 vis_pin_map.py /path/to/result/folder -m 0.2 -c neural_points.ply -o mesh_20cm.ply -n 8

# Parameters:
# -m: marching cubes resolution in meters
# -c: cropped map file (use neural_points.ply or crop in CloudCompare)
# -o: output mesh filename
# -n: mesh_min_nn (6=complete with artifacts, 15=accurate but less complete)
```

**ROS 1 Support:**
```bash
# Run with ROS topic
python3 pin_slam_ros.py ./config/lidar_slam/run.yaml /point_cloud_topic

# Visualize in Rviz
rviz -d ./config/pin_slam_ros.rviz

# Save results via ROS service
rosservice call /pin_slam/save_results
rosservice call /pin_slam/save_mesh
```

## Architecture

**Core Pipeline:**
1. **Tracker** (`utils/tracker.py`): Pose estimation via point-to-implicit registration
2. **Mapper** (`utils/mapper.py`): Incremental learning of local implicit SDF
3. **Loop Detector** (`utils/loop_detector.py`): Neural point feature-based loop detection
4. **PGO** (`utils/pgo.py`): Pose graph optimization using GTSAM
5. **Mesher** (`utils/mesher.py`): Mesh reconstruction via marching cubes

**Neural Representation:**
- **NeuralPoints** (`model/neural_points.py`): Sparse optimizable neural points with voxel hashing for efficient indexing
- **Decoder** (`model/decoder.py`): MLP decoder for implicit SDF prediction

**Data Flow:**
- Point clouds → SLAMDataset (`dataset/slam_dataset.py`) → Tracker (pose) → Mapper (update neural points) → Loop Detector → PGO → Mesher

**Configuration System:**
- Config files in `config/lidar_slam/` and `config/rgbd_slam/`
- YAML-based with dataset-specific presets
- Main config class: `utils/config.py`

**Dataloaders:**
- Generic loader for common formats (ply, pcd, bin, las)
- Specialized loaders in `dataset/dataloaders/` for KITTI, MulRan, Replica, etc.
- ROS bag support via rosbag/mcap loaders

## Key Implementation Details

**GPU Selection:**
- Default GPU is set in `pin_slam.py:8` via `os.environ['CUDA_VISIBLE_DEVICES'] = '0'`
- Modify this line to use different GPU

**Output Structure:**
- Results saved in `output_root` (configurable in YAML or via `-o` flag)
- Subdirectories: `map/` (neural points), `mesh/` (reconstructed meshes), `log/` (trajectories, metrics)

**Neural Point Map:**
- Elastic and deformable with global pose adjustment
- Voxel hashing for O(1) point lookup
- Features used for loop closure detection

**Registration:**
- Correspondence-free point-to-implicit matching
- No explicit point association required
- Runs at sensor frame rate on moderate GPU

## Testing and Evaluation

**Evaluation Scripts:**
- Trajectory evaluation: `eval/eval_traj_utils.py`
- Mesh evaluation: `eval/eval_mesh_utils.py`
- See `eval/README.md` for benchmark results

**Datasets Tested:**
- LiDAR: KITTI, MulRan, Newer College, NCLT, Boreas, Apollo, Hilti
- RGB-D: Replica, TUM, Neural-RGBD
- Dynamic scenes: KITTI-MOT with online MOS

## Notes

- Results are deterministic with fixed random seed (default 42)
- For servers without GPU, use `-c` flag but expect slower performance
- Loop closure and PGO are automatic when enabled in config
- Mesh quality controlled by `mesh_min_nn` parameter (trade-off between completeness and accuracy)

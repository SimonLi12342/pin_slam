#!/usr/bin/env python3
# @file      pin_slam_ros2.py
# @author    Yue Pan [yue.pan@igg.uni-bonn.de], ROS2 port
# Copyright (c) 2024 Yue Pan, all rights reserved

import os
import sys
import time

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import torch
import wandb
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rich import print
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation

from dataset.slam_dataset import SLAMDataset
from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.loop_detector import NeuralPointMapContextManager, detect_local_loop
from utils.mapper import Mapper
from utils.mesher import Mesher
from utils.pgo import PoseGraphManager
from utils.tools import (
    freeze_decoders,
    get_time,
    load_decoders,
    save_implicit_map,
    setup_experiment,
    split_chunks,
    track_progress,
    transform_torch,
)
from utils.tracker import Tracker

'''
    📍PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency
     Y. Pan et al.
     ROS2 Humble Port
'''


class PINSLAMer(Node):
    def __init__(self, config_path, point_cloud_topic):
        super().__init__('pin_slam')

        print("[bold green]PIN-SLAM ROS2 starts[/bold green]", "📍")

        # ROS2 parameters
        self.declare_parameter('global_frame_name', 'map')
        self.declare_parameter('body_frame_name', 'base_link')
        self.declare_parameter('sensor_frame_name', 'range_sensor')

        self.global_frame_name = self.get_parameter('global_frame_name').value
        self.body_frame_name = self.get_parameter('body_frame_name').value
        self.sensor_frame_name = self.get_parameter('sensor_frame_name').value

        # Config
        self.config = Config()
        self.config.load(config_path)
        self.config.run_with_ros = True
        #self.save_map = True
        #self.save_mesh = True
        argv = ["pin_slam_ros2.py", config_path, point_cloud_topic]
        self.run_path = setup_experiment(self.config, argv)

        # Initialize MLPs
        self.geo_mlp = Decoder(self.config, self.config.geo_mlp_hidden_dim, self.config.geo_mlp_level, 1)
        self.sem_mlp = Decoder(self.config, self.config.sem_mlp_hidden_dim, self.config.sem_mlp_level,
                               self.config.sem_class_count + 1) if self.config.semantic_on else None
        self.color_mlp = Decoder(self.config, self.config.color_mlp_hidden_dim, self.config.color_mlp_level,
                                 self.config.color_channel) if self.config.color_on else None

        self.mlp_dict = {
            "sdf": self.geo_mlp,
            "semantic": self.sem_mlp,
            "color": self.color_mlp
        }
        
        print("save_map:", self.config.save_map)
        print("save_mesh:", self.config.save_mesh)

        # Initialize neural points
        self.neural_points = NeuralPoints(self.config)

        # Load decoder model
        if self.config.load_model:
            load_decoders(self.config, self.mlp_dict)
            self.config.decoder_freezed = True

        # Dataset
        self.dataset = SLAMDataset(self.config)

        # Tracker, Mapper, Mesher
        self.tracker = Tracker(self.config, self.neural_points, self.mlp_dict)
        self.mapper = Mapper(self.config, self.dataset, self.neural_points, self.mlp_dict)
        self.mesher = Mesher(self.config, self.neural_points, self.mlp_dict)

        # PGO
        self.pgm = PoseGraphManager(self.config)
        if self.config.pgo_on:
            self.pgm.add_pose_prior(0, np.eye(4), fixed=True)

        # Loop detector
        self.lcd_npmc = NeuralPointMapContextManager(self.config)
        self.loop_corrected = False
        self.loop_reg_failed_count = 0

        # Mesh params
        self.mesh_min_nn = self.config.mesh_min_nn
        self.mc_res_m = self.config.mc_res_m

        # Publishers
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.traj_pub = self.create_publisher(Path, '~/pin_path', qos)
        self.odom_pub = self.create_publisher(Odometry, '~/odometry', qos)
        self.frame_input_pub = self.create_publisher(PointCloud2, '~/frame/input', qos)
        self.frame_map_pub = self.create_publisher(PointCloud2, '~/frame/mapping', qos)
        self.frame_reg_pub = self.create_publisher(PointCloud2, '~/frame/registration', qos)
        self.map_pub = self.create_publisher(PointCloud2, '~/map/neural_points', qos)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.global_frame_name

        self.tf_broadcaster = TransformBroadcaster(self)

        # State
        self.last_message_time = time.time()
        self.begin = False
        self.travel_dist = None

        # Services
        self.save_results_srv = self.create_service(Empty, '~/save_results', self.save_slam_result_service_callback)
        self.save_mesh_srv = self.create_service(Empty, '~/save_mesh', self.save_mesh_service_callback)

        # Subscriber
        self.pc_sub = self.create_subscription(PointCloud2, point_cloud_topic, self.frame_callback, qos)
        self.get_logger().info(f'Waiting for point cloud topic: {point_cloud_topic}')

        # Timeout checker
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)

    def save_slam_result_service_callback(self, request, response):
        self.get_logger().info("Service called, save results")
        self.save_results(terminate=False)
        return response

    def save_mesh_service_callback(self, request, response):
        self.get_logger().info("Service called, save mesh")

        global_neural_pcd_down = self.neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=23)
        self.dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()

        mc_cm_str = str(round(self.mc_res_m * 1e2))
        mesh_path = os.path.join(self.run_path, "mesh",
                                 f'mesh_frame_{self.dataset.processed_frame}_{mc_cm_str}cm.ply')

        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
        chunks_aabb = split_chunks(global_neural_pcd_down, aabb, self.mc_res_m * 300)
        cur_mesh = self.mesher.recon_aabb_collections_mesh(
            chunks_aabb, self.mc_res_m, mesh_path, False,
            self.config.semantic_on, self.config.color_on,
            filter_isolated_mesh=True, mesh_min_nn=self.mesh_min_nn
        )

        return response

    @track_progress()
    def frame_callback(self, msg):
        if self.dataset.processed_frame == 0:
            print("Begin ...")

        # I. Load and preprocess
        T0 = get_time()
        self.dataset.read_frame_ros(msg)

        T1 = get_time()
        self.dataset.preprocess_frame()

        T2 = get_time()

        # II. Odometry
        if self.dataset.processed_frame > 0:
            tracking_result = self.tracker.tracking(
                self.dataset.cur_source_points, self.dataset.cur_pose_guess_torch,
                self.dataset.cur_source_colors, self.dataset.cur_source_normals
            )
            cur_pose_torch, _, _, valid_flag = tracking_result
            self.dataset.lose_track = not valid_flag
            self.dataset.update_odom_pose(cur_pose_torch)
            self.begin = True

        self.travel_dist = self.dataset.travel_dist[:self.dataset.processed_frame + 1]
        self.neural_points.travel_dist = torch.tensor(self.travel_dist, device=self.config.device, dtype=self.config.dtype)

        T3 = get_time()

        # III. Loop detection and PGO
        if self.config.pgo_on:
            self.loop_corrected = self.detect_correct_loop()

        T4 = get_time()

        # IV. Mapping
        if not self.dataset.lose_track and not self.dataset.stop_status:
            self.mapper.process_frame(
                self.dataset.cur_point_cloud_torch, self.dataset.cur_sem_labels_torch,
                self.dataset.cur_pose_torch, self.dataset.processed_frame
            )
        else:
            self.neural_points.reset_local_map(self.dataset.cur_pose_torch[:3, 3], None, self.dataset.processed_frame)

        T5 = get_time()

        # Iterations
        cur_iter_num = self.config.iters * self.config.init_iter_ratio if self.dataset.processed_frame == 0 else self.config.iters
        if self.config.adaptive_iters and self.dataset.stop_status:
            cur_iter_num = max(1, cur_iter_num - 10)
        if self.dataset.processed_frame == self.config.freeze_after_frame:
            freeze_decoders(self.mlp_dict, self.config)
            self.config.decoder_freezed = True
            self.neural_points.compute_feature_principle_components(down_rate=17)

        # Bundle adjustment
        if self.config.ba_freq_frame > 0 and (self.dataset.processed_frame + 1) % self.config.ba_freq_frame == 0:
            self.mapper.bundle_adjustment(self.config.ba_iters, self.config.ba_frame)

        # Mapping
        if self.dataset.processed_frame % self.config.mapping_freq_frame == 0:
            self.mapper.mapping(cur_iter_num)

        T6 = get_time()

        # Publishing
        self.publish_msg(msg)

        T7 = get_time()

        if not self.config.silence:
            print(f"Frame ({self.dataset.processed_frame})")
            print(f"time for frame reading          (ms): {(T1-T0)*1e3:.2f}")
            print(f"time for frame preprocessing    (ms): {(T2-T1)*1e3:.2f}")
            print(f"time for odometry               (ms): {(T3-T2)*1e3:.2f}")
            if self.config.pgo_on:
                print(f"time for loop detection and PGO (ms): {(T4-T3)*1e3:.2f}")
            print(f"time for mapping preparation    (ms): {(T5-T4)*1e3:.2f}")
            print(f"time for training               (ms): {(T6-T5)*1e3:.2f}")
            print(f"time for publishing             (ms): {(T7-T6)*1e3:.2f}")

        cur_frame_process_time = np.array([T2-T1, T3-T2, T5-T4, T6-T5, T4-T3])
        self.dataset.time_table.append(cur_frame_process_time)

        if self.config.wandb_vis_on:
            wandb_log_content = {
                'frame': self.dataset.processed_frame,
                'timing(s)/preprocess': T2-T1,
                'timing(s)/tracking': T3-T2,
                'timing(s)/pgo': T4-T3,
                'timing(s)/mapping': T6-T4
            }
            wandb.log(wandb_log_content)

        self.dataset.processed_frame += 1
        self.last_message_time = time.time()

    def check_timeout(self):
        delta_t_s = time.time() - self.last_message_time
        if delta_t_s > self.config.timeout_duration_s and self.begin:
            self.get_logger().info('Timeout reached. Saving results and exiting.')
            self.save_results(terminate=True)
            rclpy.shutdown()

    def save_results(self, terminate: bool = False):
        self.dataset.write_results()
        if self.config.pgo_on and self.pgm.pgo_count > 0:
            print(f"# Loop corrected: {self.pgm.pgo_count}")
            self.pgm.write_g2o(os.path.join(self.run_path, "final_pose_graph.g2o"))

        if terminate:
            print("Mission completed")
            self.neural_points.prune_map(self.config.max_prune_certainty, 0)
            print("comple 2: prune map")
            self.neural_points.recreate_hash(None, None, False, False)
            print("comple 3: recreate hash")

        if self.config.save_map:
            print("Saving map ...")
            neural_pcd = self.neural_points.get_neural_points_o3d(query_global=True, color_mode=0)
            o3d.io.write_point_cloud(os.path.join(self.run_path, "map", "neural_points.ply"), neural_pcd)
            if terminate:
                self.neural_points.clear_temp()
            save_implicit_map(self.run_path, self.neural_points, self.mlp_dict)
            print("Map saved")

    def publish_msg(self, input_pc_msg):
        cur_pose = self.dataset.cur_pose_ref
        cur_q = Rotation.from_matrix(cur_pose[:3, :3]).as_quat()
        cur_t = cur_pose[:3, 3]

        now = self.get_clock().now().to_msg()

        # Pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.global_frame_name
        pose_msg.pose.orientation.x = cur_q[0]
        pose_msg.pose.orientation.y = cur_q[1]
        pose_msg.pose.orientation.z = cur_q[2]
        pose_msg.pose.orientation.w = cur_q[3]
        pose_msg.pose.position.x = float(cur_t[0])
        pose_msg.pose.position.y = float(cur_t[1])
        pose_msg.pose.position.z = float(cur_t[2])

        # Odometry
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = self.sensor_frame_name
        odom_msg.pose.pose = pose_msg.pose
        self.odom_pub.publish(odom_msg)

        # TF
        transform_msg = TransformStamped()
        transform_msg.header.stamp = now
        transform_msg.header.frame_id = self.global_frame_name
        transform_msg.child_frame_id = self.sensor_frame_name
        transform_msg.transform.rotation.x = cur_q[0]
        transform_msg.transform.rotation.y = cur_q[1]
        transform_msg.transform.rotation.z = cur_q[2]
        transform_msg.transform.rotation.w = cur_q[3]
        transform_msg.transform.translation.x = float(cur_t[0])
        transform_msg.transform.translation.y = float(cur_t[1])
        transform_msg.transform.translation.z = float(cur_t[2])
        self.tf_broadcaster.sendTransform(transform_msg)

        # Path
        self.path_msg.header.stamp = now
        if self.loop_corrected:
            self.path_msg.poses = []
            for i in range(self.dataset.processed_frame):
                cur_pose = self.dataset.pgo_poses[i]
                cur_q = Rotation.from_matrix(cur_pose[:3, :3]).as_quat()
                cur_t = cur_pose[:3, 3]

                pose_msg = PoseStamped()
                pose_msg.header.stamp = now
                pose_msg.header.frame_id = self.global_frame_name
                pose_msg.pose.orientation.x = cur_q[0]
                pose_msg.pose.orientation.y = cur_q[1]
                pose_msg.pose.orientation.z = cur_q[2]
                pose_msg.pose.orientation.w = cur_q[3]
                pose_msg.pose.position.x = float(cur_t[0])
                pose_msg.pose.position.y = float(cur_t[1])
                pose_msg.pose.position.z = float(cur_t[2])
                self.path_msg.poses.append(pose_msg)
        else:
            self.path_msg.poses.append(pose_msg)

        self.traj_pub.publish(self.path_msg)

        # Point clouds
        self.publish_point_clouds(input_pc_msg, now)

    def publish_point_clouds(self, input_pc_msg, now):
        from utils.point_cloud2_ros2 import create_cloud

        fields_xyz = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        # Neural points
        if self.neural_points.neural_points is not None and self.config.publish_np_map:
            neural_point_count = self.neural_points.count()
            down_rate_level = min(neural_point_count // 500000, len(self.config.publish_np_map_down_rate_list) - 1)
            publish_np_map_down_rate = self.config.publish_np_map_down_rate_list[down_rate_level]
            neural_points_np = self.neural_points.neural_points[::publish_np_map_down_rate].detach().cpu().numpy().astype(np.float32)

            neural_points_msg = create_cloud(self.global_frame_name, now, fields_xyz, neural_points_np)
            self.map_pub.publish(neural_points_msg)

        # Mapping frame
        if self.dataset.cur_point_cloud_torch is not None:
            frame_mapping_np = self.dataset.cur_point_cloud_torch.detach().cpu().numpy().astype(np.float32)
            frame_mapping_msg = create_cloud(self.sensor_frame_name, now, fields_xyz, frame_mapping_np)
            self.frame_map_pub.publish(frame_mapping_msg)

        # Registration frame
        if self.dataset.cur_source_points is not None:
            frame_registration_np = self.dataset.cur_source_points.detach().cpu().numpy().astype(np.float32)
            frame_registration_msg = create_cloud(self.sensor_frame_name, now, fields_xyz, frame_registration_np)
            self.frame_reg_pub.publish(frame_registration_msg)

        # Republish input
        if self.config.republish_raw_input:
            input_pc_msg.header.stamp = now
            input_pc_msg.header.frame_id = self.sensor_frame_name
            self.frame_input_pub.publish(input_pc_msg)

    def detect_correct_loop(self):
        cur_frame_id = self.dataset.processed_frame
        if self.config.global_loop_on:
            if self.config.local_map_context and cur_frame_id >= self.config.local_map_context_latency:
                local_map_frame_id = cur_frame_id - self.config.local_map_context_latency
                local_map_pose = torch.tensor(self.dataset.pgo_poses[local_map_frame_id], device=self.config.device, dtype=torch.float64)
                if self.config.local_map_context_latency > 0:
                    self.neural_points.reset_local_map(local_map_pose[:3, 3], None, local_map_frame_id, False, self.config.loop_local_map_time_window)
                context_pc_local = transform_torch(self.neural_points.local_neural_points.detach(), torch.linalg.inv(local_map_pose))
                neural_points_feature = self.neural_points.local_geo_features[:-1].detach() if self.config.loop_with_feature else None
                self.lcd_npmc.add_node(local_map_frame_id, context_pc_local, neural_points_feature)
            else:
                self.lcd_npmc.add_node(cur_frame_id, self.dataset.cur_point_cloud_torch)

        self.pgm.add_frame_node(cur_frame_id, self.dataset.pgo_poses[cur_frame_id])
        self.pgm.init_poses = self.dataset.pgo_poses[:cur_frame_id + 1]

        if cur_frame_id > 0:
            self.pgm.add_odometry_factor(cur_frame_id, cur_frame_id - 1, self.dataset.last_odom_tran)
            self.pgm.estimate_drift(self.travel_dist, cur_frame_id)
            if self.config.pgo_with_pose_prior:
                self.pgm.add_pose_prior(cur_frame_id, self.dataset.pgo_poses[cur_frame_id])

            if cur_frame_id - self.pgm.last_loop_idx > self.config.pgo_freq and not self.dataset.stop_status:
                loop_candidate_mask = ((self.travel_dist[-1] - self.travel_dist) > (self.config.min_loop_travel_dist_ratio * self.config.local_map_radius))
                loop_id = None
                local_map_context_loop = False
                if np.any(loop_candidate_mask):
                    loop_id, loop_dist, loop_transform = detect_local_loop(
                        self.dataset.pgo_poses[:cur_frame_id + 1], loop_candidate_mask, self.pgm.drift_radius,
                        cur_frame_id, self.loop_reg_failed_count, dist_thre=self.config.local_loop_dist_thre,
                        drift_thre=self.config.local_loop_dist_thre * 2.0, silence=self.config.silence
                    )
                    if loop_id is None and self.config.global_loop_on:
                        loop_id, loop_cos_dist, loop_transform, local_map_context_loop = self.lcd_npmc.detect_global_loop(
                            self.dataset.pgo_poses[:cur_frame_id + 1], self.pgm.drift_radius * self.config.loop_dist_drift_ratio_thre,
                            loop_candidate_mask, self.neural_points
                        )
                if loop_id is not None:
                    if self.config.loop_z_check_on and abs(loop_transform[2, 3]) > self.config.voxel_size_m * 3.0:
                        return False
                    pose_init_np = self.dataset.pgo_poses[loop_id] @ loop_transform
                    pose_init_torch = torch.tensor(pose_init_np, device=self.config.device, dtype=torch.float64)
                    self.neural_points.recreate_hash(pose_init_torch[:3, 3], None, True, True, loop_id)
                    pose_refine_torch, _, _, reg_valid_flag = self.tracker.tracking(self.dataset.cur_source_points, pose_init_torch, loop_reg=True)
                    pose_refine_np = pose_refine_torch.detach().cpu().numpy()
                    loop_transform = np.linalg.inv(self.dataset.pgo_poses[loop_id]) @ pose_refine_np
                    if not self.config.silence:
                        print("[bold green]Refine loop transformation succeed[/bold green]")
                    if reg_valid_flag:
                        reg_valid_flag = self.pgm.add_loop_factor(cur_frame_id, loop_id, loop_transform)
                    if reg_valid_flag:
                        self.pgm.optimize_pose_graph()
                        cur_loop_vis_id = cur_frame_id - self.config.local_map_context_latency if local_map_context_loop else cur_frame_id
                        self.pgm.loop_edges.append(np.array([loop_id, cur_loop_vis_id], dtype=np.uint32))
                        pose_diff_torch = torch.tensor(self.pgm.get_pose_diff(), device=self.config.device, dtype=self.config.dtype)
                        self.dataset.cur_pose_torch = torch.tensor(self.pgm.cur_pose, device=self.config.device, dtype=self.config.dtype)
                        self.neural_points.adjust_map(pose_diff_torch)
                        self.neural_points.recreate_hash(self.dataset.cur_pose_torch[:3, 3], None, (not self.config.pgo_merge_map), self.config.rehash_with_time, cur_frame_id)
                        self.mapper.transform_data_pool(pose_diff_torch)
                        self.dataset.update_poses_after_pgo(self.pgm.pgo_poses)
                        self.pgm.last_loop_idx = cur_frame_id
                        self.pgm.min_loop_idx = min(self.pgm.min_loop_idx, loop_id)
                        self.loop_reg_failed_count = 0
                        return True
                    else:
                        if not self.config.silence:
                            print("[bold red]Registration failed, reject the loop candidate[/bold red]")
                        self.neural_points.recreate_hash(self.dataset.cur_pose_torch[:3, 3], None, True, True, cur_frame_id)
                        self.loop_reg_failed_count += 1

        return False


def main(args=None):
    rclpy.init(args=args)

    config_path = "./config/lidar_slam/run.yaml"
    point_cloud_topic = "/velodyne_points"

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        point_cloud_topic = sys.argv[2]

    print(f"Config: {config_path}")
    print(f"Topic: {point_cloud_topic}")

    slamer = PINSLAMer(config_path, point_cloud_topic)

    try:
        rclpy.spin(slamer)
    except KeyboardInterrupt:
        pass
    finally:
        slamer.save_results(terminate=True)
        slamer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

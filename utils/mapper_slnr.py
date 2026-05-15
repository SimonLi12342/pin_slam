#!/usr/bin/env python3
# @file      mapper_slnr.py

from __future__ import annotations

import math
from collections import deque

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import torch
from rich import print
from tqdm import tqdm

from dataset.slam_dataset import SLAMDataset
from model.neural_points_slnr import NeuralPointsSLNR
from utils.config import Config
from utils.tools import transform_batch_torch


class Mapper:
    def __init__(
        self,
        config: Config,
        dataset: SLAMDataset,
        neural_points: NeuralPointsSLNR,
        decoders: dict,
    ):
        self.config = config
        self.silence = config.silence
        self.dataset = dataset
        self.neural_points = neural_points
        self.sdf_mlp = decoders["sdf"]
        self.sem_mlp = decoders["semantic"]
        self.color_mlp = decoders["color"]
        self.device = config.device
        self.dtype = config.dtype
        self.used_poses = None
        self.total_iter = 0
        self.cur_new_point_ratio = 0.0
        self.cur_sample_count = 0
        self.pool_sample_count = 0
        self.ba_done_flag = False
        self._ba_warned = False
        self.pool_frame_limit = max(4, int(self.neural_points.backend.recent_buffer_size))
        self.init_pool()

    def init_pool(self):
        self.pool_points = deque(maxlen=self.pool_frame_limit)
        self.pool_sdf = deque(maxlen=self.pool_frame_limit)
        self.pool_frame_ids = deque(maxlen=self.pool_frame_limit)

    def determine_used_pose(self):
        cur_frame = self.dataset.processed_frame
        if self.config.pgo_on:
            self.used_poses = torch.tensor(
                self.dataset.pgo_poses[: cur_frame + 1],
                device=self.device,
                dtype=torch.float64,
            )
        elif self.config.track_on:
            self.used_poses = torch.tensor(
                self.dataset.odom_poses[: cur_frame + 1],
                device=self.device,
                dtype=torch.float64,
            )
        elif self.dataset.gt_pose_provided:
            self.used_poses = torch.tensor(
                self.dataset.gt_poses[: cur_frame + 1],
                device=self.device,
                dtype=torch.float64,
            )

    def process_frame(
        self,
        point_cloud_torch: torch.Tensor,
        frame_label_torch: torch.Tensor,
        cur_pose_torch: torch.Tensor,
        frame_id: int,
        filter_dynamic: bool = False,
    ):
        self.dataset.static_mask = torch.ones(
            point_cloud_torch.shape[0],
            dtype=torch.bool,
            device=self.device,
        )
        if filter_dynamic and not self.silence:
            print("Dynamic filtering is disabled in the SLNR mapping integration path.")

        anchor_count_before = self.neural_points.count()

        points_local = point_cloud_torch[:, :3].detach().cpu().numpy().astype(np.float32)
        pose_wc = cur_pose_torch.detach().cpu().numpy().astype(np.float32)
        processed = self.neural_points.backend.ingest_frame(points_local, pose_wc, frame_id)

        self.neural_points.sync_from_backend()
        self.determine_used_pose()
        self.neural_points.reset_local_map(cur_pose_torch[:3, 3], cur_pose_torch[:3, :3], frame_id, reboot_map=True)

        anchor_count_after = self.neural_points.count()
        if anchor_count_before > 0:
            self.cur_new_point_ratio = max(0.0, float(anchor_count_after - anchor_count_before) / float(anchor_count_before))
        else:
            self.cur_new_point_ratio = 1.0 if anchor_count_after > 0 else 0.0

        if processed is not None:
            pooled_points = torch.from_numpy(processed.train_points_world).to(self.device, dtype=self.dtype)
            pooled_sdf = torch.zeros((pooled_points.shape[0],), device=self.device, dtype=self.dtype)
            self.pool_points.append(pooled_points)
            self.pool_sdf.append(pooled_sdf)
            self.pool_frame_ids.append(int(frame_id))
            self.cur_sample_count = pooled_points.shape[0]
        else:
            self.cur_sample_count = 0

        self.pool_sample_count = int(sum(points.shape[0] for points in self.pool_points))
        self.neural_points.record_memory(verbose=(not self.silence))

    def mapping(self, iter_count):
        iter_count = max(1, int(iter_count))
        for _ in tqdm(range(iter_count), disable=self.silence):
            self.neural_points.backend.optimize_step()
            self.total_iter += 1
        self.neural_points.sync_from_backend()
        if self.dataset.cur_pose_torch is not None:
            self.neural_points.reset_local_map(
                self.dataset.cur_pose_torch[:3, 3],
                self.dataset.cur_pose_torch[:3, :3],
                self.dataset.processed_frame,
                reboot_map=True,
            )

    def bundle_adjustment(self, iter_count, window_size: int = 50, use_lie_group: bool = False):
        if not self._ba_warned and not self.silence:
            print("Bundle adjustment is not implemented for the SLNR mapping integration path yet.")
            self._ba_warned = True

    def transform_data_pool(self, pose_diff_torch: torch.Tensor):
        self.neural_points.backend.transform_training_frames(pose_diff_torch)
        transformed_points = deque(maxlen=self.pool_frame_limit)
        for frame_id, points in zip(self.pool_frame_ids, self.pool_points):
            pose_diff = pose_diff_torch[min(int(frame_id), pose_diff_torch.shape[0] - 1)]
            transformed_points.append(transform_batch_torch(points, pose_diff.unsqueeze(0).expand(points.shape[0], -1, -1)))
        self.pool_points = transformed_points
        self.neural_points.sync_from_backend()
        if self.dataset.cur_pose_torch is not None:
            self.neural_points.reset_local_map(
                self.dataset.cur_pose_torch[:3, 3],
                self.dataset.cur_pose_torch[:3, :3],
                self.dataset.processed_frame,
                reboot_map=True,
            )

    def get_data_pool_o3d(self, down_rate=1, only_cur_data=False):
        if len(self.pool_points) == 0:
            return None

        if only_cur_data:
            points_torch = self.pool_points[-1]
            sdf_torch = self.pool_sdf[-1]
        else:
            points_torch = torch.cat(list(self.pool_points), dim=0)
            sdf_torch = torch.cat(list(self.pool_sdf), dim=0)

        points_np = points_torch[::down_rate].detach().cpu().numpy().astype(np.float64)
        sdf_np = sdf_torch[::down_rate].detach().cpu().numpy().astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        if sdf_np.shape[0] > 0:
            min_sdf = self.config.free_sample_end_dist_m * -2.0
            max_sdf = -min_sdf
            sdf_vis = np.clip((sdf_np - min_sdf) / (max_sdf - min_sdf), 0.0, 1.0)
            colors = cm.get_cmap("seismic")(1.0 - sdf_vis)[:, :3].astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def free_pool(self):
        self.pool_points.clear()
        self.pool_sdf.clear()
        self.pool_frame_ids.clear()

    def sdf(self, x, get_std=False, min_nn_count=1, accumulate_stability=False):
        geo_feature, _, weight_knn, nn_count, _ = self.neural_points.query_feature(
            x,
            training_mode=accumulate_stability,
        )
        sdf_pred = self.sdf_mlp.sdf(geo_feature)
        sdf_std = None
        if not self.config.weighted_first:
            sdf_pred_mean = torch.sum(sdf_pred * weight_knn, dim=1)
            if get_std:
                sdf_var = torch.sum(weight_knn * (sdf_pred - sdf_pred_mean.unsqueeze(-1)) ** 2, dim=1)
                sdf_std = torch.sqrt(sdf_var).squeeze(1)
            sdf_pred = sdf_pred_mean.squeeze(1)

        valid_mask = nn_count >= min_nn_count
        return sdf_pred, sdf_std, valid_mask

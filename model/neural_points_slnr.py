#!/usr/bin/env python3
# @file      neural_points_slnr.py

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import roma
import torch
import torch.nn as nn

from slnr.backend_slnr import SLNRMapBackend
from slnr.local_sdf_slnr import gs_searching_module, trunc_exp
from utils.config import Config
from utils.tools import feature_pca_torch, rotmat_to_quat


class NeuralPointsSLNR(nn.Module):
    def __init__(
        self,
        config: Config,
        slnr_cfg: Optional[Dict] = None,
        backend: Optional[SLNRMapBackend] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.silence = config.silence
        self.device = config.device
        self.dtype = config.dtype

        self.backend = backend if backend is not None else SLNRMapBackend(config, slnr_cfg=slnr_cfg)
        self.slnr_cfg = slnr_cfg or getattr(self.backend, "slnr_cfg", {})

        self.resolution = float(self.backend.local_sdf_resolution)
        self.color_on = False
        self.color_features = None
        self.local_color_features = None
        self.geo_feature_pca = None
        self.color_feature_pca = None

        self.temporal_local_map_on = True
        self.local_map_radius = config.local_map_radius
        self.diff_travel_dist_local = config.local_map_radius * config.local_map_travel_dist_ratio
        self.reboot_ts = 0
        self.travel_dist = None
        self.cur_ts = 0
        self.max_ts = 0

        self.neural_points = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        self.local_neural_points = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        self.point_orientations = torch.empty((0, 4), device=self.device, dtype=self.dtype)
        self.local_point_orientations = torch.empty((0, 4), device=self.device, dtype=self.dtype)
        self.geo_features = torch.empty((1, 3), device=self.device, dtype=self.dtype)
        self.local_geo_features = torch.empty((1, 3), device=self.device, dtype=self.dtype)
        self.point_ts_create = torch.empty((0,), device=self.device, dtype=torch.int)
        self.point_ts_update = torch.empty((0,), device=self.device, dtype=torch.int)
        self.local_point_ts_update = torch.empty((0,), device=self.device, dtype=torch.int)
        self.point_certainties = torch.empty((0,), device=self.device, dtype=self.dtype)
        self.local_point_certainties = torch.empty((0,), device=self.device, dtype=self.dtype)
        self.local_mask = None
        self.global2local = None
        self.local_indices = torch.empty((0,), device=self.device, dtype=torch.long)
        self.local_rotations = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        self.local_scalings = torch.empty((0, 3), device=self.device, dtype=self.dtype)

        self.cur_memory_mb = 0.0
        self.memory_footprint = []

        self.sync_from_backend()

    def rebind_config(self, config: Config, slnr_cfg: Optional[Dict] = None) -> None:
        self.config = config
        self.silence = config.silence
        self.device = config.device
        self.dtype = config.dtype
        self.local_map_radius = config.local_map_radius
        self.diff_travel_dist_local = config.local_map_radius * config.local_map_travel_dist_ratio
        self.slnr_cfg = slnr_cfg or self.slnr_cfg
        self.backend.config = config
        self.backend.slnr_cfg = self.slnr_cfg
        self.resolution = float(self.backend.local_sdf_resolution)
        self.sync_from_backend()

    def is_empty(self):
        return self.neural_points.shape[0] == 0

    def count(self):
        return self.neural_points.shape[0]

    def local_count(self):
        return self.local_neural_points.shape[0]

    def _empty_display_features(self) -> torch.Tensor:
        return torch.zeros((1, 3), device=self.device, dtype=self.dtype)

    def _rotvec_to_quat(self, rotvec: torch.Tensor) -> torch.Tensor:
        if rotvec.numel() == 0:
            return torch.empty((0, 4), device=rotvec.device, dtype=rotvec.dtype)
        rotmat = roma.rotvec_to_rotmat(rotvec)
        quat = rotmat_to_quat(rotmat)
        quat = quat / (torch.norm(quat, dim=1, keepdim=True) + 1e-12)
        return quat.to(dtype=self.dtype)

    def _display_features_from_points(self, points: torch.Tensor) -> torch.Tensor:
        if points.numel() == 0:
            return self._empty_display_features()
        pad = torch.zeros((1, 3), device=points.device, dtype=points.dtype)
        return torch.cat((points.detach(), pad), dim=0)

    def sync_from_backend(self) -> None:
        local_sdfs = self.backend.local_sdfs
        if local_sdfs is None or local_sdfs.positions.shape[0] == 0:
            self.neural_points = torch.empty((0, 3), device=self.device, dtype=self.dtype)
            self.point_orientations = torch.empty((0, 4), device=self.device, dtype=self.dtype)
            self.geo_features = self._empty_display_features()
            self.point_ts_create = torch.empty((0,), device=self.device, dtype=torch.int)
            self.point_ts_update = torch.empty((0,), device=self.device, dtype=torch.int)
            self.point_certainties = torch.empty((0,), device=self.device, dtype=self.dtype)
            self.reset_local_map(torch.zeros(3, device=self.device, dtype=self.dtype), None, self.cur_ts)
            return

        old_certainty = self.point_certainties
        old_update_ts = self.point_ts_update

        self.neural_points = local_sdfs.positions
        self.point_orientations = self._rotvec_to_quat(local_sdfs.rotations)
        self.geo_features = self._display_features_from_points(self.neural_points)

        anchor_ts = self.backend.anchor_frame_ids.to(self.device)
        self.point_ts_create = anchor_ts.clone()

        if old_update_ts.shape[0] == anchor_ts.shape[0]:
            self.point_ts_update = old_update_ts.to(self.device)
        elif old_update_ts.shape[0] < anchor_ts.shape[0]:
            appended = anchor_ts[old_update_ts.shape[0] :]
            self.point_ts_update = torch.cat((old_update_ts.to(self.device), appended), dim=0)
        else:
            self.point_ts_update = old_update_ts[: anchor_ts.shape[0]].to(self.device)

        if old_certainty.shape[0] == anchor_ts.shape[0]:
            self.point_certainties = old_certainty.to(self.device)
        elif old_certainty.shape[0] < anchor_ts.shape[0]:
            appended = torch.zeros(
                (anchor_ts.shape[0] - old_certainty.shape[0],),
                device=self.device,
                dtype=self.dtype,
            )
            self.point_certainties = torch.cat((old_certainty.to(self.device), appended), dim=0)
        else:
            self.point_certainties = old_certainty[: anchor_ts.shape[0]].to(self.device)

        if self.local_mask is not None and self.local_mask.shape[0] == self.count() + 1:
            self.reset_local_map(
                self.local_neural_points.mean(dim=0) if self.local_count() > 0 else torch.zeros(3, device=self.device, dtype=self.dtype),
                None,
                self.cur_ts,
            )
        else:
            self.reset_local_map(torch.zeros(3, device=self.device, dtype=self.dtype), None, self.cur_ts)

    def record_memory(self, verbose: bool = True, record_footprint: bool = True):
        if verbose:
            print("# Global neural point: %d" % self.count())
            print("# Local  neural point: %d" % self.local_count())
        point_dim = 3 + 3 + 3 + 1
        self.cur_memory_mb = self.count() * point_dim * 4.0 / 1024.0 / 1024.0
        if verbose:
            print("Current map memory consumption: {:.3f} MB".format(self.cur_memory_mb))
        if record_footprint:
            self.memory_footprint.append(self.cur_memory_mb)

    def compute_feature_principle_components(self, down_rate: int = 1):
        if self.geo_features.shape[0] > 4:
            _, self.geo_feature_pca = feature_pca_torch(
                self.geo_features[:-1].detach(),
                down_rate=max(1, down_rate),
                project_data=False,
            )

    def _resolve_query_map(
        self,
        query_locally: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_sdfs = self.backend.local_sdfs
        if local_sdfs is None:
            empty_xyz = torch.empty((0, 3), device=self.device, dtype=self.dtype)
            empty_idx = torch.empty((0,), device=self.device, dtype=torch.long)
            return empty_xyz, empty_xyz, empty_xyz, empty_idx

        use_local = query_locally and self.local_indices.numel() > 0
        if use_local:
            positions = self.local_neural_points
            rotations = self.local_rotations
            scalings = self.local_scalings
            global_indices = self.local_indices
        else:
            positions = local_sdfs.positions
            rotations = local_sdfs.rotations
            scalings = local_sdfs.scalings
            global_indices = torch.arange(positions.shape[0], device=self.device, dtype=torch.long)
        return positions, rotations, scalings, global_indices

    def _query_core(
        self,
        query_points: torch.Tensor,
        query_locally: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positions, rotations, scalings, global_indices = self._resolve_query_map(query_locally)
        nn_k = max(1, int(self.backend.query_nn_k))

        if positions.shape[0] == 0 or query_points.shape[0] == 0:
            geo = torch.zeros((query_points.shape[0], nn_k, 3), device=query_points.device, dtype=query_points.dtype)
            weight = torch.zeros((query_points.shape[0], nn_k), device=query_points.device, dtype=query_points.dtype)
            nn_counts = torch.zeros((query_points.shape[0],), device=query_points.device, dtype=torch.long)
            certainty = torch.zeros((query_points.shape[0],), device=query_points.device, dtype=query_points.dtype)
            global_neighbor_idx = torch.full((query_points.shape[0], nn_k), -1, device=query_points.device, dtype=torch.long)
            valid_mask = torch.zeros((query_points.shape[0], nn_k), device=query_points.device, dtype=torch.bool)
            return geo, weight, nn_counts, certainty, global_neighbor_idx, valid_mask

        geo_features, neighbor_idx, inside_mask = gs_searching_module(
            query_points,
            positions,
            rotations,
            scalings,
            nn_k,
            self.resolution,
            self.backend.gss_vox_size,
        )

        valid_mask = neighbor_idx >= 0
        nn_counts = valid_mask.sum(dim=-1)
        dists2 = (geo_features * geo_features).sum(dim=-1)
        weight = trunc_exp(-0.5 * dists2)
        weight[~valid_mask] = 0.0
        empty_row_mask = nn_counts == 0
        if empty_row_mask.any():
            weight[empty_row_mask] = 1.0 / float(nn_k)
            valid_mask[empty_row_mask] = True

        weight_row_sums = weight.sum(dim=1, keepdim=True)
        weight = weight / (weight_row_sums + 1e-12)
        weight[neighbor_idx < 0] = 0.0

        inside_global_idx = global_indices[inside_mask]
        global_neighbor_idx = torch.full_like(neighbor_idx, -1, dtype=torch.long)
        certainty_knn = torch.zeros_like(weight)
        if inside_global_idx.numel() > 0 and valid_mask.any():
            safe_neighbor_idx = neighbor_idx.clamp_min(0)
            global_neighbor_idx[valid_mask] = inside_global_idx[safe_neighbor_idx][valid_mask]
            certainty_knn[valid_mask] = self.point_certainties[global_neighbor_idx[valid_mask]]
        queried_certainty = (certainty_knn * weight).sum(dim=1)
        return geo_features, weight, nn_counts, queried_certainty, global_neighbor_idx, valid_mask

    def reset_local_map(
        self,
        sensor_position: torch.Tensor,
        sensor_orientation: torch.Tensor,
        cur_ts: int,
        use_travel_dist: bool = True,
        diff_ts_local: int = 50,
        reboot_map: bool = False,
    ):
        self.cur_ts = int(cur_ts)
        self.max_ts = max(self.max_ts, self.cur_ts)

        if self.is_empty():
            self.local_neural_points = torch.empty((0, 3), device=self.device, dtype=self.dtype)
            self.local_point_orientations = torch.empty((0, 4), device=self.device, dtype=self.dtype)
            self.local_geo_features = self._empty_display_features()
            self.local_point_certainties = torch.empty((0,), device=self.device, dtype=self.dtype)
            self.local_point_ts_update = torch.empty((0,), device=self.device, dtype=torch.int)
            self.local_mask = torch.ones((1,), device=self.device, dtype=torch.bool)
            self.global2local = torch.full((1,), -1, device=self.device, dtype=torch.long)
            self.local_indices = torch.empty((0,), device=self.device, dtype=torch.long)
            self.local_rotations = torch.empty((0, 3), device=self.device, dtype=self.dtype)
            self.local_scalings = torch.empty((0, 3), device=self.device, dtype=self.dtype)
            return

        local_sdfs = self.backend.local_sdfs
        point_ts_used = self.point_ts_create

        if self.temporal_local_map_on and self.travel_dist is not None and point_ts_used.numel() > 0:
            if use_travel_dist and point_ts_used.max().item() < len(self.travel_dist):
                cur_travel = self.travel_dist[min(self.cur_ts, len(self.travel_dist) - 1)]
                point_travel = self.travel_dist[point_ts_used.clamp_max(len(self.travel_dist) - 1)]
                time_mask = torch.abs(cur_travel - point_travel) < self.diff_travel_dist_local
            else:
                time_mask = torch.abs(point_ts_used - self.cur_ts) < diff_ts_local
            if reboot_map:
                time_mask = time_mask & (point_ts_used >= self.reboot_ts)
            if time_mask.sum().item() < 16:
                time_mask = torch.ones_like(point_ts_used, dtype=torch.bool)
        else:
            time_mask = torch.ones_like(point_ts_used, dtype=torch.bool)

        sensor_position = sensor_position.to(self.neural_points) if sensor_position is not None else torch.zeros(3, device=self.device, dtype=self.dtype)
        masked_points = self.neural_points[time_mask]
        masked_dist2 = ((masked_points - sensor_position) ** 2).sum(dim=-1)
        dist_mask = masked_dist2 < (self.local_map_radius ** 2)

        time_mask_idx = torch.nonzero(time_mask, as_tuple=False).flatten()
        local_indices = time_mask_idx[dist_mask]
        if local_indices.numel() < 16:
            local_indices = torch.arange(self.count(), device=self.device, dtype=torch.long)

        local_mask = torch.zeros((self.count(),), device=self.device, dtype=torch.bool)
        local_mask[local_indices] = True

        self.local_indices = local_indices
        self.local_neural_points = self.neural_points[local_indices]
        self.local_point_orientations = self.point_orientations[local_indices]
        self.local_geo_features = self._display_features_from_points(self.local_neural_points)
        self.local_point_certainties = self.point_certainties[local_indices]
        self.local_point_ts_update = self.point_ts_update[local_indices]
        self.local_rotations = local_sdfs.rotations[local_indices]
        self.local_scalings = local_sdfs.scalings[local_indices]

        padded_local_mask = torch.cat((local_mask, torch.ones((1,), device=self.device, dtype=torch.bool)), dim=0)
        self.local_mask = padded_local_mask
        self.global2local = torch.full((self.count() + 1,), -1, device=self.device, dtype=torch.long)
        self.global2local[local_indices] = torch.arange(local_indices.shape[0], device=self.device, dtype=torch.long)

    def assign_local_to_global(self):
        if self.local_mask is None or self.local_mask.shape[0] != self.count() + 1:
            return
        local_mask = self.local_mask[:-1]
        if local_mask.sum().item() == self.local_point_certainties.shape[0]:
            self.point_certainties[local_mask] = self.local_point_certainties
        if local_mask.sum().item() == self.local_point_ts_update.shape[0]:
            self.point_ts_update[local_mask] = self.local_point_ts_update

    def query_feature(
        self,
        query_points: torch.Tensor,
        query_ts: torch.Tensor = None,
        training_mode: bool = True,
        query_locally: bool = True,
        query_geo_feature: bool = True,
        query_color_feature: bool = False,
    ):
        if not query_geo_feature and not query_color_feature:
            raise RuntimeError("At least one feature type must be queried.")
        if query_color_feature:
            raise RuntimeError("SLNR integration path does not support color features.")

        geo_feature, weight_knn, nn_counts, queried_certainty, global_neighbor_idx, valid_mask = self._query_core(
            query_points,
            query_locally=query_locally,
        )

        if training_mode and self.count() > 0 and query_points.shape[0] > 0:
            if valid_mask.any():
                certainty_weight = weight_knn.squeeze(-1)
                flat_index = global_neighbor_idx[valid_mask]
                flat_weight = certainty_weight[valid_mask]
                self.point_certainties.scatter_add_(0, flat_index, flat_weight)
                if query_ts is not None:
                    ts_expand = query_ts.view(-1, 1).expand_as(global_neighbor_idx)
                    flat_ts = ts_expand[valid_mask].to(self.point_ts_update.dtype)
                    self.point_ts_update.scatter_reduce_(
                        dim=0,
                        index=flat_index,
                        src=flat_ts,
                        reduce="amax",
                        include_self=True,
                    )
                if query_locally and self.local_indices.numel() > 0:
                    self.local_point_certainties = self.point_certainties[self.local_indices]
                    self.local_point_ts_update = self.point_ts_update[self.local_indices]

        if self.config.weighted_first:
            geo_feature = torch.sum(geo_feature * weight_knn.unsqueeze(-1), dim=1)

        return geo_feature, None, weight_knn.unsqueeze(-1), nn_counts, queried_certainty

    def prune_map(self, prune_certainty_thre, min_prune_count=500, global_prune=False):
        return False

    def adjust_map(self, pose_diff_torch):
        if self.is_empty():
            return

        point_ts = self.point_ts_create.clamp_max(pose_diff_torch.shape[0] - 1).long()
        diff_pose = pose_diff_torch[point_ts].to(self.neural_points)
        rot_diff = diff_pose[:, :3, :3]
        tran_diff = diff_pose[:, :3, 3]

        new_points = torch.bmm(rot_diff, self.neural_points.unsqueeze(-1)).squeeze(-1) + tran_diff

        rot_old = roma.rotvec_to_rotmat(self.backend.local_sdfs.rotations)
        rot_new = torch.matmul(rot_diff, rot_old)
        rotvec_new = roma.rotmat_to_rotvec(rot_new)

        with torch.no_grad():
            self.backend.local_sdfs.positions.copy_(new_points)
            self.backend.local_sdfs.rotations.copy_(rotvec_new)

        self.sync_from_backend()

    def recreate_hash(
        self,
        sensor_position: torch.Tensor,
        sensor_orientation: torch.Tensor,
        kept_points: bool = True,
        with_ts: bool = True,
        cur_ts=0,
    ):
        self.backend.rebuild_svh_from_anchors()
        self.sync_from_backend()
        if sensor_position is None:
            sensor_position = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.reset_local_map(sensor_position, sensor_orientation, cur_ts)
        self.record_memory(verbose=(not self.silence), record_footprint=False)

    def set_search_neighborhood(self, num_nei_cells: int = 1, search_alpha: float = 1.0):
        self.search_num_nei_cells = num_nei_cells
        self.search_alpha = search_alpha

    def query_certainty(self, query_points: torch.Tensor):
        _, _, _, queried_certainty, _, _ = self._query_core(query_points, query_locally=True)
        return queried_certainty

    def get_neural_points_o3d(
        self,
        query_global: bool = True,
        color_mode: int = -1,
        random_down_ratio: int = 1,
    ):
        if query_global:
            points_torch = self.neural_points
            feat_torch = self.geo_features[:-1]
            ts_torch = self.point_ts_update
            certainty_torch = self.point_certainties
        else:
            points_torch = self.local_neural_points
            feat_torch = self.local_geo_features[:-1]
            ts_torch = self.local_point_ts_update
            certainty_torch = self.local_point_certainties

        points_np = points_torch[::random_down_ratio].detach().cpu().numpy().astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        if points_np.shape[0] == 0:
            return pcd

        if color_mode == 0:
            feat_vis = feat_torch[::random_down_ratio]
            if feat_vis.shape[0] > 4:
                feat_rgb, _ = feature_pca_torch(
                    feat_vis.detach(),
                    principal_components=self.geo_feature_pca,
                    down_rate=1,
                )
                pcd.colors = o3d.utility.Vector3dVector(feat_rgb.detach().cpu().numpy().astype(np.float64))
            else:
                pcd.paint_uniform_color([0.2, 0.6, 0.9])
        elif color_mode == 2 and ts_torch.numel() > 0:
            denom = max(1, self.max_ts)
            ts_np = (ts_torch[::random_down_ratio].detach().cpu().numpy().astype(np.float64) / denom).clip(0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(cm.get_cmap("jet")(ts_np)[:, :3].astype(np.float64))
        elif color_mode == 3 and certainty_torch.numel() > 0:
            certainty_np = (1.0 - certainty_torch[::random_down_ratio].detach().cpu().numpy().astype(np.float64) / 1000.0)
            pcd.colors = o3d.utility.Vector3dVector(np.repeat(certainty_np[:, None], 3, axis=1))
        elif color_mode == 4:
            pcd.colors = o3d.utility.Vector3dVector(np.random.rand(points_np.shape[0], 3).astype(np.float64))

        return pcd

    def clear_temp(self, clean_more: bool = False):
        self.local_mask = None
        self.global2local = None
        self.local_indices = torch.empty((0,), device=self.device, dtype=torch.long)
        self.local_neural_points = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        self.local_point_orientations = torch.empty((0, 4), device=self.device, dtype=self.dtype)
        self.local_geo_features = self._empty_display_features()
        self.local_point_certainties = torch.empty((0,), device=self.device, dtype=self.dtype)
        self.local_point_ts_update = torch.empty((0,), device=self.device, dtype=torch.int)
        self.local_rotations = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        self.local_scalings = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        if clean_more:
            self.geo_feature_pca = None
            self.color_feature_pca = None

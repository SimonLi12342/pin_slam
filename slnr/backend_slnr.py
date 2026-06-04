from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import main_util_slnr as main_util
from .common_slnr import (
    InMemoryFrameDatasetSLNR,
    ProcessedFrameSLNR,
    TrainingFrameSLNR,
    WarmupFrameSLNR,
    compute_bounds_from_voxels,
    estimate_world_normals,
    normals_to_rotvec,
    sample_fixed_points,
    sample_items,
    transform_normals,
    transform_points,
    voxel_select_first,
)
from .local_sdf_slnr import LocalSDF
from .network_slnr import NeuralMap

REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_HASH_LIB = REPO_ROOT / "modules" / "sparse_hash" / "build" / "libsvh.so"

if not SPARSE_HASH_LIB.exists():
    raise FileNotFoundError(
        f"Sparse hash extension not found: {SPARSE_HASH_LIB}. "
        "Build pin_slam/modules/sparse_hash first."
    )

torch.classes.load_library(str(SPARSE_HASH_LIB))


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _apply_pose_to_numpy_points(points: np.ndarray, pose_diff: np.ndarray) -> np.ndarray:
    rotation = pose_diff[:3, :3]
    translation = pose_diff[:3, 3]
    return (points @ rotation.T + translation[None, :]).astype(np.float32)


def _apply_pose_to_numpy_matrix(pose_wc: np.ndarray, pose_diff: np.ndarray) -> np.ndarray:
    return (pose_diff @ pose_wc).astype(np.float32)


class SLNRMapBackend(nn.Module):
    def __init__(self, config, slnr_cfg: Optional[Dict] = None):
        super().__init__()

        self.config = config
        self.device = config.device
        self.dtype = torch.float32
        self.slnr_cfg = slnr_cfg or {}
        self.rng = np.random.default_rng(config.seed)

        self.hash_voxel_size = float(self.slnr_cfg.get("hash_voxel_size", config.voxel_size_m))
        self.ht_size = int(self.slnr_cfg.get("hash_table_size", 0x80000))
        self.inval_val = int(self.slnr_cfg.get("hash_invalid_value", 999999))
        self.res_scale = max(1, int(self.slnr_cfg.get("res_scale", 1)))
        self.down_vox_size = float(
            self.slnr_cfg.get("down_vox_size", max(config.vox_down_m, config.voxel_size_m * 0.125))
        )
        self.local_sdf_resolution = float(
            self.slnr_cfg.get("local_sdf_resolution", max(config.voxel_size_m * 0.75, 0.05))
        )
        self.query_nn_k = max(1, int(self.slnr_cfg.get("query_nn_k", config.query_nn_k)))
        self.gss_vox_size = float(self.slnr_cfg.get("gss_vox_size", self.hash_voxel_size))

        self.enable_densify = _as_bool(self.slnr_cfg.get("enable_densify", False), False)
        self.add_scale_loss = _as_bool(self.slnr_cfg.get("add_scale_loss", False), False)
        self.n_iters_densitify = max(1, int(self.slnr_cfg.get("densify_every_iters", 200)))

        self.warmup_frames = max(1, int(self.slnr_cfg.get("warmup_frames", 1)))
        self.recent_buffer_size = max(1, int(self.slnr_cfg.get("recent_buffer_size", 30)))
        self.replay_buffer_size = max(1, int(self.slnr_cfg.get("replay_buffer_size", 120)))
        self.replay_insert_interval = max(1, int(self.slnr_cfg.get("replay_insert_interval", 5)))
        self.train_recent_ratio = float(self.slnr_cfg.get("train_recent_ratio", 0.7))
        self.train_frames_per_step = max(1, int(self.slnr_cfg.get("train_frames_per_step", 1)))
        self.train_points_per_frame = max(256, int(self.slnr_cfg.get("train_points_per_frame", 4096)))
        self.min_points_per_frame = max(64, int(self.slnr_cfg.get("min_points_per_frame", 256)))
        self.hash_insert_down_voxel_size = float(
            self.slnr_cfg.get("hash_insert_down_voxel_size", max(config.vox_down_m, 0.05))
        )
        self.max_new_anchors_per_frame = max(0, int(self.slnr_cfg.get("max_new_anchors_per_frame", 2000)))
        self.refresh_map_every_n_frames = max(1, int(self.slnr_cfg.get("refresh_map_every_n_frames", 1)))

        self.train_lr = float(self.slnr_cfg.get("learning_rate", 5e-3))
        self.lr_ratio = float(self.slnr_cfg.get("lr_ratio", 0.1))
        self.hidden_dim = int(self.slnr_cfg.get("hidden_dim", config.geo_mlp_hidden_dim))
        self.num_layers = int(self.slnr_cfg.get("num_layers", max(config.geo_mlp_level + 1, 2)))
        self.freeze_after_iters = int(
            self.slnr_cfg.get("freeze_after_iters", max(config.freeze_after_frame * max(config.iters, 1), 0))
        )
        self.lr_decay_iters = max(
            1,
            int(self.slnr_cfg.get("lr_decay_iters", max(config.freeze_after_frame * max(config.iters, 1), 1000))),
        )

        self.n_rays = max(64, int(self.slnr_cfg.get("n_rays", min(max(config.bs // 8, 256), 2048))))
        self.n_max_interset = max(1, int(self.slnr_cfg.get("n_max_interset", 10)))
        self.sur_behind_dis = float(
            self.slnr_cfg.get("sur_behind_dis", max(config.surface_sample_range_m * 2.0, 0.1))
        )
        self.n_surf_samples = max(2, int(self.slnr_cfg.get("n_surf_samples", max(config.surface_sample_n * 4, 8))))
        self.s_dev = float(self.slnr_cfg.get("s_dev", config.surface_sample_range_m))
        self.step_size_sdf = float(
            self.slnr_cfg.get("step_size_sdf", max(self.local_sdf_resolution * 0.25, 0.02))
        )
        self.trunc_distance = float(self.slnr_cfg.get("trunc_distance", max(config.surface_sample_range_m * 2.0, 0.1)))
        self.trunc_weight = float(self.slnr_cfg.get("trunc_weight", 1.0))

        self.warmup_raw_frames: List[WarmupFrameSLNR] = []
        self.warmup_processed_frames: List[ProcessedFrameSLNR] = []
        self.recent_frames: Deque[TrainingFrameSLNR] = deque(maxlen=self.recent_buffer_size)
        self.replay_frames: Deque[TrainingFrameSLNR] = deque(maxlen=self.replay_buffer_size)
        self.initialized = False
        self.integrated_frame_count = 0
        self.processed_frame_count = 0
        self.train_iter = 0
        self.warned_densify_skip = False

        self.svh = None
        self.local_sdfs: Optional[LocalSDF] = None
        self.neural_map: Optional[NeuralMap] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ht_info_device = None
        self.ht_info_cpu = None
        self.bounds_extents = None
        self.inv_bounds_transform = None
        self.anchor_voxel_keys = set()
        self.anchor_frame_ids = torch.empty((0,), device=self.device, dtype=torch.int)

    def is_empty(self) -> bool:
        return (self.local_sdfs is None) or self.local_sdfs.positions.shape[0] == 0

    def ingest_frame(
        self,
        points_local: np.ndarray,
        pose_wc: np.ndarray,
        frame_id: int,
        normals_local: Optional[np.ndarray] = None,
    ) -> Optional[ProcessedFrameSLNR]:
        raw_frame = WarmupFrameSLNR(
            points_local=points_local.astype(np.float32),
            normals_local=None if normals_local is None else normals_local.astype(np.float32),
            pose_wc=pose_wc.astype(np.float32),
            frame_id=frame_id,
        )
        processed = self.prepare_processed_frame(raw_frame)
        if processed is None:
            return None

        if not self.initialized:
            self.warmup_raw_frames.append(raw_frame)
            self.warmup_processed_frames.append(processed)
            if len(self.warmup_raw_frames) >= self.warmup_frames:
                self.initialize_from_warmup()
        else:
            self.integrate_processed_frame(processed)

        return processed

    def prepare_processed_frame(self, frame: WarmupFrameSLNR) -> Optional[ProcessedFrameSLNR]:
        if frame.points_local.shape[0] < self.min_points_per_frame:
            return None

        points_world = transform_points(frame.points_local, frame.pose_wc)
        if frame.normals_local is not None:
            normals_world = transform_normals(frame.normals_local, frame.pose_wc)
        else:
            normals_world = estimate_world_normals(points_world, frame.pose_wc)

        insert_points_world, insert_normals_world = voxel_select_first(
            points_world,
            normals_world,
            self.hash_insert_down_voxel_size,
        )
        train_points_world = sample_fixed_points(points_world, self.train_points_per_frame, self.rng)

        if train_points_world.shape[0] < self.min_points_per_frame:
            return None

        return ProcessedFrameSLNR(
            frame_id=frame.frame_id,
            pose_wc=frame.pose_wc.astype(np.float32),
            train_points_world=train_points_world,
            insert_points_world=insert_points_world,
            insert_normals_world=insert_normals_world,
        )

    def initialize_from_warmup(self) -> None:
        frame_dataset = InMemoryFrameDatasetSLNR(self.warmup_raw_frames)
        frame_indices = np.arange(len(frame_dataset), dtype=np.int64)

        self.svh = torch.classes.svh.HashTable(self.hash_voxel_size, self.ht_size)
        self.local_sdfs = LocalSDF(
            resolution=self.local_sdf_resolution,
            query_nn_k=self.query_nn_k,
            gss_vox_size=self.gss_vox_size,
        )
        main_util.allocate_localsdfs_in_svh(
            self.svh,
            frame_dataset,
            frame_indices,
            self.local_sdfs,
            res_scale=self.res_scale,
            down_voxel_size=self.down_vox_size,
        )
        self.refresh_hash_state()

        self.neural_map = NeuralMap(
            self.local_sdfs,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        params = [
            {"params": self.neural_map.sdf_net.parameters(), "name": "sdf_net"},
            {"params": [self.local_sdfs.positions], "name": "positions"},
            {"params": [self.local_sdfs.rotations], "name": "rotations"},
            {"params": [self.local_sdfs.scalings], "name": "scalings"},
        ]
        self.optimizer = optim.Adam(params, lr=self.train_lr)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda iteration: self.lr_ratio ** min(iteration / self.lr_decay_iters, 1.0),
        )

        base_frame_id = 0 if not self.warmup_processed_frames else int(self.warmup_processed_frames[0].frame_id)
        anchor_count = int(self.local_sdfs.positions.shape[0])
        self.anchor_frame_ids = torch.full((anchor_count,), base_frame_id, device=self.device, dtype=torch.int)
        self.rebuild_anchor_voxel_keys()

        for processed in self.warmup_processed_frames:
            self.processed_frame_count += 1
            self.add_training_frame(processed)

        self.initialized = True

    def integrate_processed_frame(self, processed: ProcessedFrameSLNR) -> None:
        self.processed_frame_count += 1
        self.integrated_frame_count += 1
        self.add_training_frame(processed)

        if processed.insert_points_world.shape[0] == 0:
            return

        insert_tensor = torch.from_numpy(processed.insert_points_world).float()
        self.svh.insert(insert_tensor)

        new_anchor_count = self.append_new_anchors(
            processed.insert_points_world,
            processed.insert_normals_world,
            processed.frame_id,
        )
        if self.integrated_frame_count % self.refresh_map_every_n_frames == 0 or new_anchor_count > 0:
            self.refresh_hash_state()

    def add_training_frame(self, processed: ProcessedFrameSLNR) -> None:
        training_frame = TrainingFrameSLNR(
            frame_id=processed.frame_id,
            points_world=processed.train_points_world.astype(np.float32),
            pose_wc=processed.pose_wc.astype(np.float32),
        )
        self.recent_frames.append(training_frame)
        if self.processed_frame_count % self.replay_insert_interval == 0 or not self.replay_frames:
            self.replay_frames.append(training_frame)

    def append_new_anchors(
        self,
        points_world: np.ndarray,
        normals_world: np.ndarray,
        frame_id: int,
    ) -> int:
        if self.local_sdfs is None or self.optimizer is None or points_world.shape[0] == 0:
            return 0

        candidate_points, candidate_normals = voxel_select_first(
            points_world,
            normals_world,
            float(self.local_sdfs.resolution),
        )
        if candidate_points.shape[0] == 0:
            return 0

        anchor_grid = np.floor(candidate_points / float(self.local_sdfs.resolution)).astype(np.int64)
        keep_indices = []
        keep_keys = []
        for idx, key in enumerate(anchor_grid):
            key_tuple = (int(key[0]), int(key[1]), int(key[2]))
            if key_tuple in self.anchor_voxel_keys:
                continue
            keep_indices.append(idx)
            keep_keys.append(key_tuple)

        if not keep_indices:
            return 0

        if 0 < self.max_new_anchors_per_frame < len(keep_indices):
            selected = self.rng.choice(len(keep_indices), size=self.max_new_anchors_per_frame, replace=False)
            keep_indices = [keep_indices[int(i)] for i in selected]
            keep_keys = [keep_keys[int(i)] for i in selected]

        new_points = torch.from_numpy(candidate_points[keep_indices]).to(self.device).float()
        new_normals = torch.from_numpy(candidate_normals[keep_indices]).to(self.device).float()
        new_rotations = normals_to_rotvec(new_normals)
        new_scalings = torch.ones((new_points.shape[0], 3), device=self.device) * math.log(self.local_sdfs.resolution)

        self.local_sdfs.densification_postfix(new_points, new_rotations, new_scalings, self.optimizer)
        new_timestamps = torch.full((new_points.shape[0],), int(frame_id), device=self.device, dtype=torch.int)
        self.anchor_frame_ids = torch.cat((self.anchor_frame_ids, new_timestamps), dim=0)
        self.anchor_voxel_keys.update(keep_keys)
        return int(new_points.shape[0])

    def rebuild_anchor_voxel_keys(self) -> None:
        if self.local_sdfs is None or self.local_sdfs.positions.shape[0] == 0:
            self.anchor_voxel_keys = set()
            return
        positions = self.local_sdfs.positions.detach().cpu().numpy()
        anchor_grid = np.floor(positions / float(self.local_sdfs.resolution)).astype(np.int64)
        self.anchor_voxel_keys = {
            (int(key[0]), int(key[1]), int(key[2]))
            for key in anchor_grid
        }

    def refresh_hash_state(self) -> None:
        if self.svh is None:
            return

        ht_info, _, _ = self.svh.get_ht_info()
        self.ht_info_cpu = ht_info
        self.ht_info_device = ht_info.to(self.device)

        vox_coords = ht_info[:, :3]
        valid_mask = vox_coords[:, 0] != self.inval_val
        valid_voxels = vox_coords[valid_mask]
        vox_world = ((valid_voxels + 0.5) * self.hash_voxel_size).cpu().numpy().astype(np.float32)

        bounds_extents_np, inv_bounds_np = compute_bounds_from_voxels(vox_world)
        bounds_extents_np = bounds_extents_np * 1.1
        self.bounds_extents = torch.from_numpy(bounds_extents_np).float().to(self.device)
        self.inv_bounds_transform = torch.from_numpy(inv_bounds_np).float().to(self.device)

    def rebuild_svh_from_anchors(self) -> None:
        self.svh = torch.classes.svh.HashTable(self.hash_voxel_size, self.ht_size)
        rebuilt = False

        # Preserve the SLNR sparse-hash semantics when the hash needs to be
        # recreated inside the PIN-SLAM control flow. The original SLNR `ht_info`
        # represents occupied support voxels, not anchor-center voxels.
        if self.ht_info_cpu is not None:
            vox_coords = self.ht_info_cpu[:, :3]
            valid_mask = vox_coords[:, 0] != self.inval_val
            valid_voxels = vox_coords[valid_mask]
            if valid_voxels.numel() > 0:
                vox_centers = (valid_voxels + 0.5) * self.hash_voxel_size
                self.svh.insert(vox_centers.float())
                rebuilt = True

        if (not rebuilt) and self.local_sdfs is not None and self.local_sdfs.positions.shape[0] > 0:
            # Fallback for paths where no SLNR hash state has been initialized yet.
            self.svh.insert(self.local_sdfs.positions.detach().cpu().float())
        self.refresh_hash_state()

    def sample_training_frames(self) -> List[TrainingFrameSLNR]:
        if len(self.recent_frames) == 0 and len(self.replay_frames) == 0:
            return []

        n_total = self.train_frames_per_step
        n_recent = min(len(self.recent_frames), max(1, int(round(n_total * self.train_recent_ratio))))
        n_replay = max(0, n_total - n_recent)

        frames = sample_items(list(self.recent_frames), n_recent, self.rng)
        frames.extend(sample_items(list(self.replay_frames), n_replay, self.rng))
        if len(frames) < n_total:
            frames.extend(sample_items(list(self.recent_frames), n_total - len(frames), self.rng))
        return frames[:n_total]

    def reset_runtime_buffers(self) -> None:
        self.recent_frames.clear()
        self.replay_frames.clear()
        if not self.initialized:
            self.warmup_raw_frames.clear()
            self.warmup_processed_frames.clear()

    def optimize_step(self) -> None:
        if not self.initialized or self.neural_map is None or self.optimizer is None:
            return
        if self.ht_info_device is None or self.bounds_extents is None or self.inv_bounds_transform is None:
            return

        frames = self.sample_training_frames()
        if len(frames) == 0:
            return

        pc_batch = torch.from_numpy(np.stack([frame.points_world for frame in frames], axis=0)).float().to(self.device)
        T_batch = torch.from_numpy(np.stack([frame.pose_wc for frame in frames], axis=0)).float().to(self.device)

        sample_pts = main_util.sample_points_svh(
            self.svh,
            self.ht_info_device,
            pc_batch,
            T_batch,
            self.bounds_extents,
            self.inv_bounds_transform,
            n_rays=self.n_rays,
            n_max=self.n_max_interset,
            sur_behind_dis=self.sur_behind_dis,
            n_surf_samples=self.n_surf_samples,
            s_dev=self.s_dev,
            step_size_sdf=self.step_size_sdf,
            device=self.device,
        )
        if sample_pts["pc_sdf"].numel() == 0 or sample_pts["sample_mask_sdf"].sum().item() == 0:
            return

        self.optimizer.zero_grad(set_to_none=True)
        loss = main_util.compute_loss(
            self.neural_map,
            sample_pts,
            iter=self.train_iter,
            trunc_distance=self.trunc_distance,
            trunc_weight=self.trunc_weight,
            add_scale_loss=self.add_scale_loss,
        )
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            return

        loss.backward()

        if self.train_iter > self.freeze_after_iters:
            main_util.freeze_model(self.neural_map.sdf_net)

        if self.enable_densify and not self.warned_densify_skip and not self.config.silence:
            print("SLNR densify/prune is disabled in the integration path for now.")
            self.warned_densify_skip = True

        self.optimizer.step()
        self.lr_scheduler.step()
        self.train_iter += 1

    def query_sdf(self, query_points: torch.Tensor):
        if self.neural_map is None:
            empty = torch.zeros(query_points.shape[0], device=query_points.device, dtype=query_points.dtype)
            count = torch.zeros(query_points.shape[0], device=query_points.device, dtype=torch.long)
            return empty, count
        return self.neural_map.query_sdf(query_points)

    def transform_training_frames(self, pose_diff_torch: torch.Tensor) -> None:
        pose_diff_np = pose_diff_torch.detach().cpu().numpy().astype(np.float32)

        def _transform_frame(frame: TrainingFrameSLNR) -> TrainingFrameSLNR:
            frame_id = min(frame.frame_id, pose_diff_np.shape[0] - 1)
            pose_diff = pose_diff_np[frame_id]
            return TrainingFrameSLNR(
                frame_id=frame.frame_id,
                points_world=_apply_pose_to_numpy_points(frame.points_world, pose_diff),
                pose_wc=_apply_pose_to_numpy_matrix(frame.pose_wc, pose_diff),
            )

        self.recent_frames = deque((_transform_frame(frame) for frame in self.recent_frames), maxlen=self.recent_buffer_size)
        self.replay_frames = deque((_transform_frame(frame) for frame in self.replay_frames), maxlen=self.replay_buffer_size)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import torch


@dataclass
class WarmupFrameSLNR:
    points_local: np.ndarray
    normals_local: Optional[np.ndarray]
    pose_wc: np.ndarray
    frame_id: int


@dataclass
class ProcessedFrameSLNR:
    frame_id: int
    pose_wc: np.ndarray
    train_points_world: np.ndarray
    insert_points_world: np.ndarray
    insert_normals_world: np.ndarray


@dataclass
class TrainingFrameSLNR:
    frame_id: int
    points_world: np.ndarray
    pose_wc: np.ndarray


class InMemoryFrameDatasetSLNR:
    def __init__(self, frames: Sequence[WarmupFrameSLNR]):
        self.frames = list(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame = self.frames[idx]
        normals = None if frame.normals_local is None else frame.normals_local.copy()
        return frame.points_local.copy(), normals, frame.pose_wc.copy()


def transform_points(points_local: np.ndarray, pose_wc: np.ndarray) -> np.ndarray:
    rotation = pose_wc[:3, :3]
    translation = pose_wc[:3, 3]
    return (points_local @ rotation.T + translation[None, :]).astype(np.float32)


def transform_normals(normals_local: np.ndarray, pose_wc: np.ndarray) -> np.ndarray:
    rotation = pose_wc[:3, :3]
    normals_world = normals_local @ rotation.T
    norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
    return (normals_world / (norms + 1e-5)).astype(np.float32)


def estimate_world_normals(points_world: np.ndarray, pose_wc: np.ndarray) -> np.ndarray:
    if points_world.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
    point_cloud.estimate_normals()
    normals_world = np.asarray(point_cloud.normals, dtype=np.float32)

    rays_o = pose_wc[:3, 3]
    rays_d = rays_o[None, :] - points_world
    rays_d[:, 2] += 3.0
    ranges = np.linalg.norm(rays_d, axis=1, keepdims=True)
    rays_d = rays_d / (ranges + 1e-5)
    dd = (normals_world * rays_d).sum(axis=-1)
    normals_world[dd < 0.0] *= -1.0

    norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
    normals_world = normals_world / (norms + 1e-5)
    return normals_world.astype(np.float32)


def voxel_select_first(
    points: np.ndarray,
    normals: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or points.shape[0] <= 1:
        return points.astype(np.float32), normals.astype(np.float32)

    grid = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(grid, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points[unique_indices].astype(np.float32), normals[unique_indices].astype(np.float32)


def sample_fixed_points(points: np.ndarray, target_count: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if target_count <= 0:
        return points.astype(np.float32)
    if points.shape[0] >= target_count:
        indices = rng.choice(points.shape[0], size=target_count, replace=False)
    else:
        indices = rng.choice(points.shape[0], size=target_count, replace=True)
    return points[indices].astype(np.float32)


def normals_to_rotvec(normals: torch.Tensor) -> torch.Tensor:
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-5)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype).view(1, 3)
    z_axis = z_axis.expand_as(normals)

    dots = torch.clamp((z_axis * normals).sum(dim=-1), -1.0, 1.0)
    angles = torch.arccos(dots)
    axis = torch.cross(z_axis, normals, dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    rotvec = torch.zeros_like(normals)
    regular_mask = axis_norm.squeeze(-1) > 1e-6
    if regular_mask.any():
        axis_regular = axis[regular_mask] / axis_norm[regular_mask]
        rotvec[regular_mask] = axis_regular * angles[regular_mask].unsqueeze(-1)

    opposite_mask = (~regular_mask) & (dots < 0.0)
    if opposite_mask.any():
        rotvec[opposite_mask, 0] = math.pi

    return rotvec


def compute_bounds_from_voxels(vox_coords_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if vox_coords_world.shape[0] == 0:
        return np.ones(3, dtype=np.float32), np.eye(4, dtype=np.float32)

    coords_min = vox_coords_world.min(axis=0)
    coords_max = vox_coords_world.max(axis=0)
    fallback_extents = np.maximum(coords_max - coords_min, 1e-3).astype(np.float32)
    fallback_center = ((coords_min + coords_max) * 0.5).astype(np.float32)
    fallback_inv = np.eye(4, dtype=np.float32)
    fallback_inv[:3, 3] = -fallback_center

    if vox_coords_world.shape[0] < 4:
        return fallback_extents, fallback_inv

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vox_coords_world.astype(np.float64))

    try:
        obb = point_cloud.get_oriented_bounding_box()
    except RuntimeError:
        return fallback_extents, fallback_inv

    bounds_extents = np.asarray(obb.extent, dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(obb.R, dtype=np.float32)
    transform[:3, 3] = np.asarray(obb.center, dtype=np.float32)
    inv_transform = np.linalg.inv(transform).astype(np.float32)
    return bounds_extents, inv_transform


def sample_items(
    items: Sequence[TrainingFrameSLNR],
    count: int,
    rng: np.random.Generator,
) -> List[TrainingFrameSLNR]:
    if count <= 0 or len(items) == 0:
        return []
    if len(items) >= count:
        indices = rng.choice(len(items), size=count, replace=False)
    else:
        indices = rng.choice(len(items), size=count, replace=True)
    return [items[int(idx)] for idx in indices]

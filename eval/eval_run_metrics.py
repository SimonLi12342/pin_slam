#!/usr/bin/env python3
# @file      eval_run_metrics.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_mesh_utils import eval_mesh


def _to_builtin(value: Any):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_dynamic_summary(summary_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not summary_path.is_file():
        return summary
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip()
        if value.lower() in {"true", "false"}:
            summary[key.strip()] = value.lower() == "true"
            continue
        try:
            if "." in value:
                summary[key.strip()] = float(value)
            else:
                summary[key.strip()] = int(value)
        except ValueError:
            summary[key.strip()] = value
    return summary


def _find_latest_mesh(run_dir: Path) -> Path | None:
    mesh_dir = run_dir / "mesh"
    if not mesh_dir.is_dir():
        return None
    candidates = sorted(mesh_dir.glob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _read_mesh_stats(mesh_path: Path) -> dict[str, Any]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    return {
        "mesh_path": str(mesh_path),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
        "mesh_file_size_mb": float(mesh_path.stat().st_size) / (1024.0 * 1024.0),
        "mesh_is_empty": bool(mesh.is_empty()),
    }


def _read_point_cloud_stats(pcd_path: Path) -> dict[str, Any]:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    return {
        "point_cloud_path": str(pcd_path),
        "point_count": int(len(pcd.points)),
        "point_cloud_file_size_mb": float(pcd_path.stat().st_size) / (1024.0 * 1024.0),
        "point_cloud_is_empty": bool(pcd.is_empty()),
    }


def _read_timing_stats(run_dir: Path) -> dict[str, Any]:
    time_table_path = run_dir / "time_table.npy"
    if not time_table_path.is_file():
        return {}
    time_table = np.load(time_table_path)
    if time_table.ndim != 2 or time_table.shape[0] == 0:
        return {}

    total_per_frame = np.sum(time_table, axis=1)
    stats = {
        "time_table_path": str(time_table_path),
        "frame_count": int(time_table.shape[0]),
        "mean_total_time_s": float(np.mean(total_per_frame)),
        "median_total_time_s": float(np.median(total_per_frame)),
        "mean_total_fps": float(1.0 / (np.mean(total_per_frame) + 1e-12)),
        "max_total_time_s": float(np.max(total_per_frame)),
    }
    if time_table.shape[0] > 1:
        total_wo_init = total_per_frame[1:]
        stats["mean_total_time_wo_init_s"] = float(np.mean(total_wo_init))
        stats["mean_total_fps_wo_init"] = float(1.0 / (np.mean(total_wo_init) + 1e-12))

    stage_names = [
        "preprocess_s",
        "tracking_s",
        "mapping_prepare_s",
        "mapping_s",
        "pgo_s",
    ]
    for idx, stage_name in enumerate(stage_names):
        if idx < time_table.shape[1]:
            stats[f"mean_{stage_name}"] = float(np.mean(time_table[:, idx]))
    return stats


def _read_memory_stats(run_dir: Path) -> dict[str, Any]:
    mem_path = run_dir / "memory_footprint.npy"
    if not mem_path.is_file():
        return {}
    mem = np.load(mem_path)
    if mem.size == 0:
        return {}
    return {
        "memory_footprint_path": str(mem_path),
        "memory_samples": int(mem.size),
        "memory_mean_mb": float(np.mean(mem)),
        "memory_max_mb": float(np.max(mem)),
        "memory_final_mb": float(mem[-1]),
    }


def evaluate_run(
    run_dir: Path,
    gt_pointcloud: Path | None,
    down_sample_res: float,
    threshold: float,
    truncation_acc: float,
    truncation_com: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
    }

    summary["timing"] = _read_timing_stats(run_dir)
    summary["memory"] = _read_memory_stats(run_dir)

    map_pcd_path = run_dir / "map" / "neural_points.ply"
    if map_pcd_path.is_file():
        summary["map"] = _read_point_cloud_stats(map_pcd_path)

    dynamic_summary_path = run_dir / "meta" / "dynamic_filter_summary.txt"
    dynamic_summary = _load_dynamic_summary(dynamic_summary_path)
    if dynamic_summary:
        summary["dynamic"] = dynamic_summary

    mesh_path = _find_latest_mesh(run_dir)
    if mesh_path is not None:
        summary["mesh"] = _read_mesh_stats(mesh_path)

    if gt_pointcloud is not None:
        if mesh_path is None:
            summary["reconstruction_eval"] = {
                "skipped": True,
                "reason": "No mesh found under run_dir/mesh.",
            }
        elif not gt_pointcloud.is_file():
            summary["reconstruction_eval"] = {
                "skipped": True,
                "reason": f"Ground-truth point cloud not found: {gt_pointcloud}",
            }
        else:
            summary["reconstruction_eval"] = _to_builtin(
                eval_mesh(
                    str(mesh_path),
                    str(gt_pointcloud),
                    down_sample_res=down_sample_res,
                    threshold=threshold,
                    truncation_acc=truncation_acc,
                    truncation_com=truncation_com,
                )
            )
            summary["reconstruction_eval"]["pred_mesh_path"] = str(mesh_path)
            summary["reconstruction_eval"]["gt_pointcloud_path"] = str(gt_pointcloud)
    else:
        summary["reconstruction_eval"] = {
            "skipped": True,
            "reason": "No ground-truth point cloud was provided.",
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a PIN-SLAM run folder and save metrics to run_dir/evals."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a run result directory, e.g. experiments/test_pin_...",
    )
    parser.add_argument(
        "--gt-pointcloud",
        type=Path,
        default=None,
        help="Optional path to a ground-truth point cloud for reconstruction metrics.",
    )
    parser.add_argument("--down-sample-res", type=float, default=0.02)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--truncation-acc", type=float, default=0.50)
    parser.add_argument("--truncation-com", type=float, default=0.50)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    gt_pointcloud = None if args.gt_pointcloud is None else args.gt_pointcloud.expanduser().resolve()
    summary = evaluate_run(
        run_dir=run_dir,
        gt_pointcloud=gt_pointcloud,
        down_sample_res=args.down_sample_res,
        threshold=args.threshold,
        truncation_acc=args.truncation_acc,
        truncation_com=args.truncation_com,
    )

    eval_dir = run_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "summary.json"
    summary_path.write_text(json.dumps(_to_builtin(summary), indent=2), encoding="utf-8")

    print(f"Saved evaluation summary to {summary_path}")
    print(json.dumps(_to_builtin(summary), indent=2))


if __name__ == "__main__":
    main()

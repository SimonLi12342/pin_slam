#!/usr/bin/env python3
# @file      pin_slam_slnr.py

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import dtyper as typer
import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
import wandb
import yaml
from rich import print
from tqdm import tqdm

from dataset.dataset_indexing import set_dataset_path
from dataset.dataloaders import available_dataloaders
from dataset.slam_dataset import SLAMDataset
from gui import slam_gui_slnr
from gui.gui_utils_slnr import ControlPacket, ParamsGUI, VisPacket, get_latest_queue
from model.decoder import Decoder
from model.decoder_slnr import DecoderSLNR
from model.neural_points import NeuralPoints
from model.neural_points_slnr import NeuralPointsSLNR
from slnr import main_util_slnr
from utils.config import Config
from utils.loop_detector import NeuralPointMapContextManager, detect_local_loop
from utils.mapper import Mapper as TrackingMapper
from utils.mapper_slnr import Mapper
from utils.mesher import Mesher
from utils.pgo import PoseGraphManager
from utils.tools import (
    create_bbx_o3d,
    freeze_decoders,
    get_gpu_memory_usage_gb,
    get_time,
    load_decoders,
    remove_gpu_cache,
    setup_experiment,
    split_chunks,
    transform_torch,
    unfreeze_decoders,
)
from utils.tracker import Tracker


app = typer.Typer(add_completion=False, rich_markup_mode="rich", context_settings={"help_option_names": ["-h", "--help"]})
_available_dl_help = available_dataloaders()

docstring = f"""
:round_pushpin: PIN-SLAM with SLNR local-SDF mapping backend \n

[bold green]Examples: [/bold green]

$ python3 pin_slam_slnr.py -i <data-dir> -vsm
$ python3 pin_slam_slnr.py ./config/lidar_slam/run_slnr.yaml rosbag -i <path-to-my-rosbag> -dvsm

# Use a specific dataloader: {", ".join(_available_dl_help)}
"""


def _load_slnr_section(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        raw_cfg = yaml.safe_load(file) or {}
    return raw_cfg.get("slnr", {})


def _enforce_slnr_constraints(config: Config):
    warnings = []
    if config.device == "cpu":
        raise RuntimeError("The SLNR integration path requires CUDA because the copied SLNR sparse-hash and gaussian-search modules are CUDA-only.")
    if config.weighted_first:
        config.weighted_first = False
        warnings.append("`weighted_first` was forced to `False` for the SLNR mapping backend.")
    if config.semantic_on:
        config.semantic_on = False
        warnings.append("Semantic mapping is disabled in `pin_slam_slnr.py`.")
    if config.color_on:
        config.color_on = False
        config.color_map_on = False
        config.color_channel = 0
        warnings.append("Color/intensity mapping is disabled in `pin_slam_slnr.py`.")
    if config.loop_with_feature:
        config.loop_with_feature = False
        warnings.append("Feature-based loop context is disabled in `pin_slam_slnr.py`.")
    if config.pgo_on:
        config.pgo_on = False
        warnings.append(
            "PGO / loop-closure is disabled in `pin_slam_slnr.py` because the "
            "current integration only replaces the mapping representation."
        )
    if config.ba_freq_frame > 0:
        config.ba_freq_frame = 0
        warnings.append(
            "Bundle adjustment is disabled in `pin_slam_slnr.py` because the "
            "SLNR mapping path does not implement native BA."
        )
    if config.mesh_min_nn > config.query_nn_k:
        config.mesh_min_nn = config.query_nn_k
        warnings.append(
            "`mesh_min_nn` was clamped to `query_nn_k` for the SLNR backend; "
            "otherwise marching-cubes masking would reject all query points."
        )
    for warning in warnings:
        print(f"[bold yellow]{warning}[/bold yellow]")


def _configure_slnr_tracking_compat(config: Config):
    warnings = []
    if not config.track_on:
        return

    if config.track_mask_query_nn_k > 2:
        config.track_mask_query_nn_k = 2
        warnings.append(
            "`valid_nn_k` was relaxed to 2 for the SLNR backend because the "
            "support-point map is sparser than native PIN-SLAM neural points."
        )
    if config.reg_min_grad_norm > 0.05:
        config.reg_min_grad_norm = 0.05
        warnings.append(
            "`min_grad_norm` was relaxed to 0.05 for the SLNR backend."
        )
    if config.reg_max_grad_norm < 50.0:
        config.reg_max_grad_norm = 50.0
        warnings.append(
            "`max_grad_norm` was increased to 50.0 for the SLNR backend."
        )
    if config.reg_lm_lambda < 1e-2:
        config.reg_lm_lambda = 1e-2
        warnings.append(
            "`lm_lambda` was increased to 0.01 for the SLNR backend."
        )
    if config.reg_iter_n > 5:
        config.reg_iter_n = 5
        warnings.append(
            "`iter_n` was reduced to 5 for the SLNR backend to avoid unstable "
            "over-optimization during scan registration."
        )
    if config.final_residual_ratio_thre < 1.0:
        config.final_residual_ratio_thre = 1.0
        warnings.append(
            "`final_residual_ratio_thre` was relaxed to 1.0 for the SLNR backend."
        )
    if config.eigenvalue_check:
        config.eigenvalue_check = False
        warnings.append(
            "Eigenvalue degeneracy check was disabled for the SLNR backend "
            "because the sparse support-point map triggers false odometry rejections."
        )
    if not config.reg_dist_div_grad_norm:
        config.reg_dist_div_grad_norm = True
        warnings.append(
            "`reg_dist_div_grad_norm` was enabled for the SLNR backend."
        )

    for warning in warnings:
        print(f"[bold yellow]{warning}[/bold yellow]")


def _allow_slnr_mapping_frame(frame_id: int, valid_mapping_flag: bool, system_rebooted: bool) -> bool:
    if system_rebooted:
        return True
    if frame_id == 0:
        return True
    return valid_mapping_flag


def _allow_slnr_mapping_by_tracking_quality(tracker: Tracker) -> bool:
    if not tracker.last_valid_flag:
        return False

    valid_points = int(tracker.last_valid_point_count)
    source_points = max(1, int(tracker.last_source_point_count))
    valid_ratio = float(valid_points) / float(source_points)
    residual_cm = float(tracker.last_sdf_residual_cm or 1e9)

    if valid_points < 700:
        return False
    if valid_ratio < 0.4:
        return False
    if residual_cm > 7.5:
        return False
    return True


def _mesh_stats(mesh) -> tuple[int, int]:
    if mesh is None:
        return 0, 0
    return len(mesh.vertices), len(mesh.triangles)


def _ensure_run_output_dirs(run_path: str) -> None:
    access = 0o755
    os.makedirs(run_path, access, exist_ok=True)
    for name in ("mesh", "map", "model", "log", "meta"):
        os.makedirs(os.path.join(run_path, name), access, exist_ok=True)


def _save_slnr_outputs(
    run_path: str,
    neural_points: NeuralPointsSLNR,
    config: Config,
    slnr_cfg: dict,
):
    backend = neural_points.backend
    if backend.neural_map is None or backend.local_sdfs is None:
        print(
            "[bold yellow]Skipping SLNR model/mesh export because the backend "
            "map was never initialized.[/bold yellow]"
        )
        return None, None, None

    train_iter = int(getattr(backend, "train_iter", 0))
    model_path = None
    mesh = None
    mesh_path = None

    if config.save_map:
        model_path = os.path.join(run_path, "model", f"model_{train_iter}.pth")
        print(f"Saving SLNR model checkpoint to {model_path}")
        torch.save(
            {
                "net_params": backend.neural_map.state_dict(),
                "n_iter": train_iter,
            },
            model_path,
        )

    if config.save_mesh:
        if backend.ht_info_device is None:
            print(
                "[bold yellow]Skipping SLNR mesh export because sparse-hash "
                "state is unavailable.[/bold yellow]"
            )
        else:
            grid_res = max(2, int(slnr_cfg.get("save_grid_res", 5)))
            chunk_size = max(1, int(slnr_cfg.get("save_mesh_chunk_size", 256)))
            mesh_min_nn = max(
                1,
                int(
                    slnr_cfg.get(
                        "save_mesh_min_nn",
                        min(int(config.mesh_min_nn), int(backend.query_nn_k)),
                    )
                ),
            )
            mesh_path = os.path.join(run_path, "mesh", f"mesh_{train_iter}.ply")
            print(
                "Reconstructing the SLNR mesh "
                f"(grid_res={grid_res}, mesh_min_nn={mesh_min_nn})"
            )
            try:
                mesh = main_util_slnr.create_mesh_svh(
                    backend.ht_info_device,
                    backend.hash_voxel_size,
                    backend.neural_map,
                    grid_res=grid_res,
                    chunk_size=chunk_size,
                    mesh_min_nn=mesh_min_nn,
                    save_path=mesh_path,
                    device=backend.device,
                )
            except Exception as exc:
                print(
                    "[bold yellow]SLNR mesh export failed: "
                    f"{exc}[/bold yellow]"
                )
                mesh = None
                mesh_path = None

    return model_path, mesh, mesh_path


class SLNRTracker(Tracker):
    def __init__(self, config: Config, neural_points, decoders: dict):
        super().__init__(config, neural_points, decoders)
        self.last_sdf_residual_cm = None
        self.last_valid_point_count = 0
        self.last_source_point_count = 0
        self.last_valid_flag = False

    def tracking(self, source_points, *args, **kwargs):
        self.last_source_point_count = int(source_points.shape[0])
        result = super().tracking(source_points, *args, **kwargs)
        self.last_valid_flag = bool(result[3])
        return result

    def registration_step(self, *args, **kwargs):
        result = super().registration_step(*args, **kwargs)
        self.last_valid_point_count = int(result[4].shape[0])
        self.last_sdf_residual_cm = float(result[5])
        return result


@app.command(help=docstring)
def run_pin_slam_slnr(
    config_path: str = typer.Argument("config/lidar_slam/run_slnr.yaml", help="Path to *.yaml config file"),
    dataset_name: Optional[str] = typer.Argument(None, help="Name of a specific dataset"),
    sequence_name: Optional[str] = typer.Argument(None, help="Name of a specific sequence"),
    input_path: Optional[str] = typer.Option(None, "--input-path", "-i", help="Path to the point cloud input directory"),
    output_path: Optional[str] = typer.Option(None, "--output-path", "-o", help="Path to the result output directory"),
    frame_range: Optional[Tuple[int, int, int]] = typer.Option(None, "--range", help="Specify the start, end and step"),
    seed: int = typer.Option(42, help="Set the random seed"),
    data_loader_on: bool = typer.Option(False, "--data-loader-on", "-d", help="Use a specific data loader"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Turn on the visualizer"),
    cpu_only: bool = typer.Option(False, "--cpu-only", "-c", help="Run only on CPU"),
    log_on: bool = typer.Option(False, "--log-on", "-l", help="Turn on the logs printing"),
    wandb_on: bool = typer.Option(False, "--wandb-on", "-w", help="Turn on the weight & bias logging"),
    save_map: bool = typer.Option(False, "--save-map", "-s", help="Save the map after SLAM"),
    save_mesh: bool = typer.Option(False, "--save-mesh", "-m", help="Save the reconstructed mesh after SLAM"),
    save_merged_pc: bool = typer.Option(False, "--save-merged-pc", "-p", help="Save the merged point cloud after SLAM"),
    deskew: bool = typer.Option(False, "--deskew", help="Try to deskew the LiDAR scans"),
):
    config = Config()
    config.load(config_path)
    slnr_cfg = _load_slnr_section(config_path)

    config.use_dataloader = data_loader_on
    config.seed = seed
    config.silence = not log_on
    config.wandb_vis_on = wandb_on
    config.o3d_vis_on = visualize
    config.save_map = save_map
    config.save_mesh = save_mesh
    config.save_merged_pc = save_merged_pc

    if not config.deskew and deskew:
        config.deskew = True
    if frame_range:
        config.begin_frame, config.end_frame, config.step_frame = frame_range
    if cpu_only:
        config.device = "cpu"
    if input_path:
        config.pc_path = input_path
    if output_path:
        config.output_root = output_path
    if dataset_name:
        set_dataset_path(config, dataset_name, sequence_name)

    _enforce_slnr_constraints(config)

    argv = sys.argv
    run_path = setup_experiment(config, argv)
    print("[bold green]PIN-SLAM-SLNR starts[/bold green]")

    if config.o3d_vis_on:
        mp.set_start_method("spawn")

    neural_points = NeuralPointsSLNR(config, slnr_cfg=slnr_cfg)
    geo_mlp = DecoderSLNR(config, neural_points)
    mlp_dict = {"sdf": geo_mlp, "semantic": None, "color": None}

    mapping_on = True
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        neural_points = loaded_model["neural_points"]
        neural_points.rebind_config(config, slnr_cfg=slnr_cfg)
        geo_mlp = DecoderSLNR(config, neural_points)
        mlp_dict = {"sdf": geo_mlp, "semantic": None, "color": None}
        load_decoders(loaded_model, mlp_dict)
        config.decoder_freezed = True
        print("SLNR-backed map loaded")
        neural_points.recreate_hash(torch.zeros(3, device=config.device), None, True, False)
        neural_points.compute_feature_principle_components(down_rate=59)
        mapping_on = False
        neural_points.temporal_local_map_on = False
        config.pgo_on = False

    use_native_tracking_map = bool(config.track_on and mapping_on)
    if not use_native_tracking_map and config.track_on:
        _configure_slnr_tracking_compat(config)
    if config.dynamic_filter_on and not use_native_tracking_map:
        config.dynamic_filter_on = False
        print(
            "[bold yellow]Dynamic filtering was disabled because the current "
            "run does not have a native PIN-SLAM tracking map to provide the "
            "dynamic mask.[/bold yellow]"
        )
    elif config.dynamic_filter_on and not config.silence:
        print(
            "Dynamic filtering enabled: the native PIN-SLAM tracking map will "
            "estimate the static mask, and SLNR mapping will ingest only the "
            "masked points."
        )
    dynamic_filter_active = bool(config.dynamic_filter_on)

    lcd_npmc = NeuralPointMapContextManager(config)
    dataset = SLAMDataset(config)
    dataset.run_path = run_path

    dynamic_filter_frame_count = 0
    dynamic_filtered_points_total = 0
    dynamic_filtered_points_max = 0
    dynamic_prune_events = 0
    dynamic_pruned_anchor_total = 0

    tracking_neural_points = None
    tracking_mlp_dict = None
    tracking_mapper = None
    if use_native_tracking_map:
        tracking_geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
        tracking_mlp_dict = {"sdf": tracking_geo_mlp, "semantic": None, "color": None}
        tracking_neural_points = NeuralPoints(config)
        tracker = SLNRTracker(config, tracking_neural_points, tracking_mlp_dict)
        tracking_mapper = TrackingMapper(config, dataset, tracking_neural_points, tracking_mlp_dict)
        if not config.silence:
            print("Using native PIN-SLAM tracking map for odometry and SLNR map for reconstruction")
    else:
        tracker = SLNRTracker(config, neural_points, mlp_dict)
        if config.load_model and not mapping_on:
            tracker.reg_local_map = False

    mapper = Mapper(config, dataset, neural_points, mlp_dict)
    mesher = Mesher(config, neural_points, mlp_dict)
    cur_mesh = None

    pgm = PoseGraphManager(config)
    init_pose = dataset.gt_poses[0] if dataset.gt_pose_provided else np.eye(4)
    pgm.add_pose_prior(0, init_pose, fixed=True)

    last_frame = dataset.total_pc_count - 1
    loop_reg_failed_count = 0

    q_main2vis = q_vis2main = None
    if config.o3d_vis_on:
        q_main2vis = mp.Queue()
        q_vis2main = mp.Queue()
        params_gui = ParamsGUI(
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            config=config,
            local_map_default_on=config.local_map_default_on,
            mesh_default_on=config.mesh_default_on,
            sdf_default_on=config.sdf_default_on,
            neural_point_map_default_on=config.neural_point_map_default_on,
        )
        gui_process = mp.Process(target=slam_gui_slnr.run, args=(params_gui,))
        gui_process.start()
        time.sleep(3)

        vis_visualize_on = True
        vis_source_pc_weight = False
        vis_global_on = not config.local_map_default_on
        vis_mesh_on = config.mesh_default_on
        vis_mesh_freq_frame = config.mesh_freq_frame
        vis_mesh_mc_res_m = config.mc_res_m
        vis_mesh_min_nn = config.mesh_min_nn
        vis_sdf_on = config.sdf_default_on
        vis_sdf_freq_frame = config.sdfslice_freq_frame
        vis_sdf_slice_height = config.sdf_slice_height
        vis_sdf_res_m = config.vis_sdf_res_m

    cur_sdf_slice = None
    cur_fps = 0.0

    # for each frame
    # frame id as the processed frame, possible skipping done in data loader
    for frame_id in tqdm(range(dataset.total_pc_count)):
        remove_gpu_cache()

        # I. Load data and preprocessing
        T0 = get_time()

        if config.use_dataloader:
            dataset.read_frame_with_loader(frame_id)
        else:
            dataset.read_frame(frame_id)

        T1 = get_time()
        valid_frame = dataset.preprocess_frame()
        if not valid_frame:
            dataset.processed_frame += 1
            continue
        T2 = get_time()

        # II. Odometry
        if frame_id > 0:
            if config.track_on:
                tracking_result = tracker.tracking(
                    dataset.cur_source_points,
                    dataset.cur_pose_guess_torch,
                    dataset.cur_source_colors,
                    dataset.cur_source_normals,
                    vis_result=config.o3d_vis_on,
                )
                cur_pose_torch, cur_odom_cov, weight_pc_o3d, valid_flag = tracking_result
                dataset.lose_track = not valid_flag
                dataset.update_odom_pose(cur_pose_torch)
            else:
                if dataset.gt_pose_provided:
                    dataset.update_odom_pose(dataset.cur_pose_guess_torch)
                else:
                    sys.exit("You are using mapping mode, but no pose is provided.")
        else:
            cur_odom_cov = None
            weight_pc_o3d = None
            valid_flag = True

        travel_dist = dataset.travel_dist[: frame_id + 1]
        travel_dist_torch = torch.tensor(travel_dist, device=config.device, dtype=config.dtype)
        neural_points.travel_dist = travel_dist_torch
        if tracking_neural_points is not None:
            tracking_neural_points.travel_dist = travel_dist_torch
        valid_tracking_map_flag = (not dataset.lose_track) and (not dataset.stop_status)
        valid_mapping_flag = valid_tracking_map_flag
        if frame_id > 0 and config.track_on:
            valid_mapping_flag = valid_mapping_flag and _allow_slnr_mapping_by_tracking_quality(tracker)
        T3 = get_time()

        # III. Loop detection and pgo (now disabled)
        if config.pgo_on:
            if config.global_loop_on:
                if config.local_map_context and frame_id >= config.local_map_context_latency:
                    local_map_frame_id = frame_id - config.local_map_context_latency
                    local_map_pose = torch.tensor(dataset.pgo_poses[local_map_frame_id], device=config.device, dtype=torch.float64)
                    if config.local_map_context_latency > 0:
                        neural_points.reset_local_map(
                            local_map_pose[:3, 3],
                            None,
                            local_map_frame_id,
                            config.loop_local_map_by_travel_dist,
                            config.loop_local_map_time_window,
                        )
                    context_pc_local = transform_torch(neural_points.local_neural_points.detach(), torch.linalg.inv(local_map_pose))
                    lcd_npmc.add_node(local_map_frame_id, context_pc_local, None, valid_flag=valid_mapping_flag)
                else:
                    lcd_npmc.add_node(frame_id, dataset.cur_point_cloud_torch, valid_flag=valid_mapping_flag)
            pgm.add_frame_node(frame_id, dataset.pgo_poses[frame_id])
            pgm.init_poses = dataset.pgo_poses[: frame_id + 1]
            if frame_id > 0:
                cur_edge_cov = cur_odom_cov if config.use_reg_cov_mat else None
                pgm.add_odometry_factor(frame_id, frame_id - 1, dataset.last_odom_tran, cov=cur_edge_cov)
                pgm.estimate_drift(travel_dist, frame_id, correct_ratio=0.01)
                if config.pgo_with_pose_prior:
                    pgm.add_pose_prior(frame_id, dataset.pgo_poses[frame_id])
            local_map_context_loop = False
            if frame_id - pgm.last_loop_idx > config.pgo_freq and not dataset.stop_status:
                loop_candidate_mask = ((travel_dist[-1] - travel_dist) > (config.min_loop_travel_dist_ratio * config.local_map_radius))
                loop_id = None
                if np.any(loop_candidate_mask):
                    loop_id, loop_dist, loop_transform = detect_local_loop(
                        dataset.pgo_poses[: frame_id + 1],
                        loop_candidate_mask,
                        pgm.drift_radius,
                        frame_id,
                        loop_reg_failed_count,
                        config.local_loop_dist_thre,
                        config.local_loop_dist_thre * 3.0,
                        config.silence,
                    )
                    if loop_id is None and config.global_loop_on:
                        loop_id, loop_cos_dist, loop_transform, local_map_context_loop = lcd_npmc.detect_global_loop(
                            dataset.pgo_poses[: frame_id + 1],
                            pgm.drift_radius * config.loop_dist_drift_ratio_thre,
                            loop_candidate_mask,
                            neural_points,
                        )
                if loop_id is not None:
                    if config.loop_z_check_on and abs(loop_transform[2, 3]) > config.voxel_size_m * 4.0:
                        loop_id = None
                    if not lcd_npmc.valid_flags[loop_id]:
                        loop_id = None
                if loop_id is not None:
                    pose_init_torch = torch.tensor((dataset.pgo_poses[loop_id] @ loop_transform), device=config.device, dtype=torch.float64)
                    neural_points.recreate_hash(pose_init_torch[:3, 3], None, True, True, loop_id)
                    loop_reg_source_point = dataset.cur_source_points.clone()
                    pose_refine_torch, loop_cov_mat, weight_pcd, reg_valid_flag = tracker.tracking(loop_reg_source_point, pose_init_torch, loop_reg=True)
                    if reg_valid_flag:
                        pose_refine_np = pose_refine_torch.detach().cpu().numpy()
                        loop_transform = np.linalg.inv(dataset.pgo_poses[loop_id]) @ pose_refine_np
                        cur_edge_cov = loop_cov_mat if config.use_reg_cov_mat else None
                        reg_valid_flag = pgm.add_loop_factor(frame_id, loop_id, loop_transform, cov=cur_edge_cov)
                    if reg_valid_flag:
                        if not config.silence:
                            print("[bold green]Refine loop transformation succeed [/bold green]")
                        pgm.optimize_pose_graph()
                        cur_loop_vis_id = frame_id - config.local_map_context_latency if local_map_context_loop else frame_id
                        pgm.loop_edges_vis.append(np.array([loop_id, cur_loop_vis_id], dtype=np.uint32))
                        pgm.loop_edges.append(np.array([loop_id, frame_id], dtype=np.uint32))
                        pgm.loop_trans.append(loop_transform)
                        pose_diff_torch = torch.tensor(pgm.get_pose_diff(), device=config.device, dtype=config.dtype)
                        dataset.cur_pose_torch = torch.tensor(pgm.cur_pose, device=config.device, dtype=config.dtype)
                        neural_points.adjust_map(pose_diff_torch)
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3, 3], None, (not config.pgo_merge_map), config.rehash_with_time, frame_id)
                        mapper.transform_data_pool(pose_diff_torch)
                        dataset.update_poses_after_pgo(pgm.pgo_poses)
                        pgm.last_loop_idx = frame_id
                        pgm.min_loop_idx = min(pgm.min_loop_idx, loop_id)
                        loop_reg_failed_count = 0
                    else:
                        if not config.silence:
                            print("[bold red]Registration failed, reject the loop candidate [/bold red]")
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3, 3], None, True, True, frame_id)
                        loop_reg_failed_count += 1
        T4 = get_time()

        # check failure and reboot the system
        system_rebooted = False
        if dataset.consecutive_lose_track_frame >= config.reboot_frame_thre:
            if not config.silence:
                print("[bold red]Lose track for a long time, reboot the system[/bold red]")
            mapper.reset_runtime_state()
            neural_points.reboot_ts = frame_id
            if tracking_mapper is not None:
                tracking_mapper.init_pool()
                tracking_neural_points.reboot_ts = frame_id
            system_rebooted = True
            dataset.consecutive_lose_track_frame = 0
            unfreeze_decoders(mlp_dict, config)
            if tracking_mlp_dict is not None:
                unfreeze_decoders(tracking_mlp_dict, config)
            config.decoder_freezed = False

        # IV. Mapping and bundle adjustment
        # if lose track, we will not update the map and data pool
        # if the robot stops, also don't process this frame
        if mapping_on and tracking_mapper is not None:
            if frame_id < 5 or valid_tracking_map_flag or system_rebooted:
                tracking_mapper.process_frame(
                    dataset.cur_point_cloud_torch,
                    dataset.cur_sem_labels_torch,
                    dataset.cur_pose_torch,
                    frame_id,
                    (config.dynamic_filter_on and frame_id > 0),
                )
            else:
                tracking_mapper.determine_used_pose()
                tracking_neural_points.reset_local_map(
                    dataset.cur_pose_torch[:3, 3],
                    None,
                    frame_id,
                    reboot_map=True,
                )

        if mapping_on and _allow_slnr_mapping_frame(frame_id, valid_mapping_flag, system_rebooted):
            mapping_point_cloud_torch = dataset.cur_point_cloud_torch
            dynamic_points_world = None
            if (
                config.dynamic_filter_on
                and frame_id > 0
                and dataset.static_mask is not None
                and dataset.static_mask.shape[0] == dataset.cur_point_cloud_torch.shape[0]
            ):
                static_count = int(dataset.static_mask.sum().item())
                filtered_count = int(dataset.static_mask.shape[0] - static_count)
                dynamic_filter_frame_count += 1
                dynamic_filtered_points_total += filtered_count
                dynamic_filtered_points_max = max(dynamic_filtered_points_max, filtered_count)
                mapping_point_cloud_torch = dataset.cur_point_cloud_torch[dataset.static_mask]
                dynamic_mask = ~dataset.static_mask
                if dynamic_mask.any():
                    dynamic_points_local = dataset.cur_point_cloud_torch[dynamic_mask][:, :3]
                    dynamic_points_world = transform_torch(dynamic_points_local, dataset.cur_pose_torch)
            mapper.process_frame(
                mapping_point_cloud_torch,
                dataset.cur_pose_torch,
                frame_id,
            )
            if dynamic_points_world is not None and dynamic_points_world.shape[0] > 0:
                dynamic_anchor_hits = neural_points.accumulate_dynamic_evidence(dynamic_points_world)
                if (
                    dynamic_anchor_hits >= int(slnr_cfg.get("dynamic_prune_min_dynamic_points", 128))
                    and frame_id > 5
                ):
                    pruned_anchor_count = neural_points.prune_dynamic_anchors(
                        min_observation_votes=float(slnr_cfg.get("dynamic_prune_min_votes", 8)),
                        dynamic_ratio_thre=float(slnr_cfg.get("dynamic_prune_ratio_thre", 0.6)),
                    )
                    if pruned_anchor_count > 0:
                        dynamic_prune_events += 1
                        dynamic_pruned_anchor_total += int(pruned_anchor_count)
        else:
            neural_points.reset_local_map(dataset.cur_pose_torch[:3, 3], None, frame_id, reboot_map=True)
        T5 = get_time()

        if mapping_on:
            cur_iter_num = config.iters * config.init_iter_ratio if (frame_id == 0 or system_rebooted) else config.iters
            if dataset.stop_status:
                cur_iter_num = max(1, cur_iter_num - 10)
            if (frame_id - neural_points.reboot_ts) == config.freeze_after_frame:
                freeze_decoders(mlp_dict, config)
                if tracking_mlp_dict is not None:
                    freeze_decoders(tracking_mlp_dict, config)
                config.decoder_freezed = True
                neural_points.compute_feature_principle_components(down_rate=17)
                if tracking_neural_points is not None:
                    tracking_neural_points.compute_feature_principle_components(down_rate=17)

            if frame_id % config.mapping_freq_frame == 0:
                if tracking_mapper is not None:
                    tracking_mapper.mapping(cur_iter_num)
                mapper.mapping(cur_iter_num)
        T6 = get_time()

        # V. Regular logging
        if not config.silence:
            print("time for frame reading          (ms): {:.2f}".format((T1 - T0) * 1e3))
            print("time for frame preprocessing    (ms): {:.2f}".format((T2 - T1) * 1e3))
            if config.track_on:
                print("time for odometry               (ms): {:.2f}".format((T3 - T2) * 1e3))
            if config.pgo_on:
                print("time for loop detection and PGO (ms): {:.2f}".format((T4 - T3) * 1e3))
            print("time for mapping preparation    (ms): {:.2f}".format((T5 - T4) * 1e3))
            print("time for mapping                (ms): {:.2f}".format((T6 - T5) * 1e3))

        if config.log_freq_frame > 0 and (frame_id + 1) % config.log_freq_frame == 0:
            dataset.write_results_log()

        # VI. Mesh reconstruction and visualization
        if config.o3d_vis_on:
            if not q_vis2main.empty():
                control_packet: ControlPacket = get_latest_queue(q_vis2main)
                vis_visualize_on = control_packet.flag_vis
                vis_global_on = control_packet.flag_global
                vis_mesh_on = control_packet.flag_mesh
                vis_sdf_on = control_packet.flag_sdf
                vis_source_pc_weight = control_packet.flag_source
                vis_mesh_mc_res_m = control_packet.mc_res_m
                vis_mesh_min_nn = control_packet.mesh_min_nn
                vis_mesh_freq_frame = control_packet.mesh_freq_frame
                vis_sdf_slice_height = control_packet.sdf_slice_height
                vis_sdf_freq_frame = control_packet.sdf_freq_frame
                vis_sdf_res_m = control_packet.sdf_res_m
                while control_packet.flag_pause:
                    time.sleep(0.1)
                    if not q_vis2main.empty():
                        control_packet = get_latest_queue(q_vis2main)
                        if not control_packet.flag_pause:
                            break

            if vis_visualize_on:
                vis_query_min_nn = max(
                    1,
                    min(int(vis_mesh_min_nn), int(neural_points.backend.query_nn_k)),
                )
                backend_ready_for_vis = (
                    neural_points.backend.neural_map is not None
                    and not neural_points.is_empty()
                )
                dataset.update_o3d_map()
                if config.track_on and frame_id > 0 and vis_source_pc_weight and (weight_pc_o3d is not None):
                    dataset.cur_frame_o3d = weight_pc_o3d
                T7 = get_time()

                if (
                    backend_ready_for_vis
                    and vis_mesh_on
                    and (
                        frame_id == 0
                        or frame_id == last_frame
                        or (frame_id + 1) % vis_mesh_freq_frame == 0
                        or pgm.last_loop_idx == frame_id
                    )
                ):
                    global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=37)
                    dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()
                    if not vis_global_on:
                        chunks_aabb = split_chunks(global_neural_pcd_down, dataset.cur_bbx, vis_mesh_mc_res_m * 100)
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, vis_mesh_mc_res_m, None, True, False, False, filter_isolated_mesh=True, mesh_min_nn=vis_query_min_nn)
                    else:
                        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                        chunks_aabb = split_chunks(global_neural_pcd_down, aabb, vis_mesh_mc_res_m * 200)
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, vis_mesh_mc_res_m, None, False, False, False, filter_isolated_mesh=True, mesh_min_nn=vis_query_min_nn)

                if (
                    backend_ready_for_vis
                    and vis_sdf_on
                    and (
                        frame_id == 0
                        or frame_id == last_frame
                        or (frame_id + 1) % vis_sdf_freq_frame == 0
                    )
                ):
                    sdf_bound = config.surface_sample_range_m * 4.0
                    vis_sdf_bbx = create_bbx_o3d(dataset.cur_pose_ref[:3, 3], config.max_range / 2)
                    cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(
                        vis_sdf_bbx,
                        dataset.cur_pose_ref[2, 3] + vis_sdf_slice_height,
                        vis_sdf_res_m,
                        True,
                        -sdf_bound,
                        sdf_bound,
                        mask_min_nn_count=vis_query_min_nn,
                    )
                    if config.vis_sdf_slice_v:
                        cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(
                            dataset.cur_bbx,
                            dataset.cur_pose_ref[0, 3],
                            vis_sdf_res_m,
                            True,
                            -sdf_bound,
                            sdf_bound,
                            mask_min_nn_count=vis_query_min_nn,
                        )
                        cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
                    else:
                        cur_sdf_slice = cur_sdf_slice_h

                pool_pcd = mapper.get_data_pool_o3d(down_rate=37)
                odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
                loop_edges = pgm.loop_edges_vis if config.pgo_on else None
                packet_to_vis = VisPacket(frame_id=frame_id, travel_dist=travel_dist[-1], gpu_mem_usage_gb=get_gpu_memory_usage_gb(), cur_fps=cur_fps)
                if not neural_points.is_empty():
                    packet_to_vis.add_neural_points_data(neural_points, only_local_map=(not vis_global_on), pca_color_on=config.decoder_freezed)
                if dataset.cur_frame_o3d is not None:
                    packet_to_vis.add_scan(np.array(dataset.cur_frame_o3d.points, dtype=np.float64), np.array(dataset.cur_frame_o3d.colors, dtype=np.float64))
                if cur_mesh is not None:
                    packet_to_vis.add_mesh(np.array(cur_mesh.vertices, dtype=np.float64), np.array(cur_mesh.triangles), np.array(cur_mesh.vertex_colors, dtype=np.float64))
                if cur_sdf_slice is not None:
                    packet_to_vis.add_sdf_slice(np.array(cur_sdf_slice.points, dtype=np.float64), np.array(cur_sdf_slice.colors, dtype=np.float64))
                if pool_pcd is not None:
                    packet_to_vis.add_sdf_training_pool(np.array(pool_pcd.points, dtype=np.float64), np.array(pool_pcd.colors, dtype=np.float64))
                packet_to_vis.add_traj(odom_poses, gt_poses, pgo_poses, loop_edges)
                q_main2vis.put(packet_to_vis)
                T8 = get_time()
                if not config.silence:
                    print("time for o3d update             (ms): {:.2f}".format((T7 - T6) * 1e3))
                    print("time for visualization          (ms): {:.2f}".format((T8 - T7) * 1e3))

        # VII. Timing, logging, and frame bookkeeping
        cur_frame_process_time = np.array([T2 - T1, T3 - T2, T5 - T4, T6 - T5, T4 - T3])
        dataset.time_table.append(cur_frame_process_time)
        cur_fps = 1.0 / (np.sum(np.array(dataset.time_table[-10:]), axis=1).mean() + 1e-6)

        if config.wandb_vis_on:
            wandb.log({"frame": frame_id, "timing(s)/preprocess": T2 - T1, "timing(s)/tracking": T3 - T2, "timing(s)/pgo": T4 - T3, "timing(s)/mapping": T6 - T4})

        dataset.processed_frame += 1

    # VI. Save results
    _ensure_run_output_dirs(run_path)
    dataset.run_path = run_path

    dynamic_summary_path = os.path.join(run_path, "meta", "dynamic_filter_summary.txt")
    with open(dynamic_summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"dynamic_filter_active: {dynamic_filter_active}\n")
        summary_file.write(f"dynamic_filter_frames: {dynamic_filter_frame_count}\n")
        summary_file.write(f"dynamic_filtered_points_total: {dynamic_filtered_points_total}\n")
        summary_file.write(f"dynamic_filtered_points_max_per_frame: {dynamic_filtered_points_max}\n")
        summary_file.write(f"dynamic_prune_events: {dynamic_prune_events}\n")
        summary_file.write(f"dynamic_pruned_anchor_total: {dynamic_pruned_anchor_total}\n")
    print(
        "Dynamic filtering summary: "
        f"active={dynamic_filter_active}, "
        f"mask_frames={dynamic_filter_frame_count}, "
        f"filtered_points_total={dynamic_filtered_points_total}, "
        f"filtered_points_max_per_frame={dynamic_filtered_points_max}, "
        f"prune_events={dynamic_prune_events}, "
        f"pruned_anchor_total={dynamic_pruned_anchor_total}"
    )

    mapper.free_pool()
    if tracking_mapper is not None:
        tracking_mapper.free_pool()
    pose_eval_results = dataset.write_results()
    if config.pgo_on and pgm.pgo_count > 0:
        print("# Loop corrected: ", pgm.pgo_count)
        pgm.write_g2o(os.path.join(run_path, "final_pose_graph.g2o"))
        pgm.write_loops(os.path.join(run_path, "loop_log.txt"))
        if config.o3d_vis_on:
            pgm.plot_loops(os.path.join(run_path, "loop_plot.png"), vis_now=False)

    output_mc_res_m = config.mc_res_m * 0.6
    mc_cm_str = str(round(output_mc_res_m * 1e2))
    model_path, exported_mesh, mesh_path = _save_slnr_outputs(
        run_path,
        neural_points,
        config,
        slnr_cfg,
    )
    if exported_mesh is not None:
        cur_mesh = exported_mesh
        mesh_vertices, mesh_triangles = _mesh_stats(cur_mesh)
        if mesh_vertices == 0 or mesh_triangles == 0:
            print(
                "[bold yellow]SLNR mesh export produced an empty mesh "
                f"(vertices={mesh_vertices}, triangles={mesh_triangles}).[/bold yellow]"
            )
        elif mesh_path is not None and not os.path.isfile(mesh_path):
            print(
                "[bold red]SLNR mesh export finished, but the mesh file was "
                f"not created at {mesh_path}.[/bold red]"
            )

    neural_points.clear_temp()
    if tracking_neural_points is not None:
        tracking_neural_points.clear_temp()
    if config.save_map:
        print(
            "Use 'python vis_pin_map.py {} -m {} -o mesh_out_{}cm.ply' to inspect the saved geometry offline.".format(
                run_path, output_mc_res_m, mc_cm_str
            )
        )

    if config.save_merged_pc:
        dataset.write_merged_point_cloud()

    remove_gpu_cache()

    if config.o3d_vis_on:
        while True:
            if not q_vis2main.empty():
                q_vis2main.get()
            packet_to_vis = VisPacket(frame_id=frame_id, travel_dist=travel_dist[-1], slam_finished=True)
            if not neural_points.is_empty():
                packet_to_vis.add_neural_points_data(neural_points, only_local_map=False, pca_color_on=config.decoder_freezed)
            if cur_mesh is not None:
                packet_to_vis.add_mesh(np.array(cur_mesh.vertices, dtype=np.float64), np.array(cur_mesh.triangles), np.array(cur_mesh.vertex_colors, dtype=np.float64))
                cur_mesh = None
            packet_to_vis.add_traj(odom_poses, gt_poses, pgo_poses, loop_edges)
            q_main2vis.put(packet_to_vis)
            time.sleep(1.0)

    return pose_eval_results


if __name__ == "__main__":
    app()

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from contactflow.dp3.aqr_dp3.config import (
    PEG_INSERTION_WORKSPACE_CROP_MAX,
    PEG_INSERTION_WORKSPACE_CROP_MIN,
    PULL_CUBE_TOOL_WORKSPACE_CROP_MAX,
    PULL_CUBE_TOOL_WORKSPACE_CROP_MIN,
    STACK_CUBE_WORKSPACE_CROP_MAX,
    STACK_CUBE_WORKSPACE_CROP_MIN,
    TASK_PROFILES,
    validate_workspace_bounds,
)


DEFAULT_SOURCE_JSON = (
    "data/maniskill_demos/PegInsertionSide-v1/rl/"
    "trajectory.pointcloud.pd_joint_delta_pos.physx_cpu.json"
)
DEFAULT_DP3_ROOT = "."

TASK_PROFILE_CONTRACTS = {
    "peg_insertion": {
        "env_id": "PegInsertionSide-v1",
        "workspace": (
            PEG_INSERTION_WORKSPACE_CROP_MIN,
            PEG_INSERTION_WORKSPACE_CROP_MAX,
        ),
    },
    "stack_cube": {
        "env_id": "StackCube-v1",
        "workspace": (
            STACK_CUBE_WORKSPACE_CROP_MIN,
            STACK_CUBE_WORKSPACE_CROP_MAX,
        ),
    },
    "pull_cube_tool": {
        "env_id": "PullCubeTool-v1",
        "workspace": (
            PULL_CUBE_TOOL_WORKSPACE_CROP_MIN,
            PULL_CUBE_TOOL_WORKSPACE_CROP_MAX,
        ),
    },
}

STACKCUBE_PHASE_NAMES = {
    0: "approach",
    1: "pre_grasp",
    2: "grasp",
    3: "lift",
    4: "align_above_support",
    5: "stacked",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an AQR-DP3 checkpoint on its configured ManiSkill task "
            "with a fixed one- to eight-action execution prefix."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dp3-root", default=DEFAULT_DP3_ROOT)
    parser.add_argument("--source-json", default=DEFAULT_SOURCE_JSON)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("auto", "ema", "model"),
        default="auto",
    )
    parser.add_argument(
        "--exec-steps", type=int, choices=tuple(range(1, 9)), default=1
    )
    parser.add_argument("--eval-start", type=int, default=876)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Use 1 for a short environment smoke run.",
    )
    parser.add_argument("--max-env-steps", type=int, default=100)
    parser.add_argument(
        "--ignore-truncated",
        action="store_true",
        help=(
            "Continue after the ManiSkill TimeLimit reports truncated=True, "
            "up to --max-env-steps or success."
        ),
    )
    parser.add_argument(
        "--query-points",
        type=int,
        choices=(0, 4096, 8192),
        default=0,
        help="0 uses the point budget stored in the checkpoint.",
    )
    parser.add_argument(
        "--point-sampling-mode",
        choices=("pooled", "fps"),
        default="pooled",
    )
    parser.add_argument(
        "--robot-seg-ids",
        default="auto",
        help=(
            "Used only with --robot-mask-source segmentation. Use auto or a "
            "comma-separated list of robot link segmentation ids."
        ),
    )
    parser.add_argument(
        "--robot-mask-source",
        choices=("off", "segmentation"),
        default="off",
        help=(
            "off is deployment-safe and supplies an all-zero mask. "
            "segmentation is a privileged simulator-only diagnostic."
        ),
    )
    parser.add_argument(
        "--allow-privileged-robot-mask",
        action="store_true",
        help=(
            "Required to use simulator segmentation as a robot-mask input. "
            "Such a run is not deployment-safe and must not be reported as a "
            "deployment result."
        ),
    )
    parser.add_argument(
        "--diagnostic-segmentation",
        action="store_true",
        help=(
            "Use simulator segmentation only to label the already-selected "
            "AQR query neighbors in evaluation logs. These labels never enter "
            "point sampling, robot masking, local tokens, or policy inputs."
        ),
    )
    parser.add_argument(
        "--diagnostic-tool-geometry",
        action="store_true",
        help=(
            "Log current/base/AQR-corrected/final slot1 tool keypoints and "
            "actions. This adds evaluator-only FK work and is off by default."
        ),
    )
    parser.add_argument(
        "--disable-relation-features",
        action="store_true",
        help=(
            "Evaluator-only A2 ablation: bypass the trained relation branch "
            "and use semantic local tokens only. The checkpoint is unchanged."
        ),
    )
    parser.add_argument(
        "--disable-post-refiner",
        action="store_true",
        help=(
            "Evaluate the completed DP3-H12 base trajectory without invoking "
            "the post-diffusion refiner. The checkpoint PointNeXt encoder and "
            "diffusion UNet remain unchanged."
        ),
    )
    parser.add_argument(
        "--diagnostic-cubeA-seg-ids",
        dest="diagnostic_cube_a_seg_ids",
        default="auto",
        help="auto or comma-separated CubeA segmentation ids.",
    )
    parser.add_argument(
        "--diagnostic-cubeB-seg-ids",
        dest="diagnostic_cube_b_seg_ids",
        default="auto",
        help="auto or comma-separated CubeB segmentation ids.",
    )
    parser.add_argument(
        "--crop-min",
        default=",".join(str(value) for value in PEG_INSERTION_WORKSPACE_CROP_MIN),
    )
    parser.add_argument(
        "--crop-max",
        default=",".join(str(value) for value in PEG_INSERTION_WORKSPACE_CROP_MAX),
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--report-out",
        default="outputs/aqr_dp3_maniskill/report.json",
    )
    parser.add_argument(
        "--csv-out",
        default="outputs/aqr_dp3_maniskill/episodes.csv",
    )
    parser.add_argument(
        "--step-csv-out",
        default="outputs/aqr_dp3_maniskill/steps.csv",
    )
    parser.add_argument(
        "--paper-trace-out",
        default="",
        help=(
            "Optional JSONL path for one row per planned AQR slot, including "
            "base/corrected actions, tool keypoints, confidence, and residuals. "
            "Enables detailed tool geometry diagnostics but never changes policy inputs."
        ),
    )
    parser.add_argument(
        "--paper-snapshot-dir",
        default="",
        help=(
            "Optional directory for publication snapshots containing the actual "
            "ManiSkill RGB render, sampled XYZRGB point cloud, evaluator-only "
            "segmentation labels, TCP/fingertip anchors, and base/corrected "
            "action horizons. This never changes policy inputs."
        ),
    )
    parser.add_argument(
        "--paper-snapshot-episodes",
        default="first",
        help=(
            "Episodes saved by --paper-snapshot-dir: first, all, or a "
            "comma-separated list of episode ids (for example 800,803)."
        ),
    )
    parser.add_argument(
        "--paper-snapshot-plan-stride",
        type=int,
        default=1,
        help="Save one publication snapshot every N policy plans.",
    )
    parser.add_argument(
        "--paper-offset-probe-action",
        default="",
        help=(
            "Optional comma-separated normalized arm-action offset for a "
            "controlled publication probe (for example "
            "0.18,-0.12,0,0,0,0,0). The same offset is injected into every "
            "refined horizon slot, AQR is rerun without executing the probe, "
            "and reference/wrong/corrected FK trajectories are saved."
        ),
    )
    parser.add_argument(
        "--video-mode",
        choices=("off", "failures", "all"),
        default="off",
        help=(
            "Record no videos, failed episodes only, or all episodes. "
            "Video and task-stage state are evaluator-only diagnostics and "
            "never enter the policy input."
        ),
    )
    parser.add_argument(
        "--video-dir",
        default="outputs/aqr_dp3_maniskill/videos",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--video-stride",
        type=int,
        default=1,
        help="Record one rendered frame every N environment steps.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_contract(
    task_profile: str,
) -> tuple[
    str,
    tuple[float, float, float],
    tuple[float, float, float],
]:
    profile = str(task_profile).strip().lower()
    if profile not in TASK_PROFILES or profile not in TASK_PROFILE_CONTRACTS:
        choices = ", ".join(sorted(TASK_PROFILE_CONTRACTS))
        raise ValueError(
            f"unsupported AQR task_profile {task_profile!r}; expected {choices}"
        )
    contract = TASK_PROFILE_CONTRACTS[profile]
    crop_min, crop_max = contract["workspace"]
    return str(contract["env_id"]), tuple(crop_min), tuple(crop_max)


def _validate_task_environment(task_profile: str, env_id: str) -> str:
    expected_env_id, _, _ = _task_contract(task_profile)
    if str(env_id) != expected_env_id:
        raise ValueError(
            "checkpoint task_profile and source trajectory environment disagree: "
            f"{task_profile!r} requires {expected_env_id!r}, found {env_id!r}"
        )
    return expected_env_id


def _policy_point_feature_dim(cfg: Any) -> int:
    try:
        value = cfg.shape_meta.obs.point_cloud.shape[-1]
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "checkpoint policy config has no point_cloud feature shape"
        ) from exc
    point_dim = int(value)
    if point_dim not in {3, 6}:
        raise ValueError(
            "AQR-DP3 online evaluation supports XYZ or XYZRGB point features; "
            f"checkpoint requires {point_dim} channels"
        )
    return point_dim


def _parse_robot_ids(value: str) -> tuple[int, ...] | None:
    text = str(value).strip().lower()
    if text == "auto":
        return None
    result = tuple(
        sorted(
            {
                int(item.strip())
                for item in text.split(",")
                if item.strip()
            }
        )
    )
    if not result or any(item < 0 for item in result):
        raise ValueError("--robot-seg-ids must be auto or non-negative ids")
    return result


def _to_numpy_local(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _to_bool(value: Any) -> bool:
    return bool(_to_numpy_local(value).reshape(-1)[0])


def _to_scalar(value: Any) -> float:
    return float(_to_numpy_local(value).reshape(-1)[0])


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a ManiSkill/SAPIEN wxyz quaternion to a rotation matrix."""

    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.size != 4 or not bool(np.isfinite(value).all()):
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = value / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _pose_from_env_attr(
    env, names: tuple[str, ...]
) -> np.ndarray | None:
    if env is None:
        return None
    unwrapped = env.unwrapped
    for name in names:
        if not hasattr(unwrapped, name):
            continue
        value = getattr(unwrapped, name)
        candidates: list[Any] = []
        if hasattr(value, "pose"):
            pose = value.pose
            candidates.extend(
                [
                    getattr(pose, "raw_pose", None),
                    getattr(pose, "p", None),
                    pose,
                ]
            )
        candidates.extend(
            [
                getattr(value, "raw_pose", None),
                getattr(value, "p", None),
                value,
            ]
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                array = _to_numpy_local(candidate).reshape(-1).astype(
                    np.float32
                )
            except Exception:
                continue
            if array.size >= 3 and np.all(np.isfinite(array[:3])):
                return array
    return None


def _available_env_pose_attrs(
    env, tokens: tuple[str, ...]
) -> list[str]:
    if env is None:
        return []
    return [
        name
        for name in dir(env.unwrapped)
        if any(token in name.lower() for token in tokens)
    ][:40]


def _object_goal_pose_from_obs_or_env(
    obs: dict[str, Any], env
) -> tuple[np.ndarray, np.ndarray]:
    extra = obs.get("extra", {})
    object_pose = None
    for key in ("obj_pose", "object_pose", "cube_pose", "cubeA_pose"):
        if key in extra:
            object_pose = (
                _to_numpy_local(extra[key]).reshape(-1).astype(np.float32)
            )
            break
    if object_pose is None:
        object_pose = _pose_from_env_attr(
            env, ("obj", "object", "cube", "box", "cubeA")
        )
    if object_pose is None:
        available = ", ".join(
            _available_env_pose_attrs(
                env, ("obj", "object", "cube", "box", "cubeA")
            )
        )
        raise KeyError(
            "StackCube evaluation could not resolve cubeA pose; "
            f"available object-like attributes: {available or '(none)'}"
        )

    goal_position = None
    for key in ("goal_pos", "target_pos", "cubeB_pose"):
        if key in extra:
            goal_position = (
                _to_numpy_local(extra[key])
                .reshape(-1)
                .astype(np.float32)[:3]
            )
            break
    if goal_position is None:
        goal_pose = _pose_from_env_attr(
            env,
            (
                "goal_region",
                "goal_site",
                "goal",
                "target_region",
                "target_site",
                "target",
                "cubeB",
            ),
        )
        if goal_pose is not None:
            goal_position = goal_pose[:3]
    if goal_position is None:
        available = ", ".join(
            _available_env_pose_attrs(env, ("goal", "target", "cubeB"))
        )
        raise KeyError(
            "StackCube evaluation could not resolve cubeB/goal pose; "
            f"available goal-like attributes: {available or '(none)'}"
        )
    return object_pose.astype(np.float32), goal_position.astype(np.float32)


def _environment_done(
    *,
    terminated: bool,
    truncated: bool,
    ignore_truncated: bool,
) -> bool:
    return bool(terminated) or (bool(truncated) and not bool(ignore_truncated))


def _robot_ids_from_env(env) -> tuple[int, ...]:
    ids: list[int] = []
    for link in env.unwrapped.agent.robot.links:
        raw = getattr(link, "per_scene_id", None)
        if raw is None:
            continue
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        ids.extend(int(item) for item in np.asarray(raw).reshape(-1))
    result = tuple(sorted(set(ids)))
    if not result:
        raise RuntimeError(
            "Could not resolve robot segmentation ids; pass --robot-seg-ids."
        )
    return result


def _scene_ids(value: Any) -> tuple[int, ...]:
    raw = getattr(value, "per_scene_id", None)
    if raw is None:
        return ()
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    return tuple(sorted(set(int(item) for item in np.asarray(raw).reshape(-1))))


def _actor_ids_from_env(
    env,
    names: tuple[str, ...],
) -> tuple[int, ...]:
    unwrapped = env.unwrapped
    for name in names:
        actor = getattr(unwrapped, name, None)
        ids = _scene_ids(actor)
        if ids:
            return ids
    raise RuntimeError(
        "Could not resolve segmentation ids for actor aliases "
        f"{', '.join(names)}; pass explicit diagnostic ids."
    )


def _state16(obs: dict[str, Any]) -> np.ndarray:
    qpos = _to_numpy_local(obs["agent"]["qpos"]).reshape(-1).astype(np.float32)
    tcp = _to_numpy_local(obs["extra"]["tcp_pose"]).reshape(-1).astype(np.float32)
    state = np.concatenate([qpos, tcp]).astype(np.float32)
    if state.shape != (16,):
        raise ValueError(
            "AQR-DP3 evaluation requires qpos(9)+TCP pose(7)=16 values; "
            f"found qpos={qpos.shape}, tcp={tcp.shape}."
        )
    if not bool(np.isfinite(state).all()):
        raise ValueError("the live qpos+TCP state contains non-finite values")
    return state


def _dense_observation(
    obs: dict[str, Any],
    *,
    query_points: int,
    point_feature_dim: int,
    rng: np.random.Generator,
    robot_seg_ids: tuple[int, ...] | None,
    robot_mask_source: str,
    point_sampling_mode: str,
    crop_min: tuple[float, float, float] | None,
    crop_max: tuple[float, float, float] | None,
    diagnostic_segmentation: bool = False,
) -> dict[str, np.ndarray]:
    from contactflow.dp3.aqr_dp3.observation import _sample_aqr_points

    pointcloud = obs["pointcloud"]
    xyzw = _to_numpy_local(pointcloud["xyzw"]).reshape(-1, 4)
    append_rgb = int(point_feature_dim) == 6
    if int(point_feature_dim) not in {3, 6}:
        raise ValueError("point_feature_dim must be 3 (XYZ) or 6 (XYZRGB)")
    rgb_value = pointcloud.get("rgb")
    if append_rgb and rgb_value is None:
        raise KeyError(
            "XYZRGB checkpoint requires deployable point-aligned "
            "obs/pointcloud/rgb"
        )
    point_rgb = None
    if rgb_value is not None:
        point_rgb_array = _to_numpy_local(rgb_value)
        point_rgb = point_rgb_array.reshape(-1, point_rgb_array.shape[-1])
    segmentation_value = pointcloud.get("segmentation")
    if robot_mask_source == "segmentation":
        if segmentation_value is None:
            raise KeyError(
                "Privileged robot-mask diagnostics require point-aligned "
                "ManiSkill segmentation."
            )
        if robot_seg_ids is None:
            raise ValueError("segmentation robot-mask mode needs robot_seg_ids.")
        segmentation = _to_numpy_local(segmentation_value).reshape(-1)
    elif robot_mask_source == "off":
        if diagnostic_segmentation:
            if segmentation_value is None:
                raise KeyError(
                    "Diagnostic query semantics require point-aligned "
                    "ManiSkill segmentation."
                )
            segmentation = _to_numpy_local(segmentation_value).reshape(-1)
        else:
            segmentation = np.zeros((xyzw.shape[0],), dtype=np.int32)
        robot_seg_ids = ()
    else:
        raise ValueError(f"unsupported robot_mask_source {robot_mask_source!r}")
    sampled = _sample_aqr_points(
        xyzw,
        int(query_points),
        rng,
        segmentation=segmentation,
        robot_seg_ids=robot_seg_ids,
        point_sampling_mode=point_sampling_mode,
        crop_min=crop_min,
        crop_max=crop_max,
        rgb=point_rgb,
        append_rgb=append_rgb,
        return_sampled_segmentation=bool(diagnostic_segmentation),
    )
    if diagnostic_segmentation:
        points, robot_mask, valid_mask, sampled_segmentation = sampled
    else:
        points, robot_mask, valid_mask = sampled
        sampled_segmentation = None
    if int(valid_mask.sum()) == 0:
        raise RuntimeError("the live dense point frame has no valid geometry")
    if bool(np.any(robot_mask & ~valid_mask)):
        raise RuntimeError("robot_mask is not a subset of point_valid_mask")
    result = {
        "point_cloud": points.astype(np.float32, copy=False),
        "agent_pos": _state16(obs),
        "robot_mask": robot_mask.astype(bool, copy=False),
        "point_valid_mask": valid_mask.astype(bool, copy=False),
    }
    if sampled_segmentation is not None:
        result["_eval_segmentation"] = sampled_segmentation.astype(
            np.int32, copy=False
        )
    return result


def _history_input(history: deque[dict[str, np.ndarray]], device):
    import torch

    keys = ("point_cloud", "agent_pos", "robot_mask", "point_valid_mask")
    return {
        key: torch.from_numpy(
            np.stack([item[key] for item in history], axis=0)[None]
        ).to(device)
        for key in keys
    }


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_mean(value: Any) -> float | None:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            tensor = value.detach().to(dtype=torch.float32)
            finite = tensor[torch.isfinite(tensor)]
            if finite.numel() == 0:
                return None
            return float(finite.mean().cpu())
    except ImportError:
        pass
    if isinstance(value, (bool, int, float, np.number)):
        result = float(value)
        return result if np.isfinite(result) else None
    return None


def _aqr_diagnostics(value: Any) -> dict[str, float | None]:
    if not isinstance(value, dict):
        return {}
    names = (
        "raw_valid_count",
        "enc_valid_count",
        "query_valid_count",
        "enc_global_count",
        "enc_tcp_swept_count",
        "enc_fill_count",
        "enc_padding_count",
        "query_padding_count",
        "tcp_tube_count",
        "tcp_swept_zero_coverage",
        "enc_robot_ratio",
        "query_robot_ratio",
        "empty_fraction",
        "robot_point_fraction",
        "nearest_distance_m",
        "tool_outside_fraction",
        "local_token_norm",
        "query_gate",
        "delta_norm",
        "raw_action_residual_norm",
        "bounded_action_residual_norm",
        "action_residual_clip_fraction",
        "action_residual_progress_limited_fraction",
        "action_residual_progress_ratio",
        "protected_non_arm_residual_norm",
        "aqr_update_count",
        "point_frontend_ms",
        "pointnext_ms",
        "query_stem_ms",
        "trajectory_ms",
        "relative_query_ms",
        "local_residual_ms_total",
        "tool_trajectory_update_change_m",
        "a0_min",
        "a0_max",
        "a0_boundary_fraction",
        "r_face_mean",
        "r_face_neg_fraction",
        "r_motion_mean",
        "normal_confidence_mean",
        "relation_gate_mean",
        "attn_entropy",
        "scale_gate_weights",
        "post_raw_residual_norm",
        "post_bounded_residual_norm",
        "post_applied_residual_norm",
        "post_confidence",
        "post_local_present",
        "post_refine_count",
        "anchor_current_weight",
        "anchor_candidate_weight",
        "anchor_left_weight",
        "anchor_right_weight",
    )
    return {
        f"aqr_{name}": _tensor_mean(value[name])
        for name in names
        if name in value
    }


def _commanded_tool_trajectory(
    policy,
    action_dict: dict[str, Any],
    obs_dict: dict[str, Any],
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray | None:
    import torch

    action_to_tool = getattr(policy, "action_to_tool", None)
    indices = getattr(policy, "_query_slot_indices", None)
    if action_to_tool is None or indices is None or "action_pred" not in action_dict:
        return None
    if (
        getattr(action_to_tool, "mode", None) == "joint_delta_fk"
        and getattr(action_to_tool, "candidate_lift", None) is None
    ):
        # current_tcp_local intentionally constructs only the static observed
        # TCP query and therefore has no dynamic candidate-FK contract. The
        # evaluator must not reintroduce that ablated path merely to compute
        # commanded-tool diagnostics.
        return None
    raw_prediction = action_dict["action_pred"]
    if not isinstance(raw_prediction, torch.Tensor) or raw_prediction.ndim != 3:
        return None
    indices = indices.to(device=raw_prediction.device)
    environment_actions = raw_prediction.index_select(1, indices)
    low = torch.as_tensor(
        action_low, dtype=environment_actions.dtype, device=environment_actions.device
    )
    high = torch.as_tensor(
        action_high, dtype=environment_actions.dtype, device=environment_actions.device
    )
    environment_actions = torch.maximum(
        torch.minimum(environment_actions, high), low
    )
    policy_actions = policy.normalizer["action"].normalize(environment_actions)
    trajectory = action_to_tool(
        policy_actions=policy_actions,
        environment_actions=environment_actions,
        state_history_raw=obs_dict["agent_pos"],
    )
    return {
        "names": tuple(trajectory.names),
        "keypoints_world": (
            trajectory.keypoints_world[0].detach().to(torch.float32).cpu().numpy()
        ),
        "rotations_world_from_tool": (
            trajectory.rotations_world_from_tool[0]
            .detach()
            .to(torch.float32)
            .cpu()
            .numpy()
        ),
        "tcp_matrices_world": (
            trajectory.tcp_matrices_world[0]
            .detach()
            .to(torch.float32)
            .cpu()
            .numpy()
        ),
    }


def _commanded_tcp_positions(
    policy,
    action_dict: dict[str, Any],
    obs_dict: dict[str, Any],
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray | None:
    trajectory = _commanded_tool_trajectory(
        policy,
        action_dict,
        obs_dict,
        action_low=action_low,
        action_high=action_high,
    )
    if trajectory is None:
        return None
    return np.asarray(trajectory["tcp_matrices_world"])[..., :3, 3]


def _diagnostic_tensor(
    diagnostics: Any,
    key: str,
) -> np.ndarray | None:
    if not isinstance(diagnostics, dict) or key not in diagnostics:
        return None
    try:
        value = _to_numpy_local(diagnostics[key])
    except Exception:
        return None
    return np.asarray(value)


def _tool_snapshot(
    *,
    prefix: str,
    names: tuple[str, ...],
    keypoints_world: np.ndarray,
    rotation_world_from_tool: np.ndarray,
    cube_a_position: np.ndarray | None,
) -> dict[str, Any]:
    points = np.asarray(keypoints_world, dtype=np.float64)
    rotation = np.asarray(rotation_world_from_tool, dtype=np.float64)
    if points.ndim != 2 or points.shape[-1] != 3:
        return {}
    if rotation.shape != (3, 3):
        return {}
    by_name = {
        name: points[index]
        for index, name in enumerate(names)
        if index < len(points)
    }
    tcp = by_name.get("tcp")
    left = by_name.get("left_fingertip")
    right = by_name.get("right_fingertip")
    midpoint = (
        0.5 * (left + right)
        if left is not None and right is not None
        else tcp
    )
    output: dict[str, Any] = {}
    for label, point in (
        ("tcp", tcp),
        ("left_fingertip", left),
        ("right_fingertip", right),
        ("fingertip_midpoint", midpoint),
    ):
        if point is None:
            continue
        for axis, value in zip(("x", "y", "z"), point):
            output[f"{prefix}_{label}_{axis}"] = float(value)
    closing_axis = None
    if left is not None and right is not None:
        vector = left - right
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            closing_axis = vector / norm
            for axis, value in zip(("x", "y", "z"), closing_axis):
                output[f"{prefix}_closing_axis_{axis}"] = float(value)
            output[f"{prefix}_model_gripper_width_m"] = norm
    if tcp is not None and midpoint is not None:
        output[f"{prefix}_midpoint_tcp_offset_m"] = float(
            np.linalg.norm(midpoint - tcp)
        )
    if cube_a_position is not None and midpoint is not None:
        delta_world = np.asarray(
            cube_a_position, dtype=np.float64
        ) - midpoint
        cube_tool = rotation.T @ delta_world
        for axis, value in zip(("x", "y", "z"), cube_tool):
            output[f"{prefix}_cubeA_in_tool_{axis}_m"] = float(value)
        output[f"{prefix}_cubeA_midpoint_distance_m"] = float(
            np.linalg.norm(delta_world)
        )
        if closing_axis is not None:
            output[f"{prefix}_cubeA_closing_axis_offset_m"] = float(
                np.dot(delta_world, closing_axis)
            )
    return output


def _finger_link_origins_from_env(env) -> dict[str, float]:
    result: dict[str, float] = {}
    aliases = {
        "left": ("leftfinger", "left_finger"),
        "right": ("rightfinger", "right_finger"),
    }
    try:
        links = env.unwrapped.agent.robot.links
    except Exception:
        return result
    for link in links:
        name = str(getattr(link, "name", "")).lower()
        side = next(
            (
                label
                for label, tokens in aliases.items()
                if any(token in name for token in tokens)
            ),
            None,
        )
        if side is None:
            continue
        pose = getattr(link, "pose", None)
        position = getattr(pose, "p", None)
        if position is None:
            continue
        try:
            value = _to_numpy_local(position).reshape(-1, 3)[0]
        except Exception:
            continue
        for axis, coordinate in zip(("x", "y", "z"), value):
            result[
                f"eval_only_sim_{side}_finger_link_origin_{axis}"
            ] = float(coordinate)
    if all(
        f"eval_only_sim_{side}_finger_link_origin_x" in result
        for side in ("left", "right")
    ):
        left = np.asarray(
            [
                result[f"eval_only_sim_left_finger_link_origin_{axis}"]
                for axis in ("x", "y", "z")
            ],
            dtype=np.float64,
        )
        right = np.asarray(
            [
                result[f"eval_only_sim_right_finger_link_origin_{axis}"]
                for axis in ("x", "y", "z")
            ],
            dtype=np.float64,
        )
        midpoint = 0.5 * (left + right)
        for axis, coordinate in zip(("x", "y", "z"), midpoint):
            result[
                f"eval_only_sim_finger_link_midpoint_{axis}"
            ] = float(coordinate)
        result["eval_only_sim_finger_link_origin_width_m"] = float(
            np.linalg.norm(left - right)
        )
    return result


def _action_tool_diagnostics(
    policy,
    action_dict: dict[str, Any],
    obs_dict: dict[str, Any],
    raw_obs: dict[str, Any],
    env,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict[str, Any]:
    action_to_tool = getattr(policy, "action_to_tool", None)
    if action_to_tool is None:
        return {}
    names = tuple(action_to_tool.names)
    extra = raw_obs.get("extra", {})
    cube_pose = extra.get("cubeA_pose")
    cube_a_position = None
    if cube_pose is not None:
        cube_a_position = (
            _to_numpy_local(cube_pose).reshape(-1)[:3].astype(np.float64)
        )
    else:
        try:
            object_pose, _ = _object_goal_pose_from_obs_or_env(raw_obs, env)
            cube_a_position = np.asarray(
                object_pose, dtype=np.float64
            ).reshape(-1)[:3]
        except Exception:
            cube_a_position = None
    diagnostics = action_dict.get("aqr_diagnostics", {})
    output: dict[str, Any] = {
        "eval_only_tool_keypoint_names": "|".join(names),
        "eval_only_tool_diagnostics_entered_policy_input": False,
    }
    snapshots = (
        (
            "eval_only_current_model",
            "current_tool_keypoints_world",
            "current_tool_rotations_world",
        ),
        (
            "eval_only_base_candidate",
            "base_candidate_tool_keypoints_world",
            "base_candidate_tool_rotations_world",
        ),
        (
            "eval_only_aqr_corrected_candidate",
            "aqr_corrected_tool_keypoints_world",
            "aqr_corrected_tool_rotations_world",
        ),
    )
    for prefix, point_key, rotation_key in snapshots:
        points = _diagnostic_tensor(diagnostics, point_key)
        rotations = _diagnostic_tensor(diagnostics, rotation_key)
        if points is None or rotations is None:
            continue
        output.update(
            _tool_snapshot(
                prefix=prefix,
                names=names,
                keypoints_world=points[0, 0],
                rotation_world_from_tool=rotations[0, 0],
                cube_a_position=cube_a_position,
            )
        )
    final_trajectory = _commanded_tool_trajectory(
        policy,
        action_dict,
        obs_dict,
        action_low=action_low,
        action_high=action_high,
    )
    if final_trajectory is not None:
        output.update(
            _tool_snapshot(
                prefix="eval_only_final_candidate",
                names=tuple(final_trajectory["names"]),
                keypoints_world=final_trajectory["keypoints_world"][0],
                rotation_world_from_tool=(
                    final_trajectory["rotations_world_from_tool"][0]
                ),
                cube_a_position=cube_a_position,
            )
        )
    for prefix, key in (
        ("eval_only_base_action", "base_candidate_actions_environment"),
        ("eval_only_aqr_corrected_action", "aqr_corrected_actions_environment"),
    ):
        values = _diagnostic_tensor(diagnostics, key)
        if values is None:
            continue
        for index, value in enumerate(values[0, 0]):
            output[f"{prefix}_{index}"] = float(value)
    base_midpoint = np.asarray(
        [
            output.get(
                f"eval_only_base_candidate_fingertip_midpoint_{axis}"
            )
            for axis in ("x", "y", "z")
        ],
        dtype=object,
    )
    corrected_midpoint = np.asarray(
        [
            output.get(
                f"eval_only_aqr_corrected_candidate_fingertip_midpoint_{axis}"
            )
            for axis in ("x", "y", "z")
        ],
        dtype=object,
    )
    if all(value is not None for value in base_midpoint) and all(
        value is not None for value in corrected_midpoint
    ):
        shift = corrected_midpoint.astype(np.float64) - base_midpoint.astype(
            np.float64
        )
        for axis, value in zip(("x", "y", "z"), shift):
            output[f"eval_only_aqr_midpoint_shift_{axis}_m"] = float(value)
        output["eval_only_aqr_midpoint_shift_m"] = float(
            np.linalg.norm(shift)
        )
    output.update(_finger_link_origins_from_env(env))
    return output


def _query_semantic_diagnostics(
    policy,
    action_dict: dict[str, Any],
    current_observation: dict[str, np.ndarray],
    *,
    robot_seg_ids: tuple[int, ...],
    cube_a_seg_ids: tuple[int, ...],
    cube_b_seg_ids: tuple[int, ...],
) -> dict[str, Any]:
    labels = current_observation.get("_eval_segmentation")
    if labels is None:
        return {}
    diagnostics = action_dict.get("aqr_diagnostics", {})
    source_indices = _diagnostic_tensor(
        diagnostics, "query_source_indices"
    )
    anchors = _diagnostic_tensor(diagnostics, "tool_keypoints_world")
    if source_indices is None or anchors is None:
        return {
            "eval_only_query_semantics_available": False,
            "eval_only_query_semantics_reason": "missing_query_diagnostics",
        }
    source_indices = source_indices[0].astype(np.int64, copy=False)
    raw_points = np.asarray(current_observation["point_cloud"])
    raw_valid = np.asarray(
        current_observation["point_valid_mask"], dtype=bool
    )
    safe_indices = np.clip(source_indices, 0, max(0, len(raw_points) - 1))
    query_xyz = raw_points[safe_indices, :3].astype(np.float64, copy=False)
    query_valid = (
        source_indices >= 0
    ) & raw_valid[safe_indices] & np.isfinite(query_xyz).all(axis=1)
    query_labels = np.asarray(labels)[safe_indices]
    centers = anchors[0].reshape(-1, 3).astype(np.float64, copy=False)
    radii = tuple(float(value) for value in policy.relative_query.radii_m)
    neighbors = tuple(int(value) for value in policy.relative_query.neighbors)
    category_ids = {
        "cubeA": set(int(value) for value in cube_a_seg_ids),
        "cubeB": set(int(value) for value in cube_b_seg_ids),
        "robot": set(int(value) for value in robot_seg_ids),
    }
    total_counts = Counter()
    per_scale: dict[str, Any] = {}
    if centers.size == 0 or query_xyz.size == 0:
        return {
            "eval_only_query_semantics_available": False,
            "eval_only_query_semantics_reason": "empty_query_or_anchors",
        }
    squared_distance = (
        (centers[:, None, :] - query_xyz[None, :, :]) ** 2
    ).sum(axis=-1)
    squared_distance[:, ~query_valid] = np.inf
    for scale_index, (radius, neighbor_count) in enumerate(
        zip(radii, neighbors)
    ):
        k = min(int(neighbor_count), int(query_xyz.shape[0]))
        nearest = np.argpartition(
            squared_distance,
            kth=max(0, k - 1),
            axis=1,
        )[:, :k]
        nearest_distance = np.take_along_axis(
            squared_distance, nearest, axis=1
        )
        valid_neighbor = np.isfinite(nearest_distance) & (
            nearest_distance <= float(radius) ** 2
        )
        selected_labels = query_labels[nearest]
        scale_counts = Counter()
        for label, valid_flag in zip(
            selected_labels.reshape(-1),
            valid_neighbor.reshape(-1),
        ):
            if not bool(valid_flag):
                continue
            label_int = int(label)
            category = "other"
            for name in ("cubeA", "cubeB", "robot"):
                if label_int in category_ids[name]:
                    category = name
                    break
            scale_counts[category] += 1
            total_counts[category] += 1
        scale_total = sum(scale_counts.values())
        per_scale[
            f"eval_only_query_scale{scale_index}_radius_m"
        ] = float(radius)
        per_scale[
            f"eval_only_query_scale{scale_index}_neighbor_count"
        ] = int(scale_total)
        for category in ("cubeA", "cubeB", "robot", "other"):
            per_scale[
                f"eval_only_query_scale{scale_index}_{category}_fraction"
            ] = (
                float(scale_counts[category]) / float(scale_total)
                if scale_total
                else None
            )
    total = sum(total_counts.values())
    result: dict[str, Any] = {
        "eval_only_query_semantics_available": True,
        "eval_only_query_semantics_entered_policy_input": False,
        "eval_only_query_semantic_neighbor_count": int(total),
        **per_scale,
    }
    for category in ("cubeA", "cubeB", "robot", "other"):
        result[f"eval_only_query_{category}_fraction"] = (
            float(total_counts[category]) / float(total)
            if total
            else None
        )
    return result


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray(
        [value for value in values if np.isfinite(value)], dtype=np.float64
    )
    if not finite.size:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _optional_scalar(
    sources: tuple[dict[str, Any] | None, ...],
    keys: tuple[str, ...],
) -> float | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key not in source:
                continue
            try:
                value = _to_numpy_local(source[key]).reshape(-1)
                if value.size:
                    result = float(value[0])
                    if np.isfinite(result):
                        return result
            except Exception:
                continue
    return None


def _stackcube_step_diagnostics(
    obs: dict[str, Any],
    info: dict[str, Any] | None,
    env,
    *,
    initial_cube_a_z: float | None,
) -> dict[str, Any]:
    """Return evaluator-only StackCube task state.

    Simulator task flags and object poses are intentionally kept outside the
    observation passed to the policy.  Geometry-derived phase labels are
    diagnostics, not deployment inputs or training targets.
    """

    extra = obs.get("extra", {})
    sources = (info, extra)
    grasped_raw = _optional_scalar(
        sources,
        ("is_cubeA_grasped", "is_grasped", "is_grasping"),
    )
    on_cube_b_raw = _optional_scalar(
        sources,
        ("is_cubeA_on_cubeB", "is_obj_on_goal"),
    )
    static_raw = _optional_scalar(
        sources,
        ("is_cubeA_static", "is_obj_static"),
    )
    result: dict[str, Any] = {
        "eval_only_is_cubeA_grasped": (
            bool(grasped_raw > 0.5) if grasped_raw is not None else None
        ),
        "eval_only_is_cubeA_on_cubeB": (
            bool(on_cube_b_raw > 0.5) if on_cube_b_raw is not None else None
        ),
        "eval_only_is_cubeA_static": (
            bool(static_raw > 0.5) if static_raw is not None else None
        ),
        "eval_only_stackcube_phase_id": None,
        "eval_only_stackcube_phase": None,
        "eval_only_grasp_source": (
            "simulator_flag" if grasped_raw is not None else "geometry_proxy"
        ),
    }
    try:
        if "cubeA_pose" in extra and "cubeB_pose" in extra:
            cube_a_pose = _to_numpy_local(extra["cubeA_pose"]).reshape(-1)
            cube_b_position = _to_numpy_local(extra["cubeB_pose"]).reshape(-1)
        else:
            cube_a_pose, cube_b_position = _object_goal_pose_from_obs_or_env(
                obs, env
            )
        cube_a_position = np.asarray(cube_a_pose[:3], dtype=np.float64)
        cube_b_position = np.asarray(cube_b_position[:3], dtype=np.float64)
        tcp_pose = _to_numpy_local(extra["tcp_pose"]).reshape(-1)
        tcp_position = np.asarray(tcp_pose[:3], dtype=np.float64)
        target_position = cube_b_position.copy()
        target_position[2] += 0.04
        tcp_cube_distance = float(
            np.linalg.norm(tcp_position - cube_a_position)
        )
        target_xy_error = float(
            np.linalg.norm(cube_a_position[:2] - target_position[:2])
        )
        target_z_error = float(
            abs(cube_a_position[2] - target_position[2])
        )
        target_error = float(
            np.linalg.norm(cube_a_position - target_position)
        )
        lift_height = (
            float(cube_a_position[2] - initial_cube_a_z)
            if initial_cube_a_z is not None
            else None
        )
        grasped_for_phase = (
            bool(grasped_raw > 0.5)
            if grasped_raw is not None
            else tcp_cube_distance <= 0.045
        )
        lifted = bool(
            grasped_for_phase
            and lift_height is not None
            and lift_height >= 0.02
        )
        aligned_above = bool(
            lifted
            and target_xy_error <= 0.025
            and cube_a_position[2] >= target_position[2] + 0.015
        )
        stacked_geometry = bool(
            target_xy_error <= 0.018 and target_z_error <= 0.010
        )
        phase_id = 0
        if tcp_cube_distance <= 0.09:
            phase_id = 1
        if grasped_for_phase:
            phase_id = 2
        if lifted:
            phase_id = 3
        if aligned_above:
            phase_id = 4
        if stacked_geometry or bool(on_cube_b_raw and on_cube_b_raw > 0.5):
            phase_id = 5
        result.update(
            {
                "eval_only_cubeA_x": float(cube_a_position[0]),
                "eval_only_cubeA_y": float(cube_a_position[1]),
                "eval_only_cubeA_z": float(cube_a_position[2]),
                "eval_only_cubeB_x": float(cube_b_position[0]),
                "eval_only_cubeB_y": float(cube_b_position[1]),
                "eval_only_cubeB_z": float(cube_b_position[2]),
                "eval_only_tcp_cubeA_distance_m": tcp_cube_distance,
                "eval_only_cubeA_target_xy_error_m": target_xy_error,
                "eval_only_cubeA_target_z_error_m": target_z_error,
                "eval_only_cubeA_target_error_m": target_error,
                "eval_only_cubeA_lift_height_m": lift_height,
                "eval_only_lifted": lifted,
                "eval_only_aligned_above_support": aligned_above,
                "eval_only_stacked_geometry": stacked_geometry,
                "eval_only_stackcube_phase_id": phase_id,
                "eval_only_stackcube_phase": STACKCUBE_PHASE_NAMES[phase_id],
            }
        )
        if cube_a_pose.size >= 7 and tcp_pose.size >= 7:
            rotation_world_from_tcp = _quaternion_wxyz_to_matrix(tcp_pose[3:7])
            rotation_world_from_cube = _quaternion_wxyz_to_matrix(
                cube_a_pose[3:7]
            )
            cube_in_tcp = rotation_world_from_tcp.T @ (
                cube_a_position - tcp_position
            )
            rotation_tcp_from_cube = (
                rotation_world_from_tcp.T @ rotation_world_from_cube
            )
            result.update(
                {
                    "eval_only_cubeA_in_tcp_x_m": float(cube_in_tcp[0]),
                    "eval_only_cubeA_in_tcp_y_m": float(cube_in_tcp[1]),
                    "eval_only_cubeA_in_tcp_z_m": float(cube_in_tcp[2]),
                    "eval_only_cubeA_tcp_lateral_error_m": float(
                        np.linalg.norm(cube_in_tcp[:2])
                    ),
                    "eval_only_cubeA_tcp_axial_offset_m": float(cube_in_tcp[2]),
                    **{
                        f"eval_only_cubeA_in_tcp_r{row}{column}": float(
                            rotation_tcp_from_cube[row, column]
                        )
                        for row in range(3)
                        for column in range(3)
                    },
                }
            )
    except Exception as exc:
        result["eval_only_geometry_unavailable"] = type(exc).__name__
    return result


def _new_stackcube_episode_state() -> dict[str, Any]:
    return {
        "max_phase_id": 0,
        "ever_grasped": False,
        "ever_stable_grasp": False,
        "ever_secure_lift": False,
        "ever_lifted": False,
        "ever_aligned_above_support": False,
        "ever_on_cubeB": False,
        "ever_static": False,
        "ever_stacked_geometry": False,
        "final_grasped": None,
        "consecutive_grasp_steps": 0,
        "max_consecutive_grasp_steps": 0,
        "first_grasp_step": None,
        "first_stable_grasp_step": None,
        "first_lift_step": None,
        "first_aligned_step": None,
        "first_on_cubeB_step": None,
        "first_stacked_geometry_step": None,
        "min_tcp_cubeA_distance_m": None,
        "min_cubeA_target_error_m": None,
        "max_cubeA_lift_height_m": None,
    }


def _update_stackcube_episode_state(
    state: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    env_step: int,
) -> None:
    phase_id = diagnostics.get("eval_only_stackcube_phase_id")
    if phase_id is not None:
        state["max_phase_id"] = max(int(state["max_phase_id"]), int(phase_id))
    events = (
        ("eval_only_is_cubeA_grasped", "ever_grasped", "first_grasp_step"),
        ("eval_only_lifted", "ever_lifted", "first_lift_step"),
        (
            "eval_only_aligned_above_support",
            "ever_aligned_above_support",
            "first_aligned_step",
        ),
        (
            "eval_only_is_cubeA_on_cubeB",
            "ever_on_cubeB",
            "first_on_cubeB_step",
        ),
        (
            "eval_only_stacked_geometry",
            "ever_stacked_geometry",
            "first_stacked_geometry_step",
        ),
    )
    for diagnostic_key, ever_key, first_key in events:
        if diagnostics.get(diagnostic_key) is True:
            state[ever_key] = True
            if state[first_key] is None:
                state[first_key] = int(env_step)
    grasp_value = diagnostics.get("eval_only_is_cubeA_grasped")
    if grasp_value is True:
        state["consecutive_grasp_steps"] += 1
        state["max_consecutive_grasp_steps"] = max(
            int(state["max_consecutive_grasp_steps"]),
            int(state["consecutive_grasp_steps"]),
        )
        if int(state["consecutive_grasp_steps"]) >= 10:
            state["ever_stable_grasp"] = True
            if state["first_stable_grasp_step"] is None:
                state["first_stable_grasp_step"] = (
                    int(env_step) - int(state["consecutive_grasp_steps"]) + 1
                )
    elif grasp_value is False:
        state["consecutive_grasp_steps"] = 0
    if diagnostics.get("eval_only_is_cubeA_static") is True:
        state["ever_static"] = True
    if diagnostics.get("eval_only_is_cubeA_grasped") is not None:
        state["final_grasped"] = bool(
            diagnostics["eval_only_is_cubeA_grasped"]
        )
    for diagnostic_key, state_key in (
        (
            "eval_only_tcp_cubeA_distance_m",
            "min_tcp_cubeA_distance_m",
        ),
        (
            "eval_only_cubeA_target_error_m",
            "min_cubeA_target_error_m",
        ),
    ):
        value = diagnostics.get(diagnostic_key)
        if value is None:
            continue
        current = state[state_key]
        state[state_key] = (
            float(value) if current is None else min(float(current), float(value))
        )
    lift_height = diagnostics.get("eval_only_cubeA_lift_height_m")
    if lift_height is not None:
        current_lift = state["max_cubeA_lift_height_m"]
        state["max_cubeA_lift_height_m"] = (
            float(lift_height)
            if current_lift is None
            else max(float(current_lift), float(lift_height))
        )
        if (
            state["ever_stable_grasp"]
            and float(lift_height) >= 0.02
            and grasp_value is True
        ):
            state["ever_secure_lift"] = True


def _stackcube_failure_stage(
    state: dict[str, Any],
    *,
    success: bool,
) -> str:
    if success:
        return "success"
    if state["ever_on_cubeB"] or state["ever_stacked_geometry"]:
        if not state["ever_static"]:
            return "stacked_not_static"
        return "stacked_without_success"
    if state["ever_aligned_above_support"]:
        return "aligned_failed_placement"
    if state["ever_stable_grasp"]:
        if state["final_grasped"] is False:
            return "late_drop_after_stable_grasp"
        if state["ever_secure_lift"] or state["ever_lifted"]:
            return "lifted_not_aligned"
        return "stable_grasp_no_lift"
    if int(state["max_consecutive_grasp_steps"]) > 0:
        if state["ever_lifted"] or (
            state["max_cubeA_lift_height_m"] is not None
            and float(state["max_cubeA_lift_height_m"]) >= 0.005
        ):
            return "marginal_grasp_brief_lift"
        return "contact_only"
    minimum_distance = state.get("min_tcp_cubeA_distance_m")
    if minimum_distance is not None and float(minimum_distance) <= 0.055:
        return "reached_cube_never_grasped"
    return "never_reached_cube"


def _render_rgb_frame(env) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, dict):
        for key in ("rgb", "image", "render"):
            if key in frame:
                frame = frame[key]
                break
        else:
            frame = next(iter(frame.values()))
    array = _to_numpy_local(frame)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"env.render() returned unsupported shape {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0 + 1e-6:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return token or "unknown"


class _EpisodeVideoRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        video_dir: Path,
        episode_id: int,
        fps: int,
        stride: int,
    ):
        self.enabled = bool(enabled)
        self.episode_id = int(episode_id)
        self.stride = int(stride)
        self._writer = None
        self._partial_path = (
            video_dir / f"episode_{self.episode_id:04d}.partial.mp4"
        )
        if self.enabled:
            try:
                import imageio.v2 as imageio
            except ImportError as exc:
                raise RuntimeError(
                    "Video recording requires imageio and imageio-ffmpeg."
                ) from exc
            video_dir.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                str(self._partial_path),
                fps=int(fps),
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )

    def append(self, env, *, env_step: int, force: bool = False) -> None:
        if self._writer is None:
            return
        if force or int(env_step) % self.stride == 0:
            self._writer.append_data(_render_rgb_frame(env))

    def finalize(
        self,
        *,
        keep: bool,
        success: bool,
        failure_stage: str,
    ) -> str | None:
        if self._writer is None:
            return None
        self._writer.close()
        self._writer = None
        if not keep:
            self._partial_path.unlink(missing_ok=True)
            return None
        outcome = "success" if success else "failure"
        final_path = self._partial_path.with_name(
            f"episode_{self.episode_id:04d}_{outcome}_"
            f"{_safe_filename_token(failure_stage)}.mp4"
        )
        self._partial_path.replace(final_path)
        return str(final_path.resolve())

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._partial_path.unlink(missing_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _paper_snapshot_episode_selected(
    specification: str,
    *,
    episode_id: int,
    episode_position: int,
) -> bool:
    value = str(specification).strip().lower()
    if value in {"", "first"}:
        return int(episode_position) == 0
    if value == "all":
        return True
    try:
        selected = {int(token.strip()) for token in value.split(",") if token.strip()}
    except ValueError as exc:
        raise ValueError(
            "--paper-snapshot-episodes must be first, all, or comma-separated "
            "integer episode ids"
        ) from exc
    if not selected:
        raise ValueError("--paper-snapshot-episodes selected no episodes")
    return int(episode_id) in selected


def _paper_snapshot_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    array = np.asarray(value)
    if array.size == 1:
        scalar = array.reshape(-1)[0]
        return scalar.item() if hasattr(scalar, "item") else scalar
    return array.tolist()


def _parse_paper_offset_probe_action(
    value: str, *, arm_dim: int
) -> tuple[float, ...] | None:
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        values = tuple(float(token.strip()) for token in text_value.split(","))
    except ValueError as exc:
        raise ValueError(
            "--paper-offset-probe-action must contain comma-separated numbers"
        ) from exc
    if len(values) != int(arm_dim):
        raise ValueError(
            "--paper-offset-probe-action must contain exactly "
            f"{int(arm_dim)} arm values; received {len(values)}"
        )
    norm = float(np.linalg.norm(np.asarray(values, dtype=np.float64)))
    if norm <= 0.0 or norm > 0.30 + 1e-6:
        raise ValueError(
            "paper offset probe norm must be in (0, 0.30] normalized action units"
        )
    return values


def _attach_paper_offset_probe(
    *,
    policy,
    action_dict: dict[str, Any],
    obs_dict: dict[str, Any],
    normalized_arm_offset: tuple[float, ...],
) -> None:
    """Rerun post-AQR on a fixed whole-horizon offset without executing it."""

    import torch

    diagnostics = action_dict.get("aqr_diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise RuntimeError("offset probe requires AQR diagnostics")
    base_action = diagnostics.get("post_base_action")
    if base_action is None or not torch.is_tensor(base_action):
        raise RuntimeError("offset probe requires post_base_action diagnostics")
    if getattr(policy, "post_refiner", None) is None:
        raise RuntimeError("offset probe requires post-diffusion AQR")

    with torch.no_grad():
        encoded = policy._encode_observation(obs_dict)
        clean_base = base_action.detach().clone()
        clean_applied_residual = diagnostics.get("post_applied_residual")
        reference = (
            clean_base + clean_applied_residual.detach()
            if torch.is_tensor(clean_applied_residual)
            else clean_base
        )
        candidate = clean_base.clone()
        slots = policy._query_slot_indices.to(candidate.device)
        arm_dim = int(policy.aqr.arm_dim)
        offset = torch.as_tensor(
            normalized_arm_offset,
            device=candidate.device,
            dtype=candidate.dtype,
        )
        selected = candidate.index_select(1, slots).clone()
        selected[..., :arm_dim] = (
            selected[..., :arm_dim] + offset.view(1, 1, arm_dim)
        ).clamp(-1.0, 1.0)
        candidate[:, slots] = selected
        corrected, probe_diagnostics = policy._post_diffusion_refine(
            base_action=candidate,
            encoded=encoded,
        )
        reference_tool, _, reference_environment = (
            policy._tool_from_normalized_query_actions(reference, encoded)
        )
        candidate_tool, _, candidate_environment = (
            policy._tool_from_normalized_query_actions(candidate, encoded)
        )
        corrected_tool, _, corrected_environment = (
            policy._tool_from_normalized_query_actions(corrected, encoded)
        )

        names = tuple(policy.action_to_tool.names)
        left = next(
            (index for index, name in enumerate(names) if "left_finger" in name),
            0,
        )
        right = next(
            (index for index, name in enumerate(names) if "right_finger" in name),
            left,
        )

        def midpoint(trajectory) -> torch.Tensor:
            points = trajectory.keypoints_world
            return 0.5 * (points[..., left, :] + points[..., right, :])

        reference_midpoint = midpoint(reference_tool)
        candidate_midpoint = midpoint(candidate_tool)
        corrected_midpoint = midpoint(corrected_tool)
        before_distance = torch.linalg.vector_norm(
            candidate_midpoint - reference_midpoint, dim=-1
        )
        after_distance = torch.linalg.vector_norm(
            corrected_midpoint - reference_midpoint, dim=-1
        )
        improvement = before_distance - after_distance
        diagnostics.update(
            {
                "offset_probe_normalized_arm_offset": offset,
                "offset_probe_clean_base_actions_normalized": clean_base,
                "offset_probe_reference_actions_environment": reference_environment,
                "offset_probe_candidate_actions_environment": candidate_environment,
                "offset_probe_corrected_actions_environment": corrected_environment,
                "offset_probe_reference_tool_keypoints_world": (
                    reference_tool.keypoints_world
                ),
                "offset_probe_candidate_tool_keypoints_world": (
                    candidate_tool.keypoints_world
                ),
                "offset_probe_corrected_tool_keypoints_world": (
                    corrected_tool.keypoints_world
                ),
                "offset_probe_before_distance_m": before_distance,
                "offset_probe_after_distance_m": after_distance,
                "offset_probe_improvement_m": improvement,
                "offset_probe_helpful_fraction": (
                    (after_distance < before_distance).to(candidate.dtype).mean()
                ),
                "offset_probe_post_confidence": probe_diagnostics[
                    "post_confidence"
                ],
                "offset_probe_is_controlled_not_executed": torch.ones(
                    (), device=candidate.device, dtype=torch.bool
                ),
            }
        )


def _save_paper_snapshot(
    *,
    snapshot_dir: Path,
    env,
    policy,
    action_dict: dict[str, Any],
    current_observation: dict[str, np.ndarray],
    scene_diagnostics: dict[str, Any],
    episode_id: int,
    plan_index: int,
    env_step_before_plan: int,
    robot_seg_ids: tuple[int, ...],
    cube_a_seg_ids: tuple[int, ...],
    cube_b_seg_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Persist evaluator-only simulation evidence for publication figures."""

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Paper snapshots require imageio.") from exc

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"episode_{int(episode_id):04d}_plan_{int(plan_index):04d}_"
        f"step_{int(env_step_before_plan):04d}"
    )
    rgb_path = snapshot_dir / f"{stem}.png"
    npz_path = snapshot_dir / f"{stem}.npz"
    imageio.imwrite(rgb_path, _render_rgb_frame(env))

    point_cloud = np.asarray(current_observation["point_cloud"], dtype=np.float32)
    valid_mask = np.asarray(
        current_observation["point_valid_mask"], dtype=bool
    )
    segmentation = current_observation.get("_eval_segmentation")
    if segmentation is None:
        raise RuntimeError(
            "Paper snapshots require sampled simulator segmentation labels."
        )

    diagnostics = action_dict.get("aqr_diagnostics", {})
    arrays: dict[str, np.ndarray] = {
        "point_cloud": point_cloud,
        "point_valid_mask": valid_mask,
        "segmentation": np.asarray(segmentation, dtype=np.int32),
        "agent_pos": np.asarray(current_observation["agent_pos"], dtype=np.float32),
    }
    diagnostic_keys = (
        "current_tool_keypoints_world",
        "current_tool_rotations_world",
        "tool_keypoints_world",
        "base_candidate_tool_keypoints_world",
        "base_candidate_tool_rotations_world",
        "aqr_corrected_tool_keypoints_world",
        "aqr_corrected_tool_rotations_world",
        "base_candidate_actions_environment",
        "aqr_corrected_actions_environment",
        "post_confidence",
        "post_raw_residual_norm",
        "post_bounded_residual_norm",
        "post_applied_residual_norm",
        "query_source_indices",
        "offset_probe_normalized_arm_offset",
        "offset_probe_clean_base_actions_normalized",
        "offset_probe_reference_actions_environment",
        "offset_probe_candidate_actions_environment",
        "offset_probe_corrected_actions_environment",
        "offset_probe_reference_tool_keypoints_world",
        "offset_probe_candidate_tool_keypoints_world",
        "offset_probe_corrected_tool_keypoints_world",
        "offset_probe_before_distance_m",
        "offset_probe_after_distance_m",
        "offset_probe_improvement_m",
        "offset_probe_helpful_fraction",
        "offset_probe_post_confidence",
        "offset_probe_is_controlled_not_executed",
    )
    for key in diagnostic_keys:
        value = _diagnostic_tensor(diagnostics, key)
        if value is not None:
            arrays[key] = np.asarray(value)

    metadata = {
        "format": "contactflow_aqr_simulation_snapshot_v1",
        "task": "StackCube-v1",
        "simulator": "ManiSkill",
        "robot": "Franka Panda",
        "robot_description": "7-DoF arm with parallel-jaw gripper",
        "episode_id": int(episode_id),
        "plan_index": int(plan_index),
        "env_step_before_plan": int(env_step_before_plan),
        "tool_keypoint_names": list(
            getattr(getattr(policy, "action_to_tool", None), "names", ())
        ),
        "robot_seg_ids": [int(value) for value in robot_seg_ids],
        "cubeA_seg_ids": [int(value) for value in cube_a_seg_ids],
        "cubeB_seg_ids": [int(value) for value in cube_b_seg_ids],
        "segmentation_is_evaluator_only": True,
        "segmentation_entered_policy_input": False,
        "rgb_path": rgb_path.name,
        "npz_path": npz_path.name,
        "scene": {
            key: _paper_snapshot_metadata_value(value)
            for key, value in scene_diagnostics.items()
        },
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False)
    )
    np.savez_compressed(npz_path, **arrays)
    metadata["rgb_path"] = str(rgb_path.resolve())
    metadata["npz_path"] = str(npz_path.resolve())
    return metadata


def _attach_paper_post_execution_render(
    *,
    metadata: dict[str, Any],
    snapshot_dir: Path,
    env,
    env_step_after_execution: int,
    executed_slots: int,
    scene_diagnostics: dict[str, Any],
) -> None:
    """Attach the rendered state reached after the corrected execution prefix."""

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Paper snapshots require imageio.") from exc

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"episode_{int(metadata['episode_id']):04d}_"
        f"plan_{int(metadata['plan_index']):04d}_"
        f"after_exec_{int(executed_slots):02d}_"
        f"step_{int(env_step_after_execution):04d}.png"
    )
    path = snapshot_dir / stem
    imageio.imwrite(path, _render_rgb_frame(env))
    metadata["post_execution_rgb_path"] = str(path.resolve())
    metadata["post_execution_env_step"] = int(env_step_after_execution)
    metadata["post_execution_slots"] = int(executed_slots)
    metadata["post_execution_scene"] = {
        key: _paper_snapshot_metadata_value(value)
        for key, value in scene_diagnostics.items()
    }


def _paper_trace_rows(
    policy,
    action_dict: dict[str, Any],
    *,
    episode_id: int,
    plan_index: int,
    env_step_before_plan: int,
) -> list[dict[str, Any]]:
    """Serialize the complete post-AQR horizon without entering policy inputs."""

    diagnostics = action_dict.get("aqr_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return []
    base_actions = _diagnostic_tensor(
        diagnostics, "base_candidate_actions_environment"
    )
    corrected_actions = _diagnostic_tensor(
        diagnostics, "aqr_corrected_actions_environment"
    )
    base_keypoints = _diagnostic_tensor(
        diagnostics, "base_candidate_tool_keypoints_world"
    )
    corrected_keypoints = _diagnostic_tensor(
        diagnostics, "aqr_corrected_tool_keypoints_world"
    )
    query_actions = _diagnostic_tensor(diagnostics, "post_query_action")
    applied_residual = _diagnostic_tensor(diagnostics, "post_applied_residual")
    query_keypoints = _diagnostic_tensor(
        diagnostics, "query_candidate_tool_keypoints_world"
    )
    if base_actions is None or corrected_actions is None:
        return []
    if base_actions.ndim != 3 or corrected_actions.shape != base_actions.shape:
        return []

    query_indices_raw = getattr(policy, "_query_slot_indices", None)
    if query_indices_raw is None:
        query_indices = list(range(base_actions.shape[1]))
    else:
        query_indices = [
            int(value)
            for value in _to_numpy_local(query_indices_raw).reshape(-1)
        ]
    slot_count = min(base_actions.shape[1], len(query_indices))
    keypoint_names = list(
        getattr(getattr(policy, "action_to_tool", None), "names", ())
    )

    def selected_scalar(name: str, position: int, horizon_slot: int) -> float | None:
        values = _diagnostic_tensor(diagnostics, name)
        if values is None or values.size == 0:
            return None
        values = np.asarray(values)
        if values.ndim >= 2 and values.shape[0] == 1:
            values = values[0]
        values = values.reshape(-1)
        index = horizon_slot if horizon_slot < values.size else position
        return float(values[index]) if index < values.size else None

    def batch_slot(values: np.ndarray | None, position: int) -> list[Any] | None:
        if values is None or values.ndim < 2 or values.shape[0] < 1:
            return None
        selected = values[0, position]
        return np.asarray(selected).astype(np.float64).tolist()

    def batch_horizon_slot(
        values: np.ndarray | None, horizon_slot: int
    ) -> list[Any] | None:
        if values is None or values.ndim < 2 or values.shape[0] < 1:
            return None
        if horizon_slot >= values.shape[1]:
            return None
        return np.asarray(values[0, horizon_slot]).astype(np.float64).tolist()

    global_scalars = {
        name: (
            float(np.asarray(value).astype(np.float64).mean())
            if value is not None and np.asarray(value).size
            else None
        )
        for name, value in {
            "relation_gate_mean": _diagnostic_tensor(
                diagnostics, "relation_gate_mean"
            ),
            "anchor_current_weight": _diagnostic_tensor(
                diagnostics, "anchor_current_weight"
            ),
            "anchor_candidate_weight": _diagnostic_tensor(
                diagnostics, "anchor_candidate_weight"
            ),
            "anchor_left_weight": _diagnostic_tensor(
                diagnostics, "anchor_left_weight"
            ),
            "anchor_right_weight": _diagnostic_tensor(
                diagnostics, "anchor_right_weight"
            ),
        }.items()
    }

    rows: list[dict[str, Any]] = []
    for position in range(slot_count):
        horizon_slot = int(query_indices[position])
        base_action = np.asarray(base_actions[0, position], dtype=np.float64)
        corrected_action = np.asarray(
            corrected_actions[0, position], dtype=np.float64
        )
        rows.append(
            {
                "format": "contactflow_aqr_paper_trace_v1",
                "episode_id": int(episode_id),
                "plan_index": int(plan_index),
                "env_step_before_plan": int(env_step_before_plan),
                "executable_slot": int(position + 1),
                "policy_horizon_slot": horizon_slot,
                "tool_keypoint_names": keypoint_names,
                "base_action": base_action.tolist(),
                "corrected_action": corrected_action.tolist(),
                "action_shift": (corrected_action - base_action).tolist(),
                "action_shift_norm": float(
                    np.linalg.norm((corrected_action - base_action)[:7])
                ),
                "base_tool_keypoints_world": batch_slot(
                    base_keypoints, position
                ),
                "corrected_tool_keypoints_world": batch_slot(
                    corrected_keypoints, position
                ),
                "query_action": batch_horizon_slot(
                    query_actions, horizon_slot
                ),
                "applied_residual": batch_horizon_slot(
                    applied_residual, horizon_slot
                ),
                "query_tool_keypoints_world": batch_slot(
                    query_keypoints, position
                ),
                "query_action_override_used": bool(
                    np.asarray(
                        _diagnostic_tensor(
                            diagnostics, "post_query_action_override_used"
                        )
                    ).any()
                )
                if _diagnostic_tensor(
                    diagnostics, "post_query_action_override_used"
                )
                is not None
                else False,
                "post_confidence": selected_scalar(
                    "post_confidence", position, horizon_slot
                ),
                "post_raw_residual_norm": selected_scalar(
                    "post_raw_residual_norm", position, horizon_slot
                ),
                "post_bounded_residual_norm": selected_scalar(
                    "post_bounded_residual_norm", position, horizon_slot
                ),
                "post_applied_residual_norm": selected_scalar(
                    "post_applied_residual_norm", position, horizon_slot
                ),
                **global_scalars,
            }
        )
    return rows


def _load_policy(args: argparse.Namespace):
    """Load either the standalone AQR-DP3 trainer format or a legacy workspace."""
    import torch

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AQR-DP3 checkpoint does not exist: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")

    if isinstance(payload, dict) and payload.get("kind") == "aqr_dp3_training_checkpoint":
        if str(args.checkpoint_policy) == "ema":
            raise ValueError(
                "standalone AQR-DP3 checkpoints contain model weights only; "
                "use --checkpoint-policy auto or model"
            )
        config = payload.get("config")
        if not isinstance(config, dict) or not isinstance(config.get("policy"), dict):
            raise ValueError("standalone AQR-DP3 checkpoint has no valid policy config")
        from omegaconf import OmegaConf

        from contactflow.tools.train_aqr_dp3 import _instantiate

        policy_config = config["policy"]
        policy = _instantiate(policy_config)
        policy.load_state_dict(payload["model_state_dict"], strict=True)
        return policy, OmegaConf.create(policy_config)

    raise ValueError(
        "This clean AQR evaluator accepts only "
        "kind='aqr_dp3_training_checkpoint'; legacy workspace fallback "
        "has been removed."
    )


def evaluate(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import gymnasium as gym
    import torch
    import mani_skill.envs  # noqa: F401

    from contactflow.dp3.aqr_dp3.observation import _parse_crop_bound
    _to_numpy = _to_numpy_local

    if int(args.eval_episodes) <= 0:
        raise ValueError("--eval-episodes must be positive")
    if int(args.max_env_steps) <= 0:
        raise ValueError("--max-env-steps must be positive")
    if int(args.video_fps) <= 0:
        raise ValueError("--video-fps must be positive")
    if int(args.video_stride) <= 0:
        raise ValueError("--video-stride must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} is unavailable")
    source_path = Path(args.source_json).expanduser().resolve()
    metadata = _load_json(source_path)
    env_info = metadata["env_info"]
    env_kwargs = dict(env_info.get("env_kwargs", {}))
    control_mode = env_kwargs.get("control_mode")
    if control_mode != "pd_joint_delta_pos":
        raise ValueError(
            "AQR-DP3 evaluator requires pd_joint_delta_pos; found "
            f"{control_mode!r}."
        )
    env_kwargs.update(
        obs_mode="pointcloud",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        sim_backend="physx_cpu",
        render_backend="cpu",
        num_envs=1,
    )
    paper_snapshot_enabled = bool(str(args.paper_snapshot_dir).strip())
    if int(args.paper_snapshot_plan_stride) < 1:
        raise ValueError("--paper-snapshot-plan-stride must be at least 1")
    diagnostic_segmentation_enabled = bool(
        args.diagnostic_segmentation or paper_snapshot_enabled
    )
    if str(args.video_mode) != "off" or paper_snapshot_enabled:
        env_kwargs["render_mode"] = "rgb_array"
    else:
        # Source demonstrations were recorded with render_mode=rgb_array. Do
        # not inherit that presentation renderer for metric-only evaluation;
        # pointcloud observations still initialize their required sensors, but
        # the extra RGB render target/fence chain is avoided.
        env_kwargs.pop("render_mode", None)
    episodes = metadata.get("episodes", [])[
        int(args.eval_start) : int(args.eval_start) + int(args.eval_episodes)
    ]
    if not episodes:
        raise ValueError("the requested evaluation episode slice is empty")

    policy, cfg = _load_policy(args)
    policy.to(device)
    policy.eval()
    tool_diagnostics_enabled = bool(
        args.diagnostic_tool_geometry
        or diagnostic_segmentation_enabled
        or str(args.paper_trace_out).strip()
        or paper_snapshot_enabled
    )
    if hasattr(policy, "_evaluation_tool_diagnostics"):
        policy._evaluation_tool_diagnostics = tool_diagnostics_enabled
    aqr = getattr(policy, "aqr", None)
    if aqr is None:
        raise TypeError("checkpoint policy is not an AQR-DP3 policy")
    if bool(args.disable_post_refiner):
        if (
            str(aqr.refinement_stage) != "post_diffusion"
            or getattr(policy, "post_refiner", None) is None
        ):
            raise ValueError(
                "--disable-post-refiner requires a post-diffusion AQR checkpoint."
            )
        policy._disable_post_refiner_for_evaluation = True
    paper_offset_probe_action = _parse_paper_offset_probe_action(
        args.paper_offset_probe_action,
        arm_dim=int(aqr.arm_dim),
    )
    if paper_offset_probe_action is not None and not paper_snapshot_enabled:
        raise ValueError(
            "--paper-offset-probe-action requires --paper-snapshot-dir"
        )
    relative_query = getattr(policy, "relative_query", None)
    checkpoint_relation_features = bool(
        getattr(relative_query, "relation_features", False)
    )
    if bool(args.disable_relation_features):
        if not checkpoint_relation_features:
            raise ValueError(
                "--disable-relation-features requires a checkpoint whose "
                "relation branch is enabled."
            )
        relative_query.relation_features = False
    effective_relation_features = bool(
        getattr(relative_query, "relation_features", False)
    )
    task_profile = str(getattr(aqr, "task_profile", "")).strip().lower()
    env_id = _validate_task_environment(
        task_profile,
        str(env_info.get("env_id", "")),
    )
    _, locked_crop_min, locked_crop_max = _task_contract(task_profile)
    point_feature_dim = _policy_point_feature_dim(cfg)
    if int(getattr(policy, "state_dim", -1)) != 16:
        raise ValueError("AQR-DP3 online state must be the deployable 16-D qpos+TCP")
    if not 1 <= int(args.exec_steps) <= int(aqr.exec_steps):
        raise ValueError(
            "runtime execution prefix must lie within the checkpoint chunk: "
            f"requested={args.exec_steps}, checkpoint_max={aqr.exec_steps}"
        )
    if int(args.num_inference_steps) > 0:
        policy.num_inference_steps = int(args.num_inference_steps)
    query_points = (
        int(aqr.front_end.p_query)
        if int(args.query_points) == 0
        else int(args.query_points)
    )
    if query_points != int(aqr.front_end.p_query):
        raise ValueError(
            "live point budget must match the checkpoint: "
            f"{query_points} != {aqr.front_end.p_query}"
        )
    n_obs_steps = int(cfg.n_obs_steps)
    if n_obs_steps < 2:
        raise ValueError("joint-delta AQR tracking requires at least two observations")
    crop_min = _parse_crop_bound(args.crop_min, "--crop-min")
    crop_max = _parse_crop_bound(args.crop_max, "--crop-max")
    crop_min, crop_max = validate_workspace_bounds(
        crop_min,
        crop_max,
        allow_none=False,
    )
    checkpoint_crop_min, checkpoint_crop_max = validate_workspace_bounds(
        aqr.front_end.workspace_crop_min,
        aqr.front_end.workspace_crop_max,
        allow_none=False,
    )
    if (crop_min, crop_max) != (locked_crop_min, locked_crop_max):
        raise ValueError(
            f"{task_profile} live evaluation must use its locked workspace bounds."
        )
    if (checkpoint_crop_min, checkpoint_crop_max) != (crop_min, crop_max):
        raise ValueError(
            "checkpoint PointCloudFrontEnd and live observation workspace bounds "
            "must match exactly."
        )

    env = gym.make(env_id, **env_kwargs)
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    paper_trace_rows: list[dict[str, Any]] = []
    paper_snapshot_rows: list[dict[str, Any]] = []
    all_latency: list[float] = []
    slot_errors: dict[int, list[float]] = {
        slot: [] for slot in range(1, int(args.exec_steps) + 1)
    }
    requested_robot_ids = _parse_robot_ids(args.robot_seg_ids)
    requested_cube_a_ids = _parse_robot_ids(
        args.diagnostic_cube_a_seg_ids
    )
    requested_cube_b_ids = _parse_robot_ids(
        args.diagnostic_cube_b_seg_ids
    )
    if (
        args.robot_mask_source == "segmentation"
        and not bool(args.allow_privileged_robot_mask)
    ):
        raise ValueError(
            "--robot-mask-source segmentation is privileged and requires "
            "--allow-privileged-robot-mask. This mode is diagnostic only."
        )
    resolved_robot_ids: tuple[int, ...] | None = (
        requested_robot_ids
        if args.robot_mask_source == "segmentation"
        else ()
    )
    resolved_diagnostic_robot_ids: tuple[int, ...] = ()
    resolved_cube_a_ids: tuple[int, ...] = ()
    resolved_cube_b_ids: tuple[int, ...] = ()
    active_video: _EpisodeVideoRecorder | None = None
    video_paths: list[str] = []
    try:
        for episode_position, episode in enumerate(episodes):
            episode_id = int(episode["episode_id"])
            episode_seed = int(episode["episode_seed"])
            policy_seed = (int(args.seed) + episode_id) % (2**31)
            torch.manual_seed(policy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(policy_seed)
            rng = np.random.default_rng(policy_seed)
            obs, info = env.reset(
                seed=episode_seed,
                options=episode.get("reset_kwargs", {}).get("options", {}),
            )
            robot_seg_ids = None
            if args.robot_mask_source == "segmentation":
                robot_seg_ids = (
                    _robot_ids_from_env(env)
                    if requested_robot_ids is None
                    else requested_robot_ids
                )
            resolved_robot_ids = robot_seg_ids
            if diagnostic_segmentation_enabled:
                resolved_diagnostic_robot_ids = _robot_ids_from_env(env)
                resolved_cube_a_ids = (
                    _actor_ids_from_env(env, ("cubeA", "cube_a"))
                    if requested_cube_a_ids is None
                    else requested_cube_a_ids
                )
                resolved_cube_b_ids = (
                    _actor_ids_from_env(env, ("cubeB", "cube_b"))
                    if requested_cube_b_ids is None
                    else requested_cube_b_ids
                )
            if hasattr(policy, "reset"):
                policy.reset()
            first = _dense_observation(
                obs,
                query_points=query_points,
                point_feature_dim=point_feature_dim,
                rng=rng,
                robot_seg_ids=robot_seg_ids,
                robot_mask_source=args.robot_mask_source,
                point_sampling_mode=args.point_sampling_mode,
                crop_min=crop_min,
                crop_max=crop_max,
                diagnostic_segmentation=diagnostic_segmentation_enabled,
            )
            history = deque([first] * n_obs_steps, maxlen=n_obs_steps)
            initial_diagnostics = (
                _stackcube_step_diagnostics(
                    obs,
                    info,
                    env,
                    initial_cube_a_z=None,
                )
                if task_profile == "stack_cube"
                else {}
            )
            initial_cube_a_z = initial_diagnostics.get("eval_only_cubeA_z")
            stackcube_state = _new_stackcube_episode_state()
            active_video = _EpisodeVideoRecorder(
                enabled=str(args.video_mode) != "off",
                video_dir=Path(args.video_dir).expanduser(),
                episode_id=episode_id,
                fps=int(args.video_fps),
                stride=int(args.video_stride),
            )
            active_video.append(env, env_step=0, force=True)
            success = False
            end_reason = "max_env_steps"
            env_steps = 0
            plan_count = 0
            episode_latency: list[float] = []
            episode_slot_errors: dict[int, list[float]] = {
                slot: [] for slot in range(1, int(args.exec_steps) + 1)
            }
            latest_diag: dict[str, float | None] = {}
            while env_steps < int(args.max_env_steps) and not success:
                obs_dict = _history_input(history, device)
                _sync(device)
                start = time.perf_counter()
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                _sync(device)
                latency_ms = 1000.0 * (time.perf_counter() - start)
                episode_latency.append(latency_ms)
                all_latency.append(latency_ms)
                latest_diag = _aqr_diagnostics(
                    action_dict.get("aqr_diagnostics", {})
                )
                plan_tool_diagnostics = (
                    _action_tool_diagnostics(
                        policy,
                        action_dict,
                        obs_dict,
                        obs,
                        env,
                        action_low=action_low,
                        action_high=action_high,
                    )
                    if task_profile == "stack_cube"
                    and tool_diagnostics_enabled
                    else {}
                )
                plan_query_semantics = (
                    _query_semantic_diagnostics(
                        policy,
                        action_dict,
                        history[-1],
                        robot_seg_ids=resolved_diagnostic_robot_ids,
                        cube_a_seg_ids=resolved_cube_a_ids,
                        cube_b_seg_ids=resolved_cube_b_ids,
                    )
                    if diagnostic_segmentation_enabled
                    else {}
                )
                commanded_tcp = _commanded_tcp_positions(
                    policy,
                    action_dict,
                    obs_dict,
                    action_low=action_low,
                    action_high=action_high,
                )
                actions = _to_numpy(action_dict["action"])
                if actions.ndim == 3 and actions.shape[0] == 1:
                    actions = actions[0]
                actions = np.asarray(actions, dtype=np.float32)
                if actions.ndim != 2:
                    raise ValueError(
                        f"policy action must be [slots,A], got {actions.shape}"
                    )
                if actions.shape[0] < int(args.exec_steps):
                    raise ValueError(
                        "policy returned fewer action slots than --exec-steps"
                    )
                if actions.shape[1] != action_low.size:
                    raise ValueError(
                        "policy and environment action dimensions disagree: "
                        f"{actions.shape[1]} != {action_low.size}"
                    )
                plan_count += 1
                snapshot_metadata: dict[str, Any] | None = None
                save_paper_snapshot = bool(
                    paper_snapshot_enabled
                    and _paper_snapshot_episode_selected(
                        args.paper_snapshot_episodes,
                        episode_id=episode_id,
                        episode_position=episode_position,
                    )
                    and (plan_count - 1) % int(args.paper_snapshot_plan_stride) == 0
                )
                if save_paper_snapshot:
                    snapshot_scene_diagnostics = (
                        _stackcube_step_diagnostics(
                            obs,
                            info,
                            env,
                            initial_cube_a_z=initial_cube_a_z,
                        )
                        if task_profile == "stack_cube"
                        else {}
                    )
                    snapshot_phase = str(
                        snapshot_scene_diagnostics.get(
                            "eval_only_stackcube_phase", ""
                        )
                    )
                    if (
                        paper_offset_probe_action is not None
                        and snapshot_phase in {"pre_grasp", "grasp", "lift"}
                    ):
                        _attach_paper_offset_probe(
                            policy=policy,
                            action_dict=action_dict,
                            obs_dict=obs_dict,
                            normalized_arm_offset=paper_offset_probe_action,
                        )
                    snapshot_metadata = _save_paper_snapshot(
                        snapshot_dir=Path(args.paper_snapshot_dir).expanduser(),
                        env=env,
                        policy=policy,
                        action_dict=action_dict,
                        current_observation=history[-1],
                        scene_diagnostics=snapshot_scene_diagnostics,
                        episode_id=episode_id,
                        plan_index=plan_count,
                        env_step_before_plan=env_steps,
                        robot_seg_ids=resolved_diagnostic_robot_ids,
                        cube_a_seg_ids=resolved_cube_a_ids,
                        cube_b_seg_ids=resolved_cube_b_ids,
                    )
                    paper_snapshot_rows.append(snapshot_metadata)
                if str(args.paper_trace_out).strip():
                    paper_trace_rows.extend(
                        _paper_trace_rows(
                            policy,
                            action_dict,
                            episode_id=episode_id,
                            plan_index=plan_count,
                            env_step_before_plan=env_steps,
                        )
                    )
                executed_slots = 0
                for slot_index in range(int(args.exec_steps)):
                    if env_steps >= int(args.max_env_steps):
                        break
                    action = np.clip(
                        actions[slot_index], action_low, action_high
                    ).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    env_steps += 1
                    executed_slots += 1
                    observed = _dense_observation(
                        obs,
                        query_points=query_points,
                        point_feature_dim=point_feature_dim,
                        rng=rng,
                        robot_seg_ids=robot_seg_ids,
                        robot_mask_source=args.robot_mask_source,
                        point_sampling_mode=args.point_sampling_mode,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        diagnostic_segmentation=diagnostic_segmentation_enabled,
                    )
                    history.append(observed)
                    realized_tcp = observed["agent_pos"][9:12].copy()
                    commanded = (
                        commanded_tcp[slot_index]
                        if commanded_tcp is not None
                        and slot_index < len(commanded_tcp)
                        else None
                    )
                    tracking_error = (
                        float(np.linalg.norm(realized_tcp - commanded))
                        if commanded is not None
                        else None
                    )
                    slot_number = slot_index + 1
                    if tracking_error is not None:
                        slot_errors[slot_number].append(tracking_error)
                        episode_slot_errors[slot_number].append(tracking_error)
                    step_success = _to_bool(info.get("success", False))
                    terminated_flag = _to_bool(terminated)
                    truncated_flag = _to_bool(truncated)
                    env_done = _environment_done(
                        terminated=terminated_flag,
                        truncated=truncated_flag,
                        ignore_truncated=bool(args.ignore_truncated),
                    )
                    stackcube_diagnostics = (
                        _stackcube_step_diagnostics(
                            obs,
                            info,
                            env,
                            initial_cube_a_z=initial_cube_a_z,
                        )
                        if task_profile == "stack_cube"
                        else {}
                    )
                    if task_profile == "stack_cube":
                        _update_stackcube_episode_state(
                            stackcube_state,
                            stackcube_diagnostics,
                            env_step=env_steps,
                        )
                    active_video.append(
                        env,
                        env_step=env_steps,
                        force=bool(step_success or env_done),
                    )
                    step_rows.append(
                        {
                            "episode_id": episode_id,
                            "episode_seed": episode_seed,
                            "policy_seed": policy_seed,
                            "env_step": env_steps,
                            "plan_index": plan_count,
                            "slot": slot_number,
                            "exec_steps": int(args.exec_steps),
                            "reward": _to_scalar(reward),
                            "success": step_success,
                            "terminated": terminated_flag,
                            "truncated": truncated_flag,
                            "policy_latency_ms": latency_ms,
                            "commanded_tcp_x": (
                                float(commanded[0]) if commanded is not None else None
                            ),
                            "commanded_tcp_y": (
                                float(commanded[1]) if commanded is not None else None
                            ),
                            "commanded_tcp_z": (
                                float(commanded[2]) if commanded is not None else None
                            ),
                            "realized_tcp_x": float(realized_tcp[0]),
                            "realized_tcp_y": float(realized_tcp[1]),
                            "realized_tcp_z": float(realized_tcp[2]),
                            "commanded_realized_tcp_error_m": tracking_error,
                            **{
                                f"executed_action_{index}": float(value)
                                for index, value in enumerate(action)
                            },
                            "diagnostics_are_evaluation_only": bool(
                                task_profile == "stack_cube"
                            ),
                            **plan_tool_diagnostics,
                            **plan_query_semantics,
                            **stackcube_diagnostics,
                            **latest_diag,
                        }
                    )
                    if step_success:
                        success = True
                        end_reason = "success"
                        break
                    if env_done:
                        end_reason = (
                            "terminated"
                            if terminated_flag
                            else "truncated"
                        )
                        break
                if snapshot_metadata is not None:
                    post_execution_scene = (
                        _stackcube_step_diagnostics(
                            obs,
                            info,
                            env,
                            initial_cube_a_z=initial_cube_a_z,
                        )
                        if task_profile == "stack_cube"
                        else {}
                    )
                    _attach_paper_post_execution_render(
                        metadata=snapshot_metadata,
                        snapshot_dir=Path(args.paper_snapshot_dir).expanduser(),
                        env=env,
                        env_step_after_execution=env_steps,
                        executed_slots=executed_slots,
                        scene_diagnostics=post_execution_scene,
                    )
                if end_reason in {"terminated", "truncated"}:
                    break

            failure_stage = (
                _stackcube_failure_stage(stackcube_state, success=success)
                if task_profile == "stack_cube"
                else ("success" if success else end_reason)
            )
            keep_video = str(args.video_mode) == "all" or (
                str(args.video_mode) == "failures" and not success
            )
            video_path = active_video.finalize(
                keep=keep_video,
                success=success,
                failure_stage=failure_stage,
            )
            active_video = None
            if video_path is not None:
                video_paths.append(video_path)
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "episode_seed": episode_seed,
                    "policy_seed": policy_seed,
                    "success": success,
                    "env_steps": env_steps,
                    "plan_count": plan_count,
                    "end_reason": end_reason,
                    "failure_stage": failure_stage,
                    "video_path": video_path,
                    "diagnostics_are_evaluation_only": bool(
                        task_profile == "stack_cube"
                    ),
                    "latency_mean_ms": (
                        float(np.mean(episode_latency))
                        if episode_latency
                        else None
                    ),
                    "latency_p95_ms": (
                        float(np.quantile(episode_latency, 0.95))
                        if episode_latency
                        else None
                    ),
                    **{
                        f"slot{slot}_tcp_error_mean_m": (
                            float(np.mean(values)) if values else None
                        )
                        for slot, values in episode_slot_errors.items()
                    },
                    **(
                        {
                            **stackcube_state,
                            "max_phase_name": STACKCUBE_PHASE_NAMES[
                                int(stackcube_state["max_phase_id"])
                            ],
                        }
                        if task_profile == "stack_cube"
                        else {}
                    ),
                    **latest_diag,
                }
            )
            print(
                f"evaluated {episode_position + 1}/{len(episodes)} "
                f"episodes: success={success}, stage={failure_stage}"
            )
    finally:
        if active_video is not None:
            active_video.abort()
        env.close()

    successes = sum(bool(row["success"]) for row in episode_rows)
    latency_summary = _quantiles(all_latency)
    tracking = {
        f"slot{slot}_commanded_realized_tcp_error_m": _quantiles(values)
        for slot, values in slot_errors.items()
        if slot <= int(args.exec_steps)
    }
    report = {
        "format": "contactflow_aqr_dp3_maniskill_evaluation_v1",
        "status": "PASS",
        "task": env_id,
        "task_profile": task_profile,
        "obs_mode": "pointcloud",
        "control_mode": "pd_joint_delta_pos",
        "execution_mode": "fixed_prefix",
        "exec_steps": int(args.exec_steps),
        "checkpoint_max_exec_steps": int(aqr.exec_steps),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_policy": str(args.checkpoint_policy),
        "experiment": str(aqr.experiment),
        "refinement_stage": str(aqr.refinement_stage),
        "post_base_frozen": bool(aqr.post_base_frozen),
        "post_refiner_input_mode": str(aqr.post_refiner_input_mode),
        "post_refiner_disabled": bool(args.disable_post_refiner),
        "effective_policy": (
            "pure_dp3_h12"
            if bool(args.disable_post_refiner)
            else str(aqr.post_refiner_input_mode)
        ),
        "observation_encoder_class": type(policy.obs_encoder).__name__,
        "relation_fusion_mode": str(aqr.relation_fusion_mode),
        "anchor_preserving_query": bool(aqr.anchor_preserving_query),
        "relation_gate_maximum": float(aqr.relation_gate_maximum),
        "bounded_action_residual": bool(aqr.bounded_action_residual),
        "action_residual_max_norm": float(aqr.action_residual_max_norm),
        "action_residual_min_progress_ratio": float(
            aqr.action_residual_min_progress_ratio
        ),
        "action_residual_protect_non_arm": bool(
            aqr.action_residual_protect_non_arm
        ),
        "improvement_loss_enabled": bool(aqr.improvement_loss_enabled),
        "improvement_loss_weight": float(aqr.improvement_loss_weight),
        "improvement_margin_ratio": float(aqr.improvement_margin_ratio),
        "improvement_loss_clean_only": bool(
            aqr.improvement_loss_clean_only
        ),
        "checkpoint_relation_features": checkpoint_relation_features,
        "effective_relation_features": effective_relation_features,
        "relation_features_disabled_for_ablation": bool(
            args.disable_relation_features
        ),
        "query_source": str(aqr.query_source),
        "local_intervention": str(aqr.local_intervention),
        "query_points": query_points,
        "point_feature_dim": point_feature_dim,
        "point_features": "xyzrgb" if point_feature_dim == 6 else "xyz",
        "workspace_crop_min": list(crop_min),
        "workspace_crop_max": list(crop_max),
        "workspace_filter_applied_before_sampling": True,
        "state_contract": "qpos9_plus_tcp_pose7",
        "robot_mask_source": str(args.robot_mask_source),
        "robot_mask_feature_enabled": bool(aqr.use_robot_mask_feature),
        "deployment_input_safe": args.robot_mask_source == "off",
        "privileged_input_warning": (
            None
            if args.robot_mask_source == "off"
            else (
                "Simulator segmentation entered the policy input. This run is "
                "an offline privileged diagnostic, not a deployment result."
            )
        ),
        "robot_seg_ids": list(resolved_robot_ids or ()),
        "diagnostic_segmentation_enabled": diagnostic_segmentation_enabled,
        "diagnostic_tool_geometry_enabled": tool_diagnostics_enabled,
        "diagnostic_segmentation_entered_policy_input": False,
        "diagnostic_robot_seg_ids": list(
            resolved_diagnostic_robot_ids
        ),
        "diagnostic_cubeA_seg_ids": list(resolved_cube_a_ids),
        "diagnostic_cubeB_seg_ids": list(resolved_cube_b_ids),
        "eval_start": int(args.eval_start),
        "eval_episodes": len(episode_rows),
        "successes": int(successes),
        "success_rate": (
            float(successes) / float(len(episode_rows))
            if episode_rows
            else 0.0
        ),
        "success_rate_pct": (
            100.0 * float(successes) / float(len(episode_rows))
            if episode_rows
            else 0.0
        ),
        "max_env_steps": int(args.max_env_steps),
        "ignore_truncated": bool(args.ignore_truncated),
        "num_inference_steps": int(policy.num_inference_steps),
        "seed": int(args.seed),
        "latency_ms": latency_summary,
        "commanded_realized_tcp_tracking": tracking,
        "aqr_diagnostics": {
            key: _mean(step_rows, key)
            for key in sorted(
                {key for row in step_rows for key in row if key.startswith("aqr_")}
            )
        },
        "tool_candidate_diagnostics": {
            key: _mean(step_rows, key)
            for key in (
                "eval_only_aqr_midpoint_shift_m",
                "eval_only_base_candidate_cubeA_closing_axis_offset_m",
                (
                    "eval_only_aqr_corrected_candidate_"
                    "cubeA_closing_axis_offset_m"
                ),
                "eval_only_final_candidate_cubeA_closing_axis_offset_m",
                "eval_only_current_model_midpoint_tcp_offset_m",
            )
        },
        "query_semantic_diagnostics": {
            key: _mean(step_rows, key)
            for key in (
                "eval_only_query_cubeA_fraction",
                "eval_only_query_cubeB_fraction",
                "eval_only_query_robot_fraction",
                "eval_only_query_other_fraction",
                "eval_only_query_semantic_neighbor_count",
            )
        },
        "tool_keypoint_offsets_tcp": {
            key: list(value)
            for key, value in aqr.keypoint_offsets_tcp.items()
        },
        "stable_grasp_diagnostic_definition": {
            "minimum_consecutive_grasp_steps": 10,
            "secure_lift_height_m": 0.02,
            "note": (
                "Evaluation-only heuristic; not a policy input or training "
                "ground-truth label."
            ),
        },
        "failure_stage_counts": dict(
            sorted(Counter(row["failure_stage"] for row in episode_rows).items())
        ),
        "evaluation_only_privileged_diagnostics": {
            "enabled": task_profile == "stack_cube",
            "entered_policy_input": False,
            "fields": (
                [
                    "is_cubeA_grasped",
                    "is_cubeA_on_cubeB",
                    "is_cubeA_static",
                    "cubeA_pose",
                    "cubeB_pose",
                    "geometry_derived_phase",
                    "model_tool_keypoints",
                    "simulator_finger_link_origins",
                    "base_vs_aqr_candidate_tool_shift",
                    "query_neighbor_segmentation_labels",
                ]
                if task_profile == "stack_cube"
                else []
            ),
        },
        "video": {
            "mode": str(args.video_mode),
            "fps": int(args.video_fps),
            "stride": int(args.video_stride),
            "directory": str(Path(args.video_dir).expanduser().resolve()),
            "saved_count": len(video_paths),
            "paths": video_paths,
        },
        "source_json": str(source_path),
        "episode_csv": str(Path(args.csv_out).expanduser().resolve()),
        "step_csv": str(Path(args.step_csv_out).expanduser().resolve()),
        "paper_trace_jsonl": (
            str(Path(args.paper_trace_out).expanduser().resolve())
            if str(args.paper_trace_out).strip()
            else None
        ),
        "paper_snapshots": {
            "enabled": paper_snapshot_enabled,
            "directory": (
                str(Path(args.paper_snapshot_dir).expanduser().resolve())
                if paper_snapshot_enabled
                else None
            ),
            "episode_selection": str(args.paper_snapshot_episodes),
            "plan_stride": int(args.paper_snapshot_plan_stride),
            "saved_count": len(paper_snapshot_rows),
            "manifest": (
                str(
                    (
                        Path(args.paper_snapshot_dir).expanduser()
                        / "manifest.jsonl"
                    ).resolve()
                )
                if paper_snapshot_enabled
                else None
            ),
            "segmentation_entered_policy_input": False,
            "controlled_offset_probe_action": (
                list(paper_offset_probe_action)
                if paper_offset_probe_action is not None
                else None
            ),
            "controlled_offset_probe_executed": False,
        },
    }
    if str(args.paper_trace_out).strip():
        _write_jsonl(Path(args.paper_trace_out).expanduser(), paper_trace_rows)
    if paper_snapshot_enabled:
        _write_jsonl(
            Path(args.paper_snapshot_dir).expanduser() / "manifest.jsonl",
            paper_snapshot_rows,
        )
    return report, episode_rows, step_rows


def main() -> int:
    args = parse_args()
    try:
        report, episode_rows, step_rows = evaluate(args)
    except Exception as exc:
        report = {
            "format": "contactflow_aqr_dp3_maniskill_evaluation_v1",
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "detail": str(exc),
        }
        episode_rows = []
        step_rows = []
    report_path = Path(args.report_out).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(Path(args.csv_out).expanduser(), episode_rows)
    _write_csv(Path(args.step_csv_out).expanduser(), step_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

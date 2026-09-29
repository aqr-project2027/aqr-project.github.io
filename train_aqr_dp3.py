from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "a6_post.yaml"
EXPERIMENTS = (
    "AQR_OFF",
    "P0",
    "P1",
    "P2",
    "P3",
    "B0",
    "B1",
    "S0",
    "S1",
    "A0",
    "A1",
    "A1_PROTECTED",
    "A1_QUERY3",
    "Q0",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7_TCP",
    "Q7_FINGERS",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6_POST",
)


def _bool_value(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected true/false, yes/no, on/off, or 1/0; got {value!r}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, train, or run an offline held-out test for the "
            "deployment-safe AQR-DP3 policy."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("validate", "train", "test"),
        default="validate",
        help="Execution mode. The wrapper scripts select this automatically.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="AQR-DP3 YAML config or a small overlay with base_config.",
    )
    parser.add_argument(
        "--validation-stage",
        choices=("config", "data", "model", "all"),
        default="all",
        help="For --mode validate, stop after the selected contract stage.",
    )
    parser.add_argument(
        "--smoke-batches",
        type=int,
        default=1,
        help="Model-stage batches used for a no-update forward/loss smoke test.",
    )

    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--exec-steps", type=int, choices=tuple(range(1, 9)))
    parser.add_argument("--query-points", type=int, choices=(4096, 8192))
    parser.add_argument("--raw-points", type=int, choices=(4096, 8192))
    parser.add_argument("--future-aux", type=_bool_value)

    parser.add_argument("--zarr-path")
    parser.add_argument("--action-contract")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--report-out")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "cuda:0", "cuda:1"),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-validation-steps", type=int)
    parser.add_argument("--progress", type=_bool_value)
    parser.add_argument("--progress-update-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "phase_balanced"),
    )
    parser.add_argument("--expert-events-csv")
    parser.add_argument(
        "--initialize-from",
        help=(
            "Load model weights from an AQR-DP3 training checkpoint while "
            "starting a fresh optimizer. This is distinct from --resume."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer/model state from the selected checkpoint.",
    )
    return parser.parse_args(argv)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config(
    path: Path,
    *,
    seen: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    import yaml

    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    if resolved in seen:
        chain = " -> ".join(str(item) for item in (*seen, resolved))
        raise ValueError(f"cyclic base_config chain: {chain}")
    if not resolved.is_file():
        raise FileNotFoundError(f"AQR-DP3 config does not exist: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"AQR-DP3 config must be a YAML mapping: {resolved}")
    overlay = dict(raw)
    base_reference = overlay.pop("base_config", None)
    if base_reference is None:
        return overlay
    base_path = Path(str(base_reference))
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    base = _load_config(base_path, seen=seen | {resolved})
    return _deep_merge(base, overlay)


def _nested(config: Mapping[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError("missing config field: " + ".".join(keys))
        current = current[key]
    return current


def _set_point_budget(config: dict[str, Any], query_points: int) -> None:
    points = int(query_points)
    channels = int(
        config["shape_meta"]["obs"]["point_cloud"]["shape"][-1]
    )
    config["shape_meta"]["obs"]["point_cloud"]["shape"] = [points, channels]
    config["policy"]["shape_meta"]["obs"]["point_cloud"]["shape"] = [
        points,
        channels,
    ]
    config["policy"]["aqr_config"]["front_end"]["query_points"] = points
    config["dataset"]["query_points"] = points


def _set_raw_point_budget(config: dict[str, Any], raw_points: int) -> None:
    points = int(raw_points)
    channels = int(
        config["shape_meta"]["obs"]["point_cloud"]["shape"][-1]
    )
    config["shape_meta"]["obs"]["point_cloud"]["shape"] = [points, channels]
    config["policy"]["shape_meta"]["obs"]["point_cloud"]["shape"] = [
        points,
        channels,
    ]


def _apply_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.experiment is not None:
        cfg["policy"]["aqr_config"]["experiment"] = args.experiment
        cfg["name"] = f"aqr_dp3_{args.experiment.lower()}"
        if args.experiment == "A1_PROTECTED":
            cfg["policy"]["aqr_config"]["front_end"]["mode"] = "dual_bank"
        elif args.experiment in {"A1", "A1_QUERY3", "AQR_OFF"}:
            cfg["policy"]["aqr_config"]["front_end"]["mode"] = (
                "baseline_fps2048"
            )
        cfg["policy"]["aqr_config"]["query_update_fractions"] = (
            [0.55, 0.75, 0.90]
            if args.experiment == "A1_QUERY3"
            else [0.75]
        )
    if args.exec_steps is not None:
        cfg["policy"]["aqr_config"]["exec_steps"] = int(args.exec_steps)
    if args.query_points is not None:
        _set_point_budget(cfg, int(args.query_points))
    if args.raw_points is not None:
        _set_raw_point_budget(cfg, int(args.raw_points))
    if args.future_aux is not None:
        cfg["policy"]["aqr_config"]["future_aux"] = bool(args.future_aux)
        if bool(args.future_aux):
            # With horizon=4 and n_obs_steps=2, current+1 and current+2 are
            # available. The longer +4 target requires horizon >= 6.
            cfg["policy"]["aqr_config"]["future_horizons"] = [1, 2]
    if args.zarr_path is not None:
        cfg["dataset"]["zarr_path"] = args.zarr_path
    if args.action_contract is not None:
        cfg["policy"]["aqr_config"]["action_contract_path"] = args.action_contract
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.checkpoint is not None:
        cfg["evaluation"]["checkpoint"] = args.checkpoint
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["dataloader"]["batch_size"] = int(args.batch_size)
        cfg["validation_dataloader"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["dataloader"]["num_workers"] = int(args.num_workers)
        cfg["validation_dataloader"]["num_workers"] = int(args.num_workers)
    if args.max_train_steps is not None:
        cfg["training"]["max_train_steps"] = int(args.max_train_steps)
    if args.max_validation_steps is not None:
        cfg["training"]["max_validation_steps"] = int(
            args.max_validation_steps
        )
    if args.progress is not None:
        cfg["training"]["progress"] = bool(args.progress)
    if args.progress_update_steps is not None:
        cfg["training"]["progress_update_steps"] = int(
            args.progress_update_steps
        )
    if args.learning_rate is not None:
        cfg["optimizer"]["lr"] = float(args.learning_rate)
    if args.sampling_mode is not None:
        cfg.setdefault("training_sampler", {})["mode"] = str(
            args.sampling_mode
        )
    if args.expert_events_csv is not None:
        cfg.setdefault("training_sampler", {})[
            "events_csv"
        ] = args.expert_events_csv
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)
        cfg["dataset"]["seed"] = int(args.seed)
        cfg["policy"]["aqr_config"]["front_end"]["seed"] = int(args.seed)
        cfg.setdefault("training_sampler", {})["seed"] = int(args.seed)
    return cfg


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _require_project_local(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{label} must live inside the new AQR project root "
            f"{PROJECT_ROOT.resolve()}; got {resolved}"
        ) from exc
    return resolved


def _resolve_runtime_paths(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["dataset"]["zarr_path"] = str(
        _resolve_project_path(cfg["dataset"]["zarr_path"])
    )
    aqr_config = cfg["policy"]["aqr_config"]
    aqr_config["action_contract_path"] = str(
        _resolve_project_path(aqr_config["action_contract_path"])
    )
    urdf_path = str(aqr_config.get("urdf_path", "")).strip()
    if urdf_path:
        aqr_config["urdf_path"] = str(_resolve_project_path(urdf_path))
    cfg["training"]["output_dir"] = str(
        _resolve_project_path(cfg["training"]["output_dir"])
    )
    cfg["evaluation"]["checkpoint"] = str(
        _resolve_project_path(cfg["evaluation"]["checkpoint"])
    )
    sampler_config = cfg.get("training_sampler")
    if isinstance(sampler_config, Mapping):
        events_csv = str(sampler_config.get("events_csv", "")).strip()
        if events_csv:
            sampler_config["events_csv"] = str(
                _resolve_project_path(events_csv)
            )
    return cfg


def _training_sampler_static_contract(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    sampler = dict(config.get("training_sampler", {"mode": "uniform"}))
    mode = str(sampler.get("mode", "uniform")).strip().lower()
    if mode not in {"uniform", "phase_balanced"}:
        raise ValueError(
            "training_sampler.mode must be uniform or phase_balanced."
        )
    if mode == "uniform":
        return {
            "mode": "uniform",
            "changes_default_dataloader": False,
        }

    events_csv = str(sampler.get("events_csv", "")).strip()
    if not events_csv:
        raise ValueError(
            "phase_balanced training requires training_sampler.events_csv."
        )
    events_path = Path(events_csv).expanduser()
    if not events_path.is_file():
        raise FileNotFoundError(
            f"phase-balanced expert events do not exist: {events_path}"
        )
    ratios = {
        "grasp_position": float(sampler.get("grasp_position_ratio", 0.45)),
        "stable_lift": float(sampler.get("stable_lift_ratio", 0.30)),
        "uniform": float(sampler.get("uniform_ratio", 0.25)),
    }
    if any(value < 0.0 for value in ratios.values()):
        raise ValueError("phase-balanced sampling ratios must be non-negative.")
    if abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("phase-balanced sampling ratios must sum to 1.")
    windows = {
        "pre_close_steps": int(sampler.get("pre_close_steps", 8)),
        "post_stable_steps": int(sampler.get("post_stable_steps", 2)),
        "post_secure_lift_steps": int(
            sampler.get("post_secure_lift_steps", 2)
        ),
    }
    if any(value < 0 for value in windows.values()):
        raise ValueError("phase-balanced window sizes must be non-negative.")
    samples_per_epoch = int(sampler.get("samples_per_epoch", 0))
    if samples_per_epoch < 0:
        raise ValueError("training_sampler.samples_per_epoch must be non-negative.")
    return {
        "mode": mode,
        "events_csv": str(events_path.resolve()),
        "ratios": ratios,
        "windows": windows,
        "samples_per_epoch": samples_per_epoch,
        "require_all_training_episodes": bool(
            sampler.get("require_all_training_episodes", True)
        ),
        "privileged_events_entered_batch": False,
        "deployment_input_safe": True,
    }


def _validate_static_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_policy = "contactflow.dp3.aqr_dp3.AQRDP3"
    expected_dataset = "contactflow.dp3.aqr_dp3.AQRDP3Dataset"
    if _nested(config, "policy", "_target_") != expected_policy:
        raise ValueError(f"policy._target_ must be {expected_policy}")
    if _nested(config, "dataset", "_target_") != expected_dataset:
        raise ValueError(f"dataset._target_ must be {expected_dataset}")

    horizon = int(_nested(config, "policy", "horizon"))
    obs_steps = int(_nested(config, "policy", "n_obs_steps"))
    action_steps = int(_nested(config, "policy", "n_action_steps"))
    experiment = str(
        _nested(config, "policy", "aqr_config", "experiment")
    ).upper()
    expected_temporal_contract = (
        (12, 2, 8) if experiment == "A6_POST" else (4, 2, 3)
    )
    if (horizon, obs_steps, action_steps) != expected_temporal_contract:
        raise ValueError(
            "AQR-DP3 temporal contract mismatch: expected "
            f"{expected_temporal_contract}, got "
            f"{(horizon, obs_steps, action_steps)}."
        )
    if int(_nested(config, "dataset", "horizon")) != horizon:
        raise ValueError("policy and dataset horizons disagree.")
    if int(_nested(config, "dataset", "pad_before")) != obs_steps - 1:
        raise ValueError("dataset.pad_before must equal n_obs_steps-1.")
    if int(_nested(config, "dataset", "pad_after")) != action_steps - 1:
        raise ValueError("dataset.pad_after must equal n_action_steps-1.")

    root_shape = _nested(config, "shape_meta")
    policy_shape = _nested(config, "policy", "shape_meta")
    if root_shape != policy_shape:
        raise ValueError("root and policy shape_meta must be identical.")
    state_shape = list(_nested(root_shape, "obs", "agent_pos", "shape"))
    action_shape = list(_nested(root_shape, "action", "shape"))
    point_shape = list(_nested(root_shape, "obs", "point_cloud", "shape"))
    if state_shape != [16]:
        raise ValueError("AQR-DP3 agent_pos must have 16 deployable values.")
    if action_shape != [8]:
        raise ValueError("AQR-DP3 action dimension must be 8.")
    if len(point_shape) != 2 or int(point_shape[1]) not in {3, 6}:
        raise ValueError(
            "AQR-DP3 point_cloud must have shape [P_query,3 or 6]."
        )
    point_channels = int(point_shape[1])
    use_pc_color = bool(_nested(config, "policy", "use_pc_color"))
    if use_pc_color != (point_channels == 6):
        raise ValueError(
            "policy.use_pc_color must be false for XYZ and true for XYZRGB."
        )
    if point_channels != 6:
        raise ValueError(
            "Formal three-task AQR-DP3 training requires a rebuilt XYZRGB "
            "high-resolution point cloud."
        )
    query_points = int(
        _nested(config, "policy", "aqr_config", "front_end", "query_points")
    )
    if query_points not in {4096, 8192}:
        raise ValueError("P_query must be 4096 or 8192.")
    if int(point_shape[0]) < query_points:
        raise ValueError("shape_meta raw point count must cover P_query.")
    if int(_nested(config, "dataset", "query_points")) != query_points:
        raise ValueError("dataset query_points must equal policy P_query.")

    aqr_mapping = dict(_nested(config, "policy", "aqr_config"))
    from contactflow.dp3.aqr_dp3.config import (
        AQRDP3Config,
        PEG_INSERTION_WORKSPACE_CROP_MAX,
        PEG_INSERTION_WORKSPACE_CROP_MIN,
        PULL_CUBE_TOOL_WORKSPACE_CROP_MAX,
        PULL_CUBE_TOOL_WORKSPACE_CROP_MIN,
        STACK_CUBE_WORKSPACE_CROP_MAX,
        STACK_CUBE_WORKSPACE_CROP_MIN,
        validate_workspace_bounds,
    )

    aqr = AQRDP3Config.from_mapping(aqr_mapping)
    dataset_path = _require_project_local(
        _resolve_project_path(_nested(config, "dataset", "zarr_path")),
        label="dataset.zarr_path",
    )
    output_dir = _require_project_local(
        _resolve_project_path(_nested(config, "training", "output_dir")),
        label="training.output_dir",
    )
    expected_task_marker = {
        "peg_insertion": "peginsertion",
        "stack_cube": "stackcube",
        "pull_cube_tool": "pullcubetool",
    }[aqr.task_profile]
    normalized_dataset_name = "".join(
        character
        for character in dataset_path.name.lower()
        if character.isalnum()
    )
    if expected_task_marker not in normalized_dataset_name:
        raise ValueError(
            f"dataset filename does not match task_profile={aqr.task_profile}: "
            f"{dataset_path.name}"
        )
    dataset_crop_min, dataset_crop_max = validate_workspace_bounds(
        _nested(config, "dataset", "workspace_crop_min"),
        _nested(config, "dataset", "workspace_crop_max"),
        allow_none=False,
    )
    front_crop_min, front_crop_max = validate_workspace_bounds(
        aqr.front_end.workspace_crop_min,
        aqr.front_end.workspace_crop_max,
        allow_none=False,
    )
    task_bounds = {
        "peg_insertion": (
            PEG_INSERTION_WORKSPACE_CROP_MIN,
            PEG_INSERTION_WORKSPACE_CROP_MAX,
        ),
        "stack_cube": (
            STACK_CUBE_WORKSPACE_CROP_MIN,
            STACK_CUBE_WORKSPACE_CROP_MAX,
        ),
        "pull_cube_tool": (
            PULL_CUBE_TOOL_WORKSPACE_CROP_MIN,
            PULL_CUBE_TOOL_WORKSPACE_CROP_MAX,
        ),
    }
    locked_bounds = task_bounds[aqr.task_profile]
    if (dataset_crop_min, dataset_crop_max) != locked_bounds:
        raise ValueError(
            f"{aqr.task_profile} AQR dataset workspace must equal the locked bounds "
            f"{locked_bounds}; got {(dataset_crop_min, dataset_crop_max)}."
        )
    if (front_crop_min, front_crop_max) != (
        dataset_crop_min,
        dataset_crop_max,
    ):
        raise ValueError(
            "dataset and PointCloudFrontEnd workspace bounds must match exactly."
        )
    if not bool(_nested(config, "dataset", "reject_workspace_violations")):
        raise ValueError(
            "Formal AQR-DP3 training requires reject_workspace_violations=true; "
            "mask-only legacy loading is diagnostic, not a clean training dataset."
        )
    maximum_future = horizon - obs_steps
    if bool(aqr.future_aux) and max(aqr.future_horizons) > maximum_future:
        raise ValueError(
            "future_aux horizon exceeds the sampled sequence. With "
            "horizon=4/n_obs_steps=2, use future_horizons=[1,2]."
        )

    contract_path = _require_project_local(
        _resolve_project_path(aqr.action_contract_path),
        label="policy.aqr_config.action_contract_path",
    )
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"validated AQR action contract does not exist: {contract_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_source = str(contract.get("source_trajectory", "")).strip()
    normalized_contract_source = "".join(
        character
        for character in contract_source.lower()
        if character.isalnum()
    )
    if not contract_source or expected_task_marker not in normalized_contract_source:
        raise ValueError(
            "action contract source task does not match "
            f"task_profile={aqr.task_profile}: {contract_source or '<missing>'}"
        )
    if int(_nested(contract, "control", "action_dim")) != 8:
        raise ValueError("action contract must describe an 8-D controller action.")
    alignment = _nested(contract, "dp3_alignment")
    actual_alignment = (
        int(alignment["horizon"]),
        int(alignment["n_obs_steps"]),
        int(alignment["n_action_steps"]),
    )
    if actual_alignment != (horizon, obs_steps, action_steps):
        raise ValueError(
            "action contract DP3 alignment disagrees with the policy: "
            f"contract={actual_alignment}, "
            f"policy={(horizon, obs_steps, action_steps)}"
        )
    expected_execution_slots = list(
        range(obs_steps - 1, obs_steps - 1 + action_steps)
    )
    actual_execution_slots = [
        int(value) for value in alignment.get("execution_slots", ())
    ]
    if actual_execution_slots != expected_execution_slots:
        raise ValueError(
            "action contract execution slots disagree with the policy: "
            f"contract={actual_execution_slots}, "
            f"policy={expected_execution_slots}"
        )
    return {
        "policy_target": expected_policy,
        "dataset_target": expected_dataset,
        "experiment": aqr.experiment,
        "task_profile": aqr.task_profile,
        "query_points": query_points,
        "point_channels": point_channels,
        "use_pc_color": use_pc_color,
        "state_dim": 16,
        "action_dim": 8,
        "horizon": horizon,
        "n_obs_steps": obs_steps,
        "n_action_steps": action_steps,
        "exec_steps": int(aqr.exec_steps),
        "post_refiner_input_mode": str(aqr.post_refiner_input_mode),
        "future_aux": bool(aqr.future_aux),
        "future_horizons": list(aqr.future_horizons),
        "action_contract": str(contract_path),
        "action_contract_source": contract_source,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "workspace_crop_min": list(dataset_crop_min),
        "workspace_crop_max": list(dataset_crop_max),
        "reject_workspace_violations": True,
        "training_sampler": _training_sampler_static_contract(config),
    }


def _instantiate(config: Mapping[str, Any]):
    import hydra
    from omegaconf import OmegaConf

    return hydra.utils.instantiate(OmegaConf.create(copy.deepcopy(dict(config))))


def _instantiate_optimizer(config: Mapping[str, Any], parameters):
    import hydra
    from omegaconf import OmegaConf

    return hydra.utils.instantiate(
        OmegaConf.create(copy.deepcopy(dict(config))),
        params=parameters,
    )


def _dataset_contract(config: Mapping[str, Any]):
    import torch

    dataset = _instantiate(_nested(config, "dataset"))
    if len(dataset) <= 0:
        raise ValueError("AQR-DP3 training split is empty.")
    sample = dataset[0]
    obs = sample["obs"]
    expected_horizon = int(_nested(config, "policy", "horizon"))
    expected_points = int(
        _nested(
            config,
            "policy",
            "shape_meta",
            "obs",
            "point_cloud",
            "shape",
        )[0]
    )
    expected_channels = int(
        _nested(
            config,
            "policy",
            "shape_meta",
            "obs",
            "point_cloud",
            "shape",
        )[1]
    )
    expected = {
        "point_cloud": (
            expected_horizon,
            expected_points,
            expected_channels,
        ),
        "agent_pos": (expected_horizon, 16),
        "robot_mask": (expected_horizon, expected_points),
        "point_valid_mask": (expected_horizon, expected_points),
        "action": (expected_horizon, 8),
    }
    actual = {
        "point_cloud": tuple(obs["point_cloud"].shape),
        "agent_pos": tuple(obs["agent_pos"].shape),
        "robot_mask": tuple(obs["robot_mask"].shape),
        "point_valid_mask": tuple(obs["point_valid_mask"].shape),
        "action": tuple(sample["action"].shape),
    }
    mismatches = {
        key: {"expected": list(expected[key]), "actual": list(actual[key])}
        for key in expected
        if expected[key] != actual[key]
    }
    if mismatches:
        raise ValueError(f"AQR-DP3 sampled tensor contract mismatch: {mismatches}")
    if obs["robot_mask"].dtype != torch.bool:
        raise ValueError("robot_mask must remain boolean.")
    if obs["point_valid_mask"].dtype != torch.bool:
        raise ValueError("point_valid_mask must remain boolean.")
    valid_per_frame = obs["point_valid_mask"].sum(dim=-1)
    minimum_valid_points = int(valid_per_frame.min().item())
    required_valid_points = int(
        _nested(config, "dataset", "query_points")
    )
    if minimum_valid_points < required_valid_points:
        raise ValueError(
            "AQR-DP3 formal data needs a complete filtered Query Bank in every "
            f"sampled frame: minimum={minimum_valid_points}, "
            f"required={required_valid_points}."
        )
    validation_dataset = dataset.get_validation_dataset()
    if len(validation_dataset) <= 0:
        raise ValueError(
            "AQR-DP3 validation split is empty; configure a positive val_ratio."
        )
    report = {
        "zarr_path": str(_nested(config, "dataset", "zarr_path")),
        "training_samples": len(dataset),
        "validation_samples": len(validation_dataset),
        "sample_shapes": {key: list(value) for key, value in actual.items()},
        "minimum_valid_points_in_first_sample": minimum_valid_points,
        "robot_points_in_first_sample": int(obs["robot_mask"].sum().item()),
        "workspace_scan": copy.deepcopy(dataset.workspace_scan),
    }
    return dataset, validation_dataset, report


def _resolve_device(value: str):
    import torch

    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is not available.")
    return torch.device(requested)


def _loader(dataset, config: Mapping[str, Any], *, sampler=None):
    from torch.utils.data import DataLoader

    kwargs = copy.deepcopy(dict(config))
    if sampler is not None:
        kwargs.pop("shuffle", None)
        kwargs["sampler"] = sampler
    workers = int(kwargs.get("num_workers", 0))
    kwargs["num_workers"] = workers
    if workers == 0:
        kwargs["persistent_workers"] = False
    return DataLoader(dataset, **kwargs)


def _training_sampler_contract(
    dataset,
    config: Mapping[str, Any],
):
    static = _training_sampler_static_contract(config)
    if static["mode"] == "uniform":
        return None, static

    sampler_config = dict(_nested(config, "training_sampler"))
    pools = dataset.build_phase_sampling_pools(
        events_csv=str(static["events_csv"]),
        n_obs_steps=int(_nested(config, "policy", "n_obs_steps")),
        pre_close_steps=int(static["windows"]["pre_close_steps"]),
        post_stable_steps=int(static["windows"]["post_stable_steps"]),
        post_secure_lift_steps=int(
            static["windows"]["post_secure_lift_steps"]
        ),
        require_all_training_episodes=bool(
            static["require_all_training_episodes"]
        ),
    )
    from contactflow.dp3.aqr_dp3.sampling import (
        PhaseBalancedIndexSampler,
    )

    samples_per_epoch = int(static["samples_per_epoch"])
    if samples_per_epoch == 0:
        samples_per_epoch = len(dataset)
    sampler = PhaseBalancedIndexSampler(
        pools=pools,
        ratios=static["ratios"],
        num_samples=samples_per_epoch,
        seed=int(sampler_config.get("seed", _nested(config, "training", "seed"))),
    )
    report = copy.deepcopy(static)
    report["pools"] = copy.deepcopy(pools.report)
    report["sampler"] = sampler.report()
    report["dataset_samples"] = len(dataset)
    report["dataloader_shuffle_disabled"] = True
    return sampler, report


def _move_to_device(value: Any, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {
            key: _move_to_device(item, device) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _mean_metrics(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {
        key: float(sum(float(row[key]) for row in rows if key in row))
        / float(sum(1 for row in rows if key in row))
        for key in keys
    }


def _diagnostic_metrics(value: Any) -> dict[str, float]:
    import numbers
    import numpy as np
    import torch

    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        name = str(key)
        if not name.startswith("aqr_"):
            name = f"aqr_{name}"
        if torch.is_tensor(item):
            if item.numel() == 0:
                continue
            scalar = float(item.detach().to(torch.float32).mean().cpu())
        elif isinstance(item, numbers.Real):
            scalar = float(item)
        else:
            continue
        if np.isfinite(scalar):
            result[name] = scalar
    return result


def _evaluate_loss(model, loader, device, max_steps: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    losses: list[float] = []
    metrics: list[Mapping[str, float]] = []
    expert_query = str(getattr(getattr(model, "aqr", None), "query_source", "")) == (
        "expert"
    )
    future_encoder_state = None
    future_encoder = getattr(model, "future_target_encoder", None)
    if future_encoder is not None:
        future_encoder_state = {
            key: value.detach().clone()
            for key, value in future_encoder.state_dict().items()
        }
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if int(max_steps) > 0 and batch_index >= int(max_steps):
                    break
                batch = _move_to_device(batch, device)
                if expert_query:
                    expert_obs = dict(batch["obs"])
                    expert_obs["expert_action"] = batch["action"]
                    prediction = model.predict_action(expert_obs)
                    loss = F.mse_loss(prediction["action_pred"], batch["action"])
                    loss_dict = {
                        "expert_query_rollout_mse": float(loss.detach().cpu()),
                        **_diagnostic_metrics(
                            prediction.get("aqr_diagnostics", {})
                        ),
                    }
                else:
                    loss_result = model.compute_loss(batch)
                    if isinstance(loss_result, tuple):
                        loss, loss_dict = loss_result
                    else:
                        loss = loss_result
                        loss_dict = {"bc_loss": float(loss.detach().cpu())}
                losses.append(float(loss.detach().cpu()))
                metrics.append(
                    {key: float(value) for key, value in loss_dict.items()}
                )
    finally:
        if future_encoder_state is not None:
            future_encoder.load_state_dict(future_encoder_state)
    if not losses:
        raise ValueError("offline test/validation loader produced no batches.")
    return {
        "batches": len(losses),
        "loss": float(sum(losses) / len(losses)),
        "objective": (
            "expert_query_rollout_mse"
            if expert_query
            else (
                "post_diffusion_final_horizon_refinement"
                if str(
                    getattr(
                        getattr(model, "aqr", None),
                        "refinement_stage",
                        "denoising",
                    )
                )
                == "post_diffusion"
                else (
                (
                    "diffusion_mse_plus_clean_relative_action_improvement"
                    if bool(
                        getattr(
                            getattr(model, "aqr", None),
                            "improvement_loss_clean_only",
                            False,
                        )
                    )
                    else "diffusion_mse_plus_relative_action_improvement"
                )
                if bool(
                    getattr(
                        getattr(model, "aqr", None),
                        "improvement_loss_enabled",
                        False,
                    )
                )
                else "diffusion_training_objective_mse"
                )
            )
        ),
        "metrics": _mean_metrics(metrics),
    }


def _prediction_contract(model, dataset, loader_config, device) -> dict[str, Any]:
    import torch

    batch = next(iter(_loader(dataset, loader_config)))
    batch = _move_to_device(batch, device)
    obs = dict(batch["obs"])
    if str(model.aqr.query_source) == "expert":
        obs["expert_action"] = batch["action"]

    calls = {"pointnext": 0, "future_target": 0, "future_aux": 0}
    if bool(model.aqr.strict_dp3_off):
        handle = model.obs_encoder.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__(
                "pointnext", calls["pointnext"] + 1
            )
        )
        try:
            model.eval()
            with torch.no_grad():
                prediction = model.predict_action(obs)
        finally:
            handle.remove()
        expected_action = (
            int(batch["action"].shape[0]),
            int(model.n_action_steps),
            int(model.action_dim),
        )
        expected_prediction = (
            int(batch["action"].shape[0]),
            int(model.horizon),
            int(model.action_dim),
        )
        if tuple(prediction["action"].shape) != expected_action:
            raise AssertionError("AQR_OFF did not preserve SimpleDP3 action shape.")
        if tuple(prediction["action_pred"].shape) != expected_prediction:
            raise AssertionError(
                "AQR_OFF did not preserve SimpleDP3 prediction shape."
            )
        if calls["pointnext"] != 1:
            raise AssertionError(
                "AQR_OFF original DP3 encoder must execute exactly once."
            )
        return {
            "action_shape": list(prediction["action"].shape),
            "action_pred_shape": list(prediction["action_pred"].shape),
            "finite": bool(
                torch.isfinite(prediction["action"]).all()
                and torch.isfinite(prediction["action_pred"]).all()
            ),
            "pointnext_forward_calls": calls["pointnext"],
            "future_target_forward_calls": 0,
            "future_aux_forward_calls": 0,
            "strict_simple_dp3_path": True,
            "aqr_update_count": 0,
        }
    workspace_chain = {
        "frontend_calls": 0,
        "raw_workspace_rejected_points": 0,
        "encoder_bank_outside_points": 0,
        "query_bank_outside_points": 0,
        "pointnext_input_outside_points": 0,
        "encoder_bank_shape": None,
        "query_bank_shape": None,
        "encoder_tcp_swept_points": 0,
        "query_unique_source_points": 0,
        "query_sources_outside_encoder": 0,
        "query_source_index_max": -1,
    }
    crop_min = torch.as_tensor(
        model.aqr.front_end.workspace_crop_min,
        device=device,
        dtype=obs["point_cloud"].dtype,
    )
    crop_max = torch.as_tensor(
        model.aqr.front_end.workspace_crop_max,
        device=device,
        dtype=obs["point_cloud"].dtype,
    )

    def outside_count(
        xyz: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> int:
        inside = (xyz > crop_min).all(dim=-1) & (xyz < crop_max).all(dim=-1)
        return int((valid_mask.to(torch.bool) & ~inside).sum().detach().cpu())

    def count(name: str):
        def hook(_module, _inputs, _output):
            calls[name] += 1

        return hook

    def capture_frontend(_module, _inputs, output):
        workspace_chain["frontend_calls"] += 1
        workspace_chain["encoder_bank_shape"] = list(output.enc_points.shape)
        workspace_chain["query_bank_shape"] = list(output.query_points.shape)
        workspace_chain["encoder_tcp_swept_points"] += int(
            output.diagnostics["enc_tcp_swept_count"].sum().detach().cpu()
        )
        # Source indices are local to each batch item. Comparing a flattened
        # batch makes unrelated FPS2048 banks collectively cover all raw
        # indices and falsely reports that Query4096 was reduced to FPS2048.
        query_unique_per_sample: list[int] = []
        query_outside_encoder = 0
        valid_query_sources = output.query_source_indices[
            output.query_source_indices.ge(0)
        ]
        for batch_index in range(int(output.query_source_indices.shape[0])):
            sample_query = output.query_source_indices[batch_index]
            sample_query = sample_query[sample_query.ge(0)]
            sample_encoder = output.enc_source_indices[batch_index]
            sample_encoder = sample_encoder[sample_encoder.ge(0)]
            query_unique_per_sample.append(
                int(torch.unique(sample_query).numel())
            )
            query_outside_encoder += int(
                (~torch.isin(sample_query, sample_encoder)).sum().detach().cpu()
            )
        workspace_chain["query_unique_source_points"] = (
            min(query_unique_per_sample) if query_unique_per_sample else 0
        )
        workspace_chain["query_sources_outside_encoder"] = query_outside_encoder
        if valid_query_sources.numel():
            workspace_chain["query_source_index_max"] = int(
                valid_query_sources.max().detach().cpu()
            )
        workspace_chain["raw_workspace_rejected_points"] += int(
            output.diagnostics["raw_workspace_rejected_count"].sum().detach().cpu()
        )
        workspace_chain["encoder_bank_outside_points"] += outside_count(
            output.enc_points[..., :3],
            output.enc_valid_mask,
        )
        workspace_chain["query_bank_outside_points"] += outside_count(
            output.query_points[..., :3],
            output.query_valid_mask,
        )

    def capture_pointnext_input(_module, _args, kwargs):
        workspace_chain["pointnext_input_outside_points"] += outside_count(
            kwargs["enc_xyz_metric"],
            kwargs["enc_valid_mask"],
        )

    handles = [
        model.point_frontend.register_forward_hook(capture_frontend),
        model.obs_encoder.register_forward_hook(count("pointnext")),
        model.obs_encoder.register_forward_pre_hook(
            capture_pointnext_input,
            with_kwargs=True,
        ),
    ]
    if model.future_target_encoder is not None:
        handles.append(
            model.future_target_encoder.register_forward_hook(
                count("future_target")
            )
        )
    if model.future_auxiliary is not None:
        handles.append(
            model.future_auxiliary.predictor.register_forward_hook(
                count("future_aux")
            )
        )
    try:
        model.eval()
        with torch.no_grad():
            prediction = model.predict_action(obs)
    finally:
        for handle in handles:
            handle.remove()

    action = prediction["action"]
    action_pred = prediction["action_pred"]
    expected_action = (
        int(action.shape[0]),
        int(model.exec_steps),
        int(model.action_dim),
    )
    expected_prediction = (
        int(action.shape[0]),
        int(model.horizon),
        int(model.action_dim),
    )
    if tuple(action.shape) != expected_action:
        raise ValueError(
            f"prediction action shape {tuple(action.shape)} != {expected_action}"
        )
    if tuple(action_pred.shape) != expected_prediction:
        raise ValueError(
            "full action prediction shape "
            f"{tuple(action_pred.shape)} != {expected_prediction}"
        )
    if not bool(torch.isfinite(action).all() and torch.isfinite(action_pred).all()):
        raise FloatingPointError("AQR-DP3 prediction contains non-finite actions.")
    if calls["pointnext"] != 1:
        raise AssertionError(
            "predict_action must execute the current-observation PointNeXt "
            f"exactly once; observed {calls['pointnext']} calls."
        )
    if calls["future_target"] or calls["future_aux"]:
        raise AssertionError(
            "future_aux modules were called from the deployment prediction path."
        )
    if any(
        int(workspace_chain[key]) != 0
        for key in (
            "encoder_bank_outside_points",
            "query_bank_outside_points",
            "pointnext_input_outside_points",
        )
    ):
        raise AssertionError(
            "workspace-outside geometry reached FPS/query/PointNeXt: "
            f"{workspace_chain}"
        )
    update_value = prediction.get("aqr_diagnostics", {}).get(
        "aqr_update_count",
        torch.zeros((), device=action.device),
    )
    diagnostics = prediction.get("aqr_diagnostics", {})
    local_token_norm = diagnostics.get("local_token_norm")
    if model.aqr.refinement_stage == "post_diffusion":
        delta_norm = diagnostics.get("post_applied_residual_norm")
        slot_mask = diagnostics.get("post_refiner_slot_mask")
    else:
        delta_norm = diagnostics.get("delta_norm")
        slot_mask = diagnostics.get("local_residual_slot_mask")
    expected_slot_mask = torch.zeros(
        model.horizon, device=action.device, dtype=torch.bool
    )
    expected_slot_mask[model._query_slot_indices] = True
    if local_token_norm is None or tuple(local_token_norm.shape) != (
        int(action.shape[0]),
        int(model.horizon),
    ):
        raise AssertionError("AQR local_token_norm must preserve [B,H].")
    if bool(local_token_norm[:, ~expected_slot_mask].ne(0).any()):
        raise AssertionError("AQR local tokens leaked beyond configured slots.")
    if delta_norm is None or tuple(delta_norm.shape) != (
        int(action.shape[0]),
        int(model.horizon),
    ):
        raise AssertionError("AQR delta_norm must preserve [B,H].")
    if bool(delta_norm[:, ~expected_slot_mask].ne(0).any()):
        raise AssertionError("AQR local residual leaked beyond configured slots.")
    if slot_mask is None or not torch.equal(
        slot_mask.to(device=action.device, dtype=torch.bool),
        expected_slot_mask,
    ):
        raise AssertionError("AQR local residual slot mask is incorrect.")
    expected_query_shape = [
        int(action.shape[0]),
        int(model.aqr.front_end.query_points),
        int(model.point_channels),
    ]
    if workspace_chain["query_bank_shape"] != expected_query_shape:
        raise AssertionError(
            "Query Bank did not use configured P_query: "
            f"{workspace_chain['query_bank_shape']} != {expected_query_shape}"
        )
    if model.aqr.front_end.mode == "baseline_fps2048" and int(
        workspace_chain["encoder_tcp_swept_points"]
    ) != 0:
        raise AssertionError("Protected encoder points reached baseline FPS2048.")
    if int(workspace_chain["query_unique_source_points"]) != int(
        model.aqr.front_end.query_points
    ):
        raise AssertionError("Query Bank did not contain 4096 unique raw sources.")
    if int(workspace_chain["query_sources_outside_encoder"]) <= 0:
        raise AssertionError("Query Bank was incorrectly reduced to FPS2048.")
    return {
        "action_shape": list(action.shape),
        "action_pred_shape": list(action_pred.shape),
        "finite": True,
        "pointnext_forward_calls": calls["pointnext"],
        "future_target_forward_calls": calls["future_target"],
        "future_aux_forward_calls": calls["future_aux"],
        "workspace_chain": workspace_chain,
        "local_token_norm_shape": list(local_token_norm.shape),
        "local_delta_norm_shape": list(delta_norm.shape),
        "local_residual_slot_mask": expected_slot_mask.tolist(),
        "slot1_query_anchor_count": int(
            torch.as_tensor(
                diagnostics["slot1_query_anchor_count"]
            ).detach().cpu()
        ),
        "aqr_update_count": int(
            torch.as_tensor(update_value).detach().to(torch.long).max().cpu()
        ),
    }


def _model_contract(
    config: Mapping[str, Any],
    dataset,
    *,
    smoke_batches: int,
    training_sampler=None,
    initialize_from: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    model = _instantiate(_nested(config, "policy"))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    report: dict[str, Any] = {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "parameters": int(parameter_count),
        "trainable_parameters": int(trainable_count),
    }
    initialization_report: dict[str, Any]
    if int(smoke_batches) > 0:
        if dataset is None:
            dataset, _, _ = _dataset_contract(config)
        normalizer = dataset.get_normalizer()
        model.set_normalizer(normalizer)
        device = _resolve_device(str(_nested(config, "training", "device")))
        model.to(device)
        # Match the formal fine-tuning order exactly: install the dataset
        # normalizer, move to the requested device, then load epoch weights.
        # This catches checkpoint loaders that reconstruct CPU-only children.
        _, initialization_report, _ = _initialize_model_from_checkpoint(
            model,
            config,
            initialize_from,
        )
        smoke_loader = _loader(
            dataset,
            _nested(config, "dataloader"),
            sampler=training_sampler,
        )
        smoke = _evaluate_loss(model, smoke_loader, device, int(smoke_batches))
        smoke["training_sampler"] = (
            training_sampler.epoch_report()
            if training_sampler is not None
            else {"mode": "uniform"}
        )
        report["device"] = str(device)
        report["smoke"] = smoke
        report["prediction"] = _prediction_contract(
            model,
            dataset,
            _nested(config, "dataloader"),
            device,
        )
    else:
        _, initialization_report, _ = _initialize_model_from_checkpoint(
            model,
            config,
            initialize_from,
        )
    report["initialization"] = initialization_report
    return model, report


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _evaluation_contract_mismatches(
    current: Mapping[str, Any],
    saved: Mapping[str, Any],
) -> list[str]:
    paths = (
        ("policy", "horizon"),
        ("policy", "n_obs_steps"),
        ("policy", "n_action_steps"),
        ("policy", "shape_meta", "obs", "point_cloud", "shape"),
        ("policy", "shape_meta", "obs", "agent_pos", "shape"),
        ("policy", "shape_meta", "action", "shape"),
        ("policy", "encoder_output_dim"),
        ("policy", "diffusion_step_embed_dim"),
        ("policy", "down_dims"),
        ("policy", "kernel_size"),
        ("policy", "n_groups"),
        ("policy", "condition_type"),
        ("policy", "use_down_condition"),
        ("policy", "use_mid_condition"),
        ("policy", "use_up_condition"),
        ("policy", "num_inference_steps"),
        ("policy", "noise_scheduler", "_target_"),
        ("policy", "noise_scheduler", "num_train_timesteps"),
        ("policy", "noise_scheduler", "beta_start"),
        ("policy", "noise_scheduler", "beta_end"),
        ("policy", "noise_scheduler", "beta_schedule"),
        ("policy", "noise_scheduler", "prediction_type"),
        ("policy", "noise_scheduler", "clip_sample"),
        ("policy", "noise_scheduler", "set_alpha_to_one"),
        ("policy", "noise_scheduler", "steps_offset"),
        ("policy", "aqr_config", "front_end", "query_points"),
        ("policy", "aqr_config", "action_to_tool_mode"),
        ("policy", "aqr_config", "exec_steps"),
        ("policy", "aqr_config", "query_source"),
        ("policy", "aqr_config", "refinement_stage"),
        ("policy", "aqr_config", "local_action_slots"),
        ("policy", "aqr_config", "anchor_preserving_query"),
    )
    mismatches: list[str] = []
    missing = object()

    def optional(mapping: Mapping[str, Any], path: tuple[str, ...]):
        value: Any = mapping
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                return missing
            value = value[key]
        return value

    for path in paths:
        current_value = optional(current, path)
        saved_value = optional(saved, path)
        if current_value is missing and saved_value is missing:
            continue
        if current_value is missing or saved_value is missing:
            mismatches.append(".".join(path) + " (missing)")
            continue
        if current_value != saved_value:
            mismatches.append(
                f"{'.'.join(path)}: current={current_value!r}, "
                f"checkpoint={saved_value!r}"
            )
    return mismatches


def _load_evaluation_state(model, state_dict: Mapping[str, Any]) -> None:
    # Q7 intentionally changes only the fixed TCP-frame keypoint layout while
    # reusing the trained network weights. Keep that explicit ablation buffer
    # from the current config and require every other tensor to load strictly.
    adapted = dict(state_dict)
    current = model.state_dict()
    key = "action_to_tool.keypoint_offsets_tcp"
    if (
        key in adapted
        and key in current
        and tuple(adapted[key].shape) != tuple(current[key].shape)
    ):
        adapted[key] = current[key]
    model.load_state_dict(adapted, strict=True)


def _initialize_model_from_checkpoint(
    model,
    config: Mapping[str, Any],
    checkpoint: str | Path | None,
) -> tuple[Mapping[str, Any] | None, dict[str, Any], Path | None]:
    if checkpoint is None:
        return None, {"mode": "random_initialization"}, None

    checkpoint_path = _resolve_project_path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"initialization checkpoint does not exist: {checkpoint_path}"
        )
    payload = _torch_load(checkpoint_path)
    if payload.get("kind") != "aqr_dp3_training_checkpoint":
        raise ValueError(
            "--initialize-from requires an AQR-DP3 trainer checkpoint."
        )
    saved_config = payload.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError(
            "initialization checkpoint is missing its resolved config."
        )
    mismatches = _evaluation_contract_mismatches(config, saved_config)
    try:
        early_current_experiment = str(
            _nested(config, "policy", "aqr_config", "experiment")
        ).upper()
        early_saved_experiment = str(
            _nested(saved_config, "policy", "aqr_config", "experiment")
        ).upper()
    except ValueError:
        early_current_experiment = ""
        early_saved_experiment = ""
    early_training_config = config.get("training", {})
    early_raw_excluded_prefixes = (
        early_training_config.get("initialize_exclude_prefixes", ())
        if isinstance(early_training_config, Mapping)
        else ()
    )
    if isinstance(early_raw_excluded_prefixes, str):
        early_raw_excluded_prefixes = (early_raw_excluded_prefixes,)
    early_excluded_prefixes = tuple(
        str(prefix).strip()
        for prefix in early_raw_excluded_prefixes
        if str(prefix).strip()
    )
    a1_to_post_required_prefixes = {
        "post_refiner.",
        "query_stem.",
        "relative_query.",
        "local_residual.",
        "action_to_tool.",
    }
    a1_to_post_extension = (
        early_current_experiment == "A6_POST"
        and early_saved_experiment == "A1"
    )
    if a1_to_post_extension and not a1_to_post_required_prefixes.issubset(
        early_excluded_prefixes
    ):
        missing_prefixes = sorted(
            a1_to_post_required_prefixes.difference(early_excluded_prefixes)
        )
        raise ValueError(
            "A1 to A6_POST initialization may reuse only the global DP3 base; "
            "fresh local/refiner prefixes are missing: " + ", ".join(missing_prefixes)
        )
    if (
        early_current_experiment == "A6_POST"
        and early_saved_experiment in {"A1", "A2", "A3", "A4", "A5"}
    ):
        # Temporal convolution weights are horizon-independent. A6 deliberately
        # expands the sampled sequence and executable prefix while reusing the
        # verified observation/action/controller contract.
        allowed_temporal_migrations = (
            "policy.horizon",
            "policy.n_action_steps",
            "policy.aqr_config.exec_steps",
            "policy.aqr_config.query_source",
            "policy.aqr_config.refinement_stage",
            "policy.aqr_config.local_action_slots",
            "policy.aqr_config.anchor_preserving_query",
        )
        mismatches = [
            mismatch
            for mismatch in mismatches
            if not mismatch.startswith(allowed_temporal_migrations)
        ]
    if mismatches:
        raise ValueError(
            "initialization checkpoint changes the model/action contract: "
            + "; ".join(mismatches)
        )
    # LinearNormalizer's custom state-dict loader rebuilds its ParameterDict
    # from checkpoint tensors.  torch.load(map_location="cpu") therefore puts
    # that rebuilt child back on CPU even when the policy was already moved to
    # CUDA.  Reapply the policy device after loading so normalization and point
    # masks cannot diverge on the first fine-tuning batch.
    model_device = model.device
    def experiment_name(mapping: Mapping[str, Any]) -> str:
        try:
            return str(
                _nested(mapping, "policy", "aqr_config", "experiment")
            ).upper()
        except ValueError:
            return ""

    current_experiment = experiment_name(config)
    saved_experiment = experiment_name(saved_config)
    excluded_prefixes = early_excluded_prefixes
    if any(prefix in {"model.", "obs_encoder."} for prefix in excluded_prefixes):
        raise ValueError(
            "Refiner initialization may not exclude the shared DP3 base "
            "(model./obs_encoder.)."
        )
    attention_extension = (
        current_experiment == "A4" and saved_experiment == "A3"
    )
    post_extension = (
        current_experiment == "A6_POST"
        and saved_experiment in {"A2", "A3", "A4", "A5"}
    )
    missing_keys: list[str] = []
    excluded_source_keys: list[str] = []
    if excluded_prefixes:
        source_state = payload["model_state_dict"]
        excluded_source_keys = sorted(
            key for key in source_state if key.startswith(excluded_prefixes)
        )
        adapted_state = {
            key: value
            for key, value in source_state.items()
            if not key.startswith(excluded_prefixes)
        }
        incompatible = model.load_state_dict(adapted_state, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        disallowed_missing = [
            key for key in missing_keys if not key.startswith(excluded_prefixes)
        ]
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                "Selective initialization changed parameters outside the fresh "
                "refiner contract; "
                f"disallowed_missing={disallowed_missing}, "
                f"unexpected={unexpected_keys}."
            )
        if not any(key.startswith("post_refiner.") for key in missing_keys):
            raise RuntimeError(
                "Selective refiner initialization must leave post_refiner.* fresh."
            )
    elif attention_extension or post_extension:
        incompatible = model.load_state_dict(
            payload["model_state_dict"], strict=False
        )
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        allowed_missing_prefixes = (
            (
                "post_refiner.",
                "relative_query.anchor_role_embedding.",
                "relative_query.anchor_phase_embedding.",
                "relative_query.anchor_sweep_embedding.",
                "relative_query.anchor_attention.",
            )
            if post_extension
            else (
                "relative_query.action_query_encoder.",
                "relative_query.fused_key_encoder.",
            )
        )
        disallowed_missing = [
            key
            for key in missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                "Architecture extension contains unexpected parameters; "
                f"disallowed_missing={disallowed_missing}, "
                f"unexpected={unexpected_keys}."
            )
        if not missing_keys:
            raise RuntimeError(
                "Architecture extension expected new parameters."
            )
    else:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(model_device)
    report = {
        "mode": "model_only_fresh_optimizer",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "checkpoint_global_step": int(payload.get("global_step", 0)),
        "optimizer_state_loaded": False,
        "model_device_reapplied_after_load": str(model_device),
        "architecture_extension": (
            "A1_base_to_A6_POST_fresh_local_refiner"
            if a1_to_post_extension
            else (
                "fresh_refiner_ablation"
                if excluded_prefixes
                else (
                    "legacy_to_A6_POST"
                    if post_extension
                    else ("A3_to_A4" if attention_extension else "none")
                )
            )
        ),
        "new_parameter_keys": missing_keys,
        "excluded_prefixes": list(excluded_prefixes),
        "excluded_checkpoint_keys": excluded_source_keys,
    }
    return payload, report, checkpoint_path


def _checkpoint_payload(
    *,
    config: Mapping[str, Any],
    model,
    optimizer,
    epoch: int,
    global_step: int,
    validation: Mapping[str, Any] | None,
    curriculum_scheduler=None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "aqr_dp3_training_checkpoint",
        "config": copy.deepcopy(dict(config)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation": copy.deepcopy(validation),
    }
    if curriculum_scheduler is not None:
        payload["curriculum_state"] = curriculum_scheduler.state_dict()
    return payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_validate(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage = str(args.validation_stage)
    report: dict[str, Any] = {
        "mode": "validate",
        "validation_stage": stage,
        "config": _validate_static_config(config),
    }
    dataset = None
    training_sampler = None
    if stage in {"data", "model", "all"}:
        dataset, _, data_report = _dataset_contract(config)
        training_sampler, sampler_report = _training_sampler_contract(
            dataset,
            config,
        )
        data_report["training_sampler"] = sampler_report
        report["data"] = data_report
    if stage in {"model", "all"}:
        _, model_report = _model_contract(
            config,
            dataset,
            smoke_batches=int(args.smoke_batches),
            training_sampler=training_sampler,
            initialize_from=args.initialize_from,
        )
        report["model"] = model_report
    report["status"] = "pass"
    return report


def _update_curriculum(
    model,
    *,
    global_step: int,
    total_steps: int,
) -> dict[str, float]:
    """DEPRECATED: replaced by CurriculumScheduler."""
    raise NotImplementedError("use CurriculumScheduler instead")


class CurriculumScheduler:
    """Validation-gated curriculum with exponential smoothing.

    Phases advance only when validation loss plateaus (stops improving
    significantly), not on a fixed schedule.  Within each phase, parameters
    transition to their targets with exp(-speed * epoch_in_phase), reaching
    >95% within 1-2 epochs so the model gets maximum adaptation time.

    Phase targets are fractions of the selected experiment's configured
    ceilings.  Therefore A3/A4 can never accidentally enable relation gain;
    only A5 has a non-zero gain target.
    """

    PHASES = (
        # (min_epochs, patience, speed, offset_fraction, gain_fraction)
        # speed: within-phase transition rate (higher = faster convergence to target)
        # patience: number of validations without improvement before advancing
        # min_epochs: minimum epochs in phase before plateau detection starts
        (3, 2, 5.0, 0.00, 0.0),
        (3, 3, 5.0, 0.25, 0.0),
        (3, 3, 5.0, 0.60, 0.0),
        (5, 4, 3.0, 1.00, 1.0),
    )

    def __init__(self, model, *, plateau_threshold: float = 0.005):
        self._model = model
        self._plateau_threshold = float(plateau_threshold)

        # Mutable state (serialized in checkpoint)
        self.phase: int = 0
        self.epochs_in_phase: int = 0
        self._best_val_loss: float = float("inf")
        self._patience_counter: int = 0
        # Previous-phase values for smooth transition
        self._prev_prob: float = 0.0
        self._prev_max_action_norm: float = 0.0
        self._prev_gain: float = 0.0

    # ---- serialization ----
    def state_dict(self) -> dict:
        return {
            "phase": self.phase,
            "epochs_in_phase": self.epochs_in_phase,
            "best_val_loss": self._best_val_loss,
            "patience_counter": self._patience_counter,
            "prev_prob": self._prev_prob,
            "prev_max_action_norm": self._prev_max_action_norm,
            "prev_gain": self._prev_gain,
        }

    def load_state_dict(self, state: dict) -> None:
        self.phase = int(state["phase"])
        self.epochs_in_phase = int(state["epochs_in_phase"])
        self._best_val_loss = float(state["best_val_loss"])
        self._patience_counter = int(state["patience_counter"])
        self._prev_prob = float(state.get("prev_prob", 0.0))
        self._prev_max_action_norm = float(
            state.get("prev_max_action_norm", 0.0)
        )
        self._prev_gain = float(state.get("prev_gain", 0.0))

    # ---- called before each epoch ----
    def step(self) -> dict[str, float]:
        """Apply current curriculum params to model, return state dict."""
        self.epochs_in_phase += 1

        phase_cfg = self.PHASES[self.phase]
        _, _, speed, offset_fraction, gain_fraction = phase_cfg
        tgt_prob = self._model._offset_target_prob * offset_fraction
        tgt_max_action_norm = (
            self._model._offset_target_max_action_norm * offset_fraction
        )
        tgt_gain = self._model._relation_gain_target * gain_fraction

        # Exponential smoothing: reach target within ~1 epoch
        ref_epochs = max(1.0, float(phase_cfg[0]))
        frac = 1.0 - math.exp(-speed * self.epochs_in_phase / ref_epochs)

        current = {
            "offset_prob": self._prev_prob + (tgt_prob - self._prev_prob) * frac,
            "offset_max_action_norm": self._prev_max_action_norm
            + (tgt_max_action_norm - self._prev_max_action_norm) * frac,
            "relation_gain_scale": self._prev_gain
            + (tgt_gain - self._prev_gain) * frac,
        }

        # Apply to model
        model = self._model
        model._offset_prob = current["offset_prob"]
        model._offset_max_action_norm = current["offset_max_action_norm"]
        if model.local_residual is not None:
            model.local_residual.relation_gain_scale.fill_(
                current["relation_gain_scale"]
            )

        return {
            "curriculum_phase": self.phase,
            **current,
            "curriculum_patience": self._patience_counter,
        }

    # ---- called after each validation ----
    def on_validation(self, val_loss: float) -> dict[str, object]:
        """Check for plateau and possibly advance phase.

        Returns a dict with 'advanced': bool and 'reason': str.
        """
        phase_cfg = self.PHASES[self.phase]
        min_epochs, patience = phase_cfg[0], phase_cfg[1]

        improved = val_loss < self._best_val_loss * (1.0 - self._plateau_threshold)

        if improved:
            self._best_val_loss = val_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        advanced = False
        reason = ""
        if (
            self.epochs_in_phase >= min_epochs
            and self._patience_counter >= patience
            and self.phase < len(self.PHASES) - 1
        ):
            epochs_spent = self.epochs_in_phase
            # Save current values as starting point for next phase
            self._prev_prob = self._model._offset_prob
            self._prev_max_action_norm = (
                self._model._offset_max_action_norm
            )
            self._prev_gain = (
                float(self._model.local_residual.relation_gain_scale.item())
                if self._model.local_residual is not None
                else 0.0
            )

            self.phase += 1
            self.epochs_in_phase = 0
            self._best_val_loss = float("inf")
            self._patience_counter = 0
            advanced = True
            reason = (
                f"val_loss plateaued at {val_loss:.6f} after "
                f"{epochs_spent} epochs in phase "
                f"{self.phase - 1} (patience={patience})"
            )

        return {
            "curriculum_advanced": advanced,
            "curriculum_reason": reason,
            "curriculum_val_improved": improved,
        }


def _run_train(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch
    from tqdm.auto import tqdm

    static_report = _validate_static_config(config)
    if bool(args.resume) and args.initialize_from is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive.")
    if str(_nested(config, "policy", "aqr_config", "experiment")).upper() == "Q3":
        raise ValueError(
            "Q3 uses expert future actions and is evaluation-only. "
            "Train a causal checkpoint such as A1, then run --mode test "
            "--experiment Q3 with that checkpoint."
        )
    dataset, validation_dataset, data_report = _dataset_contract(config)
    seed = int(_nested(config, "training", "seed"))
    _seed_everything(seed)
    device = _resolve_device(str(_nested(config, "training", "device")))

    model = _instantiate(_nested(config, "policy"))
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)
    model.to(device)
    (
        initialization_payload,
        initialization_report,
        initialization_path,
    ) = _initialize_model_from_checkpoint(
        model,
        config,
        args.initialize_from,
    )
    optimizer = _instantiate_optimizer(
        _nested(config, "optimizer"), model.parameters()
    )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer config did not instantiate torch.optim.Optimizer")

    training_sampler, training_sampler_report = _training_sampler_contract(
        dataset,
        config,
    )
    train_loader = _loader(
        dataset,
        _nested(config, "dataloader"),
        sampler=training_sampler,
    )
    validation_loader = _loader(
        validation_dataset, _nested(config, "validation_dataloader")
    )
    output_dir = Path(str(_nested(config, "training", "output_dir")))
    checkpoint_dir = output_dir / "checkpoints"
    latest_path = (
        _resolve_project_path(args.checkpoint)
        if args.checkpoint is not None
        else checkpoint_dir / "latest.pt"
    )
    if (
        initialization_path is not None
        and latest_path.resolve() == initialization_path
    ):
        raise ValueError(
            "--initialize-from checkpoint cannot also be the output checkpoint."
        )
    best_path = checkpoint_dir / "best.pt"
    report_path = (
        _resolve_project_path(args.report_out)
        if args.report_out is not None
        else output_dir / "train_report.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_validation = float("inf")
    reset_initialization_progress = bool(
        _nested(config, "training").get("initialize_reset_progress", False)
    )
    initialization_report["training_progress_reset"] = bool(
        reset_initialization_progress
    )
    if initialization_payload is not None and not reset_initialization_progress:
        start_epoch = int(initialization_payload.get("epoch", 0))
        global_step = int(initialization_payload.get("global_step", 0))
    if bool(args.resume):
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {latest_path}")
        payload = _torch_load(latest_path)
        if payload.get("kind") != "aqr_dp3_training_checkpoint":
            raise ValueError("resume checkpoint is not an AQR-DP3 trainer checkpoint.")
        saved_config = payload.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("resume checkpoint is missing its resolved config.")
        resume_mismatches = [
            key
            for key in ("policy", "dataset", "optimizer")
            if saved_config.get(key) != config.get(key)
        ]
        if resume_mismatches:
            raise ValueError(
                "resume requires the original policy/data/optimizer contract; "
                "changed sections: " + ", ".join(resume_mismatches)
            )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        model.to(device)
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best_payload = _torch_load(best_path) if best_path.is_file() else payload
        best_result = best_payload.get("validation")
        if isinstance(best_result, Mapping) and best_result.get("loss") is not None:
            best_validation = float(best_result["loss"])

    epochs = int(_nested(config, "training", "epochs"))
    max_train_steps = int(_nested(config, "training", "max_train_steps"))
    max_validation_steps = int(
        _nested(config, "training", "max_validation_steps")
    )
    validation_every = int(_nested(config, "training", "validation_every"))
    checkpoint_every = int(_nested(config, "training", "checkpoint_every"))
    gradient_clip = float(_nested(config, "training", "gradient_clip"))
    training_config = _nested(config, "training")
    progress_enabled = bool(training_config.get("progress", True))
    progress_update_steps = int(
        training_config.get("progress_update_steps", 20)
    )
    if min(epochs, validation_every, checkpoint_every) <= 0:
        raise ValueError(
            "epochs, validation_every, and checkpoint_every must be positive."
        )
    if min(max_train_steps, max_validation_steps) < 0:
        raise ValueError("maximum step limits must be non-negative.")
    if gradient_clip < 0:
        raise ValueError("gradient_clip must be non-negative.")
    if progress_update_steps <= 0:
        raise ValueError("progress_update_steps must be positive.")

    total_train_steps = epochs * len(train_loader)

    # ── Curriculum scheduler (validation-gated) ──
    curriculum_scheduler: CurriculumScheduler | None = None
    if model._offset_training_enabled:
        curriculum_scheduler = CurriculumScheduler(model)
    if curriculum_scheduler is not None and initialization_payload is not None:
        saved_curriculum = initialization_payload.get("curriculum_state")
        load_initialization_curriculum = bool(
            _nested(config, "training").get(
                "initialize_load_curriculum_state", True
            )
        )
        if load_initialization_curriculum and isinstance(saved_curriculum, Mapping):
            curriculum_scheduler.load_state_dict(dict(saved_curriculum))
    if bool(args.resume) and curriculum_scheduler is not None:
        payload = _torch_load(latest_path)
        saved_curriculum = payload.get("curriculum_state")
        if isinstance(saved_curriculum, Mapping):
            curriculum_scheduler.load_state_dict(dict(saved_curriculum))

    history: list[dict[str, Any]] = []
    stop = False
    for epoch in range(start_epoch, epochs):
        if curriculum_scheduler is not None:
            curriculum = curriculum_scheduler.step()
        else:
            curriculum = {}

        model.train()
        if training_sampler is not None:
            training_sampler.set_epoch(epoch)
        epoch_losses: list[float] = []
        epoch_metrics: list[Mapping[str, float]] = []
        progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{epochs}",
            total=len(train_loader),
            disable=not progress_enabled,
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
        )
        for epoch_step, batch in enumerate(progress, start=1):
            if max_train_steps > 0 and global_step >= max_train_steps:
                stop = True
                break
            batch = _move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss_result = model.compute_loss(batch)
            if isinstance(loss_result, tuple):
                loss, loss_dict = loss_result
            else:
                loss = loss_result
                loss_dict = {"bc_loss": float(loss.detach().cpu())}
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite training loss at global step {global_step}"
                )
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_metrics.append(
                {key: float(value) for key, value in loss_dict.items()}
            )
            global_step += 1
            if progress_enabled and (
                epoch_step == 1
                or epoch_step % progress_update_steps == 0
            ):
                progress.set_postfix(
                    loss=f"{epoch_losses[-1]:.6f}",
                    step=global_step,
                    refresh=False,
                )
        if not epoch_losses and not stop:
            raise ValueError("AQR-DP3 training loader produced no batches.")

        validation = None
        if (epoch + 1) % validation_every == 0:
            validation = _evaluate_loss(
                model,
                validation_loader,
                device,
                max_validation_steps,
            )
            model.train()
            # ── Curriculum: check for plateau ──
            if curriculum_scheduler is not None and validation is not None:
                plateau_info = curriculum_scheduler.on_validation(
                    float(validation["loss"])
                )
                curriculum = {**curriculum, **plateau_info}
        epoch_report = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": (
                float(sum(epoch_losses) / len(epoch_losses))
                if epoch_losses
                else None
            ),
            "train_metrics": _mean_metrics(epoch_metrics),
            "validation": validation,
            "training_sampler": (
                training_sampler.epoch_report()
                if training_sampler is not None
                else {"mode": "uniform"}
            ),
            "curriculum": curriculum,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)

        payload = _checkpoint_payload(
            config=config,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            validation=validation,
            curriculum_scheduler=curriculum_scheduler,
        )
        if (epoch + 1) % checkpoint_every == 0 or stop:
            _atomic_torch_save(payload, latest_path)
        if validation is not None and float(validation["loss"]) < best_validation:
            best_validation = float(validation["loss"])
            _atomic_torch_save(payload, best_path)
        if stop:
            break

    if not latest_path.is_file():
        payload = _checkpoint_payload(
            config=config,
            model=model,
            optimizer=optimizer,
            epoch=(history[-1]["epoch"] if history else start_epoch),
            global_step=global_step,
            validation=(history[-1]["validation"] if history else None),
            curriculum_scheduler=curriculum_scheduler,
        )
        _atomic_torch_save(payload, latest_path)

    report = {
        "mode": "train",
        "status": "pass",
        "contract": static_report,
        "data": data_report,
        "initialization": initialization_report,
        "training_sampler": training_sampler_report,
        "device": str(device),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "global_step": global_step,
        "latest_checkpoint": str(latest_path.resolve()),
        "best_checkpoint": (
            str(best_path.resolve()) if best_path.is_file() else None
        ),
        "history": history,
    }
    _write_json(report_path, report)
    report["report_out"] = str(report_path.resolve())
    return report


def _run_test(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    static_report = _validate_static_config(config)
    dataset, validation_dataset, data_report = _dataset_contract(config)
    del dataset
    checkpoint_path = _resolve_project_path(
        args.checkpoint
        if args.checkpoint is not None
        else _nested(config, "evaluation", "checkpoint")
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"AQR-DP3 test checkpoint does not exist: {checkpoint_path}"
        )
    payload = _torch_load(checkpoint_path)
    if payload.get("kind") != "aqr_dp3_training_checkpoint":
        raise ValueError(
            "checkpoint is not an AQR-DP3 training checkpoint produced by this tool."
        )
    saved_config = payload.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("checkpoint is missing its resolved training config.")
    mismatches = _evaluation_contract_mismatches(config, saved_config)
    if mismatches:
        raise ValueError(
            "offline-test config changes a checkpoint-locked contract: "
            + "; ".join(mismatches)
        )

    model = _instantiate(_nested(config, "policy"))
    normalizer = validation_dataset.get_normalizer()
    model.set_normalizer(normalizer)
    _load_evaluation_state(model, payload["model_state_dict"])
    device = _resolve_device(str(_nested(config, "training", "device")))
    model.to(device)
    loader = _loader(
        validation_dataset, _nested(config, "validation_dataloader")
    )
    configured_max = int(_nested(config, "evaluation", "max_batches"))
    cli_max = args.max_validation_steps
    max_batches = int(cli_max) if cli_max is not None else configured_max
    _seed_everything(int(_nested(config, "training", "seed")))
    offline = _evaluate_loss(model, loader, device, max_batches)

    output_dir = Path(str(_nested(config, "training", "output_dir")))
    report_path = (
        _resolve_project_path(args.report_out)
        if args.report_out is not None
        else output_dir / "test_report.json"
    )
    report = {
        "mode": "test",
        "status": "pass",
        "contract": static_report,
        "data": data_report,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "offline_validation": offline,
    }
    _write_json(report_path, report)
    report["report_out"] = str(report_path.resolve())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.smoke_batches) < 0:
        raise ValueError("--smoke-batches must be non-negative.")
    config = _load_config(Path(args.config))
    config = _apply_overrides(config, args)
    config = _resolve_runtime_paths(config)
    if args.mode == "validate":
        report = _run_validate(config, args)
        report_path = (
            _resolve_project_path(args.report_out)
            if args.report_out is not None
            else PROJECT_ROOT / "outputs" / "aqr_dp3_validation_report.json"
        )
        _write_json(report_path, report)
        report["report_out"] = str(report_path.resolve())
    elif args.mode == "train":
        report = _run_train(config, args)
    else:
        report = _run_test(config, args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import zarr

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

from contactflow.dp3.aqr_dp3.config import (
    PEG_INSERTION_WORKSPACE_CROP_MAX,
    PEG_INSERTION_WORKSPACE_CROP_MIN,
    validate_workspace_bounds,
)


def _workspace_mask_np(
    points: np.ndarray,
    crop_min: tuple[float, float, float] | None,
    crop_max: tuple[float, float, float] | None,
) -> np.ndarray:
    xyz = np.asarray(points)[..., :3]
    valid = np.isfinite(xyz).all(axis=-1)
    if crop_min is not None:
        valid &= np.all(
            xyz > np.asarray(crop_min, dtype=xyz.dtype),
            axis=-1,
        )
        valid &= np.all(
            xyz < np.asarray(crop_max, dtype=xyz.dtype),
            axis=-1,
        )
    return valid


def _scan_workspace_violations(
    point_array,
    point_valid_mask,
    *,
    crop_min: tuple[float, float, float] | None,
    crop_max: tuple[float, float, float] | None,
    chunk_steps: int,
    example_limit: int = 8,
) -> dict:
    """Scan stored-valid XYZ without materializing a dense training set."""

    if int(chunk_steps) <= 0:
        raise ValueError("workspace_scan_chunk_steps must be positive.")
    steps = int(point_array.shape[0])
    point_count = int(point_array.shape[1])
    outside_count = 0
    affected_frames = 0
    stored_valid_count = 0
    minimum_in_workspace_valid_per_frame = point_count
    valid_xyz_min = np.full(3, np.inf, dtype=np.float64)
    valid_xyz_max = np.full(3, -np.inf, dtype=np.float64)
    examples: list[dict] = []
    for start in range(0, steps, int(chunk_steps)):
        stop = min(start + int(chunk_steps), steps)
        points = np.asarray(point_array[start:stop])
        xyz = points[..., :3]
        finite = np.isfinite(xyz).all(axis=-1)
        if point_valid_mask is None:
            stored_valid = finite
        else:
            stored_valid = np.asarray(point_valid_mask[start:stop])
            if stored_valid.ndim == 3 and stored_valid.shape[-1] == 1:
                stored_valid = stored_valid[..., 0]
            if tuple(stored_valid.shape) != (stop - start, point_count):
                raise ValueError(
                    "point_valid_mask shape is not aligned with point_cloud."
                )
            stored_valid = stored_valid.astype(bool, copy=False)
        stored_valid_count += int(np.count_nonzero(stored_valid))
        workspace_valid = _workspace_mask_np(points, crop_min, crop_max)
        in_workspace_valid = stored_valid & workspace_valid
        if in_workspace_valid.shape[0]:
            minimum_in_workspace_valid_per_frame = min(
                minimum_in_workspace_valid_per_frame,
                int(in_workspace_valid.sum(axis=1).min()),
            )
        outside = stored_valid & ~workspace_valid
        outside_count += int(np.count_nonzero(outside))
        affected_frames += int(np.count_nonzero(outside.any(axis=1)))
        finite_valid = stored_valid & finite
        if bool(finite_valid.any()):
            selected = xyz[finite_valid].astype(np.float64, copy=False)
            valid_xyz_min = np.minimum(valid_xyz_min, selected.min(axis=0))
            valid_xyz_max = np.maximum(valid_xyz_max, selected.max(axis=0))
        if len(examples) < int(example_limit) and bool(outside.any()):
            for local_step, point_index in np.argwhere(outside):
                examples.append(
                    {
                        "step": int(start + local_step),
                        "point_index": int(point_index),
                        "xyz": [
                            float(value) for value in xyz[local_step, point_index]
                        ],
                    }
                )
                if len(examples) >= int(example_limit):
                    break
    return {
        "workspace_crop_min": list(crop_min) if crop_min is not None else None,
        "workspace_crop_max": list(crop_max) if crop_max is not None else None,
        "steps": steps,
        "stored_valid_points": int(stored_valid_count),
        "outside_points": int(outside_count),
        "affected_frames": int(affected_frames),
        "outside_fraction_of_stored_valid": (
            float(outside_count) / float(stored_valid_count)
            if stored_valid_count
            else 0.0
        ),
        "minimum_in_workspace_valid_points_per_frame": int(
            minimum_in_workspace_valid_per_frame
        ),
        "stored_valid_xyz_min": (
            valid_xyz_min.tolist() if bool(np.isfinite(valid_xyz_min).all()) else None
        ),
        "stored_valid_xyz_max": (
            valid_xyz_max.tolist() if bool(np.isfinite(valid_xyz_max).all()) else None
        ),
        "examples": examples,
    }


def _streaming_stats(
    array,
    *,
    chunk_steps: int,
    row_mask=None,
    step_mask: np.ndarray | None = None,
    workspace_crop_min: tuple[float, float, float] | None = None,
    workspace_crop_max: tuple[float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-channel statistics without loading dense point banks at once."""

    if int(chunk_steps) <= 0:
        raise ValueError("normalizer_chunk_steps must be positive.")
    feature_dim = int(array.shape[-1])
    minimum = np.full(feature_dim, np.inf, dtype=np.float64)
    maximum = np.full(feature_dim, -np.inf, dtype=np.float64)
    total = np.zeros(feature_dim, dtype=np.float64)
    squared = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    if step_mask is not None:
        step_mask = np.asarray(step_mask, dtype=bool).reshape(-1)
        if step_mask.shape[0] != int(array.shape[0]):
            raise ValueError("normalizer step mask and values are misaligned.")
    rows_per_step = int(np.prod(array.shape[1:-1])) if array.ndim > 2 else 1
    for start in range(0, int(array.shape[0]), int(chunk_steps)):
        stop = min(start + int(chunk_steps), int(array.shape[0]))
        values = np.asarray(array[start:stop], dtype=np.float64).reshape(
            -1, feature_dim
        )
        mask = None
        if row_mask is not None:
            mask = np.asarray(row_mask[start:stop], dtype=bool).reshape(-1)
            if mask.shape[0] != values.shape[0]:
                raise ValueError("normalizer validity mask and values are misaligned.")
        if step_mask is not None:
            selected_steps = np.repeat(
                step_mask[start:stop],
                rows_per_step,
            )
            if selected_steps.shape[0] != values.shape[0]:
                raise ValueError("normalizer step mask and values are misaligned.")
            mask = selected_steps if mask is None else (mask & selected_steps)
        if workspace_crop_min is not None:
            if feature_dim < 3:
                raise ValueError("workspace filtering requires XYZ feature channels.")
            workspace = _workspace_mask_np(
                values,
                workspace_crop_min,
                workspace_crop_max,
            )
            mask = workspace if mask is None else (mask & workspace)
        if mask is not None:
            values = values[mask]
        finite = np.isfinite(values).all(axis=1)
        values = values[finite]
        if values.shape[0] == 0:
            continue
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
        count += int(values.shape[0])
    if count == 0:
        raise ValueError("cannot fit an AQR normalizer without finite valid rows.")
    mean = total / float(count)
    if count > 1:
        variance = (squared - float(count) * np.square(mean)) / float(count - 1)
    else:
        variance = np.zeros_like(mean)
    return {
        "min": minimum.astype(np.float32),
        "max": maximum.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
    }


def _episode_step_mask(
    episode_ends,
    episode_mask: np.ndarray,
) -> np.ndarray:
    ends = np.asarray(episode_ends[:], dtype=np.int64).reshape(-1)
    selected = np.asarray(episode_mask, dtype=bool).reshape(-1)
    if ends.shape != selected.shape:
        raise ValueError("episode mask and episode_ends are misaligned.")
    if ends.size == 0:
        return np.zeros((0,), dtype=bool)
    if np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing.")
    result = np.zeros((int(ends[-1]),), dtype=bool)
    start = 0
    for use_episode, stop in zip(selected, ends):
        if bool(use_episode):
            result[start : int(stop)] = True
        start = int(stop)
    return result


def _normalizer_from_stats(
    stats: dict[str, np.ndarray],
    *,
    mode: str,
    output_min: float,
    output_max: float,
    range_eps: float,
) -> SingleFieldLinearNormalizer:
    if mode not in {"limits", "gaussian"}:
        raise ValueError("normalizer mode must be limits or gaussian.")
    minimum = stats["min"]
    maximum = stats["max"]
    if mode == "limits":
        input_range = maximum - minimum
        ignored = input_range < float(range_eps)
        safe_range = input_range.copy()
        safe_range[ignored] = float(output_max) - float(output_min)
        scale = (float(output_max) - float(output_min)) / safe_range
        offset = float(output_min) - scale * minimum
        offset[ignored] = (
            (float(output_max) + float(output_min)) / 2.0 - minimum[ignored]
        )
    else:
        standard_deviation = stats["std"].copy()
        ignored = standard_deviation < float(range_eps)
        standard_deviation[ignored] = 1.0
        scale = 1.0 / standard_deviation
        offset = -stats["mean"] * scale
    tensor_stats = {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in stats.items()
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=torch.as_tensor(scale, dtype=torch.float32),
        offset=torch.as_tensor(offset, dtype=torch.float32),
        input_stats_dict=tensor_stats,
    )


class AQRDP3Dataset(BaseDataset):
    """Dense-point DP3 dataset with aligned boolean masks kept unnormalized.

    ``point_valid_mask`` is part of the sensor contract.  A stored
    ``robot_mask`` may be a privileged offline label; the policy ignores it by
    default unless ``use_robot_mask_feature`` is explicitly enabled for a
    separately validated deployable source.
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        query_points: int = 4096,
        require_robot_mask: bool = True,
        require_point_valid_mask: bool = True,
        load_to_memory: bool = False,
        normalizer_chunk_steps: int = 32,
        workspace_crop_min=PEG_INSERTION_WORKSPACE_CROP_MIN,
        workspace_crop_max=PEG_INSERTION_WORKSPACE_CROP_MAX,
        reject_workspace_violations: bool = True,
        workspace_scan_chunk_steps: int = 32,
    ):
        super().__init__()
        path = Path(zarr_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"AQR-DP3 zarr dataset does not exist: {path}")
        if int(query_points) not in {4096, 8192}:
            raise ValueError("query_points must be 4096 or 8192.")
        root = zarr.open(str(path), mode="r")
        if "data" not in root:
            raise ValueError("AQR-DP3 zarr is missing the data group.")
        available = set(root["data"].keys())
        required = {"state", "action", "point_cloud"}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"AQR-DP3 zarr is missing arrays: {missing}")
        point_shape = tuple(root["data"]["point_cloud"].shape)
        if len(point_shape) != 3 or point_shape[-1] not in {3, 6}:
            raise ValueError(
                "point_cloud must have shape [steps,raw_points,3 or 6], got "
                f"{point_shape}."
            )
        if int(point_shape[1]) < int(query_points):
            raise ValueError(
                "AQR-DP3 needs a dense raw point bank at least as large as P_query: "
                f"raw={point_shape[1]}, query={int(query_points)}."
            )
        if require_robot_mask and "robot_mask" not in available:
            raise ValueError(
                "This dataset configuration requires data/robot_mask. Rebuild "
                "the dataset with the AQR preparation script or set "
                "require_robot_mask=false. A segmentation-derived mask remains "
                "an offline label and must not be enabled as a deployment input."
            )
        if require_point_valid_mask and "point_valid_mask" not in available:
            raise ValueError(
                "AQR-DP3 full mode requires data/point_valid_mask so padding cannot "
                "be mistaken for geometry."
            )
        crop_min, crop_max = validate_workspace_bounds(
            workspace_crop_min,
            workspace_crop_max,
            allow_none=False,
        )
        self.workspace_crop_min = crop_min
        self.workspace_crop_max = crop_max
        self.reject_workspace_violations = bool(reject_workspace_violations)
        self.workspace_scan_chunk_steps = int(workspace_scan_chunk_steps)
        self.workspace_scan = _scan_workspace_violations(
            root["data"]["point_cloud"],
            (
                root["data"]["point_valid_mask"]
                if "point_valid_mask" in available
                else None
            ),
            crop_min=crop_min,
            crop_max=crop_max,
            chunk_steps=self.workspace_scan_chunk_steps,
        )
        if (
            self.reject_workspace_violations
            and int(self.workspace_scan["outside_points"]) > 0
        ):
            raise ValueError(
                "AQR-DP3 dataset contains stored-valid points outside the locked "
                "workspace: "
                f"outside_points={self.workspace_scan['outside_points']}, "
                f"affected_frames={self.workspace_scan['affected_frames']}/"
                f"{self.workspace_scan['steps']}, "
                f"valid_xyz_min={self.workspace_scan['stored_valid_xyz_min']}, "
                f"valid_xyz_max={self.workspace_scan['stored_valid_xyz_max']}. "
                "Rebuild from raw H5 with "
                f"--crop-min {','.join(str(v) for v in crop_min)} "
                f"--crop-max {','.join(str(v) for v in crop_max)}. "
                "Load-time masking is diagnostic only because it cannot recover "
                "in-workspace samples displaced by the contaminated FPS/query bank."
            )
        if (
            self.reject_workspace_violations
            and int(
                self.workspace_scan[
                    "minimum_in_workspace_valid_points_per_frame"
                ]
            )
            < int(query_points)
        ):
            raise ValueError(
                "AQR-DP3 dataset does not provide a full filtered Query Bank "
                f"in every frame: minimum_valid="
                f"{self.workspace_scan['minimum_in_workspace_valid_points_per_frame']}, "
                f"required={int(query_points)}. Rebuild from the raw dense "
                "XYZRGB trajectory; padded points are not a valid Query4096 source."
            )

        episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
        if "episode_ids" in root["meta"]:
            episode_ids = np.asarray(root["meta"]["episode_ids"], dtype=np.int64)
            episode_id_source = "meta/episode_ids"
        else:
            episode_ids = np.arange(len(episode_ends), dtype=np.int64)
            episode_id_source = "sequential_fallback"
        if episode_ids.shape != episode_ends.shape:
            raise ValueError(
                "meta/episode_ids must align with meta/episode_ends; got "
                f"{episode_ids.shape} and {episode_ends.shape}."
            )

        data_keys = ["state", "action", "point_cloud"]
        data_keys.extend(
            key for key in ("robot_mask", "point_valid_mask") if key in available
        )
        if bool(load_to_memory):
            self.replay_buffer = ReplayBuffer.copy_from_path(
                str(path), keys=data_keys
            )
        else:
            # Dense 4096/8192-point sequences are intentionally disk-backed;
            # copying a full training set to RAM can consume tens of gigabytes.
            self.replay_buffer = ReplayBuffer.create_from_path(str(path), mode="r")
        validation_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=float(val_ratio),
            seed=int(seed),
        )
        training_mask = downsample_mask(
            mask=~validation_mask,
            max_n=max_train_episodes,
            seed=int(seed),
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=int(horizon),
            pad_before=int(pad_before),
            pad_after=int(pad_after),
            episode_mask=training_mask,
            keys=data_keys,
        )
        self.train_mask = training_mask
        # Normalization is fitted on the selected training episodes only.
        # Validation trajectories must not influence action/state/point scales.
        self.normalizer_step_mask = _episode_step_mask(
            self.replay_buffer.episode_ends,
            training_mask,
        )
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.query_points = int(query_points)
        self.has_robot_mask = "robot_mask" in data_keys
        self.has_point_valid_mask = "point_valid_mask" in data_keys
        self.data_keys = tuple(data_keys)
        self.normalizer_chunk_steps = int(normalizer_chunk_steps)
        self.episode_ids = episode_ids
        self.episode_id_source = episode_id_source

    def get_validation_dataset(self) -> "AQRDP3Dataset":
        validation = copy.copy(self)
        validation.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            keys=self.data_keys,
        )
        validation.train_mask = ~self.train_mask
        return validation

    def get_normalizer(
        self,
        mode: str = "limits",
        *,
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        **kwargs,
    ) -> LinearNormalizer:
        # Masks deliberately stay outside the normalizer. They are boolean causal
        # metadata, not observations with train-set statistics.
        normalizer = LinearNormalizer()
        point_mask = (
            self.replay_buffer["point_valid_mask"]
            if self.has_point_valid_mask
            else None
        )
        arrays = {
            "action": (self.replay_buffer["action"], None),
            "agent_pos": (self.replay_buffer["state"], None),
            "point_cloud": (self.replay_buffer["point_cloud"], point_mask),
        }
        for key, (array, mask) in arrays.items():
            stats = _streaming_stats(
                array,
                chunk_steps=self.normalizer_chunk_steps,
                row_mask=mask,
                step_mask=self.normalizer_step_mask,
                workspace_crop_min=(
                    self.workspace_crop_min if key == "point_cloud" else None
                ),
                workspace_crop_max=(
                    self.workspace_crop_max if key == "point_cloud" else None
                ),
            )
            normalizer[key] = _normalizer_from_stats(
                stats,
                mode=mode,
                output_min=float(output_min),
                output_max=float(output_max),
                range_eps=float(range_eps),
            )
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def build_phase_sampling_pools(
        self,
        *,
        events_csv: str,
        n_obs_steps: int,
        pre_close_steps: int,
        post_stable_steps: int,
        post_secure_lift_steps: int,
        require_all_training_episodes: bool = True,
    ):
        from contactflow.dp3.aqr_dp3.sampling import (
            build_phase_sampling_pools,
        )

        return build_phase_sampling_pools(
            sequence_indices=self.sampler.indices,
            episode_ends=self.replay_buffer.episode_ends,
            episode_ids=self.episode_ids,
            training_episode_mask=self.train_mask,
            events_csv=events_csv,
            n_obs_steps=int(n_obs_steps),
            pre_close_steps=int(pre_close_steps),
            post_stable_steps=int(post_stable_steps),
            post_secure_lift_steps=int(post_secure_lift_steps),
            require_all_training_episodes=bool(
                require_all_training_episodes
            ),
        )

    @staticmethod
    def _mask(value: np.ndarray, *, steps: int, points: int, name: str) -> np.ndarray:
        mask = np.asarray(value)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if tuple(mask.shape) != (steps, points):
            raise ValueError(
                f"{name} must have sampled shape {(steps, points)}, got {mask.shape}."
            )
        return mask.astype(bool, copy=False)

    def _sample_to_data(self, sample) -> dict:
        state = sample["state"].astype(np.float32)
        points = sample["point_cloud"].astype(np.float32)
        steps, point_count = points.shape[:2]
        if self.has_robot_mask:
            robot_mask = self._mask(
                sample["robot_mask"],
                steps=steps,
                points=point_count,
                name="robot_mask",
            )
        else:
            robot_mask = np.zeros((steps, point_count), dtype=bool)
        if self.has_point_valid_mask:
            point_valid_mask = self._mask(
                sample["point_valid_mask"],
                steps=steps,
                points=point_count,
                name="point_valid_mask",
            )
        else:
            point_valid_mask = np.isfinite(points[..., :3]).all(axis=-1)
        point_valid_mask &= _workspace_mask_np(
            points,
            self.workspace_crop_min,
            self.workspace_crop_max,
        )
        robot_mask &= point_valid_mask
        return {
            "obs": {
                "point_cloud": points,
                "agent_pos": state,
                "robot_mask": robot_mask,
                "point_valid_mask": point_valid_mask,
            },
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data = self._sample_to_data(self.sampler.sample_sequence(index))
        return dict_apply(data, torch.from_numpy)

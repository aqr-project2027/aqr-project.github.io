from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from contactflow.dp3.action_query.knn_query import sample_raw_xyz


@dataclass(frozen=True)
class AQRBCSampleIndex:
    episode_id: int
    timestep: int


def load_metadata(traj_path: str | Path) -> dict[str, Any]:
    path = Path(traj_path).expanduser().resolve().with_suffix(".json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def available_episode_ids(
    traj_path: str | Path,
    metadata: dict[str, Any],
) -> list[int]:
    import h5py

    with h5py.File(Path(traj_path).expanduser().resolve(), "r") as h5:
        return [
            int(item["episode_id"])
            for item in metadata.get("episodes", [])
            if f"traj_{int(item['episode_id'])}" in h5
        ]


class AQRBCHDF5Dataset(Dataset):
    """Deterministic raw-2048 action-chunk dataset with no Oracle inputs."""

    def __init__(
        self,
        traj_path: str | Path,
        episode_ids: Sequence[int],
        *,
        chunk_steps: int,
        normalizer_scale: np.ndarray,
        normalizer_offset: np.ndarray,
        point_seed: int,
        max_samples: int = 0,
    ):
        import h5py

        self.traj_path = Path(traj_path).expanduser().resolve()
        self.chunk_steps = int(chunk_steps)
        self.normalizer_scale = np.asarray(
            normalizer_scale, dtype=np.float32
        ).reshape(-1)
        self.normalizer_offset = np.asarray(
            normalizer_offset, dtype=np.float32
        ).reshape(-1)
        self.point_seed = int(point_seed)
        self._h5 = None
        indices: list[AQRBCSampleIndex] = []
        with h5py.File(self.traj_path, "r") as h5:
            for episode_id in episode_ids:
                key = f"traj_{int(episode_id)}"
                if key not in h5:
                    continue
                action_count = int(h5[key]["actions"].shape[0])
                final_start = action_count - self.chunk_steps
                indices.extend(
                    AQRBCSampleIndex(int(episode_id), timestep)
                    for timestep in range(0, final_start + 1)
                )
        if max_samples > 0:
            indices = indices[: int(max_samples)]
        if not indices:
            raise ValueError("AQR-BC dataset selection produced no samples.")
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _data(self):
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self.traj_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.indices[int(index)]
        traj = self._data()[f"traj_{item.episode_id}"]
        timestep = item.timestep
        qpos = np.asarray(
            traj["obs/agent/qpos"][timestep], dtype=np.float32
        ).reshape(-1)
        previous_qpos = np.asarray(
            traj["obs/agent/qpos"][max(0, timestep - 1)], dtype=np.float32
        ).reshape(-1)
        tcp = np.asarray(
            traj["obs/extra/tcp_pose"][timestep], dtype=np.float32
        ).reshape(-1)
        state = np.concatenate([qpos, tcp], axis=0).astype(np.float32)
        raw_actions = np.asarray(
            traj["actions"][timestep : timestep + self.chunk_steps],
            dtype=np.float32,
        )
        policy_actions = (
            raw_actions * self.normalizer_scale + self.normalizer_offset
        ).astype(np.float32)
        seed = (
            self.point_seed
            + item.episode_id * 1_000_003
            + timestep * 9_176
        ) & 0xFFFFFFFF
        points = sample_raw_xyz(
            np.asarray(traj["obs/pointcloud/xyzw"][timestep]),
            np.random.default_rng(seed),
        )
        return {
            "point_xyz": torch.from_numpy(points.xyz),
            "point_valid_mask": torch.from_numpy(points.valid_mask),
            "state": torch.from_numpy(state),
            "previous_arm_qpos": torch.from_numpy(previous_qpos[:7].copy()),
            "target_action": torch.from_numpy(policy_actions),
            "episode_id": torch.tensor(item.episode_id, dtype=torch.long),
            "timestep": torch.tensor(timestep, dtype=torch.long),
        }


def compute_state_statistics(
    traj_path: str | Path,
    episode_ids: Sequence[int],
    chunk_steps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    import h5py

    total = np.zeros(16, dtype=np.float64)
    total_square = np.zeros(16, dtype=np.float64)
    count = 0
    with h5py.File(Path(traj_path).expanduser().resolve(), "r") as h5:
        for episode_id in episode_ids:
            key = f"traj_{int(episode_id)}"
            if key not in h5:
                continue
            traj = h5[key]
            action_count = int(traj["actions"].shape[0])
            final_start = action_count - int(chunk_steps)
            if final_start < 0:
                continue
            qpos = np.asarray(
                traj["obs/agent/qpos"][0 : final_start + 1],
                dtype=np.float64,
            )
            tcp = np.asarray(
                traj["obs/extra/tcp_pose"][0 : final_start + 1],
                dtype=np.float64,
            )
            state = np.concatenate([qpos, tcp], axis=-1)
            total += state.sum(axis=0)
            total_square += np.square(state).sum(axis=0)
            count += len(state)
    if count == 0:
        raise ValueError("Cannot compute state statistics from an empty split.")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count

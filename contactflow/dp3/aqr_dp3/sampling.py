from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PHASE_NAMES = ("grasp_position", "stable_lift", "uniform")


@dataclass(frozen=True)
class PhaseSamplingPools:
    grasp_position: np.ndarray
    stable_lift: np.ndarray
    uniform: np.ndarray
    report: dict[str, Any]

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            "grasp_position": self.grasp_position,
            "stable_lift": self.stable_lift,
            "uniform": self.uniform,
        }


def _read_event_actions(path: str | Path) -> dict[int, dict[str, int]]:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expert event CSV does not exist: {csv_path}")
    events: dict[int, dict[str, int]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                episode_id = int(row["episode_id"])
                event = str(row["event"])
                action_index = int(row["action_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed expert event row in {csv_path}: {row}"
                ) from exc
            if event not in {"first_close", "stable_grasp", "secure_lift"}:
                continue
            episode_events = events.setdefault(episode_id, {})
            if event in episode_events:
                raise ValueError(
                    f"Duplicate {event} event for episode {episode_id} in "
                    f"{csv_path}"
                )
            episode_events[event] = action_index
    return events


def sequence_sample_anchor_steps(
    sequence_indices: np.ndarray,
    *,
    n_obs_steps: int,
) -> np.ndarray:
    """Return the replay-buffer action step supervised as policy slot1.

    SequenceSampler stores:
      [buffer_start, buffer_end, sample_start, sample_end].
    The unpadded logical sequence start is ``buffer_start - sample_start``.
    DP3 executes its first action at sequence offset ``n_obs_steps - 1``.
    """

    indices = np.asarray(sequence_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 4:
        raise ValueError("sequence_indices must have shape [samples,4]")
    if int(n_obs_steps) <= 0:
        raise ValueError("n_obs_steps must be positive")
    anchors = indices[:, 0] - indices[:, 2] + int(n_obs_steps) - 1
    if len(np.unique(anchors)) != len(anchors):
        raise ValueError(
            "SequenceSampler does not provide one unique sample per action step."
        )
    return anchors


def _inclusive_range(start: int, end: int) -> Iterable[int]:
    if end < start:
        return ()
    return range(int(start), int(end) + 1)


def build_phase_sampling_pools(
    *,
    sequence_indices: np.ndarray,
    episode_ends: np.ndarray,
    episode_ids: np.ndarray,
    training_episode_mask: np.ndarray,
    events_csv: str | Path,
    n_obs_steps: int,
    pre_close_steps: int = 8,
    post_stable_steps: int = 2,
    post_secure_lift_steps: int = 2,
    require_all_training_episodes: bool = True,
) -> PhaseSamplingPools:
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    training_mask = np.asarray(training_episode_mask, dtype=bool)
    if episode_ends.ndim != 1 or len(episode_ends) == 0:
        raise ValueError("episode_ends must be a non-empty vector")
    if episode_ids.shape != episode_ends.shape:
        raise ValueError("episode_ids must align with episode_ends")
    if len(np.unique(episode_ids)) != len(episode_ids):
        raise ValueError("episode_ids must be unique")
    if training_mask.shape != episode_ends.shape:
        raise ValueError("training_episode_mask must align with episode_ends")
    if min(
        int(pre_close_steps),
        int(post_stable_steps),
        int(post_secure_lift_steps),
    ) < 0:
        raise ValueError("phase sampling window sizes must be non-negative")

    anchors = sequence_sample_anchor_steps(
        sequence_indices,
        n_obs_steps=int(n_obs_steps),
    )
    dataset_index_by_global_step = {
        int(global_step): int(dataset_index)
        for dataset_index, global_step in enumerate(anchors)
    }
    events_by_episode = _read_event_actions(events_csv)

    grasp_indices: list[int] = []
    lift_indices: list[int] = []
    missing_events: dict[int, list[str]] = {}
    invalid_events: dict[int, str] = {}
    covered_episode_ids: list[int] = []

    episode_start = 0
    for episode_index, episode_end in enumerate(episode_ends):
        episode_end = int(episode_end)
        episode_length = episode_end - episode_start
        episode_id = int(episode_ids[episode_index])
        if not bool(training_mask[episode_index]):
            episode_start = episode_end
            continue

        episode_events = events_by_episode.get(episode_id, {})
        missing = [
            event
            for event in ("first_close", "stable_grasp", "secure_lift")
            if event not in episode_events
        ]
        if missing:
            missing_events[episode_id] = missing
            episode_start = episode_end
            continue

        first_close = int(episode_events["first_close"])
        stable_grasp = int(episode_events["stable_grasp"])
        secure_lift = int(episode_events["secure_lift"])
        if not (
            0 <= first_close <= stable_grasp <= secure_lift < episode_length
        ):
            invalid_events[episode_id] = (
                f"expected 0 <= close <= stable <= lift < {episode_length}; "
                f"got {(first_close, stable_grasp, secure_lift)}"
            )
            episode_start = episode_end
            continue

        grasp_start = max(0, first_close - int(pre_close_steps))
        grasp_end = min(
            episode_length - 1,
            stable_grasp + int(post_stable_steps),
        )
        lift_start = stable_grasp
        lift_end = min(
            episode_length - 1,
            secure_lift + int(post_secure_lift_steps),
        )

        for relative_step in _inclusive_range(grasp_start, grasp_end):
            global_step = episode_start + relative_step
            if global_step not in dataset_index_by_global_step:
                raise ValueError(
                    f"No SequenceSampler sample anchors episode {episode_id} "
                    f"action step {relative_step}."
                )
            grasp_indices.append(dataset_index_by_global_step[global_step])
        for relative_step in _inclusive_range(lift_start, lift_end):
            global_step = episode_start + relative_step
            if global_step not in dataset_index_by_global_step:
                raise ValueError(
                    f"No SequenceSampler sample anchors episode {episode_id} "
                    f"action step {relative_step}."
                )
            lift_indices.append(dataset_index_by_global_step[global_step])
        covered_episode_ids.append(episode_id)
        episode_start = episode_end

    if require_all_training_episodes and (missing_events or invalid_events):
        details: list[str] = []
        if missing_events:
            preview = list(sorted(missing_events.items()))[:10]
            details.append(f"missing={preview}")
        if invalid_events:
            preview = list(sorted(invalid_events.items()))[:10]
            details.append(f"invalid={preview}")
        raise ValueError(
            "Expert events do not cover every selected training episode: "
            + "; ".join(details)
        )

    grasp = np.unique(np.asarray(grasp_indices, dtype=np.int64))
    stable_lift = np.unique(np.asarray(lift_indices, dtype=np.int64))
    uniform = np.arange(len(sequence_indices), dtype=np.int64)
    if grasp.size == 0 or stable_lift.size == 0 or uniform.size == 0:
        raise ValueError(
            "Phase-balanced sampling requires non-empty grasp, lift, and "
            "uniform pools."
        )
    report = {
        "events_csv": str(Path(events_csv).expanduser().resolve()),
        "deployment_input_safe": True,
        "privileged_events_entered_batch": False,
        "selected_training_episodes": int(np.sum(training_mask)),
        "covered_training_episodes": len(covered_episode_ids),
        "missing_training_episodes": len(missing_events),
        "invalid_training_episodes": len(invalid_events),
        "windows": {
            "grasp_position": {
                "start": f"first_close-{int(pre_close_steps)}",
                "end": f"stable_grasp+{int(post_stable_steps)}",
            },
            "stable_lift": {
                "start": "stable_grasp",
                "end": f"secure_lift+{int(post_secure_lift_steps)}",
            },
        },
        "pool_sizes": {
            "grasp_position": int(grasp.size),
            "stable_lift": int(stable_lift.size),
            "uniform": int(uniform.size),
        },
        "pool_overlap": {
            "grasp_and_stable_lift": int(
                np.intersect1d(grasp, stable_lift).size
            ),
        },
        "episode_id_range": {
            "minimum": (
                int(min(covered_episode_ids)) if covered_episode_ids else None
            ),
            "maximum": (
                int(max(covered_episode_ids)) if covered_episode_ids else None
            ),
        },
    }
    return PhaseSamplingPools(
        grasp_position=grasp,
        stable_lift=stable_lift,
        uniform=uniform,
        report=report,
    )


def _exact_counts(
    ratios: Mapping[str, float],
    total: int,
) -> dict[str, int]:
    total = int(total)
    if total <= 0:
        raise ValueError("total samples must be positive")
    values = np.asarray([float(ratios[name]) for name in PHASE_NAMES])
    if np.any(values < 0.0) or not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("phase sampling ratios must be non-negative and sum to 1")
    raw = values * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    fractional = raw - counts
    order = np.argsort(-fractional, kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return {
        name: int(count)
        for name, count in zip(PHASE_NAMES, counts.tolist())
    }


class PhaseBalancedIndexSampler:
    """Deterministic exact-ratio index sampler for one training epoch."""

    def __init__(
        self,
        *,
        pools: PhaseSamplingPools,
        ratios: Mapping[str, float],
        num_samples: int,
        seed: int,
    ):
        self.pools = pools.as_mapping()
        self.ratios = {name: float(ratios[name]) for name in PHASE_NAMES}
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        self.counts = _exact_counts(self.ratios, self.num_samples)
        for name, count in self.counts.items():
            if count > 0 and len(self.pools[name]) == 0:
                raise ValueError(f"phase pool {name!r} is empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        selected: list[np.ndarray] = []
        for name in PHASE_NAMES:
            count = self.counts[name]
            if count == 0:
                continue
            pool = self.pools[name]
            selected.append(
                rng.choice(
                    pool,
                    size=count,
                    replace=count > len(pool),
                ).astype(np.int64, copy=False)
            )
        indices = np.concatenate(selected, axis=0)
        if len(indices) != self.num_samples:
            raise RuntimeError("phase sampler produced the wrong epoch length")
        rng.shuffle(indices)
        return iter(indices.tolist())

    def report(self) -> dict[str, Any]:
        return {
            "mode": "phase_balanced",
            "seed": self.seed,
            "samples_per_epoch": self.num_samples,
            "ratios": dict(self.ratios),
            "exact_counts_per_epoch": dict(self.counts),
        }

    def epoch_report(self) -> dict[str, Any]:
        result = self.report()
        result["epoch_seed_index"] = self.epoch
        return result

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from contactflow.dp3.aqr_dp3.sampling import (
    PhaseBalancedIndexSampler,
    build_phase_sampling_pools,
    sequence_sample_anchor_steps,
)


def _sequence_indices(
    episode_lengths: list[int],
    *,
    horizon: int = 4,
    pad_before: int = 1,
    pad_after: int = 2,
) -> np.ndarray:
    rows = []
    episode_start = 0
    for length in episode_lengths:
        for logical_start in range(
            -pad_before,
            length - horizon + pad_after + 1,
        ):
            buffer_start = max(logical_start, 0) + episode_start
            buffer_end = min(logical_start + horizon, length) + episode_start
            sample_start = buffer_start - (logical_start + episode_start)
            sample_end = horizon - (
                logical_start + horizon + episode_start - buffer_end
            )
            rows.append(
                [buffer_start, buffer_end, sample_start, sample_end]
            )
        episode_start += length
    return np.asarray(rows, dtype=np.int64)


class AQRDP3PhaseSamplerTest(unittest.TestCase):
    def _events_csv(self, directory: Path, *, omit_episode: int | None = None):
        path = directory / "events.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("episode_id", "event", "action_index"),
            )
            writer.writeheader()
            for episode_id in (100, 101):
                if episode_id == omit_episode:
                    continue
                for event, action_index in (
                    ("first_close", 5),
                    ("stable_grasp", 6),
                    ("secure_lift", 8),
                ):
                    writer.writerow(
                        {
                            "episode_id": episode_id,
                            "event": event,
                            "action_index": action_index,
                        }
                    )
        return path

    def test_anchor_mapping_covers_every_action_step_once(self):
        indices = _sequence_indices([10, 10])
        anchors = sequence_sample_anchor_steps(indices, n_obs_steps=2)
        np.testing.assert_array_equal(anchors, np.arange(20))

    def test_builds_expected_phase_windows(self):
        indices = _sequence_indices([10, 10])
        with tempfile.TemporaryDirectory() as directory:
            events = self._events_csv(Path(directory))
            pools = build_phase_sampling_pools(
                sequence_indices=indices,
                episode_ends=np.asarray([10, 20]),
                episode_ids=np.asarray([100, 101]),
                training_episode_mask=np.asarray([True, True]),
                events_csv=events,
                n_obs_steps=2,
                pre_close_steps=2,
                post_stable_steps=1,
                post_secure_lift_steps=1,
            )

        self.assertEqual(len(pools.grasp_position), 10)
        self.assertEqual(len(pools.stable_lift), 8)
        self.assertEqual(len(pools.uniform), 20)
        self.assertEqual(
            pools.report["pool_overlap"]["grasp_and_stable_lift"],
            4,
        )
        np.testing.assert_array_equal(
            pools.grasp_position[:5],
            [3, 4, 5, 6, 7],
        )
        np.testing.assert_array_equal(
            pools.stable_lift[:4],
            [6, 7, 8, 9],
        )

    def test_sampler_has_exact_counts_and_is_epoch_deterministic(self):
        indices = _sequence_indices([10, 10])
        with tempfile.TemporaryDirectory() as directory:
            events = self._events_csv(Path(directory))
            pools = build_phase_sampling_pools(
                sequence_indices=indices,
                episode_ends=np.asarray([10, 20]),
                episode_ids=np.asarray([100, 101]),
                training_episode_mask=np.asarray([True, True]),
                events_csv=events,
                n_obs_steps=2,
                pre_close_steps=2,
                post_stable_steps=1,
                post_secure_lift_steps=1,
            )

        sampler = PhaseBalancedIndexSampler(
            pools=pools,
            ratios={
                "grasp_position": 0.45,
                "stable_lift": 0.30,
                "uniform": 0.25,
            },
            num_samples=20,
            seed=42,
        )
        self.assertEqual(
            sampler.report()["exact_counts_per_epoch"],
            {
                "grasp_position": 9,
                "stable_lift": 6,
                "uniform": 5,
            },
        )
        sampler.set_epoch(7)
        first = list(iter(sampler))
        sampler.set_epoch(7)
        second = list(iter(sampler))
        self.assertEqual(first, second)
        sampler.set_epoch(8)
        third = list(iter(sampler))
        self.assertNotEqual(first, third)
        self.assertEqual(len(first), 20)
        self.assertTrue(all(0 <= index < 20 for index in first))

    def test_missing_training_episode_events_fail_closed(self):
        indices = _sequence_indices([10, 10])
        with tempfile.TemporaryDirectory() as directory:
            events = self._events_csv(Path(directory), omit_episode=101)
            with self.assertRaisesRegex(
                ValueError,
                "do not cover every selected training episode",
            ):
                build_phase_sampling_pools(
                    sequence_indices=indices,
                    episode_ends=np.asarray([10, 20]),
                    episode_ids=np.asarray([100, 101]),
                    training_episode_mask=np.asarray([True, True]),
                    events_csv=events,
                    n_obs_steps=2,
                    pre_close_steps=2,
                    post_stable_steps=1,
                    post_secure_lift_steps=1,
                )

    def test_duplicate_episode_ids_fail_closed(self):
        indices = _sequence_indices([10, 10])
        with tempfile.TemporaryDirectory() as directory:
            events = self._events_csv(Path(directory))
            with self.assertRaisesRegex(
                ValueError,
                "episode_ids must be unique",
            ):
                build_phase_sampling_pools(
                    sequence_indices=indices,
                    episode_ends=np.asarray([10, 20]),
                    episode_ids=np.asarray([100, 100]),
                    training_episode_mask=np.asarray([True, True]),
                    events_csv=events,
                    n_obs_steps=2,
                    pre_close_steps=2,
                    post_stable_steps=1,
                    post_secure_lift_steps=1,
                )


if __name__ == "__main__":
    unittest.main()

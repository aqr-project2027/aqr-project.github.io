from __future__ import annotations

import unittest
from unittest import mock

import torch

from contactflow.dp3.aqr_dp3.config import PointCloudFrontEndConfig
from contactflow.dp3.aqr_dp3.pointcloud import (
    FILL_BANK_ID,
    GLOBAL_BANK_ID,
    TCP_SWEPT_BANK_ID,
    PointCloudFrontEnd,
    _batched_farthest_point_sample,
    _farthest_point_sample,
)


def _raw_cloud(point_count: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    xyz = 0.06 * (torch.rand((1, point_count, 3), generator=generator) - 0.5)
    source_id = torch.arange(point_count, dtype=torch.float32).reshape(1, -1, 1)
    return torch.cat([xyz, source_id], dim=-1)


class PointCloudFrontEndTest(unittest.TestCase):
    def test_full_valid_fps_uses_one_batched_backend_call(self):
        xyz = torch.linspace(-0.2, 0.2, steps=4 * 32 * 3).reshape(4, 32, 3)
        eligible = torch.ones((4, 32), dtype=torch.bool)
        expected = torch.arange(12).reshape(1, -1).expand(4, -1)
        with mock.patch(
            "contactflow.dp3.aqr_dp3.pointcloud.farthest_point_indices",
            return_value=expected,
        ) as backend:
            actual, used_fast_path = _batched_farthest_point_sample(
                xyz,
                eligible,
                12,
            )
        self.assertTrue(bool(used_fast_path.all()))
        backend.assert_called_once()
        torch.testing.assert_close(actual, expected)

    def test_batched_fps_matches_previous_scalar_indices(self):
        generator = torch.Generator().manual_seed(17)
        xyz = torch.rand((3, 37, 3), generator=generator)
        eligible = torch.ones((3, 37), dtype=torch.bool)
        actual, used_fast_path = _batched_farthest_point_sample(
            xyz,
            eligible,
            16,
        )
        expected = torch.stack(
            [
                _farthest_point_sample(xyz[index], eligible[index], 16)
                for index in range(3)
            ]
        )
        self.assertTrue(bool(used_fast_path.all()))
        torch.testing.assert_close(actual, expected)

    def test_masked_fps_preserves_exact_per_sample_fallback(self):
        generator = torch.Generator().manual_seed(5)
        xyz = torch.rand((3, 31, 3), generator=generator)
        eligible = torch.ones((3, 31), dtype=torch.bool)
        eligible[0, 2] = False
        eligible[1, 7] = False
        actual, used_fast_path = _batched_farthest_point_sample(
            xyz,
            eligible,
            12,
        )
        expected = torch.stack(
            [
                _farthest_point_sample(xyz[index], eligible[index], 12)
                for index in range(3)
            ]
        )
        torch.testing.assert_close(
            used_fast_path,
            torch.tensor([False, False, True]),
        )
        torch.testing.assert_close(actual, expected)
        for batch_index, invalid_index in enumerate((2, 7)):
            self.assertNotIn(invalid_index, actual[batch_index].tolist())

    def test_design_budgets_and_staged_modes_are_locked(self):
        for mode in ("baseline_fps2048", "cache_only", "dual_bank"):
            config = PointCloudFrontEndConfig(mode=mode)
            self.assertEqual(config.p_enc, 2048)
            self.assertEqual(config.enc_global_points, 1280)
            self.assertEqual(config.enc_tcp_swept_points, 768)
            self.assertEqual(config.p_query, 4096)
        self.assertFalse(
            PointCloudFrontEndConfig(mode="cache_only").aqr_enabled
        )
        self.assertTrue(PointCloudFrontEndConfig(mode="dual_bank").aqr_enabled)
        with self.assertRaisesRegex(ValueError, "4096 or 8192"):
            PointCloudFrontEndConfig(query_points=2048)
        with self.assertRaisesRegex(ValueError, "configured together"):
            PointCloudFrontEndConfig(workspace_crop_min=(-0.5, -0.5, -0.05))
        with self.assertRaisesRegex(ValueError, "strictly below"):
            PointCloudFrontEndConfig(
                workspace_crop_min=(0.5, -0.5, -0.05),
                workspace_crop_max=(0.5, 0.65, 0.5),
            )

    def test_validation_quota_presets_keep_the_encoder_budget_fixed(self):
        for global_points, tcp_points in (
            (1536, 512),
            (1280, 768),
            (1024, 1024),
        ):
            with self.subTest(
                global_points=global_points, tcp_points=tcp_points
            ):
                config = PointCloudFrontEndConfig(
                    enc_global_points=global_points,
                    enc_tcp_swept_points=tcp_points,
                )
                self.assertEqual(
                    config.enc_global_points
                    + config.enc_tcp_swept_points,
                    config.enc_points,
                )
        with self.assertRaisesRegex(ValueError, "Protected encoder quotas"):
            PointCloudFrontEndConfig(
                enc_global_points=1408,
                enc_tcp_swept_points=640,
            )

    def test_disabled_protected_sampling_does_not_report_false_failure(self):
        result = PointCloudFrontEnd(
            PointCloudFrontEndConfig(mode="baseline_fps2048")
        )(
            _raw_cloud(4096),
            torch.zeros((1, 2, 3)),
        )
        self.assertFalse(
            bool(result.diagnostics["tcp_swept_zero_coverage"].any())
        )
        self.assertTrue(
            bool(result.diagnostics["encoder_fps_batched_fast_path"].all())
        )

    def test_sampling_is_deterministic_and_preserves_masks(self):
        raw = _raw_cloud(4300)
        valid = torch.ones((1, 4300), dtype=torch.bool)
        valid[:, 11] = False
        robot = torch.zeros_like(valid)
        robot[:, :137] = True
        tcp = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]])
        frontend = PointCloudFrontEnd()
        first = frontend(raw, tcp, valid, robot, sample_key=19)
        second = frontend(raw, tcp, valid, robot, sample_key=19)
        for name in (
            "enc_points",
            "query_points",
            "enc_valid_mask",
            "query_valid_mask",
            "enc_robot_mask",
            "query_robot_mask",
            "enc_source_indices",
            "query_source_indices",
            "enc_bank_ids",
        ):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))
        self.assertNotIn(11, first.enc_source_indices.tolist()[0])
        self.assertNotIn(11, first.query_source_indices.tolist()[0])
        self.assertFalse(
            bool(first.diagnostics["encoder_fps_batched_fast_path"].any())
        )
        self.assertGreaterEqual(
            int(first.diagnostics["query_robot_ratio"][0] * 4096), 0
        )

    def test_query_bank_is_independent_and_gathered_directly_from_raw(self):
        raw = _raw_cloud(5000, seed=13)
        result = PointCloudFrontEnd()(
            raw,
            torch.zeros((1, 2, 3)),
            sample_key=23,
        )
        query_indices = result.query_source_indices[0]
        enc_indices = result.enc_source_indices[0]
        self.assertEqual(int(query_indices.ge(0).sum()), 4096)
        self.assertEqual(int(torch.unique(query_indices).numel()), 4096)
        self.assertTrue(
            bool((~torch.isin(query_indices, enc_indices)).any()),
            "P_query must not be a resampling of P_enc.",
        )
        torch.testing.assert_close(
            result.query_points[0, :, 3],
            query_indices.to(dtype=torch.float32),
        )

    def test_xyzrgb_features_never_change_fps_or_query_indices(self):
        generator = torch.Generator().manual_seed(53)
        xyz = 0.06 * (
            torch.rand((1, 4300, 3), generator=generator) - 0.5
        )
        rgb_a = torch.rand((1, 4300, 3), generator=generator)
        rgb_b = 1.0 - rgb_a
        frontend = PointCloudFrontEnd()
        first = frontend(
            torch.cat([xyz, rgb_a], dim=-1),
            torch.zeros((1, 2, 3)),
            sample_key=59,
        )
        second = frontend(
            torch.cat([xyz, rgb_b], dim=-1),
            torch.zeros((1, 2, 3)),
            sample_key=59,
        )
        torch.testing.assert_close(
            first.enc_source_indices, second.enc_source_indices
        )
        torch.testing.assert_close(
            first.query_source_indices, second.query_source_indices
        )
        torch.testing.assert_close(
            first.enc_points[..., :3], second.enc_points[..., :3]
        )
        self.assertGreater(
            int(
                torch.count_nonzero(
                    first.query_points[..., 3:6]
                    - second.query_points[..., 3:6]
                )
            ),
            0,
        )

    def test_protected_union_deduplicates_and_meets_both_quotas(self):
        raw = _raw_cloud(4300, seed=29)
        tcp = torch.tensor(
            [[[-0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]]
        )
        result = PointCloudFrontEnd(
            PointCloudFrontEndConfig(mode="dual_bank")
        )(raw, tcp, sample_key=31)
        indices = result.enc_source_indices[0]
        self.assertEqual(int(result.enc_valid_mask.sum()), 2048)
        self.assertEqual(int(torch.unique(indices).numel()), 2048)
        self.assertEqual(
            int((result.enc_bank_ids == GLOBAL_BANK_ID).sum()), 1280
        )
        self.assertEqual(
            int((result.enc_bank_ids == TCP_SWEPT_BANK_ID).sum()), 768
        )
        self.assertEqual(int((result.enc_bank_ids == FILL_BANK_ID).sum()), 0)
        self.assertEqual(
            int(result.diagnostics["enc_duplicate_count"][0]), 0
        )

    def test_workspace_filter_blocks_outliers_from_fps_and_query_bank(self):
        inside_count = 4300
        outside_count = 23
        raw = _raw_cloud(inside_count)
        outside_xyz = torch.tensor(
            [[[5.0, 3.0, 0.1]]], dtype=raw.dtype
        ).expand(1, outside_count, 3)
        outside_ids = torch.arange(
            inside_count,
            inside_count + outside_count,
            dtype=raw.dtype,
        ).reshape(1, -1, 1)
        raw = torch.cat(
            [raw, torch.cat([outside_xyz, outside_ids], dim=-1)],
            dim=1,
        )
        frontend = PointCloudFrontEnd(
            PointCloudFrontEndConfig(
                workspace_crop_min=(-0.5, -0.5, -0.05),
                workspace_crop_max=(0.5, 0.65, 0.5),
            )
        )
        result = frontend(
            raw,
            torch.zeros((1, 2, 3)),
            raw_valid_mask=torch.ones(
                (1, inside_count + outside_count), dtype=torch.bool
            ),
            sample_key=41,
        )
        self.assertEqual(
            int(result.diagnostics["raw_workspace_rejected_count"][0]),
            outside_count,
        )
        self.assertTrue(bool((result.enc_source_indices < inside_count).all()))
        self.assertTrue(bool((result.query_source_indices < inside_count).all()))
        self.assertTrue(
            bool(
                (
                    result.enc_points[..., :3]
                    < torch.tensor([0.5, 0.65, 0.5])
                ).all()
            )
        )


if __name__ == "__main__":
    unittest.main()

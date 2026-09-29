from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from contactflow.dp3.aqr_dp3.config import AQRDP3Config
from contactflow.dp3.aqr_dp3.dataset import (
    AQRDP3Dataset,
    _episode_step_mask,
    _streaming_stats,
)
from contactflow.dp3.aqr_dp3.geometry import (
    ActionToTool,
    RelativeMultiScaleQuery,
    ToolTrajectory,
    anchor_identity_ids,
    build_current_tcp_only_trajectory,
    build_slot1_sweep_trajectory,
)
from contactflow.dp3.aqr_dp3.model import (
    PostDiffusionActionRefiner,
    SlotWiseLocalResidual,
    predicted_clean_sample,
    relative_action_improvement_loss,
)
from contactflow.dp3.aqr_dp3.policy import AQRDP3
from contactflow.dp3.aqr_bc.torch_trajectory import (
    pose_wxyz_to_matrix_torch,
)


class AQRDP3ConfigTest(unittest.TestCase):
    def test_post_diffusion_contract_is_explicit(self):
        legacy = AQRDP3Config.from_mapping({"experiment": "A3"})
        self.assertEqual(legacy.refinement_stage, "denoising")
        post = AQRDP3Config.from_mapping({"experiment": "A6_POST"})
        self.assertEqual(post.refinement_stage, "post_diffusion")
        self.assertEqual(post.query_source, "final_base")
        self.assertTrue(post.post_base_frozen)
        self.assertTrue(post.anchor_preserving_query)
        self.assertEqual(post.post_refiner_input_mode, "full")
        with self.assertRaisesRegex(ValueError, "query_source=final_base"):
            AQRDP3Config.from_mapping(
                {
                    "experiment": "A3",
                    "refinement_stage": "post_diffusion",
                }
            )

    def test_post_refiner_structural_modes_validate_query_requirements(self):
        action_only = AQRDP3Config.from_mapping(
            {
                "experiment": "A6_POST",
                "post_refiner_input_mode": "action_only",
                "cache_query_features": False,
            }
        )
        self.assertFalse(action_only.post_refiner_uses_global_feature)
        self.assertFalse(action_only.post_refiner_uses_local_query)
        global_only = AQRDP3Config.from_mapping(
            {
                "experiment": "A6_POST",
                "post_refiner_input_mode": "global",
                "cache_query_features": False,
            }
        )
        self.assertTrue(global_only.post_refiner_uses_global_feature)
        self.assertFalse(global_only.post_refiner_uses_local_query)
        current_tcp = AQRDP3Config.from_mapping(
            {
                "experiment": "A6_POST",
                "post_refiner_input_mode": "current_tcp_local",
            }
        )
        self.assertTrue(current_tcp.post_refiner_uses_local_query)
        with self.assertRaisesRegex(ValueError, "cache_query_features"):
            AQRDP3Config.from_mapping(
                {
                    "experiment": "A6_POST",
                    "post_refiner_input_mode": "current_tcp_local",
                    "cache_query_features": False,
                }
            )

    def test_bounded_residual_is_explicit_and_legacy_safe(self):
        legacy = AQRDP3Config.from_mapping({"experiment": "A3"})
        self.assertEqual(legacy.relation_fusion_mode, "convex")
        self.assertFalse(legacy.bounded_action_residual)

        bounded = AQRDP3Config.from_mapping(
            {
                "experiment": "A3",
                "relation_fusion_mode": "semantic_residual",
                "relation_gate_maximum": 0.50,
                "bounded_action_residual": True,
                "action_residual_max_norm": 0.30,
                "action_residual_min_progress_ratio": 0.50,
                "action_residual_protect_non_arm": True,
            }
        )
        self.assertEqual(bounded.relation_fusion_mode, "semantic_residual")
        self.assertEqual(bounded.relation_gate_maximum, 0.50)
        self.assertTrue(bounded.bounded_action_residual)
        with self.assertRaisesRegex(ValueError, "relation_gate_maximum"):
            AQRDP3Config.from_mapping(
                {"experiment": "A3", "relation_gate_maximum": 1.1}
            )
        with self.assertRaisesRegex(ValueError, "progress_ratio"):
            AQRDP3Config.from_mapping(
                {
                    "experiment": "A3",
                    "action_residual_min_progress_ratio": -0.1,
                }
            )

    def test_improvement_loss_contract_is_explicit_and_validated(self):
        legacy = AQRDP3Config.from_mapping({"experiment": "A3"})
        self.assertFalse(legacy.improvement_loss_enabled)
        self.assertEqual(legacy.improvement_loss_weight, 0.0)
        self.assertFalse(legacy.improvement_loss_clean_only)

        enabled = AQRDP3Config.from_mapping(
            {
                "experiment": "A3",
                "improvement_loss_enabled": True,
                "improvement_loss_weight": 0.2,
                "improvement_margin_ratio": 0.05,
                "improvement_loss_clean_only": True,
            }
        )
        self.assertTrue(enabled.improvement_loss_enabled)
        self.assertEqual(enabled.improvement_loss_weight, 0.2)
        self.assertEqual(enabled.improvement_margin_ratio, 0.05)
        self.assertTrue(enabled.improvement_loss_clean_only)
        with self.assertRaisesRegex(ValueError, "positive weight"):
            AQRDP3Config.from_mapping(
                {
                    "experiment": "A3",
                    "improvement_loss_enabled": True,
                }
            )


    def test_stage_presets_resolve_to_the_causal_experiment_contract(self):
        expected = {
            "AQR_OFF": (
                False,
                "baseline_fps2048",
                False,
                "off",
                "normal",
            ),
            "P0": (False, "baseline_fps2048", False, "off", "normal"),
            "P1": (False, "dual_bank", False, "off", "normal"),
            "P2": (False, "cache_only", True, "off", "normal"),
            "P3": (True, "dual_bank", True, "static", "normal"),
            "B0": (False, "baseline_fps2048", False, "off", "normal"),
            "B1": (False, "dual_bank", False, "off", "normal"),
            "S0": (True, "baseline_fps2048", True, "static", "normal"),
            "S1": (True, "dual_bank", True, "static", "normal"),
            "A0": (
                True,
                "baseline_fps2048",
                True,
                "predicted_clean",
                "normal",
            ),
            "A1": (
                True,
                "baseline_fps2048",
                True,
                "predicted_clean",
                "normal",
            ),
            "A1_PROTECTED": (
                True,
                "dual_bank",
                True,
                "predicted_clean",
                "normal",
            ),
            "Q5": (
                True,
                "dual_bank",
                True,
                "predicted_clean",
                "shuffle_slots",
            ),
            "Q6": (
                True,
                "dual_bank",
                True,
                "predicted_clean",
                "zero",
            ),
        }
        for stage, contract in expected.items():
            with self.subTest(stage=stage):
                config = AQRDP3Config.from_mapping({"experiment": stage})
                actual = (
                    config.enabled,
                    config.front_end.mode,
                    config.cache_query_features,
                    config.query_source,
                    config.local_intervention,
                )
                self.assertEqual(actual, contract)

    def test_mainline_defaults_lock_slot1_query_and_one_late_update(self):
        config = AQRDP3Config.from_mapping({"experiment": "A1"})
        self.assertEqual(config.front_end.mode, "baseline_fps2048")
        self.assertEqual(config.front_end.query_points, 4096)
        self.assertEqual(config.local_action_slots, (1,))
        self.assertEqual(config.query_update_fractions, (0.75,))
        self.assertTrue(config.slot1_sweep_enabled)
        self.assertTrue(config.current_tcp_fallback_enabled)
        self.assertEqual(config.exec_steps, 1)
        self.assertFalse(config.future_aux)
        stackcube = AQRDP3Config.from_mapping(
            {"experiment": "A1", "task_profile": "stack_cube"}
        )
        self.assertEqual(stackcube.task_profile, "stack_cube")
        pullcubetool = AQRDP3Config.from_mapping(
            {"experiment": "A1", "task_profile": "pull_cube_tool"}
        )
        self.assertEqual(pullcubetool.task_profile, "pull_cube_tool")
        with self.assertRaisesRegex(ValueError, "task_profile"):
            AQRDP3Config.from_mapping(
                {"experiment": "A1", "task_profile": "unknown_task"}
            )

        query3 = AQRDP3Config.from_mapping({"experiment": "A1_QUERY3"})
        self.assertEqual(
            query3.query_update_fractions,
            (0.55, 0.75, 0.90),
        )
        with self.assertRaisesRegex(ValueError, "only permits"):
            AQRDP3Config.from_mapping(
                {"experiment": "A1", "local_action_slots": [1, 2]}
            )
        with self.assertRaisesRegex(ValueError, "sweep and/or"):
            AQRDP3Config.from_mapping(
                {
                    "experiment": "A1",
                    "slot1_sweep_enabled": False,
                    "current_tcp_fallback_enabled": False,
                }
            )

    def test_removed_and_unknown_mainline_keys_are_rejected(self):
        for key in ("adaptive_commit", "hazard", "consequence_model", "recovery"):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "Removed legacy switches"
            ):
                AQRDP3Config.from_mapping({"experiment": "A1", key: True})
        with self.assertRaisesRegex(ValueError, "Unknown AQR-DP3 settings"):
            AQRDP3Config.from_mapping(
                {"experiment": "A1", "silent_stale_override": True}
            )

    def test_privileged_robot_mask_feature_is_off_by_default(self):
        config = AQRDP3Config.from_mapping({"experiment": "A1"})
        self.assertFalse(config.use_robot_mask_feature)
        enabled = AQRDP3Config.from_mapping(
            {"experiment": "A1", "use_robot_mask_feature": True}
        )
        self.assertTrue(enabled.use_robot_mask_feature)
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            AQRDP3Config.from_mapping(
                {"experiment": "A1", "use_robot_mask_feature": "true"}
            )


class DeploymentRobotMaskSafetyTest(unittest.TestCase):
    @staticmethod
    def _policy(use_robot_mask_feature: bool) -> AQRDP3:
        policy = object.__new__(AQRDP3)
        torch.nn.Module.__init__(policy)
        # Match ModuleAttrMixin's device anchor in this lightweight fixture.
        policy.register_parameter("_dummy_variable", torch.nn.Parameter(torch.empty(0)))
        policy.aqr = AQRDP3Config.from_mapping(
            {
                "experiment": "A1",
                "use_robot_mask_feature": use_robot_mask_feature,
            }
        )
        policy.state_dim = 16
        policy.point_channels = 3
        policy.n_obs_steps = 2
        return policy

    @staticmethod
    def _obs(include_mask: bool) -> dict[str, torch.Tensor]:
        obs = {
            "point_cloud": torch.zeros((1, 2, 4096, 3)),
            "agent_pos": torch.zeros((1, 2, 16)),
            "point_valid_mask": torch.ones((1, 2, 4096), dtype=torch.bool),
        }
        if include_mask:
            obs["robot_mask"] = torch.ones(
                (1, 2, 4096), dtype=torch.bool
            )
        return obs

    def test_default_ignores_supplied_privileged_mask(self):
        _, _, _, robot = self._policy(False)._validate_observation(
            self._obs(include_mask=True)
        )
        self.assertEqual(int(torch.count_nonzero(robot)), 0)

    def test_deployable_mask_opt_in_requires_explicit_input(self):
        policy = self._policy(True)
        with self.assertRaisesRegex(KeyError, "validated deployable source"):
            policy._validate_observation(self._obs(include_mask=False))
        _, _, _, robot = policy._validate_observation(
            self._obs(include_mask=True)
        )
        self.assertTrue(bool(robot.all()))


class RelativeActionImprovementLossTest(unittest.TestCase):
    def test_zero_residual_is_pushed_toward_expert_when_base_is_wrong(self):
        residual = torch.zeros((1, 1, 2), requires_grad=True)
        loss, diagnostics = relative_action_improvement_loss(
            base_action=torch.tensor([[[1.0, 0.0]]]),
            applied_residual=residual,
            expert_action=torch.zeros((1, 1, 2)),
            active_gate=torch.ones(1),
            margin_ratio=0.05,
        )
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertGreater(residual.grad[0, 0, 0].item(), 0.0)
        self.assertEqual(
            diagnostics["improvement_active_fraction"].item(), 1.0
        )

    def test_harmful_correction_is_penalized_and_helpful_one_satisfies_margin(self):
        kwargs = {
            "base_action": torch.tensor([[[1.0, 0.0]]]),
            "expert_action": torch.zeros((1, 1, 2)),
            "active_gate": torch.ones(1),
            "margin_ratio": 0.05,
        }
        harmful, harmful_diag = relative_action_improvement_loss(
            applied_residual=torch.tensor([[[0.1, 0.0]]]), **kwargs
        )
        helpful, helpful_diag = relative_action_improvement_loss(
            applied_residual=torch.tensor([[[-0.2, 0.0]]]), **kwargs
        )
        self.assertGreater(harmful.item(), 0.0)
        self.assertEqual(helpful.item(), 0.0)
        self.assertEqual(
            harmful_diag["improvement_no_harm_violation_fraction"].item(),
            1.0,
        )
        self.assertEqual(
            helpful_diag["improvement_margin_satisfied_fraction"].item(),
            1.0,
        )

    def test_inactive_diffusion_step_contributes_exactly_zero(self):
        residual = torch.tensor([[[0.5, 0.0]]], requires_grad=True)
        loss, diagnostics = relative_action_improvement_loss(
            base_action=torch.tensor([[[1.0, 0.0]]]),
            applied_residual=residual,
            expert_action=torch.zeros((1, 1, 2)),
            active_gate=torch.zeros(1),
            margin_ratio=0.05,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(
            diagnostics["improvement_active_fraction"].item(), 0.0
        )

    def test_sample_mask_excludes_synthetic_offset_from_loss_and_gradient(self):
        residual = torch.zeros((2, 1, 2), requires_grad=True)
        loss, diagnostics = relative_action_improvement_loss(
            base_action=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]),
            applied_residual=residual,
            expert_action=torch.zeros((2, 1, 2)),
            active_gate=torch.ones(2),
            margin_ratio=0.05,
            # First item represents a synthetic-offset sample; only the second
            # naturally predicted item is eligible for improvement supervision.
            sample_mask=torch.tensor([False, True]),
        )
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(residual.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(residual.grad[1, 0, 0].item(), 0.0)
        self.assertEqual(
            diagnostics["improvement_eligible_fraction"].item(), 0.5
        )
        self.assertEqual(
            diagnostics[
                "improvement_active_among_eligible_fraction"
            ].item(),
            1.0,
        )

    def test_all_ineligible_batch_returns_differentiable_zero(self):
        residual = torch.ones((2, 1, 2), requires_grad=True)
        loss, diagnostics = relative_action_improvement_loss(
            base_action=torch.ones((2, 1, 2)),
            applied_residual=residual,
            expert_action=torch.zeros((2, 1, 2)),
            active_gate=torch.ones(2),
            margin_ratio=0.05,
            sample_mask=torch.zeros(2, dtype=torch.bool),
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(residual.grad.abs().sum().item(), 0.0)
        self.assertEqual(
            diagnostics["improvement_eligible_fraction"].item(), 0.0
        )


class PredictedCleanSampleTest(unittest.TestCase):
    def test_epsilon_sample_and_v_prediction_recover_the_same_clean_action(self):
        clean = torch.tensor(
            [
                [[-0.2, 0.3], [0.4, -0.1]],
                [[0.1, -0.4], [0.2, 0.5]],
            ],
            dtype=torch.float32,
        )
        noise = torch.tensor(
            [
                [[0.6, -0.3], [0.2, 0.1]],
                [[-0.4, 0.7], [0.3, -0.2]],
            ],
            dtype=torch.float32,
        )
        alphas_cumprod = torch.tensor([0.81, 0.49], dtype=torch.float32)
        timestep = torch.tensor([0, 1], dtype=torch.long)
        alpha = alphas_cumprod[timestep].sqrt().reshape(2, 1, 1)
        sigma = (1.0 - alphas_cumprod[timestep]).sqrt().reshape(2, 1, 1)
        noisy = alpha * clean + sigma * noise
        outputs = {
            "epsilon": noise,
            "sample": clean,
            "v_prediction": alpha * noise - sigma * clean,
        }
        for prediction_type, model_output in outputs.items():
            with self.subTest(prediction_type=prediction_type):
                scheduler = SimpleNamespace(
                    config=SimpleNamespace(
                        prediction_type=prediction_type,
                        clip_sample=False,
                    ),
                    alphas_cumprod=alphas_cumprod,
                )
                recovered = predicted_clean_sample(
                    sample=noisy,
                    model_output=model_output,
                    timestep=timestep,
                    scheduler=scheduler,
                )
                torch.testing.assert_close(
                    recovered, clean, atol=1e-6, rtol=1e-6
                )


class ActionToToolTest(unittest.TestCase):
    def test_invalid_zero_quaternion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero quaternion"):
            pose_wxyz_to_matrix_torch(torch.zeros((1, 7)))

    def test_tcp_delta_pose_composes_translation_in_the_rotated_tool_frame(self):
        action_to_tool = ActionToTool(
            mode="tcp_delta_pose",
            horizon=2,
            state_dim=7,
            tcp_pose_start=0,
            keypoint_offsets_tcp={
                "tcp": (0.0, 0.0, 0.0),
                "tool_tip": (0.1, 0.0, 0.0),
            },
        )
        state = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        environment_action = torch.zeros((1, 2, 6), dtype=torch.float32)
        environment_action[0, 0, 5] = math.pi / 2.0
        environment_action[0, 1, 0] = 1.0
        trajectory = action_to_tool(
            policy_actions=environment_action.clone(),
            environment_actions=environment_action,
            state_history_raw=state,
        )
        torch.testing.assert_close(
            trajectory.keypoints_world[0, :, 0],
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            trajectory.keypoints_world[0, :, 1],
            torch.tensor([[0.0, 0.1, 0.0], [0.0, 1.1, 0.0]]),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertEqual(trajectory.names, ("tcp", "tool_tip"))

    def test_slot1_sweep_and_current_tcp_fallback_are_independent(self):
        matrices = torch.eye(4).reshape(1, 1, 4, 4)
        current = ToolTrajectory(
            keypoints_world=torch.tensor(
                [[[[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]]]
            ),
            rotations_world_from_tool=matrices[..., :3, :3],
            tcp_matrices_world=matrices,
            names=("tcp", "tool_tip"),
        )
        candidate_matrices = matrices.clone()
        candidate_matrices[..., 0, 3] = 0.2
        candidate = ToolTrajectory(
            keypoints_world=current.keypoints_world
            + torch.tensor([0.2, 0.0, 0.0]),
            rotations_world_from_tool=candidate_matrices[..., :3, :3],
            tcp_matrices_world=candidate_matrices,
            names=current.names,
        )

        sweep_only = build_slot1_sweep_trajectory(
            current=current,
            candidate_slot1=candidate,
            sweep_enabled=True,
            current_tcp_fallback_enabled=False,
            sweep_samples=4,
        )
        fallback_only = build_slot1_sweep_trajectory(
            current=current,
            candidate_slot1=candidate,
            sweep_enabled=False,
            current_tcp_fallback_enabled=True,
            sweep_samples=4,
        )
        both = build_slot1_sweep_trajectory(
            current=current,
            candidate_slot1=candidate,
            sweep_enabled=True,
            current_tcp_fallback_enabled=True,
            sweep_samples=4,
        )
        self.assertEqual(tuple(sweep_only.keypoints_world.shape), (1, 1, 8, 3))
        self.assertEqual(
            tuple(fallback_only.keypoints_world.shape), (1, 1, 1, 3)
        )
        self.assertEqual(tuple(both.keypoints_world.shape), (1, 1, 9, 3))
        torch.testing.assert_close(
            sweep_only.keypoints_world[:, :, :2],
            current.keypoints_world,
        )
        torch.testing.assert_close(
            sweep_only.keypoints_world[:, :, -2:],
            candidate.keypoints_world,
        )
        torch.testing.assert_close(
            fallback_only.keypoints_world[:, :, 0],
            current.keypoints_world[:, :, 0],
        )
        with self.assertRaisesRegex(ValueError, "sweep and/or"):
            build_slot1_sweep_trajectory(
                current=current,
                candidate_slot1=candidate,
                sweep_enabled=False,
                current_tcp_fallback_enabled=False,
                sweep_samples=4,
            )

    def test_policy_lifts_base_and_corrected_query_actions_with_same_fk_path(self):
        class IdentityNormalizer:
            @staticmethod
            def unnormalize(value):
                return value

        policy = object.__new__(AQRDP3)
        torch.nn.Module.__init__(policy)
        policy.action_to_tool = ActionToTool(
            mode="tcp_delta_pose",
            horizon=1,
            state_dim=7,
            tcp_pose_start=0,
            keypoint_offsets_tcp={
                "tcp": (0.0, 0.0, 0.0),
                "left_fingertip": (0.0, 0.03, 0.0),
                "right_fingertip": (0.0, -0.03, 0.0),
            },
        )
        policy.normalizer = {"action": IdentityNormalizer()}
        policy.register_buffer(
            "_query_slot_indices",
            torch.tensor([1], dtype=torch.long),
            persistent=False,
        )
        actions = torch.zeros((1, 4, 6), dtype=torch.float32)
        actions[0, 1, 0] = 0.02
        encoded = SimpleNamespace(
            state_history_raw=torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            )
        )
        trajectory, selected, environment = (
            policy._tool_from_normalized_query_actions(actions, encoded)
        )
        self.assertEqual(tuple(selected.shape), (1, 1, 6))
        torch.testing.assert_close(selected, environment)
        torch.testing.assert_close(
            trajectory.keypoints_world[0, 0, 0],
            torch.tensor([0.02, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            0.5
            * (
                trajectory.keypoints_world[0, 0, 1]
                + trajectory.keypoints_world[0, 0, 2]
            ),
            trajectory.keypoints_world[0, 0, 0],
        )


class RelativeGeometryTest(unittest.TestCase):
    def test_empty_neighborhood_returns_exact_zero_tokens(self):
        batch, horizon = 1, 2
        matrices = torch.eye(4).reshape(1, 1, 4, 4).repeat(
            batch, horizon, 1, 1
        )
        matrices[..., 0, 3] = 10.0
        trajectory = ToolTrajectory(
            keypoints_world=matrices[..., :3, 3].unsqueeze(2),
            rotations_world_from_tool=matrices[..., :3, :3],
            tcp_matrices_world=matrices,
            names=("tcp",),
        )
        query = RelativeMultiScaleQuery(
            query_feature_dim=3,
            output_dim=8,
            hidden_dim=8,
            radii_m=(0.01,),
            neighbors=(2,),
        )
        token, diagnostics = query(
            query_xyz=torch.zeros((batch, 4, 3)),
            query_features=torch.randn((batch, 4, 3)),
            query_valid_mask=torch.ones((batch, 4), dtype=torch.bool),
            query_robot_mask=torch.zeros((batch, 4), dtype=torch.bool),
            trajectory=trajectory,
        )
        self.assertEqual(tuple(token.shape), (batch, horizon, 8))
        self.assertEqual(int(torch.count_nonzero(token)), 0)
        self.assertEqual(int(diagnostics.valid_neighbor_count.sum()), 0)
        torch.testing.assert_close(
            diagnostics.empty_fraction, torch.ones(batch)
        )
        torch.testing.assert_close(
            diagnostics.nearest_distance_m, torch.zeros(batch)
        )


class SlotWiseLocalResidualTest(unittest.TestCase):
    def test_zero_local_intervention_is_exactly_zero_even_without_zero_init(self):
        torch.manual_seed(37)
        residual = SlotWiseLocalResidual(
            action_dim=2,
            local_dim=5,
            global_dim=3,
            hidden_dim=16,
            time_dim=8,
            zero_init=False,
        )
        arguments = {
            "noisy_action": torch.randn((2, 4, 2)),
            "timestep": torch.tensor([5, 7]),
            "global_condition": torch.randn((2, 3)),
        }
        zero_delta, diagnostics = residual(
            **arguments,
            local_tokens=torch.zeros((2, 4, 5)),
        )
        self.assertEqual(int(torch.count_nonzero(zero_delta)), 0)
        self.assertEqual(int(torch.count_nonzero(diagnostics["delta_norm"])), 0)
        nonzero_delta, _ = residual(
            **arguments,
            local_tokens=torch.ones((2, 4, 5)),
        )
        self.assertGreater(int(torch.count_nonzero(nonzero_delta)), 0)

    def test_policy_masks_local_residual_to_slot1_exactly(self):
        class ConstantResidual(torch.nn.Module):
            def forward(self, **kwargs):
                delta = torch.ones_like(kwargs["noisy_action"])
                return delta, {"delta_norm": delta.norm(dim=-1)}

        policy = object.__new__(AQRDP3)
        torch.nn.Module.__init__(policy)
        policy.aqr = AQRDP3Config.from_mapping(
            {
                "experiment": "A1",
                "action_to_tool_mode": "tcp_delta_pose",
            }
        )
        policy.horizon = 4
        policy.local_residual = ConstantResidual()
        policy.register_buffer(
            "_query_slot_indices",
            torch.tensor([1], dtype=torch.long),
            persistent=False,
        )
        policy.noise_scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=100)
        )
        noisy = torch.zeros((2, 4, 3))
        output, diagnostics = policy._apply_local_residual(
            noisy_action=noisy,
            timestep=torch.zeros((2,), dtype=torch.long),
            global_condition=torch.zeros((2, 5)),
            global_output=torch.zeros_like(noisy),
            local_tokens=torch.ones((2, 4, 7)),
        )
        self.assertEqual(int(torch.count_nonzero(output[:, 0])), 0)
        self.assertGreater(int(torch.count_nonzero(output[:, 1])), 0)
        self.assertEqual(int(torch.count_nonzero(output[:, 2:])), 0)
        torch.testing.assert_close(
            diagnostics["local_residual_slot_mask"],
            torch.tensor([False, True, False, False]),
        )

    def test_policy_bounded_residual_preserves_non_arm_actuator(self):
        class ConstantResidual(torch.nn.Module):
            def forward(self, **kwargs):
                delta = torch.ones_like(kwargs["noisy_action"])
                return delta, {"delta_norm": delta.norm(dim=-1)}

        policy = object.__new__(AQRDP3)
        torch.nn.Module.__init__(policy)
        policy.aqr = AQRDP3Config.from_mapping(
            {
                "experiment": "A3",
                "action_to_tool_mode": "tcp_delta_pose",
                "arm_dim": 2,
                "bounded_action_residual": True,
                "action_residual_max_norm": 0.3,
                "action_residual_min_progress_ratio": 0.5,
                "action_residual_protect_non_arm": True,
            }
        )
        policy.horizon = 4
        policy.local_residual = ConstantResidual()
        policy.register_buffer(
            "_query_slot_indices",
            torch.tensor([1], dtype=torch.long),
            persistent=False,
        )
        policy.noise_scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=100)
        )
        base = torch.ones((1, 4, 3))
        output, diagnostics = policy._apply_local_residual(
            noisy_action=torch.zeros_like(base),
            timestep=torch.zeros((1,), dtype=torch.long),
            global_condition=torch.zeros((1, 5)),
            global_output=base,
            local_tokens=torch.ones((1, 4, 7)),
        )

        torch.testing.assert_close(output[..., 2], base[..., 2])
        self.assertLessEqual(
            float(diagnostics["bounded_action_residual_norm"].max()),
            0.300001,
        )


class AQRDP3DatasetMaskTest(unittest.TestCase):
    @staticmethod
    def _uninitialized_dataset(
        *, has_robot_mask: bool, has_point_valid_mask: bool
    ) -> AQRDP3Dataset:
        dataset = object.__new__(AQRDP3Dataset)
        dataset.has_robot_mask = has_robot_mask
        dataset.has_point_valid_mask = has_point_valid_mask
        dataset.workspace_crop_min = (-0.5, -0.5, -0.05)
        dataset.workspace_crop_max = (0.5, 0.65, 0.5)
        return dataset

    def test_sample_masks_are_boolean_and_fallback_validity_is_finite_xyz(self):
        sample = {
            "state": np.zeros((2, 4), dtype=np.float64),
            "action": np.zeros((2, 2), dtype=np.float64),
            "point_cloud": np.zeros((2, 5, 3), dtype=np.float64),
            "robot_mask": np.asarray(
                [
                    [[1], [0], [0], [1], [0]],
                    [[0], [1], [0], [0], [0]],
                ],
                dtype=np.uint8,
            ),
            "point_valid_mask": np.asarray(
                [
                    [[1], [1], [0], [1], [1]],
                    [[1], [1], [1], [0], [1]],
                ],
                dtype=np.uint8,
            ),
        }
        dataset = self._uninitialized_dataset(
            has_robot_mask=True, has_point_valid_mask=True
        )
        data = dataset._sample_to_data(sample)
        self.assertEqual(data["obs"]["robot_mask"].dtype, np.bool_)
        self.assertEqual(data["obs"]["point_valid_mask"].dtype, np.bool_)
        self.assertEqual(data["obs"]["robot_mask"].shape, (2, 5))
        self.assertEqual(data["obs"]["point_valid_mask"].shape, (2, 5))
        self.assertFalse(bool(data["obs"]["point_valid_mask"][0, 2]))

        sample_without_masks = dict(sample)
        sample_without_masks.pop("robot_mask")
        sample_without_masks.pop("point_valid_mask")
        sample_without_masks["point_cloud"] = sample["point_cloud"].copy()
        sample_without_masks["point_cloud"][1, 3, 0] = np.nan
        fallback = self._uninitialized_dataset(
            has_robot_mask=False, has_point_valid_mask=False
        )._sample_to_data(sample_without_masks)
        self.assertFalse(bool(fallback["obs"]["robot_mask"].any()))
        self.assertFalse(bool(fallback["obs"]["point_valid_mask"][1, 3]))

    def test_sample_mask_removes_stored_valid_workspace_outlier(self):
        sample = {
            "state": np.zeros((1, 4), dtype=np.float32),
            "action": np.zeros((1, 2), dtype=np.float32),
            "point_cloud": np.asarray(
                [[[0.0, 0.0, 0.1], [4.9, 3.0, 0.2], [0.1, 0.1, 0.1]]],
                dtype=np.float32,
            ),
            "robot_mask": np.asarray([[0, 1, 0]], dtype=np.uint8),
            "point_valid_mask": np.ones((1, 3), dtype=np.uint8),
        }
        data = self._uninitialized_dataset(
            has_robot_mask=True,
            has_point_valid_mask=True,
        )._sample_to_data(sample)
        np.testing.assert_array_equal(
            data["obs"]["point_valid_mask"],
            np.asarray([[True, False, True]]),
        )
        self.assertFalse(bool(data["obs"]["robot_mask"][0, 1]))

    def test_streaming_point_statistics_ignore_invalid_padding(self):
        values = np.asarray(
            [
                [[1.0], [2.0], [1000.0]],
                [[3.0], [-1000.0], [2.0]],
            ],
            dtype=np.float32,
        )
        valid = np.asarray(
            [[True, True, False], [True, False, True]], dtype=bool
        )
        stats = _streaming_stats(values, chunk_steps=1, row_mask=valid)
        np.testing.assert_allclose(stats["min"], np.asarray([1.0]))
        np.testing.assert_allclose(stats["max"], np.asarray([3.0]))
        np.testing.assert_allclose(stats["mean"], np.asarray([2.0]))

    def test_streaming_point_statistics_ignore_workspace_outliers(self):
        values = np.asarray(
            [
                [[0.0, 0.0, 0.1], [4.9, 3.0, 0.2]],
                [[0.2, 0.1, 0.3], [-4.9, 2.0, 0.1]],
            ],
            dtype=np.float32,
        )
        stats = _streaming_stats(
            values,
            chunk_steps=1,
            row_mask=np.ones((2, 2), dtype=bool),
            workspace_crop_min=(-0.5, -0.5, -0.05),
            workspace_crop_max=(0.5, 0.65, 0.5),
        )
        np.testing.assert_allclose(stats["min"], np.asarray([0.0, 0.0, 0.1]))
        np.testing.assert_allclose(stats["max"], np.asarray([0.2, 0.1, 0.3]))

    def test_normalizer_step_mask_excludes_validation_episodes(self):
        selected = _episode_step_mask(
            np.asarray([2, 5, 6], dtype=np.int64),
            np.asarray([True, False, True], dtype=bool),
        )
        np.testing.assert_array_equal(
            selected,
            np.asarray([True, True, False, False, False, True]),
        )
        values = np.asarray([[1.0], [3.0], [100.0], [100.0], [100.0], [5.0]])
        stats = _streaming_stats(
            values,
            chunk_steps=2,
            row_mask=selected,
        )
        np.testing.assert_allclose(stats["min"], np.asarray([1.0]))
        np.testing.assert_allclose(stats["max"], np.asarray([5.0]))
        np.testing.assert_allclose(stats["mean"], np.asarray([3.0]))


class PostDiffusionActionRefinerTest(unittest.TestCase):
    def test_anchor_identities_preserve_endpoints_and_fingers(self):
        names = (
            "sweep0_tcp",
            "sweep0_left_fingertip",
            "sweep0_right_fingertip",
            "sweep1_tcp",
            "sweep2_tcp",
            "segment_start_tcp_fallback",
        )
        roles, phases, sweeps = anchor_identity_ids(names)
        self.assertEqual(roles, (1, 2, 3, 1, 1, 1))
        self.assertEqual(phases, (1, 1, 1, 0, 2, 3))
        self.assertEqual(sweeps, (1, 1, 1, 2, 3, 0))

    def test_multislot_sweep_is_piecewise(self):
        batch, horizon = 1, 3
        current_points = torch.zeros(batch, horizon, 1, 3)
        candidate_points = torch.tensor(
            [[[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]]]
        )
        rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(
            batch, horizon, 3, 3
        )
        matrices = torch.eye(4).reshape(1, 1, 4, 4).expand(
            batch, horizon, 4, 4
        )
        current = ToolTrajectory(
            keypoints_world=current_points,
            rotations_world_from_tool=rotations,
            tcp_matrices_world=matrices,
            names=("tcp",),
        )
        candidate = ToolTrajectory(
            keypoints_world=candidate_points,
            rotations_world_from_tool=rotations,
            tcp_matrices_world=matrices,
            names=("tcp",),
        )
        swept = build_slot1_sweep_trajectory(
            current=current,
            candidate_slot1=candidate,
            sweep_enabled=True,
            current_tcp_fallback_enabled=False,
            sweep_samples=2,
        )
        torch.testing.assert_close(
            swept.keypoints_world[0, :, :, 0],
            torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]]),
        )

    def test_current_tcp_only_repeats_observed_tcp_without_candidate_path(self):
        horizon = 3
        matrices = torch.eye(4).reshape(1, 1, 4, 4).expand(1, horizon, 4, 4)
        points = torch.zeros(1, horizon, 3, 3)
        points[:, :, 0] = torch.tensor([0.1, 0.2, 0.3])
        points[:, :, 1] = torch.tensor([9.0, 9.0, 9.0])
        points[:, :, 2] = torch.tensor([-9.0, -9.0, -9.0])
        current = ToolTrajectory(
            keypoints_world=points,
            rotations_world_from_tool=matrices[..., :3, :3],
            tcp_matrices_world=matrices,
            names=("tcp", "left_fingertip", "right_fingertip"),
        )
        current_only = build_current_tcp_only_trajectory(
            current, horizon=horizon
        )
        self.assertEqual(tuple(current_only.keypoints_world.shape), (1, 3, 1, 3))
        self.assertEqual(current_only.names, ("segment_start_tcp_fallback",))
        torch.testing.assert_close(
            current_only.keypoints_world[0, :, 0],
            torch.tensor([[0.1, 0.2, 0.3]]).expand(3, 3),
        )

    def test_anchor_preserving_pool_keeps_identity_weights(self):
        query = RelativeMultiScaleQuery(
            query_feature_dim=3,
            output_dim=8,
            hidden_dim=8,
            radii_m=(0.20,),
            neighbors=(4,),
            relation_scale_gate=False,
            anchor_preserving_query=True,
        )
        names = (
            "sweep0_tcp",
            "sweep0_left_fingertip",
            "sweep1_tcp",
            "sweep1_right_fingertip",
        )
        centers = torch.zeros(1, 1, len(names), 3)
        rotations = torch.eye(3).reshape(1, 1, 3, 3)
        matrices = torch.eye(4).reshape(1, 1, 4, 4)
        trajectory = ToolTrajectory(
            keypoints_world=centers,
            rotations_world_from_tool=rotations,
            tcp_matrices_world=matrices,
            names=names,
        )
        token, diagnostics = query(
            query_xyz=torch.randn(1, 8, 3) * 0.01,
            query_features=torch.randn(1, 8, 3),
            query_valid_mask=torch.ones(1, 8, dtype=torch.bool),
            query_robot_mask=torch.zeros(1, 8, dtype=torch.bool),
            trajectory=trajectory,
        )
        self.assertEqual(tuple(token.shape), (1, 1, 8))
        weights = diagnostics.anchor_attention_weights
        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertEqual(tuple(weights.shape), (1, 1, 1, 4, 2))
        torch.testing.assert_close(
            weights.sum(dim=3), torch.ones(1, 1, 1, 2)
        )

    def test_zero_local_token_is_exact_noop(self):
        torch.manual_seed(91)
        refiner = PostDiffusionActionRefiner(
            action_dim=3,
            local_dim=5,
            global_dim=4,
            hidden_dim=16,
        )
        torch.nn.init.normal_(refiner.delta_head[-1].weight)
        torch.nn.init.normal_(refiner.delta_head[-1].bias)
        delta, confidence, _ = refiner(
            base_action=torch.randn(2, 4, 3),
            global_condition=torch.randn(2, 4),
            local_tokens=torch.zeros(2, 4, 5),
        )
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        torch.testing.assert_close(confidence, torch.zeros_like(confidence))

    def test_confidence_and_delta_preserve_slots(self):
        refiner = PostDiffusionActionRefiner(
            action_dim=2,
            local_dim=3,
            global_dim=4,
            hidden_dim=8,
        )
        delta, confidence, diagnostics = refiner(
            base_action=torch.randn(2, 4, 2),
            global_condition=torch.randn(2, 4),
            local_tokens=torch.randn(2, 4, 3),
        )
        self.assertEqual(tuple(delta.shape), (2, 4, 2))
        self.assertEqual(tuple(confidence.shape), (2, 4, 1))
        self.assertEqual(tuple(diagnostics["post_confidence"].shape), (2, 4))

    def test_action_and_global_modes_are_trainable_without_local_tokens(self):
        for mode in ("action_only", "global"):
            with self.subTest(mode=mode):
                refiner = PostDiffusionActionRefiner(
                    action_dim=2,
                    local_dim=3,
                    global_dim=4,
                    hidden_dim=8,
                    input_mode=mode,
                )
                torch.nn.init.constant_(refiner.delta_head[-1].bias, 0.25)
                delta, confidence, diagnostics = refiner(
                    base_action=torch.randn(2, 4, 2),
                    global_condition=torch.randn(2, 4),
                    local_tokens=torch.zeros(2, 4, 3),
                )
                self.assertGreater(float(delta.abs().sum()), 0.0)
                self.assertGreater(float(confidence.min()), 0.0)
                self.assertFalse(bool(diagnostics["post_local_query_enabled"].any()))
                self.assertEqual(refiner.local_projection, None)
                self.assertEqual(
                    refiner.global_projection is not None, mode == "global"
                )


if __name__ == "__main__":
    unittest.main()

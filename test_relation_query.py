"""Unit tests for AQR Relation Query (Layers 0-4)."""
import pytest
import torch

# Assume project root is on sys.path
from contactflow.dp3.aqr_dp3.config import AQRDP3Config
from contactflow.dp3.aqr_dp3.geometry import (
    ActionQueryEncoder,
    RelativeMultiScaleQuery,
    estimate_selected_normals,
    fuse_semantic_relation_tokens,
    ToolTrajectory,
    build_slot1_sweep_trajectory,
)
from contactflow.dp3.aqr_dp3.model import (
    SlotWiseLocalResidual,
    bound_action_residual,
    predicted_clean_sample,
    denoising_progress,
    query_gate,
    SinusoidalTimeEmbedding,
)


class TestSemanticPreservingFusion:
    def test_residual_fusion_never_scales_down_semantic_token(self):
        semantic = torch.tensor([[[[2.0, -3.0]]]])
        relation = torch.tensor([[[[4.0, 5.0]]]])
        logits = torch.zeros((1, 1, 1, 1))

        fused, gate = fuse_semantic_relation_tokens(
            semantic,
            relation,
            logits,
            mode="semantic_residual",
            gate_maximum=0.25,
        )

        torch.testing.assert_close(gate, torch.full_like(gate, 0.125))
        contribution = fused - semantic
        self_norm = semantic.norm(dim=-1)
        contribution_norm = contribution.norm(dim=-1)
        assert torch.all(contribution_norm <= gate.squeeze(-1) * self_norm + 1e-6)
        # The semantic stream is added exactly, never multiplied by (1-g).
        expected_relation = relation * (
            semantic.norm(dim=-1, keepdim=True)
            / relation.norm(dim=-1, keepdim=True)
        ).clamp(max=1.0)
        torch.testing.assert_close(fused, semantic + gate * expected_relation)

    def test_convex_mode_remains_checkpoint_compatible(self):
        semantic = torch.randn(2, 3, 4)
        relation = torch.randn(2, 3, 4)
        logits = torch.randn(2, 3, 1)
        fused, gate = fuse_semantic_relation_tokens(
            semantic,
            relation,
            logits,
            mode="convex",
            gate_maximum=1.0,
        )
        torch.testing.assert_close(gate, torch.sigmoid(logits))
        torch.testing.assert_close(
            fused,
            (1.0 - torch.sigmoid(logits)) * semantic
            + torch.sigmoid(logits) * relation,
        )


class TestBoundedActionResidual:
    def test_norm_progress_and_actuator_protection(self):
        delta = torch.tensor([[[-1.0, 0.0, 0.8]]])
        reference = torch.tensor([[[1.0, 0.0, -1.0]]])
        bounded, diagnostics = bound_action_residual(
            delta,
            reference_action=reference,
            arm_dim=2,
            maximum_norm=1.0,
            minimum_progress_ratio=0.75,
            protect_non_arm=True,
        )

        torch.testing.assert_close(bounded, torch.tensor([[[-0.25, 0.0, 0.0]]]))
        assert diagnostics["action_residual_progress_limited_fraction"].item()
        torch.testing.assert_close(
            diagnostics["action_residual_progress_ratio"],
            torch.tensor([[0.75]]),
        )
        torch.testing.assert_close(
            diagnostics["protected_non_arm_residual_norm"],
            torch.tensor([[0.8]]),
        )

    def test_lateral_correction_keeps_full_budget(self):
        delta = torch.tensor([[[0.0, 2.0, -0.5]]])
        reference = torch.tensor([[[1.0, 0.0, 1.0]]])
        bounded, diagnostics = bound_action_residual(
            delta,
            reference_action=reference,
            arm_dim=2,
            maximum_norm=0.3,
            minimum_progress_ratio=0.5,
            protect_non_arm=True,
        )

        torch.testing.assert_close(
            bounded,
            torch.tensor([[[0.0, 0.3, 0.0]]]),
            atol=1e-6,
            rtol=1e-6,
        )
        assert diagnostics["action_residual_clip_fraction"].item()
        assert not diagnostics["action_residual_progress_limited_fraction"].item()


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Surface normal estimation
# ═══════════════════════════════════════════════════════════════════════════

class TestEstimateSelectedNormals:
    def test_flat_plane_z(self):
        """Normals on a z=0 plane should point along ±z."""
        B, N = 2, 200
        xyz = torch.randn(B, N, 3)
        xyz[..., 2] = 0.01 * torch.randn(B, N)  # near-flat at z=0
        valid = torch.ones(B, N, dtype=torch.bool)
        sel = torch.arange(50)
        normals, conf, weight = estimate_selected_normals(xyz, valid, sel, k=8)
        assert normals.shape == (B, 50, 3)
        z_abs = normals[..., 2].abs().mean()
        assert z_abs > 0.8, f"z component too small: {z_abs:.3f}"
        assert conf.mean() < 0.3, f"confidence too high: {conf.mean():.3f}"

    def test_sphere(self):
        """Normals on a sphere should be radial."""
        B, N = 1, 500
        # Points on unit sphere
        theta = torch.rand(N) * 2 * torch.pi
        phi = torch.rand(N) * torch.pi
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        xyz = torch.stack([x, y, z], dim=-1).unsqueeze(0)  # [1, N, 3]
        valid = torch.ones(1, N, dtype=torch.bool)
        sel = torch.arange(100)
        normals, conf, weight = estimate_selected_normals(xyz, valid, sel, k=16)
        # Radial alignment: dot(normal, position) should be ≈ 1 or ≈ -1
        sel_pos = xyz[0, sel]  # [100, 3]
        sel_normals = normals[0]  # [100, 3]
        alignment = (sel_normals * sel_pos).sum(dim=-1).abs().mean()
        assert alignment > 0.7, f"Radial alignment too low: {alignment:.3f}"

    def test_empty_selection(self):
        """Empty selection should return zero-shaped tensors."""
        xyz = torch.randn(2, 100, 3)
        valid = torch.ones(2, 100, dtype=torch.bool)
        sel = torch.tensor([], dtype=torch.long)
        normals, conf, weight = estimate_selected_normals(xyz, valid, sel, k=8)
        assert normals.shape == (2, 0, 3)
        assert conf.shape == (2, 0)

    def test_masked_points(self):
        """Invalid points should be excluded from KNN."""
        B, N = 1, 100
        xyz = torch.randn(B, N, 3)
        xyz[0, :, 2] = 0.0  # flat
        valid = torch.ones(B, N, dtype=torch.bool)
        valid[0, 50:] = False  # second half invalid
        sel = torch.arange(10)  # select from first half
        normals, conf, weight = estimate_selected_normals(xyz, valid, sel, k=8)
        assert normals.shape == (1, 10, 3)
        # Should still get valid normals from the valid half
        assert conf.mean() < 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: Action Query & Key Encoders
# ═══════════════════════════════════════════════════════════════════════════

class TestActionQueryEncoder:
    def test_output_shape(self):
        enc = ActionQueryEncoder(action_dim=8, query_dim=128)
        B = 4
        q = enc(
            torch.randn(B, 8),
            torch.randn(B, 6),
            torch.randint(0, 100, (B,)),
        )
        assert q.shape == (B, 128)

    def test_deterministic(self):
        enc = ActionQueryEncoder(action_dim=8, query_dim=128)
        x = (torch.randn(2, 8), torch.randn(2, 6), torch.tensor([50, 75]))
        q1 = enc(*x)
        q2 = enc(*x)
        assert torch.allclose(q1, q2)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Relation-conditioned gain
# ═══════════════════════════════════════════════════════════════════════════

class TestSlotWiseLocalResidual:
    def test_relation_gain_zero(self):
        """scale=0: gain always 1.0 regardless of stats."""
        m = SlotWiseLocalResidual(action_dim=8, local_dim=128, global_dim=512,
                                   relation_gain_scale=0.0)
        B = 4
        x = torch.randn(B, 4, 8)
        t = torch.randint(0, 100, (B,))
        gc = torch.randn(B, 512)
        lt = torch.randn(B, 4, 128)
        stats = {'r_face_neg_fraction': torch.tensor([0.9, 0.9, 0.9, 0.9]),
                 'nearest_abnormal': torch.tensor([0.5, 0.5, 0.5, 0.5])}
        _, diag = m(noisy_action=x, timestep=t, global_condition=gc,
                     local_tokens=lt, relation_stats=stats)
        assert torch.allclose(diag['relation_gain'], torch.ones(B))

    def test_relation_gain_positive(self):
        """scale=2.0: gain > 1 for high deviation."""
        m = SlotWiseLocalResidual(action_dim=8, local_dim=128, global_dim=512,
                                   relation_gain_scale=2.0)
        B = 4
        x = torch.randn(B, 4, 8)
        t = torch.randint(0, 100, (B,))
        gc = torch.randn(B, 512)
        lt = torch.randn(B, 4, 128)
        stats = {
            'r_face_neg_fraction': torch.tensor([0.0, 0.3, 0.7, 1.0]),
            'nearest_abnormal': torch.tensor([0.0, 0.2, 0.5, 1.0]),
        }
        _, diag = m(noisy_action=x, timestep=t, global_condition=gc,
                     local_tokens=lt, relation_stats=stats)
        gains = diag['relation_gain']
        assert gains[0].item() == 1.0  # zero deviation
        assert gains[3].item() > 2.0   # max deviation
        assert gains[1].item() < gains[2].item() < gains[3].item()  # monotonic

    def test_relation_gain_no_stats(self):
        """No stats: gain defaults to 1.0."""
        m = SlotWiseLocalResidual(action_dim=8, local_dim=128, global_dim=512,
                                   relation_gain_scale=2.0)
        B = 4
        x = torch.randn(B, 4, 8)
        t = torch.randint(0, 100, (B,))
        gc = torch.randn(B, 512)
        lt = torch.randn(B, 4, 128)
        _, diag = m(noisy_action=x, timestep=t, global_condition=gc,
                     local_tokens=lt)  # no relation_stats
        assert torch.allclose(diag['relation_gain'], torch.ones(B))

    def test_zero_init_bias_baseline(self):
        """Q6 intervention: zero local tokens -> exactly zero delta."""
        m = SlotWiseLocalResidual(action_dim=8, local_dim=128, global_dim=512,
                                   zero_init=True)
        x = torch.randn(2, 4, 8)
        t = torch.tensor([50, 75])
        gc = torch.randn(2, 512)
        lt = torch.zeros(2, 4, 128)
        delta, _ = m(noisy_action=x, timestep=t, global_condition=gc,
                      local_tokens=lt)
        assert delta.abs().max().item() < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Layer 0+2: RelativeMultiScaleQuery forward
# ═══════════════════════════════════════════════════════════════════════════

class TestRelativeMultiScaleQuery:
    @pytest.fixture
    def query_module(self):
        return RelativeMultiScaleQuery(
            query_feature_dim=128, output_dim=128, hidden_dim=128,
            radii_m=(0.01, 0.03, 0.05, 0.10),
            neighbors=(8, 16, 32, 64),
            geometry_mode='tool_frame_relative',
            relation_features=True,
            relation_attention=True,
            relation_scale_gate=True,
            relation_scale_gate_geo_bias=True,
            action_dim=8,
        )

    def test_forward_without_relation(self):
        """Basic forward without action query params (falls back to distance pool)."""
        qmod = RelativeMultiScaleQuery(
            query_feature_dim=128, output_dim=128, hidden_dim=128,
            radii_m=(0.01, 0.03, 0.05, 0.10),
            neighbors=(8, 16, 32, 64),
            geometry_mode='tool_frame_relative',
            relation_features=False,
            relation_attention=False,
            relation_scale_gate=False,
        )
        B, N = 2, 500
        xyz = torch.randn(B, N, 3)
        feat = torch.randn(B, N, 128)
        valid = torch.ones(B, N, dtype=torch.bool)
        robot = torch.zeros(B, N, dtype=torch.bool)

        # Build a dummy trajectory
        kp = torch.randn(B, 1, 5, 3)  # 5 keypoints
        rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        tcp = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        traj = ToolTrajectory(
            keypoints_world=kp,
            rotations_world_from_tool=rot,
            tcp_matrices_world=tcp,
            names=tuple(f"kp{i}" for i in range(5)),
        )

        token, diag = qmod(
            query_xyz=xyz, query_features=feat,
            query_valid_mask=valid, query_robot_mask=robot,
            trajectory=traj,
        )
        assert token.shape == (B, 1, 128)  # H=1
        # Diagnostics should be present
        assert diag.nearest_distance_m is not None
        # No relation features → these are None
        assert diag.r_face_mean is None
        assert diag.r_face_neg_fraction is None

    def test_forward_with_relation_no_action(self):
        """Relation features enabled but no action query → features computed, no attention."""
        qmod = RelativeMultiScaleQuery(
            query_feature_dim=128, output_dim=128, hidden_dim=128,
            radii_m=(0.01, 0.03, 0.05, 0.10),
            neighbors=(8, 16, 32, 64),
            geometry_mode='tool_frame_relative',
            relation_features=True,
            relation_attention=False,
            relation_scale_gate=False,
        )
        B, N = 2, 500
        xyz = torch.randn(B, N, 3) * 0.2
        xyz[..., 2] = 0.01 * torch.randn(B, N)  # near z=0
        feat = torch.randn(B, N, 128)
        valid = torch.ones(B, N, dtype=torch.bool)
        robot = torch.zeros(B, N, dtype=torch.bool)
        # Place keypoints near the point cloud
        kp = torch.randn(B, 1, 5, 3) * 0.05
        rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        tcp = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        traj = ToolTrajectory(
            keypoints_world=kp,
            rotations_world_from_tool=rot,
            tcp_matrices_world=tcp,
            names=tuple(f"kp{i}" for i in range(5)),
        )

        token, diag = qmod(
            query_xyz=xyz, query_features=feat,
            query_valid_mask=valid, query_robot_mask=robot,
            trajectory=traj,
            # No action params
        )
        assert token.shape == (B, 1, 128)
        # r_face should be computed (flat plane, tools above)
        assert diag.r_face_mean is not None
        assert diag.r_face_neg_fraction is not None
        assert diag.normal_confidence_mean is not None
        assert diag.relation_gate_mean is not None
        expected_gate = torch.sigmoid(torch.tensor(-1.5))
        nonempty_scale = diag.valid_neighbor_count.sum(dim=(1, 3)) > 0
        assert torch.allclose(
            diag.relation_gate_mean[nonempty_scale],
            torch.full_like(
                diag.relation_gate_mean[nonempty_scale],
                float(expected_gate.item()),
            ),
            atol=1e-5,
        )
        # No attention → entropy is None
        assert diag.attn_entropy is None
        assert diag.scale_gate_weights is None

    def test_forward_with_full_relation(self, query_module):
        """Full relation: features + attention + scale gate."""
        qmod = query_module
        B, N = 2, 500
        # Place points in a dense blob near origin for reliable ball queries
        xyz = torch.randn(B, N, 3) * 0.2
        xyz[..., 2] = 0.02 * torch.randn(B, N)  # near z=0
        feat = torch.randn(B, N, 128)
        valid = torch.ones(B, N, dtype=torch.bool)
        robot = torch.zeros(B, N, dtype=torch.bool)
        # Place tool keypoints near the point cloud center
        kp = torch.randn(B, 1, 5, 3) * 0.05  # small spread near origin
        rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        tcp = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        traj = ToolTrajectory(
            keypoints_world=kp,
            rotations_world_from_tool=rot,
            tcp_matrices_world=tcp,
            names=tuple(f"kp{i}" for i in range(5)),
        )

        token, diag = qmod(
            query_xyz=xyz, query_features=feat,
            query_valid_mask=valid, query_robot_mask=robot,
            trajectory=traj,
            action_slot=torch.randn(B, 8),
            tool_pose_slot=torch.randn(B, 6),
            timestep=torch.randint(0, 100, (B,)),
            action_direction=torch.randn(B, 3),
        )
        assert token.shape[0] == B
        assert token.shape[-1] == 128
        gw = diag.scale_gate_weights
        assert gw is not None
        assert gw.shape[0] == B
        assert gw.shape[-1] == 4
        assert torch.allclose(gw.sum(dim=-1), torch.ones_like(gw.sum(dim=-1)))
        # attn_entropy should exist (may contain NaN for empty queries)
        assert diag.attn_entropy is not None

    def test_scale_gate_fallback(self):
        """Without scale gate: concat all scales."""
        qmod = RelativeMultiScaleQuery(
            query_feature_dim=128, output_dim=128, hidden_dim=128,
            radii_m=(0.01, 0.03, 0.05, 0.10),
            neighbors=(8, 16, 32, 64),
            geometry_mode='tool_frame_relative',
            relation_features=False,
            relation_attention=False,
            relation_scale_gate=False,
        )
        B, N = 2, 500
        xyz = torch.randn(B, N, 3)
        feat = torch.randn(B, N, 128)
        valid = torch.ones(B, N, dtype=torch.bool)
        robot = torch.zeros(B, N, dtype=torch.bool)
        kp = torch.randn(B, 1, 5, 3)
        rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        tcp = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        traj = ToolTrajectory(
            keypoints_world=kp,
            rotations_world_from_tool=rot,
            tcp_matrices_world=tcp,
            names=tuple(f"kp{i}" for i in range(5)),
        )

        token, diag = qmod(
            query_xyz=xyz, query_features=feat,
            query_valid_mask=valid, query_robot_mask=robot,
            trajectory=traj,
        )
        assert token.shape == (B, 1, 128)

    def test_r_face_uses_world_frame_for_rotated_tool(self):
        qmod = RelativeMultiScaleQuery(
            query_feature_dim=8,
            output_dim=16,
            hidden_dim=16,
            radii_m=(0.05,),
            neighbors=(64,),
            relation_features=True,
            relation_attention=False,
            relation_scale_gate=False,
        )
        grid = torch.linspace(-0.015, 0.015, 12)
        xx, yy = torch.meshgrid(grid, grid, indexing="ij")
        xyz = torch.stack(
            [xx.flatten(), yy.flatten(), torch.zeros(xx.numel())], dim=-1
        ).unsqueeze(0)
        feature = torch.randn(1, xyz.shape[1], 8)
        valid = torch.ones(1, xyz.shape[1], dtype=torch.bool)
        robot = torch.zeros_like(valid)
        keypoints = torch.tensor([[[[0.0, 0.0, 0.01]]]])
        tcp = torch.eye(4).reshape(1, 1, 4, 4)

        def run(rotation):
            trajectory = ToolTrajectory(
                keypoints_world=keypoints,
                rotations_world_from_tool=rotation.reshape(1, 1, 3, 3),
                tcp_matrices_world=tcp,
                names=("tcp",),
            )
            return qmod(
                query_xyz=xyz,
                query_features=feature,
                query_valid_mask=valid,
                query_robot_mask=robot,
                trajectory=trajectory,
            )[1].r_face_mean

        identity_face = run(torch.eye(3))
        rotated_face = run(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
            )
        )
        assert identity_face is not None and rotated_face is not None
        assert identity_face.item() > 0.7
        assert torch.allclose(identity_face, rotated_face, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: Offset training logic
# ═══════════════════════════════════════════════════════════════════════════

class TestOffsetLogic:
    def test_r_face_sign_flip_on_offset(self):
        """When tool is offset behind an object, r_face should flip sign."""
        # This tests the geometric intuition underlying Layer 4:
        # pointing at a surface from in front   → r_face > 0
        # pointing at the same surface from behind → r_face < 0

        # Create a dense flat surface near z=0
        N = 200
        xyz = torch.randn(1, N, 3) * 0.02  # very tight cluster in x,y
        xyz[..., 2] = 0.0 + 0.001 * torch.randn(1, N)  # nearly perfect plane at z=0
        valid = torch.ones(1, N, dtype=torch.bool)
        sel = torch.arange(30)
        normals, _, _ = estimate_selected_normals(xyz, valid, sel, k=8)

        # Tool 1cm above surface (close enough for strong directional signal)
        tool_front = torch.tensor([[[0.0, 0.0, 0.01]]])
        d_front = xyz[:, sel] - tool_front
        d_norm = d_front.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        r_face_front = -(normals * d_front / d_norm).sum(dim=-1)

        # Tool 1cm below surface
        tool_back = torch.tensor([[[0.0, 0.0, -0.01]]])
        d_back = xyz[:, sel] - tool_back
        d_norm_b = d_back.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        r_face_back = -(normals * d_back / d_norm_b).sum(dim=-1)

        rf_mean = r_face_front.mean()
        rb_mean = r_face_back.mean()
        # Front and back should have opposite signs (tool on opposite sides)
        assert (rf_mean * rb_mean) < 0, \
            f"Front({rf_mean:.3f}) and back({rb_mean:.3f}) should have opposite signs"

    def test_r_motion_consistency(self):
        """r_motion should be positive when moving toward surface, negative when away."""
        # On a flat z=0 surface with normal (0,0,1):
        #   moving in +z direction → r_motion = +1
        #   moving in -z direction → r_motion = -1
        N = 100
        xyz = torch.randn(1, N, 3) * 0.3
        xyz[..., 2] = 0.0
        valid = torch.ones(1, N, dtype=torch.bool)
        sel = torch.arange(20)
        normals, _, _ = estimate_selected_normals(xyz, valid, sel, k=8)

        # Surface normal points +z
        v_toward = torch.tensor([0.0, 0.0, 1.0])
        rm_t = (normals[0] * v_toward).sum(dim=-1)
        assert rm_t.mean() > 0.5

        v_away = torch.tensor([0.0, 0.0, -1.0])
        rm_a = (normals[0] * v_away).sum(dim=-1)
        assert rm_a.mean() < -0.5


# ═══════════════════════════════════════════════════════════════════════════
# Config validation
# ═══════════════════════════════════════════════════════════════════════════

class TestConfig:
    @pytest.mark.parametrize(
        ("experiment", "offset", "attention", "gain"),
        [
            ("A2", False, False, 0.0),
            ("A3", True, False, 0.0),
            ("A4", True, True, 0.0),
            ("A5", True, True, 2.0),
        ],
    )
    def test_clean_ablation_line(
        self, experiment, offset, attention, gain
    ):
        cfg = AQRDP3Config.from_mapping({"experiment": experiment})
        assert cfg.relation_features is True
        assert cfg.offset_training_enabled is offset
        assert cfg.relation_attention is attention
        assert cfg.relation_gain_scale == gain

    def test_attention_requires_features(self):
        with pytest.raises(ValueError, match='relation_attention requires'):
            AQRDP3Config.from_mapping({
                'experiment': 'A1',
                'relation_features': False,
                'relation_attention': True,
            })


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

class TestUtilities:
    def test_query_gate(self):
        progress = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        gate = query_gate(progress, start_fraction=0.5, maximum=1.0)
        assert gate[2].item() == 0.0  # at start
        assert gate[4].item() == 1.0  # at end
        assert gate[3].item() == 0.5  # halfway through active region

    def test_denoising_progress(self):
        p = denoising_progress(
            timestep=torch.tensor([0, 50, 99]),
            num_train_timesteps=100,
            batch_size=3,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        assert p[0].item() == 1.0   # t=0: fully denoised
        assert p[-1].item() == 0.0  # t=99: fully noisy

    def test_sinusoidal_time_embedding(self):
        emb = SinusoidalTimeEmbedding(64)
        t = torch.tensor([0, 50, 999])
        out = emb(t)
        assert out.shape == (3, 64)
        assert out.dtype == torch.float32

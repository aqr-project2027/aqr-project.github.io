from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from contactflow.dp3.aqr_bc.torch_trajectory import (
    TorchCandidateTrajectoryLift,
    pose_wxyz_to_matrix_torch,
)
from contactflow.dp3.aqr_dp3.features import gather_points


# ═══════════════════════════════════════════════════════════════════════════
# ToolTrajectory & sweep helpers (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ToolTrajectory:
    keypoints_world: torch.Tensor
    rotations_world_from_tool: torch.Tensor
    tcp_matrices_world: torch.Tensor
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        points = self.keypoints_world
        rotations = self.rotations_world_from_tool
        matrices = self.tcp_matrices_world
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("keypoints_world must have shape [B,H,M,3].")
        if tuple(rotations.shape) != (points.shape[0], points.shape[1], 3, 3):
            raise ValueError("tool rotations must have shape [B,H,3,3].")
        if tuple(matrices.shape) != (points.shape[0], points.shape[1], 4, 4):
            raise ValueError("TCP matrices must have shape [B,H,4,4].")
        if len(self.names) != points.shape[2]:
            raise ValueError("tool keypoint names and positions disagree.")


def build_slot1_sweep_trajectory(
    *,
    current: ToolTrajectory,
    candidate_slot1: ToolTrajectory,
    sweep_enabled: bool,
    current_tcp_fallback_enabled: bool,
    sweep_samples: int,
) -> ToolTrajectory:
    """Build a piecewise sweep for one or more executable action slots.

    For slot zero in this local window the segment starts at the observed tool
    pose.  Every later segment starts at the preceding candidate pose, so a
    multi-slot query represents the actual integrated action path instead of
    repeatedly drawing rays from the original TCP.
    """

    if current.keypoints_world.shape[1] < 1:
        raise ValueError("current tool trajectory must contain at least one slot.")
    if current.names != candidate_slot1.names:
        raise ValueError("current and candidate tool keypoints must align.")
    if current.keypoints_world.shape != candidate_slot1.keypoints_world.shape:
        raise ValueError("current and candidate tool trajectory shapes must align.")
    if not isinstance(sweep_enabled, bool):
        raise ValueError("sweep_enabled must be boolean.")
    if not isinstance(current_tcp_fallback_enabled, bool):
        raise ValueError("current_tcp_fallback_enabled must be boolean.")
    if not (sweep_enabled or current_tcp_fallback_enabled):
        raise ValueError("slot1 query needs sweep and/or current-TCP fallback.")
    if int(sweep_samples) < 2:
        raise ValueError("sweep_samples must be at least 2.")

    anchors: list[torch.Tensor] = []
    names: list[str] = []
    segment_start = torch.cat(
        [
            current.keypoints_world[:, :1],
            candidate_slot1.keypoints_world[:, :-1],
        ],
        dim=1,
    )
    if sweep_enabled:
        alpha = torch.linspace(
            0.0, 1.0, int(sweep_samples),
            dtype=current.keypoints_world.dtype,
            device=current.keypoints_world.device,
        ).reshape(1, 1, int(sweep_samples), 1, 1)
        start = segment_start.unsqueeze(2)
        end = candidate_slot1.keypoints_world.unsqueeze(2)
        swept = start + alpha * (end - start)
        anchors.append(
            swept.reshape(
                swept.shape[0],
                swept.shape[1],
                int(sweep_samples) * swept.shape[3],
                3,
            )
        )
        for sample_index in range(int(sweep_samples)):
            names.extend(f"sweep{sample_index}_{name}" for name in current.names)
    if current_tcp_fallback_enabled:
        try:
            tcp_index = current.names.index("tcp")
        except ValueError as exc:
            raise ValueError(
                "current-TCP fallback requires a keypoint named 'tcp'."
            ) from exc
        anchors.append(segment_start[:, :, tcp_index : tcp_index + 1])
        names.append("segment_start_tcp_fallback")

    return ToolTrajectory(
        keypoints_world=torch.cat(anchors, dim=2),
        rotations_world_from_tool=candidate_slot1.rotations_world_from_tool,
        tcp_matrices_world=candidate_slot1.tcp_matrices_world,
        names=tuple(names),
    )


def build_current_tcp_only_trajectory(
    current: ToolTrajectory, *, horizon: int
) -> ToolTrajectory:
    """Repeat the observed TCP anchor without exposing a candidate path.

    This is the structural current-TCP-only ablation: every executable slot
    queries the same observed TCP location and orientation. No candidate FK,
    fingertip, sweep, or action-direction geometry enters the local token.
    """

    if int(horizon) <= 0:
        raise ValueError("current-TCP-only horizon must be positive.")
    try:
        tcp_index = current.names.index("tcp")
    except ValueError as exc:
        raise ValueError(
            "current-TCP-only query requires a keypoint named 'tcp'."
        ) from exc
    points = current.keypoints_world[
        :, :1, tcp_index : tcp_index + 1
    ].expand(-1, int(horizon), -1, -1)
    rotations = current.rotations_world_from_tool[:, :1].expand(
        -1, int(horizon), -1, -1
    )
    matrices = current.tcp_matrices_world[:, :1].expand(
        -1, int(horizon), -1, -1
    )
    return ToolTrajectory(
        keypoints_world=points,
        rotations_world_from_tool=rotations,
        tcp_matrices_world=matrices,
        names=("segment_start_tcp_fallback",),
    )


def _axis_angle_to_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    if rotation_vector.shape[-1] != 3:
        raise ValueError("axis-angle vectors must end in three values.")
    theta = rotation_vector.norm(dim=-1, keepdim=True)
    axis = rotation_vector / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*rotation_vector.shape[:-1], 3, 3)
    identity = torch.eye(
        3, dtype=rotation_vector.dtype, device=rotation_vector.device
    ).expand(*rotation_vector.shape[:-1], 3, 3)
    sin = torch.sin(theta).unsqueeze(-1)
    cos = torch.cos(theta).unsqueeze(-1)
    matrix = identity + sin * skew + (1.0 - cos) * (skew @ skew)
    small = theta.squeeze(-1) < 1e-7
    return torch.where(small[..., None, None], identity, matrix)


def _matrix_from_rt(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros(
        (*translation.shape[:-1], 4, 4),
        dtype=translation.dtype, device=translation.device,
    )
    matrix[..., :3, :3] = rotation
    matrix[..., :3, 3] = translation
    matrix[..., 3, 3] = 1.0
    return matrix


class ActionToTool(nn.Module):
    """Shared train/inference action-to-tool implementation."""

    def __init__(
        self,
        *,
        mode: str,
        horizon: int,
        state_dim: int,
        tcp_pose_start: int,
        arm_dim: int = 7,
        candidate_lift: TorchCandidateTrajectoryLift | None = None,
        keypoint_offsets_tcp: Mapping[str, Sequence[float]] | None = None,
        translation_action_indices: tuple[int, int, int] = (0, 1, 2),
        rotation_action_indices: tuple[int, int, int] = (3, 4, 5),
        translation_scale_m: float = 1.0,
        rotation_scale_rad: float = 1.0,
    ):
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in {"joint_delta_fk", "tcp_delta_pose"}:
            raise ValueError(
                "ActionToTool mode must be joint_delta_fk or tcp_delta_pose."
            )
        if int(horizon) <= 0 or int(state_dim) <= 0:
            raise ValueError("horizon and state_dim must be positive.")
        if not 0 <= int(tcp_pose_start) <= int(state_dim) - 7:
            raise ValueError("tcp_pose_start does not fit state_dim.")
        offsets = (
            {"tcp": (0.0, 0.0, 0.0)}
            if keypoint_offsets_tcp is None
            else {
                str(k): tuple(float(x) for x in v)
                for k, v in keypoint_offsets_tcp.items()
            }
        )
        if not offsets or any(len(value) != 3 for value in offsets.values()):
            raise ValueError("every tool keypoint needs one 3-D TCP-frame offset.")
        self.mode = mode
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.tcp_pose_start = int(tcp_pose_start)
        self.arm_dim = int(arm_dim)
        self.candidate_lift = candidate_lift
        self.names = tuple(offsets)
        self.register_buffer(
            "keypoint_offsets_tcp",
            torch.tensor(list(offsets.values()), dtype=torch.float32),
        )
        self.register_buffer(
            "translation_action_indices",
            torch.tensor(translation_action_indices, dtype=torch.long),
        )
        self.register_buffer(
            "rotation_action_indices",
            torch.tensor(rotation_action_indices, dtype=torch.long),
        )
        self.translation_scale_m = float(translation_scale_m)
        self.rotation_scale_rad = float(rotation_scale_rad)

    def _keypoints(self, matrices: torch.Tensor) -> ToolTrajectory:
        offsets = self.keypoint_offsets_tcp.to(matrices)
        rotations = matrices[..., :3, :3]
        translations = matrices[..., :3, 3]
        positions = torch.einsum("bhij,mj->bhmi", rotations, offsets)
        positions = positions + translations.unsqueeze(2)
        return ToolTrajectory(
            keypoints_world=positions,
            rotations_world_from_tool=rotations,
            tcp_matrices_world=matrices,
            names=self.names,
        )

    def static(self, state_history_raw: torch.Tensor) -> ToolTrajectory:
        current = state_history_raw[
            :, -1, self.tcp_pose_start : self.tcp_pose_start + 7
        ]
        matrix = pose_wxyz_to_matrix_torch(current)
        return self._keypoints(matrix[:, None].expand(-1, self.horizon, -1, -1))

    def forward(
        self,
        *,
        policy_actions: torch.Tensor,
        environment_actions: torch.Tensor,
        state_history_raw: torch.Tensor,
    ) -> ToolTrajectory:
        if policy_actions.ndim != 3 or policy_actions.shape[1] != self.horizon:
            raise ValueError("policy_actions must have shape [B,H,A].")
        if environment_actions.shape != policy_actions.shape:
            raise ValueError("policy and environment action tensors must align.")
        expected_state = (policy_actions.shape[0], None, self.state_dim)
        if (
            state_history_raw.ndim != 3
            or state_history_raw.shape[0] != expected_state[0]
            or state_history_raw.shape[-1] != expected_state[2]
            or state_history_raw.shape[1] < 1
        ):
            raise ValueError("state_history_raw must have shape [B,To,state_dim].")

        if self.mode == "joint_delta_fk":
            if state_history_raw.shape[1] < 2:
                raise ValueError(
                    "joint_delta_fk requires current and previous qpos."
                )
            if self.candidate_lift is None:
                raise RuntimeError(
                    "A dynamic joint_delta_fk query requires a verified action "
                    "contract. Set aqr_config.action_contract_path."
                )
            current = state_history_raw[:, -1]
            previous = state_history_raw[:, -2]
            lifted = self.candidate_lift(
                policy_actions=policy_actions,
                current_arm_qpos=current[:, : self.arm_dim],
                previous_arm_qpos=previous[:, : self.arm_dim],
                current_tcp_pose_wxyz=current[
                    :, self.tcp_pose_start : self.tcp_pose_start + 7
                ],
            )
            return self._keypoints(lifted.tcp_matrices_world)

        current_pose = state_history_raw[
            :, -1, self.tcp_pose_start : self.tcp_pose_start + 7
        ]
        current = pose_wxyz_to_matrix_torch(current_pose)
        translation_delta = environment_actions.index_select(
            -1, self.translation_action_indices
        ) * self.translation_scale_m
        rotation_delta = environment_actions.index_select(
            -1, self.rotation_action_indices
        ) * self.rotation_scale_rad
        matrices: list[torch.Tensor] = []
        pose = current
        for slot in range(self.horizon):
            delta = _matrix_from_rt(
                _axis_angle_to_matrix(rotation_delta[:, slot]),
                translation_delta[:, slot],
            )
            pose = pose @ delta
            matrices.append(pose)
        return self._keypoints(torch.stack(matrices, dim=1))


def offset_tool_trajectory(
    trajectory: ToolTrajectory, offset_world: Sequence[float] | torch.Tensor
) -> ToolTrajectory:
    offset = torch.as_tensor(
        offset_world,
        dtype=trajectory.keypoints_world.dtype,
        device=trajectory.keypoints_world.device,
    )
    if tuple(offset.shape) != (3,):
        raise ValueError("wrong-query offset must contain exactly three values.")
    matrices = trajectory.tcp_matrices_world.clone()
    matrices[..., :3, 3] += offset
    return ToolTrajectory(
        keypoints_world=trajectory.keypoints_world + offset,
        rotations_world_from_tool=trajectory.rotations_world_from_tool,
        tcp_matrices_world=matrices,
        names=trajectory.names,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Surface normal estimation (algorithmic, no learnable parameters)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_selected_normals(
    query_xyz: torch.Tensor,           # [B, N, 3]
    query_valid_mask: torch.Tensor,    # [B, N]
    selected_indices: torch.Tensor,    # [U]  unique indices of query points
    k: int = 8,
    *,
    orientation_centers: torch.Tensor | None = None,   # [B, C, 3] tool centers
    outlier_max_mean_dist: float = 0.03,               # 3 cm threshold for floaters
    soft_gate_temperature: float = 0.05,               # sigmoid temperature
    soft_gate_threshold: float = 0.5,                  # center of sigmoid
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate surface normals for selected query points via local PCA.

    Fix 1: orient normals toward nearest tool center (resolves PCA sign ambiguity).
    Fix 2: filter outlier points whose K=8 neighbors avg distance > threshold.
    Fix 3: soft confidence weight instead of hard threshold mask.

    Returns:
        normals:     [B, U, 3]  oriented unit surface normals
        confidence:  [B, U]     λ_min / λ_max  (0 = perfect plane)
        weight:      [B, U]     soft reliability weight in [0, 1]
                                (low for outliers, low for high confidence)
    """
    B, N, _ = query_xyz.shape
    U = int(selected_indices.numel())
    if U == 0:
        return (
            torch.zeros(B, 0, 3, device=query_xyz.device, dtype=query_xyz.dtype),
            torch.ones(B, 0, device=query_xyz.device, dtype=query_xyz.dtype),
            torch.zeros(B, 0, device=query_xyz.device, dtype=query_xyz.dtype),
        )

    # Gather selected points  [B, U, 3]
    safe_idx = selected_indices.clamp(min=0, max=N - 1)
    sel_xyz = query_xyz[
        torch.arange(B, device=query_xyz.device)[:, None],
        safe_idx[None, :].expand(B, -1),
    ]

    # KNN of selected points within full query cloud: dist  [B, U, N]
    dist = torch.cdist(sel_xyz.float(), query_xyz.float())
    dist = dist.masked_fill(~query_valid_mask[:, None, :], torch.inf)
    actual_k = min(k, N)
    knn_dist, knn_idx = dist.topk(k=actual_k, dim=-1, largest=False)  # [B, U, K]

    # ── Fix 2: outlier filter ──
    mean_knn_dist = knn_dist.mean(dim=-1)  # [B, U]
    outlier = mean_knn_dist > float(outlier_max_mean_dist)

    # Gather neighbours  [B, U, K, 3]
    knn_flat = knn_idx.reshape(B, U * actual_k)
    neighbors = gather_points(query_xyz, knn_flat).reshape(B, U, actual_k, 3)

    # Centroid and covariance  [B, U, 3, 3]
    centroid = neighbors.mean(dim=-2, keepdim=True)
    centered = neighbors - centroid
    cov = torch.einsum("buki,bukj->buij", centered, centered) / max(1, actual_k - 1)

    # Eigendecomposition (ascending: λ₀ ≤ λ₁ ≤ λ₂)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Normal = eigenvector of smallest eigenvalue  [B, U, 3]
    normals_raw = eigenvectors[..., 0]

    # ── Fix 1: orient normals toward nearest tool center ──
    if orientation_centers is not None:
        # Find nearest tool center for each selected point  [B, U, 3]
        sel_for_dist = sel_xyz.float()  # [B, U, 3]
        ctr_for_dist = orientation_centers.float()  # [B, C, 3]
        ctr_dist = torch.cdist(sel_for_dist, ctr_for_dist)  # [B, U, C]
        nearest_ctr_idx = ctr_dist.argmin(dim=-1)  # [B, U]
        nearest_ctr = torch.gather(
            ctr_for_dist, 1,
            nearest_ctr_idx[:, :, None].expand(-1, -1, 3),
        )  # [B, U, 3]
        # direction from point to nearest tool center
        to_tool = nearest_ctr - sel_xyz  # [B, U, 3]
        dot = (normals_raw * to_tool).sum(dim=-1)  # [B, U]
        flip = dot < 0.0
        normals = normals_raw.clone()
        normals[flip] = -normals[flip]
    else:
        normals = normals_raw

    # Confidence: λ_min / λ_max  (0 = planar, 1 = isotropic)
    confidence = eigenvalues[..., 0] / eigenvalues[..., 2].clamp_min(1e-8)  # [B, U]

    # ── Fix 3: soft weight from confidence + outlier ──
    # sigmoid: weight ≈ 1 when confidence << threshold, ≈ 0 when >> threshold
    tau = float(soft_gate_temperature)
    thresh = float(soft_gate_threshold)
    conf_weight = torch.sigmoid((thresh - confidence) / tau)  # [B, U]
    # Outlier points get near-zero weight
    outlier_weight = (~outlier).to(conf_weight.dtype)
    weight = conf_weight * outlier_weight

    return normals, confidence, weight


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: Action Query Encoder
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time-step embedding (copied from model.py for local use)."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep[None]
        half = self.dimension // 2
        if half == 0:
            return timestep.to(torch.float32).unsqueeze(-1)
        exponent = -math.log(10000.0) * torch.arange(
            half, device=timestep.device, dtype=torch.float32
        ) / max(1, half - 1)
        angle = timestep.to(torch.float32).unsqueeze(-1) * exponent.exp()
        embedding = torch.cat([angle.sin(), angle.cos()], dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[-1]))
        return embedding


class ActionQueryEncoder(nn.Module):
    """Encodes (action, tool_pose, timestep) → action-conditioned query vector.

    The query represents "what geometric relations this action cares about"
    rather than "what the action is".
    """

    def __init__(
        self,
        *,
        action_dim: int,
        tool_pose_dim: int = 6,
        time_dim: int = 128,
        query_dim: int = 128,
    ):
        super().__init__()
        self.action_proj = nn.Linear(int(action_dim), int(query_dim))
        self.pose_proj = nn.Linear(int(tool_pose_dim), int(query_dim))
        self.time_proj = nn.Sequential(
            SinusoidalTimeEmbedding(int(time_dim)),
            nn.Linear(int(time_dim), int(query_dim)),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(int(query_dim)),
            nn.SiLU(),
            nn.Linear(int(query_dim), int(query_dim)),
        )

    def forward(
        self,
        action: torch.Tensor,       # [B, A]
        tool_pose: torch.Tensor,    # [B, 6]  TCP position(3) + z-axis direction(3)
        timestep: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        """Return action query  [B, D_q]."""
        return self.fusion(
            self.action_proj(action)
            + self.pose_proj(tool_pose)
            + self.time_proj(timestep).to(action.dtype)
        )


# ═══════════════════════════════════════════════════════════════════════════
# Relation Query Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RelationQueryDiagnostics:
    # ── existing fields ──
    valid_neighbor_count: torch.Tensor
    empty_fraction: torch.Tensor
    robot_point_fraction: torch.Tensor
    nearest_distance_m: torch.Tensor
    # ── Layer 1: relation features ──
    r_face_mean: torch.Tensor | None = None
    r_face_neg_fraction: torch.Tensor | None = None
    r_motion_mean: torch.Tensor | None = None
    normal_confidence_mean: torch.Tensor | None = None
    relation_gate_mean: torch.Tensor | None = None
    # ── Layer 2: attention ──
    attn_entropy: torch.Tensor | None = None
    scale_gate_weights: torch.Tensor | None = None
    # A4.1 anchor-preserving pooling. The two learned heads remain visible for
    # diagnosing current/candidate or left/right collapse.
    anchor_attention_weights: torch.Tensor | None = None
    anchor_role_ids: torch.Tensor | None = None
    anchor_phase_ids: torch.Tensor | None = None
    anchor_sweep_ids: torch.Tensor | None = None


def anchor_identity_ids(
    names: Sequence[str],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Encode keypoint role, trajectory phase, and sweep index per anchor.

    Roles are unknown/TCP/left/right. Phases are interior/start/candidate/
    fallback. Sweep id zero is reserved for the explicit segment-start TCP;
    sampled sweep positions use one-based ids. The combination distinguishes
    current_tcp and candidate_tcp while retaining left/right fingertip identity.
    """

    parsed: list[tuple[int | None, str, bool]] = []
    maximum_sweep = -1
    for raw_name in names:
        name = str(raw_name)
        is_fallback = name == "segment_start_tcp_fallback"
        sweep_index: int | None = None
        keypoint_name = name
        if name.startswith("sweep") and "_" in name:
            prefix, keypoint_name = name.split("_", 1)
            suffix = prefix[len("sweep") :]
            if suffix.isdigit():
                sweep_index = int(suffix)
                maximum_sweep = max(maximum_sweep, sweep_index)
        parsed.append((sweep_index, keypoint_name, is_fallback))

    role_ids: list[int] = []
    phase_ids: list[int] = []
    sweep_ids: list[int] = []
    for sweep_index, keypoint_name, is_fallback in parsed:
        if "left_fingertip" in keypoint_name:
            role = 2
        elif "right_fingertip" in keypoint_name:
            role = 3
        elif "tcp" in keypoint_name:
            role = 1
        else:
            role = 0

        if is_fallback:
            phase = 3
            sweep_id = 0
        elif sweep_index is None:
            phase = 0
            sweep_id = 0
        else:
            phase = (
                1
                if sweep_index == 0
                else (2 if sweep_index == maximum_sweep else 0)
            )
            sweep_id = sweep_index + 1
        role_ids.append(role)
        phase_ids.append(phase)
        sweep_ids.append(sweep_id)
    return tuple(role_ids), tuple(phase_ids), tuple(sweep_ids)


def fuse_semantic_relation_tokens(
    semantic_token: torch.Tensor,
    relation_token: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    mode: str,
    gate_maximum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse local tokens while preserving an exact legacy mode."""

    if semantic_token.shape != relation_token.shape:
        raise ValueError("semantic and relation tokens must have matching shapes.")
    if gate_logits.shape != semantic_token.shape[:-1] + (1,):
        raise ValueError("gate_logits must match the token prefix and end in one.")
    mode = str(mode).strip().lower()
    if mode not in {"convex", "semantic_residual"}:
        raise ValueError("mode must be convex or semantic_residual.")
    gate_maximum = float(gate_maximum)
    if not 0.0 <= gate_maximum <= 1.0:
        raise ValueError("gate_maximum must be in [0,1].")
    gate = gate_maximum * torch.sigmoid(gate_logits)
    if mode == "semantic_residual":
        # Prevent the relation encoder from bypassing the gate by growing its
        # token norm.  Its additive contribution is capped relative to the
        # semantic token while keeping the semantic stream untouched.
        semantic_norm = semantic_token.norm(dim=-1, keepdim=True).detach()
        relation_norm = relation_token.norm(dim=-1, keepdim=True)
        relation_scale = (
            semantic_norm / relation_norm.clamp_min(1e-8)
        ).clamp(max=1.0)
        relation_residual = relation_token * relation_scale
        fused = semantic_token + gate * relation_residual
    else:
        fused = (1.0 - gate) * semantic_token + gate * relation_token
    return fused, gate


# ═══════════════════════════════════════════════════════════════════════════
# RelativeMultiScaleQuery  (Layer 0+1+2 modified)
# ═══════════════════════════════════════════════════════════════════════════

class RelativeMultiScaleQuery(nn.Module):
    """Slot-aligned multi-scale local geometry with action-conditioned attention.

    Layer 0: query radii  (0.01, 0.03, 0.05, 0.10) m  with cross-scale gate
    Layer 1: r_face / r_motion  algorithmic relation primitives
    Layer 2: ActionQueryEncoder + gated-token cross-attention pooling
    """

    def __init__(
        self,
        *,
        query_feature_dim: int,
        output_dim: int = 128,
        hidden_dim: int = 128,
        radii_m: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10),
        neighbors: tuple[int, ...] = (16, 64, 128, 256),
        geometry_mode: str = "tool_frame_relative",
        # ── Layer 2 options ──
        relation_features: bool = False,
        relation_gate_initial_bias: float = -1.5,
        relation_fusion_mode: str = "convex",
        relation_gate_maximum: float = 1.0,
        relation_attention: bool = False,
        relation_scale_gate: bool = True,
        relation_scale_gate_geo_bias: bool = True,
        relation_query_dim: int = 128,
        relation_key_dim: int = 128,
        anchor_preserving_query: bool = False,
        action_dim: int = 22,
        surface_normal_k: int = 8,
        surface_normal_confidence_threshold: float = 0.5,
    ):
        super().__init__()
        if len(radii_m) == 0 or len(radii_m) != len(neighbors):
            raise ValueError("radii_m and neighbors must be non-empty and aligned.")
        if any(float(x) <= 0.0 for x in radii_m):
            raise ValueError("query radii must be positive.")
        if any(int(x) <= 0 for x in neighbors):
            raise ValueError("query neighbor counts must be positive.")
        geometry_mode = str(geometry_mode).strip().lower()
        if geometry_mode not in {
            "absolute", "translation_relative", "tool_frame_relative",
        }:
            raise ValueError(
                "geometry_mode must be absolute, translation_relative, or "
                "tool_frame_relative."
            )

        self.radii_m = tuple(float(x) for x in radii_m)
        self.neighbors = tuple(int(x) for x in neighbors)
        self.num_scales = len(self.radii_m)
        self.geometry_mode = geometry_mode
        self.relation_features = bool(relation_features)
        self.relation_fusion_mode = str(relation_fusion_mode).strip().lower()
        if self.relation_fusion_mode not in {"convex", "semantic_residual"}:
            raise ValueError(
                "relation_fusion_mode must be convex or semantic_residual."
            )
        self.relation_gate_maximum = float(relation_gate_maximum)
        if not 0.0 <= self.relation_gate_maximum <= 1.0:
            raise ValueError("relation_gate_maximum must be in [0,1].")
        self.relation_attention = bool(relation_attention)
        self.anchor_preserving_query = bool(anchor_preserving_query)
        self.relation_scale_gate = bool(relation_scale_gate)
        self.surface_normal_k = int(surface_normal_k)
        self.surface_normal_confidence_threshold = float(
            surface_normal_confidence_threshold
        )
        if self.relation_attention and not self.relation_features:
            raise ValueError("relation_attention requires relation_features=True.")
        if int(relation_query_dim) != int(relation_key_dim):
            raise ValueError(
                "relation_query_dim and relation_key_dim must match."
            )

        # A2 semantic/relation branches with semantic-favouring gated fusion.
        self.semantic_encoder = nn.Sequential(
            nn.Linear(int(query_feature_dim) + 6, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )
        self.relation_encoder: nn.Module | None = None
        self.relation_gate: nn.Module | None = None
        if self.relation_features:
            # relative xyz(3), distance/radius(1), scale ratio(1),
            # robot mask(1), r_face(1), r_motion(1)
            self.relation_encoder = nn.Sequential(
                nn.Linear(8, int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), int(output_dim)),
            )
            self.relation_gate = nn.Sequential(
                nn.Linear(int(output_dim) * 2, int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), 1),
            )
            nn.init.zeros_(self.relation_gate[-1].weight)
            nn.init.constant_(
                self.relation_gate[-1].bias, float(relation_gate_initial_bias)
            )

        # ── per-scale projection (unchanged structure) ──
        self.scale_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(output_dim) * 4, int(output_dim)),
                nn.LayerNorm(int(output_dim)),
                nn.SiLU(),
            )
            for _ in self.radii_m
        ])

        self.anchor_role_embedding: nn.Embedding | None = None
        self.anchor_phase_embedding: nn.Embedding | None = None
        self.anchor_sweep_embedding: nn.Embedding | None = None
        self.anchor_attention: nn.Module | None = None
        if self.anchor_preserving_query:
            anchor_dim = int(output_dim) * 2
            self.anchor_role_embedding = nn.Embedding(4, anchor_dim)
            self.anchor_phase_embedding = nn.Embedding(4, anchor_dim)
            # More than enough for the configured four-point sweep while still
            # failing closed if a future config silently creates huge sweeps.
            self.anchor_sweep_embedding = nn.Embedding(32, anchor_dim)
            self.anchor_attention = nn.Sequential(
                nn.LayerNorm(anchor_dim),
                nn.Linear(anchor_dim, int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), 2),
            )

        # ── Layer 2: action query & key encoders ──
        self.action_query_encoder: ActionQueryEncoder | None = None
        self.fused_key_encoder: nn.Module | None = None
        if self.relation_attention:
            self.action_query_encoder = ActionQueryEncoder(
                action_dim=int(action_dim),
                query_dim=int(relation_query_dim),
            )
            self.fused_key_encoder = nn.Sequential(
                nn.LayerNorm(int(output_dim)),
                nn.Linear(int(output_dim), int(relation_key_dim)),
            )

        # ── Layer 2: cross-scale gate ──
        self.scale_gate: nn.Module | None = None
        self.scale_gate_geo: nn.Module | None = None
        if self.relation_scale_gate:
            self.scale_gate = nn.Sequential(
                nn.Linear(int(output_dim) * self.num_scales, int(output_dim)),
                nn.LayerNorm(int(output_dim)),
                nn.SiLU(),
                nn.Linear(int(output_dim), self.num_scales),
            )
            if bool(relation_scale_gate_geo_bias):
                self.scale_gate_geo = nn.Sequential(
                    nn.Linear(self.num_scales * 2, 64),
                    nn.SiLU(),
                    nn.Linear(64, self.num_scales),
                )

        # ── final output projection ──
        output_in_dim = (
            int(output_dim) * self.num_scales
            if not self.relation_scale_gate
            else int(output_dim)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(output_in_dim, int(output_dim)),
            nn.LayerNorm(int(output_dim)),
            nn.SiLU(),
            nn.Linear(int(output_dim), int(output_dim)),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        query_xyz: torch.Tensor,
        query_features: torch.Tensor,
        query_valid_mask: torch.Tensor,
        query_robot_mask: torch.Tensor,
        trajectory: ToolTrajectory,
        # ── Layer 2: action-conditioned query parameters ──
        action_slot: torch.Tensor | None = None,
        tool_pose_slot: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        action_direction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RelationQueryDiagnostics]:
        # ── validation ──
        if query_xyz.ndim != 3 or query_xyz.shape[-1] != 3:
            raise ValueError("query_xyz must have shape [B,N,3].")
        if query_features.ndim != 3 or query_features.shape[:2] != query_xyz.shape[:2]:
            raise ValueError("query_features must align with query_xyz.")
        if tuple(query_valid_mask.shape) != tuple(query_xyz.shape[:2]):
            raise ValueError("query_valid_mask must have shape [B,N].")
        if tuple(query_robot_mask.shape) != tuple(query_xyz.shape[:2]):
            raise ValueError("query_robot_mask must have shape [B,N].")

        centers = trajectory.keypoints_world
        batch, horizon, anchors, _ = centers.shape
        if batch != query_xyz.shape[0]:
            raise ValueError("tool trajectory and point-query batch sizes disagree.")
        flat_centers = centers.reshape(batch, horizon * anchors, 3)

        distance_compute_dtype = (
            torch.float32
            if query_xyz.dtype in {torch.float16, torch.bfloat16}
            else query_xyz.dtype
        )
        distance = torch.cdist(
            flat_centers.to(dtype=distance_compute_dtype),
            query_xyz.to(dtype=distance_compute_dtype),
        )
        distance = distance.masked_fill(
            ~query_valid_mask.to(torch.bool)[:, None], torch.inf
        )

        # ── Layer 1: collect unique selected indices → estimate normals once ──
        query_normals: torch.Tensor | None = None
        normal_confidence: torch.Tensor | None = None
        normal_weight: torch.Tensor | None = None
        if self.relation_features:
            selected_per_scale: list[torch.Tensor] = []
            for radius, neighbor_count in zip(self.radii_m, self.neighbors):
                k = min(int(neighbor_count), int(query_xyz.shape[1]))
                _, nn_idx = distance.topk(k=k, dim=-1, largest=False, sorted=False)
                nn_valid = torch.isfinite(
                    distance.gather(-1, nn_idx.clamp(min=0))
                ) & (distance.gather(-1, nn_idx.clamp(min=0)) <= radius)
                for b in range(batch):
                    valid_idx = nn_idx[b][nn_valid[b]].unique()
                    if valid_idx.numel():
                        selected_per_scale.append(valid_idx)
            if selected_per_scale:
                all_selected = torch.cat(selected_per_scale).unique()
                if all_selected.numel():
                    query_normals, normal_confidence, normal_weight = (
                        estimate_selected_normals(
                            query_xyz.float(),
                            query_valid_mask,
                            all_selected,
                            k=self.surface_normal_k,
                            orientation_centers=flat_centers.float(),
                            soft_gate_threshold=(
                                self.surface_normal_confidence_threshold
                            ),
                        )
                    )
                    # Build a lookup: index → normal  [B, N, 3]
                    _full_normals = torch.zeros(
                        batch, query_xyz.shape[1], 3,
                        device=query_xyz.device, dtype=query_xyz.dtype,
                    )
                    _full_conf = torch.ones(
                        batch, query_xyz.shape[1],
                        device=query_xyz.device, dtype=query_xyz.dtype,
                    )
                    _full_weight = torch.zeros(
                        batch, query_xyz.shape[1],
                        device=query_xyz.device, dtype=query_xyz.dtype,
                    )
                    for b in range(batch):
                        _full_normals[b, all_selected] = query_normals[b].to(
                            query_xyz.dtype
                        )
                        _full_conf[b, all_selected] = normal_confidence[b].to(
                            query_xyz.dtype
                        )
                        _full_weight[b, all_selected] = normal_weight[b].to(
                            query_xyz.dtype
                        )
                    query_normals = _full_normals
                    normal_confidence = _full_conf
                    normal_weight = _full_weight
                else:
                    query_normals = torch.zeros(
                        batch, query_xyz.shape[1], 3,
                        device=query_xyz.device, dtype=query_xyz.dtype,
                    )
                    normal_confidence = torch.ones(
                        batch, query_xyz.shape[1],
                        device=query_xyz.device, dtype=query_xyz.dtype,
                    )
                    normal_weight = torch.zeros(
                        batch, query_xyz.shape[1],
                        device=query_xyz.device, dtype=query_xyz.dtype,
                    )

        # ── Layer 2: action query ──
        action_query: torch.Tensor | None = None
        if self.relation_attention:
            if action_slot is None or tool_pose_slot is None or timestep is None:
                raise ValueError(
                    "relation_attention requires action_slot, tool_pose_slot, "
                    "and timestep."
                )
            action_query = self.action_query_encoder(
                action_slot, tool_pose_slot, timestep
            )  # [B, D_q]

        # ── Per-scale processing ──
        scale_tokens: list[torch.Tensor] = []
        counts: list[torch.Tensor] = []
        empty: list[torch.Tensor] = []
        robot_fractions: list[torch.Tensor] = []
        nearest_values: list[torch.Tensor] = []
        # diagnostics accumulators
        r_face_means: list[torch.Tensor] = []
        r_face_neg_fracs: list[torch.Tensor] = []
        r_motion_means: list[torch.Tensor] = []
        normal_conf_means: list[torch.Tensor] = []
        relation_gate_means: list[torch.Tensor] = []
        attn_entropies: list[torch.Tensor] = []
        anchor_attention_weights: list[torch.Tensor] = []
        anchor_role_ids_tensor: torch.Tensor | None = None
        anchor_phase_ids_tensor: torch.Tensor | None = None
        anchor_sweep_ids_tensor: torch.Tensor | None = None
        if self.anchor_preserving_query:
            role_ids, phase_ids, sweep_ids = anchor_identity_ids(trajectory.names)
            if max(sweep_ids, default=0) >= 32:
                raise ValueError("anchor sweep identity exceeds the supported range.")
            anchor_role_ids_tensor = torch.tensor(
                role_ids, device=query_xyz.device, dtype=torch.long
            )
            anchor_phase_ids_tensor = torch.tensor(
                phase_ids, device=query_xyz.device, dtype=torch.long
            )
            anchor_sweep_ids_tensor = torch.tensor(
                sweep_ids, device=query_xyz.device, dtype=torch.long
            )

        for scale_index, (radius, neighbor_count) in enumerate(
            zip(self.radii_m, self.neighbors)
        ):
            k = min(int(neighbor_count), int(query_xyz.shape[1]))
            nearest_distance, nearest_index = distance.topk(
                k=k, dim=-1, largest=False, sorted=False
            )
            valid = torch.isfinite(nearest_distance) & (nearest_distance <= radius)
            safe_index = nearest_index.masked_fill(~valid, -1)

            neighbor_xyz = gather_points(query_xyz, safe_index)
            neighbor_features = gather_points(query_features, safe_index)
            neighbor_robot = gather_points(
                query_robot_mask.to(query_xyz.dtype).unsqueeze(-1), safe_index
            ).squeeze(-1)

            delta_world = neighbor_xyz - flat_centers.unsqueeze(-2)
            delta_world = delta_world.reshape(batch, horizon, anchors, k, 3)

            if self.geometry_mode == "absolute":
                geometry = neighbor_xyz.reshape(batch, horizon, anchors, k, 3)
            elif self.geometry_mode == "translation_relative":
                geometry = delta_world
            else:
                rotation_t = trajectory.rotations_world_from_tool.transpose(-1, -2)
                geometry = torch.einsum(
                    "bhij,bhmkj->bhmki", rotation_t, delta_world
                )

            valid_h = valid.reshape(batch, horizon, anchors, k)
            distance_h = nearest_distance.reshape(batch, horizon, anchors, k)
            feature_h = neighbor_features.reshape(
                batch, horizon, anchors, k, query_features.shape[-1]
            )
            robot_h = neighbor_robot.reshape(batch, horizon, anchors, k)
            distance_feature = distance_h.to(dtype=feature_h.dtype)
            scale_value = torch.full_like(
                distance_feature.unsqueeze(-1), float(radius)
            )

            # ── Layer 1: compute r_face, r_motion ──
            r_face_h: torch.Tensor | None = None
            r_motion_h: torch.Tensor | None = None
            if self.relation_features and query_normals is not None:
                # Gather normals for selected neighbours
                neighbor_normal = gather_points(query_normals, safe_index)
                neighbor_normal = neighbor_normal.reshape(batch, horizon, anchors, k, 3)
                neighbor_conf = gather_points(
                    normal_confidence.unsqueeze(-1), safe_index
                ).reshape(batch, horizon, anchors, k)

                # ── Fix 3: soft weight from normal quality ──
                if normal_weight is not None:
                    neighbor_w = gather_points(
                        normal_weight.unsqueeze(-1), safe_index
                    ).reshape(batch, horizon, anchors, k)
                else:
                    neighbor_w = torch.ones_like(neighbor_conf)

                # Both vectors are world-frame: r_face = n · point_to_tool.
                # Positive values mean the selected surface faces the tool.
                d_norm = distance_h.clamp_min(1e-8)
                point_to_tool_world = -delta_world / d_norm.unsqueeze(-1)
                r_face_h = (
                    neighbor_normal * point_to_tool_world
                ).sum(dim=-1)
                r_face_h = r_face_h * neighbor_w  # soft attenuation

                # r_motion = n̂ · v̂_action
                if action_direction is not None:
                    v_norm = action_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    v_hat = action_direction / v_norm
                    if v_hat.ndim == 2:
                        direction = v_hat[:, None, None, None]
                    elif v_hat.ndim == 3 and v_hat.shape[1] == horizon:
                        direction = v_hat[:, :, None, None]
                    else:
                        raise ValueError(
                            "action_direction must have shape [B,3] or [B,H,3]."
                        )
                    r_motion_h = (neighbor_normal * direction).sum(dim=-1)
                else:
                    r_motion_h = torch.zeros_like(r_face_h)
                r_motion_h = r_motion_h * neighbor_w  # soft attenuation

                # Keep relation diagnostics per sample. Collapsing the batch
                # here makes every trajectory receive the same relation gain
                # and leaks abnormal geometry between unrelated samples.
                valid_float = valid_h.to(r_face_h.dtype)
                valid_count = valid_float.sum(dim=(1, 2, 3)).clamp_min(1.0)
                r_face_means.append(
                    (r_face_h * valid_float).sum(dim=(1, 2, 3)) / valid_count
                )
                r_face_neg_fracs.append(
                    ((r_face_h < -0.3).to(r_face_h.dtype) * valid_float).sum(
                        dim=(1, 2, 3)
                    )
                    / valid_count
                )
                r_motion_means.append(
                    (r_motion_h * valid_float).sum(dim=(1, 2, 3)) / valid_count
                )
                normal_conf_means.append(
                    (neighbor_conf * valid_float).sum(dim=(1, 2, 3))
                    / valid_count
                )

            # ── Build neighbour encoder input ──
            semantic_input = torch.cat(
                [
                    feature_h,
                    geometry,
                    distance_feature.masked_fill(
                        ~valid_h, 0.0
                    ).unsqueeze(-1),
                    scale_value,
                    robot_h.unsqueeze(-1),
                ],
                dim=-1,
            ).to(dtype=feature_h.dtype)
            semantic_token = self.semantic_encoder(semantic_input)
            if self.relation_features:
                assert self.relation_encoder is not None
                assert self.relation_gate is not None
                zero_relation = torch.zeros_like(distance_feature)
                face = r_face_h if r_face_h is not None else zero_relation
                motion = r_motion_h if r_motion_h is not None else zero_relation
                relation_input = torch.cat(
                    [
                        geometry / float(radius),
                        (
                            distance_feature.masked_fill(~valid_h, 0.0)
                            / float(radius)
                        ).unsqueeze(-1),
                        torch.full_like(
                            distance_feature.unsqueeze(-1),
                            float(radius) / max(self.radii_m),
                        ),
                        robot_h.unsqueeze(-1),
                        face.unsqueeze(-1),
                        motion.unsqueeze(-1),
                    ],
                    dim=-1,
                ).to(dtype=feature_h.dtype)
                relation_token = self.relation_encoder(relation_input)
                encoded, relation_gate = fuse_semantic_relation_tokens(
                    semantic_token,
                    relation_token,
                    self.relation_gate(
                        torch.cat([semantic_token, relation_token], dim=-1)
                    ),
                    mode=self.relation_fusion_mode,
                    gate_maximum=self.relation_gate_maximum,
                )
                valid_float = valid_h.to(encoded.dtype)
                relation_gate_means.append(
                    (
                        relation_gate.squeeze(-1) * valid_float
                    ).sum(dim=(1, 2, 3))
                    / valid_float.sum(dim=(1, 2, 3)).clamp_min(1.0)
                )
            else:
                encoded = semantic_token
                    # Fallback: no normals computed → zero relation features
            encoded = encoded.masked_fill(~valid_h.unsqueeze(-1), 0.0)

            # ── Layer 2: action-conditioned cross-attention ──
            if self.relation_attention and action_query is not None:
                # Encode relation keys
                assert self.fused_key_encoder is not None
                keys = self.fused_key_encoder(encoded)
                    # Fallback: use zero placeholders for r_face/r_motion
                # Scaled dot-product attention
                D_k = keys.shape[-1]
                attn_logits = torch.einsum(
                    "bd,bhmkd->bhmk", action_query, keys
                ) / (D_k ** 0.5)
                attn_logits = attn_logits.masked_fill(
                    ~valid_h, torch.finfo(attn_logits.dtype).min
                )
                attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, M, K]

                # Multiplicative fusion: distance prior × attention
                soft_weight = torch.exp(
                    -0.5 * (distance_h / float(radius)).square()
                ).masked_fill(~valid_h, 0.0)
                combined_weight = attn_weights * soft_weight.to(
                    attn_weights.dtype
                )
                combined_weight = combined_weight / combined_weight.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)

                # Weighted aggregation
                attended = (encoded * combined_weight.unsqueeze(-1)).sum(dim=-2)
                max_val = encoded.amax(dim=-2)
                keypoint_features = torch.cat([attended, max_val], dim=-1)

                # Entropy diagnostic
                ent_valid = valid_h.any(dim=-1)
                ent = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1)
                attn_entropies.append(ent[ent_valid].mean())
            else:
                # Original: distance-weighted mean + max
                soft_weight = torch.exp(
                    -0.5 * (distance_h / float(radius)).square()
                ).masked_fill(~valid_h, 0.0)
                feature_weight = soft_weight.to(dtype=encoded.dtype)
                mean = (encoded * feature_weight.unsqueeze(-1)).sum(dim=-2)
                mean = mean / feature_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                minimum = torch.finfo(encoded.dtype).min
                maximum = encoded.masked_fill(
                    ~valid_h.unsqueeze(-1), minimum
                ).amax(dim=-2)
                any_valid = valid_h.any(dim=-1)
                maximum = torch.where(
                    any_valid.unsqueeze(-1), maximum, torch.zeros_like(maximum)
                )
                keypoint_features = torch.cat([mean, maximum], dim=-1)

            # ── Cross-keypoint aggregation ──
            if self.anchor_preserving_query:
                assert self.anchor_role_embedding is not None
                assert self.anchor_phase_embedding is not None
                assert self.anchor_sweep_embedding is not None
                assert self.anchor_attention is not None
                assert anchor_role_ids_tensor is not None
                assert anchor_phase_ids_tensor is not None
                assert anchor_sweep_ids_tensor is not None
                identity = (
                    self.anchor_role_embedding(anchor_role_ids_tensor)
                    + self.anchor_phase_embedding(anchor_phase_ids_tensor)
                    + self.anchor_sweep_embedding(anchor_sweep_ids_tensor)
                ).to(dtype=keypoint_features.dtype)
                identified = keypoint_features + identity[None, None]
                anchor_valid = valid_h.any(dim=-1)
                logits = self.anchor_attention(identified).masked_fill(
                    ~anchor_valid.unsqueeze(-1), -1e9
                )
                weights = F.softmax(logits, dim=2)
                weights = weights * anchor_valid.unsqueeze(-1).to(weights.dtype)
                weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
                # Two learned identity-aware pooling heads preserve the same
                # 4*output_dim interface as legacy mean+max scale_projection.
                pooled = torch.einsum("bhmq,bhmd->bhqd", weights, identified)
                scale_input = pooled.flatten(start_dim=2)
                anchor_attention_weights.append(weights)
            else:
                keypoint_mean = keypoint_features.mean(dim=2)  # [B, H, D*2]
                keypoint_max = keypoint_features.amax(dim=2)
                scale_input = torch.cat([keypoint_mean, keypoint_max], dim=-1)
            scale_tokens.append(
                self.scale_projection[scale_index](scale_input)
            )

            # Diagnostics (per scale)
            count = valid_h.sum(dim=-1)
            counts.append(count)
            empty.append(~valid_h.any(dim=-1))
            robot_fractions.append(
                (robot_h * valid_h.to(robot_h.dtype)).sum(dim=-1)
                / count.clamp_min(1).to(robot_h.dtype)
            )
            nearest = distance_h.masked_fill(~valid_h, torch.inf).amin(dim=-1)
            nearest_values.append(
                torch.where(
                    torch.isfinite(nearest), nearest, torch.zeros_like(nearest)
                )
            )

        # ── Layer 2: cross-scale gate ──
        if self.relation_scale_gate and self.num_scales > 1:
            r_all = torch.cat(scale_tokens, dim=-1)  # [B, H, D*num_scales]
            gate_logits = self.scale_gate(r_all)       # [B, H, num_scales]

            if self.scale_gate_geo is not None:
                # Collect per-scale r_face_neg_fraction
                if r_face_neg_fracs:
                    rfn_per_scale = torch.stack(
                        r_face_neg_fracs, dim=-1
                    )  # [B, num_scales]
                    rfn_concat = torch.cat(
                        [rfn_per_scale, torch.zeros_like(rfn_per_scale)], dim=-1
                    )  # [B, num_scales*2]  (mean also but zero for simplicity)
                    geo_bias = self.scale_gate_geo(rfn_concat)  # [B, num_scales]
                    # Reshape to [B, 1, num_scales] to avoid broadcast doubling H
                    geo_bias = geo_bias[:, None, :]
                    gate_logits = gate_logits + geo_bias

            gate_weights = F.softmax(
                gate_logits.masked_fill(~torch.isfinite(gate_logits), -1e9),
                dim=-1,
            )  # [B, H, num_scales]

            # Weighted mix
            r_mixed = (
                torch.stack(scale_tokens, dim=-1) * gate_weights[:, :, None, :]
            ).sum(dim=-1)  # [B, H, D]
            scale_gate_weights = gate_weights
        else:
            r_mixed = torch.cat(scale_tokens, dim=-1)
            scale_gate_weights = None

        token = self.output_projection(r_mixed)  # [B, H, D_out]

        # ── Assemble diagnostics ──
        stacked_counts = torch.stack(counts, dim=2)
        any_neighbor = stacked_counts.gt(0).any(dim=3).any(dim=2)
        token = token.masked_fill(~any_neighbor.unsqueeze(-1), 0.0)

        stacked_empty = torch.stack(empty, dim=2)
        stacked_robot = torch.stack(robot_fractions, dim=2)
        stacked_nearest = torch.stack(nearest_values, dim=2)
        total_neighbors = stacked_counts.sum(dim=(1, 2, 3))
        robot_point_fraction = (
            stacked_robot * stacked_counts.to(dtype=stacked_robot.dtype)
        ).sum(dim=(1, 2, 3)) / total_neighbors.clamp_min(1).to(
            dtype=stacked_robot.dtype
        )
        nonempty = ~stacked_empty
        nearest_distance = (
            stacked_nearest * nonempty.to(stacked_nearest.dtype)
        ).sum(dim=(1, 2, 3)) / nonempty.sum(dim=(1, 2, 3)).clamp_min(1).to(
            stacked_nearest.dtype
        )

        return token, RelationQueryDiagnostics(
            valid_neighbor_count=stacked_counts,
            empty_fraction=stacked_empty.to(query_xyz.dtype).mean(dim=(1, 2, 3)),
            robot_point_fraction=robot_point_fraction,
            nearest_distance_m=nearest_distance,
            r_face_mean=(
                torch.stack(r_face_means, dim=-1) if r_face_means else None
            ),
            r_face_neg_fraction=(
                torch.stack(r_face_neg_fracs, dim=-1)
                if r_face_neg_fracs else None
            ),
            r_motion_mean=(
                torch.stack(r_motion_means, dim=-1)
                if r_motion_means else None
            ),
            normal_confidence_mean=(
                torch.stack(normal_conf_means, dim=-1)
                if normal_conf_means else None
            ),
            relation_gate_mean=(
                torch.stack(relation_gate_means, dim=-1)
                if relation_gate_means else None
            ),
            attn_entropy=(
                torch.stack(attn_entropies) if attn_entropies else None
            ),
            scale_gate_weights=scale_gate_weights,
            anchor_attention_weights=(
                torch.stack(anchor_attention_weights, dim=2)
                if anchor_attention_weights else None
            ),
            anchor_role_ids=anchor_role_ids_tensor,
            anchor_phase_ids=anchor_phase_ids_tensor,
            anchor_sweep_ids=anchor_sweep_ids_tensor,
        )

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def pose_wxyz_to_matrix_torch(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 7:
        raise ValueError("pose must end in [xyz, qw, qx, qy, qz].")
    xyz = pose[..., :3]
    quaternion = pose[..., 3:]
    quaternion_norm = quaternion.norm(dim=-1, keepdim=True)
    if bool((~torch.isfinite(quaternion_norm) | (quaternion_norm < 1e-12)).any()):
        raise ValueError(
            "pose contains a non-finite or zero quaternion; an explicit valid "
            "rotation is required."
        )
    quaternion = quaternion / quaternion_norm
    w, x, y, z = quaternion.unbind(dim=-1)
    matrix = torch.zeros(
        (*pose.shape[:-1], 4, 4), dtype=pose.dtype, device=pose.device
    )
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    matrix[..., :3, 3] = xyz
    matrix[..., 3, 3] = 1.0
    return matrix


def invert_rigid_matrix_torch(matrix: torch.Tensor) -> torch.Tensor:
    """Invert an SE(3) matrix without a general-purpose matrix inverse."""

    if matrix.shape[-2:] != (4, 4):
        raise ValueError("matrix must end in [4,4].")
    inverse = torch.zeros_like(matrix)
    rotation_t = matrix[..., :3, :3].transpose(-1, -2)
    inverse[..., :3, :3] = rotation_t
    inverse[..., :3, 3] = -(
        rotation_t @ matrix[..., :3, 3].unsqueeze(-1)
    ).squeeze(-1)
    inverse[..., 3, 3] = 1.0
    return inverse


@dataclass(frozen=True)
class TorchCandidateTrajectory:
    policy_actions: torch.Tensor
    env_actions: torch.Tensor
    command_joint_deltas: torch.Tensor
    achieved_joint_deltas: torch.Tensor
    arm_qpos: torch.Tensor
    tcp_matrices_world: torch.Tensor

    @property
    def tcp_xyz_world(self) -> torch.Tensor:
        return self.tcp_matrices_world[..., :3, 3]


class TorchCandidateTrajectoryLift(nn.Module):
    """Torch equivalent of the verified P0 linear-response + exact-FK lift.

    World alignment uses the observed current TCP pose:
    ``world_T_candidate = world_T_current @ inv(base_T_current) @ base_T_candidate``.
    This avoids a hidden fixed-base-pose input while retaining exact relative FK.
    """

    def __init__(
        self,
        *,
        normalizer_scale: np.ndarray,
        normalizer_offset: np.ndarray,
        env_action_low: np.ndarray,
        env_action_high: np.ndarray,
        arm_action_indices: tuple[int, ...],
        arm_delta_low: np.ndarray,
        arm_delta_high: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        response_weights: np.ndarray,
        fk_base_matrices: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        arm_dim = len(arm_action_indices)
        expected_weight_shape = (2 * arm_dim + 1, arm_dim)
        weights = np.asarray(response_weights, dtype=np.float32)
        if weights.shape != expected_weight_shape:
            raise ValueError(
                f"response_weights must have shape {expected_weight_shape}."
            )
        self.register_buffer(
            "normalizer_scale",
            torch.as_tensor(normalizer_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "normalizer_offset",
            torch.as_tensor(normalizer_offset, dtype=torch.float32),
        )
        self.register_buffer(
            "env_action_low",
            torch.as_tensor(env_action_low, dtype=torch.float32),
        )
        self.register_buffer(
            "env_action_high",
            torch.as_tensor(env_action_high, dtype=torch.float32),
        )
        self.register_buffer(
            "arm_action_indices",
            torch.as_tensor(arm_action_indices, dtype=torch.long),
        )
        self.register_buffer(
            "arm_delta_low",
            torch.as_tensor(arm_delta_low, dtype=torch.float32),
        )
        self.register_buffer(
            "arm_delta_high",
            torch.as_tensor(arm_delta_high, dtype=torch.float32),
        )
        self.register_buffer(
            "joint_lower",
            torch.as_tensor(joint_lower, dtype=torch.float32),
        )
        self.register_buffer(
            "joint_upper",
            torch.as_tensor(joint_upper, dtype=torch.float32),
        )
        self.register_buffer(
            "response_weights",
            torch.as_tensor(weights, dtype=torch.float32),
        )
        self.fk_base_matrices = fk_base_matrices

    @property
    def action_dim(self) -> int:
        return int(self.normalizer_scale.numel())

    @property
    def arm_dim(self) -> int:
        return int(self.arm_action_indices.numel())

    def forward(
        self,
        policy_actions: torch.Tensor,
        current_arm_qpos: torch.Tensor,
        previous_arm_qpos: torch.Tensor,
        current_tcp_pose_wxyz: torch.Tensor,
    ) -> TorchCandidateTrajectory:
        if policy_actions.ndim != 3 or policy_actions.shape[-1] != self.action_dim:
            raise ValueError("policy_actions must have shape [B,S,action_dim].")
        batch, steps, _ = policy_actions.shape
        expected_qpos = (batch, self.arm_dim)
        if tuple(current_arm_qpos.shape) != expected_qpos:
            raise ValueError(f"current_arm_qpos must have shape {expected_qpos}.")
        if tuple(previous_arm_qpos.shape) != expected_qpos:
            raise ValueError(f"previous_arm_qpos must have shape {expected_qpos}.")
        if tuple(current_tcp_pose_wxyz.shape) != (batch, 7):
            raise ValueError("current_tcp_pose_wxyz must have shape [B,7].")

        scale = self.normalizer_scale.to(policy_actions)
        offset = self.normalizer_offset.to(policy_actions)
        env_low = self.env_action_low.to(policy_actions)
        env_high = self.env_action_high.to(policy_actions)
        env_actions = torch.clamp(
            (policy_actions - offset) / scale,
            min=env_low,
            max=env_high,
        )
        arm = env_actions.index_select(-1, self.arm_action_indices)
        arm = arm.clamp(-1.0, 1.0)
        delta_low = self.arm_delta_low.to(policy_actions)
        delta_high = self.arm_delta_high.to(policy_actions)
        command = 0.5 * (delta_high + delta_low) + 0.5 * (
            delta_high - delta_low
        ) * arm

        previous_delta = current_arm_qpos - previous_arm_qpos
        qpos = current_arm_qpos
        states: list[torch.Tensor] = []
        achieved_deltas: list[torch.Tensor] = []
        weights = self.response_weights.to(policy_actions)
        lower = self.joint_lower.to(policy_actions)
        upper = self.joint_upper.to(policy_actions)
        ones = torch.ones((batch, 1), dtype=policy_actions.dtype, device=policy_actions.device)
        for slot in range(steps):
            features = torch.cat([command[:, slot], previous_delta, ones], dim=-1)
            achieved = features @ weights
            next_qpos = torch.clamp(qpos + achieved, min=lower, max=upper)
            achieved = next_qpos - qpos
            qpos = next_qpos
            previous_delta = achieved
            states.append(qpos)
            achieved_deltas.append(achieved)
        state_tensor = torch.stack(states, dim=1)
        achieved_tensor = torch.stack(achieved_deltas, dim=1)

        # One vectorized FK call avoids paying the kinematics traversal launch
        # overhead separately for the current and four candidate poses.
        all_qpos = torch.cat(
            [current_arm_qpos[:, None], state_tensor], dim=1
        )
        all_base_tcp = self.fk_base_matrices(
            all_qpos.reshape(batch * (steps + 1), self.arm_dim)
        ).reshape(batch, steps + 1, 4, 4)
        current_base_tcp = all_base_tcp[:, 0]
        candidate_base_tcp = all_base_tcp[:, 1:]
        current_world_tcp = pose_wxyz_to_matrix_torch(current_tcp_pose_wxyz)
        current_base_inverse = invert_rigid_matrix_torch(current_base_tcp)
        relative = current_base_inverse[:, None] @ candidate_base_tcp
        candidate_world = current_world_tcp[:, None] @ relative
        return TorchCandidateTrajectory(
            policy_actions=policy_actions,
            env_actions=env_actions,
            command_joint_deltas=command,
            achieved_joint_deltas=achieved_tensor,
            arm_qpos=state_tensor,
            tcp_matrices_world=candidate_world,
        )


class PyTorchKinematicsFK:
    """Small callable adapter that keeps a pytorch_kinematics chain on-device."""

    def __init__(self, chain: Any):
        self.chain = chain
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    def __call__(self, qpos: torch.Tensor) -> torch.Tensor:
        if qpos.device != self._device or qpos.dtype != self._dtype:
            self.chain = self.chain.to(device=qpos.device, dtype=qpos.dtype)
            self._device = qpos.device
            self._dtype = qpos.dtype
        return self.chain.forward_kinematics(qpos).get_matrix()


def lift_kwargs_from_action_contract(
    artifact: Mapping[str, Any],
    fk_base_matrices: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    control = artifact["control"]
    normalizer = artifact["normalizer"]
    response = artifact["controller_response"]
    kinematics = artifact["kinematics"]
    if response.get("model") != "linear_ar1":
        raise ValueError("AQR-BC requires the verified linear_ar1 response model.")
    return {
        "normalizer_scale": np.asarray(normalizer["scale"], dtype=np.float32),
        "normalizer_offset": np.asarray(normalizer["offset"], dtype=np.float32),
        "env_action_low": np.asarray(
            control["environment_action_low"], dtype=np.float32
        ),
        "env_action_high": np.asarray(
            control["environment_action_high"], dtype=np.float32
        ),
        "arm_action_indices": tuple(int(x) for x in control["arm_action_indices"]),
        "arm_delta_low": np.asarray(control["arm_delta_low_rad"], dtype=np.float32),
        "arm_delta_high": np.asarray(
            control["arm_delta_high_rad"], dtype=np.float32
        ),
        "joint_lower": np.asarray(
            kinematics["joint_lower_rad"], dtype=np.float32
        ),
        "joint_upper": np.asarray(
            kinematics["joint_upper_rad"], dtype=np.float32
        ),
        "response_weights": np.asarray(response["weights"], dtype=np.float32),
        "fk_base_matrices": fk_base_matrices,
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


class ContractError(ValueError):
    """Raised when action, controller, or kinematics semantics are ambiguous."""


def _vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ContractError(f"{name} must be a non-empty finite vector.")
    return array


def execution_slot_indices(
    horizon: int,
    n_obs_steps: int,
    n_action_steps: int,
    exec_start_index: int | None = None,
) -> np.ndarray:
    """Return the exact DP3 slots exposed by ``predict_action``.

    DP3 aligns observations and actions in one horizon.  With the project default
    ``H=4, To=2, Ta=3``, the executed/query slots are therefore ``[1, 2, 3]``.
    Silent truncation is forbidden because it would train/query different slots.
    """

    horizon = int(horizon)
    n_obs_steps = int(n_obs_steps)
    n_action_steps = int(n_action_steps)
    if horizon <= 0:
        raise ContractError("horizon must be positive.")
    if n_obs_steps <= 0:
        raise ContractError("n_obs_steps must be positive.")
    if n_action_steps <= 0:
        raise ContractError("n_action_steps must be positive.")
    start = n_obs_steps - 1 if exec_start_index is None else int(exec_start_index)
    if start < 0:
        raise ContractError("execution start must be non-negative.")
    end = start + n_action_steps
    if end > horizon:
        raise ContractError(
            "execution window exceeds the diffusion horizon: "
            f"start={start}, steps={n_action_steps}, horizon={horizon}."
        )
    return np.arange(start, end, dtype=np.int64)


@dataclass(frozen=True)
class LinearActionNormalizer:
    """DP3's affine normalizer: normalized = raw * scale + offset."""

    scale: np.ndarray
    offset: np.ndarray

    def __post_init__(self) -> None:
        scale = _vector(self.scale, "normalizer scale")
        offset = _vector(self.offset, "normalizer offset")
        if scale.shape != offset.shape:
            raise ContractError("normalizer scale and offset shapes disagree.")
        if np.any(np.abs(scale) < 1e-12):
            raise ContractError("normalizer scale contains a zero dimension.")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)

    @classmethod
    def identity(cls, action_dim: int) -> "LinearActionNormalizer":
        if int(action_dim) <= 0:
            raise ContractError("action_dim must be positive.")
        return cls(np.ones(int(action_dim)), np.zeros(int(action_dim)))

    @classmethod
    def fit_limits(
        cls,
        actions: np.ndarray,
        output_min: float = -1.0,
        output_max: float = 1.0,
        range_eps: float = 1e-4,
    ) -> "LinearActionNormalizer":
        """Match the limits-mode ``LinearNormalizer`` used by DP3."""

        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ContractError("normalizer fit actions must have shape [N, A].")
        if not np.isfinite(actions).all():
            raise ContractError("normalizer fit actions contain non-finite values.")
        input_min = actions.min(axis=0)
        input_max = actions.max(axis=0)
        input_range = input_max - input_min
        ignored = input_range < float(range_eps)
        safe_range = input_range.copy()
        safe_range[ignored] = float(output_max) - float(output_min)
        scale = (float(output_max) - float(output_min)) / safe_range
        offset = float(output_min) - scale * input_min
        offset[ignored] = (
            (float(output_max) + float(output_min)) / 2.0 - input_min[ignored]
        )
        return cls(scale=scale, offset=offset)

    @property
    def action_dim(self) -> int:
        return int(self.scale.size)

    def normalize(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = np.asarray(raw_action, dtype=np.float64)
        if raw_action.shape[-1] != self.action_dim:
            raise ContractError(
                f"raw action has dim {raw_action.shape[-1]}, expected {self.action_dim}."
            )
        return raw_action * self.scale + self.offset

    def unnormalize(self, normalized_action: np.ndarray) -> np.ndarray:
        normalized_action = np.asarray(normalized_action, dtype=np.float64)
        if normalized_action.shape[-1] != self.action_dim:
            raise ContractError(
                "normalized action has dim "
                f"{normalized_action.shape[-1]}, expected {self.action_dim}."
            )
        return (normalized_action - self.offset) / self.scale


@dataclass(frozen=True)
class ActionTrajectoryContract:
    """Explicit policy -> environment -> arm-joint action contract.

    The contract currently supports ManiSkill ``pd_joint_delta_pos`` only.  It
    deliberately excludes gripper dimensions from the arm trajectory.
    """

    normalizer: LinearActionNormalizer
    env_action_low: np.ndarray
    env_action_high: np.ndarray
    arm_action_indices: tuple[int, ...]
    gripper_action_indices: tuple[int, ...]
    arm_delta_low: np.ndarray
    arm_delta_high: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    control_mode: str = "pd_joint_delta_pos"
    controller_use_delta: bool = True
    controller_use_target: bool = False

    def __post_init__(self) -> None:
        env_low = _vector(self.env_action_low, "environment action low")
        env_high = _vector(self.env_action_high, "environment action high")
        arm_low = _vector(self.arm_delta_low, "arm delta low")
        arm_high = _vector(self.arm_delta_high, "arm delta high")
        joint_low = _vector(self.joint_lower, "joint lower")
        joint_high = _vector(self.joint_upper, "joint upper")
        arm_indices = tuple(int(x) for x in self.arm_action_indices)
        gripper_indices = tuple(int(x) for x in self.gripper_action_indices)
        if env_low.shape != env_high.shape:
            raise ContractError("environment action bound shapes disagree.")
        if env_low.size != self.normalizer.action_dim:
            raise ContractError("environment action bounds do not match normalizer dim.")
        if np.any(env_low >= env_high):
            raise ContractError("environment action lows must be below highs.")
        if not arm_indices:
            raise ContractError("at least one arm action index is required.")
        if len(set(arm_indices)) != len(arm_indices):
            raise ContractError("arm action indices contain duplicates.")
        if set(arm_indices).intersection(gripper_indices):
            raise ContractError("arm and gripper action indices overlap.")
        if min(arm_indices + gripper_indices) < 0 or max(
            arm_indices + gripper_indices
        ) >= env_low.size:
            raise ContractError("arm/gripper action index is out of bounds.")
        if arm_low.size == 1:
            arm_low = np.repeat(arm_low, len(arm_indices))
        if arm_high.size == 1:
            arm_high = np.repeat(arm_high, len(arm_indices))
        if arm_low.size != len(arm_indices) or arm_high.size != len(arm_indices):
            raise ContractError("controller arm bounds do not match arm action dims.")
        if joint_low.size != len(arm_indices) or joint_high.size != len(arm_indices):
            raise ContractError("joint limits do not match arm action dims.")
        if np.any(arm_low >= arm_high):
            raise ContractError("controller arm lows must be below highs.")
        if np.any(joint_low >= joint_high):
            raise ContractError("joint lows must be below highs.")
        if str(self.control_mode) != "pd_joint_delta_pos":
            raise ContractError(
                "P0 candidate lift supports only pd_joint_delta_pos; got "
                f"{self.control_mode!r}."
            )
        if not bool(self.controller_use_delta) or bool(self.controller_use_target):
            raise ContractError(
                "P0 assumes use_delta=True and use_target=False.  Target-delta "
                "controllers require a different rollout state."
            )
        # ManiSkill normalizes this controller input to [-1, 1].  Refuse a
        # superficially similar action space with different semantics.
        if not np.allclose(env_low[list(arm_indices)], -1.0) or not np.allclose(
            env_high[list(arm_indices)], 1.0
        ):
            raise ContractError(
                "pd_joint_delta_pos arm actions must be normalized to [-1, 1]."
            )
        object.__setattr__(self, "env_action_low", env_low)
        object.__setattr__(self, "env_action_high", env_high)
        object.__setattr__(self, "arm_delta_low", arm_low)
        object.__setattr__(self, "arm_delta_high", arm_high)
        object.__setattr__(self, "joint_lower", joint_low)
        object.__setattr__(self, "joint_upper", joint_high)
        object.__setattr__(self, "arm_action_indices", arm_indices)
        object.__setattr__(self, "gripper_action_indices", gripper_indices)

    @property
    def action_dim(self) -> int:
        return int(self.env_action_low.size)

    @property
    def arm_dim(self) -> int:
        return len(self.arm_action_indices)

    def policy_to_env(self, policy_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.normalizer.unnormalize(policy_action)
        clipped = np.clip(raw, self.env_action_low, self.env_action_high)
        return clipped, np.not_equal(clipped, raw)

    def env_to_policy(self, env_action: np.ndarray) -> np.ndarray:
        env_action = np.asarray(env_action, dtype=np.float64)
        if env_action.shape[-1] != self.action_dim:
            raise ContractError(
                f"environment action has dim {env_action.shape[-1]}, "
                f"expected {self.action_dim}."
            )
        return self.normalizer.normalize(env_action)

    def env_to_arm_delta(self, env_action: np.ndarray) -> np.ndarray:
        env_action = np.asarray(env_action, dtype=np.float64)
        if env_action.shape[-1] != self.action_dim:
            raise ContractError(
                f"environment action has dim {env_action.shape[-1]}, "
                f"expected {self.action_dim}."
            )
        arm = np.clip(env_action[..., self.arm_action_indices], -1.0, 1.0)
        return 0.5 * (self.arm_delta_high + self.arm_delta_low) + 0.5 * (
            self.arm_delta_high - self.arm_delta_low
        ) * arm


@dataclass(frozen=True)
class LinearControllerResponse:
    """Non-neural one-step arm response used to correct ideal-target overshoot.

    ``weights`` maps ``[command_delta, previous_achieved_delta, 1]`` to the next
    achieved joint delta.  ``None`` means the ideal-tracking diagnostic baseline.
    """

    arm_dim: int
    weights: np.ndarray | None = None
    ridge: float = 0.0

    def __post_init__(self) -> None:
        arm_dim = int(self.arm_dim)
        if arm_dim <= 0:
            raise ContractError("response arm_dim must be positive.")
        object.__setattr__(self, "arm_dim", arm_dim)
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64)
            expected = (2 * arm_dim + 1, arm_dim)
            if weights.shape != expected or not np.isfinite(weights).all():
                raise ContractError(
                    f"response weights must be finite with shape {expected}."
                )
            object.__setattr__(self, "weights", weights)

    @classmethod
    def ideal(cls, arm_dim: int) -> "LinearControllerResponse":
        return cls(arm_dim=int(arm_dim), weights=None)

    @property
    def mode(self) -> str:
        return "ideal_command_accumulation" if self.weights is None else "linear_ar1"

    def rollout(
        self,
        current_qpos: np.ndarray,
        command_deltas: np.ndarray,
        previous_qpos: np.ndarray | None,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current = _vector(current_qpos, "current arm qpos")
        commands = np.asarray(command_deltas, dtype=np.float64)
        lower = _vector(joint_lower, "joint lower")
        upper = _vector(joint_upper, "joint upper")
        if current.size != self.arm_dim:
            raise ContractError("current qpos does not match response arm dim.")
        if commands.ndim != 2 or commands.shape[1] != self.arm_dim:
            raise ContractError("command deltas must have shape [K, arm_dim].")
        if lower.size != self.arm_dim or upper.size != self.arm_dim:
            raise ContractError("joint limits do not match response arm dim.")
        if previous_qpos is None:
            previous_delta = np.zeros(self.arm_dim, dtype=np.float64)
        else:
            previous = _vector(previous_qpos, "previous arm qpos")
            if previous.size != self.arm_dim:
                raise ContractError("previous qpos does not match response arm dim.")
            previous_delta = current - previous

        states: list[np.ndarray] = []
        achieved_deltas: list[np.ndarray] = []
        limit_hits: list[np.ndarray] = []
        qpos = current.copy()
        for command in commands:
            if self.weights is None:
                achieved = command
            else:
                features = np.concatenate([command, previous_delta, np.ones(1)])
                achieved = features @ self.weights
            unclipped = qpos + achieved
            qpos = np.clip(unclipped, lower, upper)
            achieved = qpos - (unclipped - achieved)
            previous_delta = achieved
            states.append(qpos.copy())
            achieved_deltas.append(achieved.copy())
            limit_hits.append(np.not_equal(qpos, unclipped))
        return (
            np.asarray(states, dtype=np.float64),
            np.asarray(achieved_deltas, dtype=np.float64),
            np.asarray(limit_hits, dtype=bool),
        )


def fit_linear_controller_response(
    command_deltas: np.ndarray,
    previous_achieved_deltas: np.ndarray,
    next_achieved_deltas: np.ndarray,
    ridge: float = 1e-5,
) -> LinearControllerResponse:
    commands = np.asarray(command_deltas, dtype=np.float64)
    previous = np.asarray(previous_achieved_deltas, dtype=np.float64)
    targets = np.asarray(next_achieved_deltas, dtype=np.float64)
    if commands.ndim != 2 or commands.shape[0] == 0:
        raise ContractError("response fit commands must have shape [N, J].")
    if previous.shape != commands.shape or targets.shape != commands.shape:
        raise ContractError("response fit arrays must have identical [N, J] shapes.")
    if not (
        np.isfinite(commands).all()
        and np.isfinite(previous).all()
        and np.isfinite(targets).all()
    ):
        raise ContractError("response fit arrays contain non-finite values.")
    features = np.concatenate(
        [commands, previous, np.ones((commands.shape[0], 1))], axis=1
    )
    regularizer = float(ridge) * np.eye(features.shape[1], dtype=np.float64)
    # Do not regularize the intercept.
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(
        features.T @ features + regularizer,
        features.T @ targets,
    )
    return LinearControllerResponse(
        arm_dim=commands.shape[1], weights=weights, ridge=float(ridge)
    )


def pose_wxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != 7:
        raise ContractError("pose must use [xyz, qw, qx, qy, qz].")
    flat = pose.reshape(-1, 7)
    out = np.repeat(np.eye(4, dtype=np.float64)[None], len(flat), axis=0)
    out[:, :3, 3] = flat[:, :3]
    q = flat[:, 3:]
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ContractError("pose contains a zero quaternion.")
    w, x, y, z = (q / norms).T
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out.reshape(*pose.shape[:-1], 4, 4)


def matrix_to_pose_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (4, 4):
        raise ContractError("transform matrices must end in [4, 4].")
    flat = matrix.reshape(-1, 4, 4)
    result = np.zeros((len(flat), 7), dtype=np.float64)
    result[:, :3] = flat[:, :3, 3]
    for index, rotation in enumerate(flat[:, :3, :3]):
        trace = float(np.trace(rotation))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            quat = np.asarray(
                [
                    0.25 * s,
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[1, 0] - rotation[0, 1]) / s,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                quat = np.asarray(
                    [
                        (rotation[2, 1] - rotation[1, 2]) / s,
                        0.25 * s,
                        (rotation[0, 1] + rotation[1, 0]) / s,
                        (rotation[0, 2] + rotation[2, 0]) / s,
                    ]
                )
            elif axis == 1:
                s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                quat = np.asarray(
                    [
                        (rotation[0, 2] - rotation[2, 0]) / s,
                        (rotation[0, 1] + rotation[1, 0]) / s,
                        0.25 * s,
                        (rotation[1, 2] + rotation[2, 1]) / s,
                    ]
                )
            else:
                s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                quat = np.asarray(
                    [
                        (rotation[1, 0] - rotation[0, 1]) / s,
                        (rotation[0, 2] + rotation[2, 0]) / s,
                        (rotation[1, 2] + rotation[2, 1]) / s,
                        0.25 * s,
                    ]
                )
        quat /= np.linalg.norm(quat)
        # Quaternion signs are equivalent.  A fixed non-negative w makes saved
        # trajectories deterministic and easier to compare.
        if quat[0] < 0.0:
            quat = -quat
        result[index, 3:] = quat
    return result.reshape(*matrix.shape[:-2], 7)


def transform_anchor_positions(
    tcp_matrices: np.ndarray,
    anchors_tcp: Mapping[str, Sequence[float]] | None,
) -> dict[str, np.ndarray]:
    matrices = np.asarray(tcp_matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ContractError("TCP matrices must have shape [K, 4, 4].")
    anchors = {"tcp": (0.0, 0.0, 0.0)} if anchors_tcp is None else anchors_tcp
    if not anchors:
        raise ContractError("at least one TCP-frame anchor is required.")
    output: dict[str, np.ndarray] = {}
    for name, offset in anchors.items():
        vector = _vector(offset, f"anchor {name!r}")
        if vector.size != 3:
            raise ContractError(f"anchor {name!r} must be a 3-D TCP-frame offset.")
        output[str(name)] = (
            matrices[:, :3, :3] @ vector.reshape(1, 3, 1)
        ).squeeze(-1) + matrices[:, :3, 3]
    return output


@dataclass(frozen=True)
class CandidateTrajectory:
    slot_indices: np.ndarray
    policy_actions: np.ndarray
    env_actions: np.ndarray
    action_clipped: np.ndarray
    command_joint_deltas: np.ndarray
    achieved_joint_deltas: np.ndarray
    arm_qpos: np.ndarray
    joint_limit_hits: np.ndarray
    tcp_pose_wxyz: np.ndarray
    tcp_matrices_world: np.ndarray
    anchors_world: dict[str, np.ndarray]
    response_mode: str


def build_candidate_trajectory(
    action_horizon: np.ndarray,
    action_space: str,
    current_arm_qpos: np.ndarray,
    previous_arm_qpos: np.ndarray | None,
    robot_base_pose_wxyz: np.ndarray,
    contract: ActionTrajectoryContract,
    response: LinearControllerResponse,
    fk_base_matrices: Callable[[np.ndarray], np.ndarray],
    horizon: int,
    n_obs_steps: int,
    n_action_steps: int,
    exec_start_index: int | None = None,
    anchors_tcp: Mapping[str, Sequence[float]] | None = None,
) -> CandidateTrajectory:
    """Lift only actually executed action slots into full world-frame SE(3)."""

    actions = np.asarray(action_horizon, dtype=np.float64)
    if actions.shape != (int(horizon), contract.action_dim):
        raise ContractError(
            "action horizon must have shape "
            f"({int(horizon)}, {contract.action_dim}), got {actions.shape}."
        )
    slots = execution_slot_indices(
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        exec_start_index=exec_start_index,
    )
    selected = actions[slots]
    action_space = str(action_space).strip().lower()
    if action_space == "policy":
        policy_actions = selected
        env_actions, clipped = contract.policy_to_env(policy_actions)
    elif action_space == "env":
        unclipped = selected
        env_actions = np.clip(
            unclipped, contract.env_action_low, contract.env_action_high
        )
        clipped = np.not_equal(env_actions, unclipped)
        policy_actions = contract.env_to_policy(env_actions)
    else:
        raise ContractError("action_space must be 'policy' or 'env'.")

    command_deltas = contract.env_to_arm_delta(env_actions)
    qpos, achieved_deltas, limit_hits = response.rollout(
        current_qpos=current_arm_qpos,
        command_deltas=command_deltas,
        previous_qpos=previous_arm_qpos,
        joint_lower=contract.joint_lower,
        joint_upper=contract.joint_upper,
    )
    base_to_tcp = np.asarray(fk_base_matrices(qpos), dtype=np.float64)
    expected_fk_shape = (len(slots), 4, 4)
    if base_to_tcp.shape != expected_fk_shape:
        raise ContractError(
            f"FK callback returned {base_to_tcp.shape}, expected {expected_fk_shape}."
        )
    world_from_base = pose_wxyz_to_matrix(
        np.asarray(robot_base_pose_wxyz, dtype=np.float64).reshape(1, 7)
    )[0]
    world_tcp = world_from_base[None] @ base_to_tcp
    tcp_pose = matrix_to_pose_wxyz(world_tcp)
    anchors = transform_anchor_positions(world_tcp, anchors_tcp)
    return CandidateTrajectory(
        slot_indices=slots,
        policy_actions=policy_actions,
        env_actions=env_actions,
        action_clipped=clipped,
        command_joint_deltas=command_deltas,
        achieved_joint_deltas=achieved_deltas,
        arm_qpos=qpos,
        joint_limit_hits=limit_hits,
        tcp_pose_wxyz=tcp_pose,
        tcp_matrices_world=world_tcp,
        anchors_world=anchors,
        response_mode=response.mode,
    )

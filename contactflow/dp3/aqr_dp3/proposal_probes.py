from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class CartesianPerturbationResult:
    """Result of a differentiable proposal-space Cartesian perturbation."""

    actions: torch.Tensor
    requested_twist: torch.Tensor
    achieved_twist: torch.Tensor
    translation_error_m: float
    rotation_error_rad: float
    action_delta_norm: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class QueryMembership:
    """Evaluator-only reconstruction of the points visited by an AQR query."""

    unique_indices: torch.Tensor
    neighbor_count: torch.Tensor
    centroid_world: torch.Tensor
    covariance_world: torch.Tensor
    group_indices: tuple[torch.Tensor, ...]


def _skew_vector(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )


def _rotation_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Stable SO(3) logarithm for evaluator-scale rotations."""

    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    skew = _skew_vector(matrix)
    sine = torch.sin(angle)
    scale = angle / sine.clamp_min(1e-7)
    vector = skew * scale.unsqueeze(-1)
    small = angle < 1e-5
    return torch.where(small.unsqueeze(-1), skew, vector)


def solve_cartesian_action_perturbation(
    *,
    base_actions: torch.Tensor,
    pose_from_actions: Callable[[torch.Tensor], torch.Tensor],
    requested_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
    requested_rotation_rad: Sequence[float] = (0.0, 0.0, 0.0),
    reference_frame_rotation: torch.Tensor | None = None,
    variable_slots: Sequence[int],
    arm_dim: int,
    damping: float = 1e-3,
    maximum_step_norm: float = 0.08,
    maximum_action_delta_norm: float = 0.30,
    translation_tolerance_m: float = 5e-4,
    rotation_tolerance_rad: float = 0.01,
    max_iterations: int = 12,
) -> CartesianPerturbationResult:
    """Find the smallest normalized-action change that realizes an SE(3) offset.

    The callback must return one ``[4,4]`` pose for a ``[1,H,A]`` action
    tensor. Translation and rotation requests are expressed in
    ``reference_frame_rotation``; by default the unperturbed tool frame is
    used. The routine is intended for offline mechanism probes and never
    executes the resulting action.
    """

    if base_actions.ndim != 3 or base_actions.shape[0] != 1:
        raise ValueError("base_actions must have shape [1,H,A].")
    if not 0 < int(arm_dim) <= int(base_actions.shape[-1]):
        raise ValueError("arm_dim must fit the action dimension.")
    slots = tuple(int(value) for value in variable_slots)
    if not slots or len(set(slots)) != len(slots):
        raise ValueError("variable_slots must be non-empty and unique.")
    if min(slots) < 0 or max(slots) >= base_actions.shape[1]:
        raise ValueError("variable_slots fall outside the action horizon.")
    if min(float(damping), float(maximum_step_norm), float(maximum_action_delta_norm)) <= 0:
        raise ValueError("DLS damping and action limits must be positive.")

    device = base_actions.device
    dtype = base_actions.dtype
    requested = torch.tensor(
        tuple(float(value) for value in requested_translation_m)
        + tuple(float(value) for value in requested_rotation_rad),
        device=device,
        dtype=dtype,
    )
    if requested.numel() != 6 or not torch.isfinite(requested).all():
        raise ValueError("requested translation and rotation must contain 3 values each.")

    base = base_actions.detach().clone()
    with torch.no_grad():
        base_pose = pose_from_actions(base).detach()
    if tuple(base_pose.shape) != (4, 4):
        raise ValueError("pose_from_actions must return one [4,4] pose.")
    frame_rotation = (
        base_pose[:3, :3]
        if reference_frame_rotation is None
        else reference_frame_rotation.to(device=device, dtype=dtype)
    )
    if tuple(frame_rotation.shape) != (3, 3):
        raise ValueError("reference_frame_rotation must have shape [3,3].")

    slot_tensor = torch.tensor(slots, device=device, dtype=torch.long)
    initial_variables = base.index_select(1, slot_tensor)[..., : int(arm_dim)]
    variables = initial_variables.detach().clone()

    def compose(flat_variables: torch.Tensor) -> torch.Tensor:
        selected = flat_variables.reshape(1, len(slots), int(arm_dim))
        actions = base.clone()
        current = actions.index_select(1, slot_tensor).clone()
        current[..., : int(arm_dim)] = selected
        actions[:, slot_tensor] = current
        return actions.clamp(-1.0, 1.0)

    def achieved_feature(flat_variables: torch.Tensor) -> torch.Tensor:
        pose = pose_from_actions(compose(flat_variables))
        translation_world = pose[:3, 3] - base_pose[:3, 3]
        translation_frame = frame_rotation.transpose(0, 1) @ translation_world
        relative_world = pose[:3, :3] @ base_pose[:3, :3].transpose(0, 1)
        relative_frame = (
            frame_rotation.transpose(0, 1) @ relative_world @ frame_rotation
        )
        rotation_frame = _skew_vector(relative_frame)
        return torch.cat((translation_frame, rotation_frame), dim=0)

    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        flat = variables.reshape(-1).detach().requires_grad_(True)
        current_feature = achieved_feature(flat)
        error = requested - current_feature
        if (
            float(error[:3].norm().detach()) <= float(translation_tolerance_m)
            and float(error[3:].norm().detach()) <= float(rotation_tolerance_rad)
        ):
            variables = flat.detach().reshape_as(variables)
            break
        jacobian = torch.autograd.functional.jacobian(
            achieved_feature, flat, create_graph=False, vectorize=False
        )
        identity = torch.eye(6, device=device, dtype=dtype)
        system = jacobian @ jacobian.transpose(0, 1) + float(damping) * identity
        step = jacobian.transpose(0, 1) @ torch.linalg.solve(system, error.detach())
        step_norm = step.norm().clamp_min(1e-12)
        step = step * min(1.0, float(maximum_step_norm) / float(step_norm))
        candidate = flat.detach() + step
        total_delta = candidate - initial_variables.reshape(-1)
        total_norm = total_delta.norm().clamp_min(1e-12)
        total_delta = total_delta * min(
            1.0, float(maximum_action_delta_norm) / float(total_norm)
        )
        variables = (initial_variables.reshape(-1) + total_delta).reshape_as(
            variables
        )

    actions = compose(variables.reshape(-1)).detach()
    with torch.no_grad():
        final_pose = pose_from_actions(actions).detach()
        translation_world = final_pose[:3, 3] - base_pose[:3, 3]
        translation_frame = frame_rotation.transpose(0, 1) @ translation_world
        relative_world = final_pose[:3, :3] @ base_pose[:3, :3].transpose(0, 1)
        relative_frame = (
            frame_rotation.transpose(0, 1) @ relative_world @ frame_rotation
        )
        rotation_frame = _rotation_vector(relative_frame)
        achieved = torch.cat((translation_frame, rotation_frame), dim=0)
        translation_error = float((achieved[:3] - requested[:3]).norm())
        rotation_error = float((achieved[3:] - requested[3:]).norm())
        action_delta_norm = float((actions - base).norm())
        converged = (
            translation_error <= float(translation_tolerance_m)
            and rotation_error <= float(rotation_tolerance_rad)
        )
    return CartesianPerturbationResult(
        actions=actions,
        requested_twist=requested.detach(),
        achieved_twist=achieved.detach(),
        translation_error_m=translation_error,
        rotation_error_rad=rotation_error,
        action_delta_norm=action_delta_norm,
        converged=bool(converged),
        iterations=int(iterations),
    )


@torch.no_grad()
def reconstruct_query_membership(
    *,
    query_xyz: torch.Tensor,
    query_valid_mask: torch.Tensor,
    centers_world: torch.Tensor,
    radii_m: Sequence[float],
    neighbors: Sequence[int],
) -> QueryMembership:
    """Reproduce AQR radius-plus-top-k membership without changing inference."""

    if query_xyz.ndim != 2 or query_xyz.shape[-1] != 3:
        raise ValueError("query_xyz must have shape [N,3].")
    if tuple(query_valid_mask.shape) != (query_xyz.shape[0],):
        raise ValueError("query_valid_mask must have shape [N].")
    if centers_world.ndim != 3 or centers_world.shape[-1] != 3:
        raise ValueError("centers_world must have shape [H,M,3].")
    radii = tuple(float(value) for value in radii_m)
    counts = tuple(int(value) for value in neighbors)
    if not radii or len(radii) != len(counts):
        raise ValueError("radii_m and neighbors must be non-empty and aligned.")

    horizon, anchors, _ = centers_world.shape
    distance = torch.cdist(centers_world.reshape(-1, 3), query_xyz)
    distance = distance.masked_fill(~query_valid_mask.to(torch.bool)[None], torch.inf)
    membership: list[torch.Tensor] = []
    group_indices: list[torch.Tensor] = []
    neighbor_count = torch.zeros(
        horizon,
        len(radii),
        anchors,
        device=query_xyz.device,
        dtype=torch.long,
    )
    for scale_index, (radius, count) in enumerate(zip(radii, counts)):
        k = min(int(count), int(query_xyz.shape[0]))
        values, indices = distance.topk(k=k, dim=-1, largest=False, sorted=False)
        selected = torch.isfinite(values) & values.le(float(radius))
        group_indices.extend(
            indices[row, selected[row]].unique(sorted=True)
            for row in range(indices.shape[0])
        )
        neighbor_count[:, scale_index] = selected.sum(dim=-1).reshape(
            horizon, anchors
        )
        if selected.any():
            membership.append(indices[selected])
    unique = (
        torch.cat(membership).unique(sorted=True)
        if membership
        else torch.empty(0, device=query_xyz.device, dtype=torch.long)
    )
    if unique.numel() == 0:
        centroid = torch.zeros(3, device=query_xyz.device, dtype=query_xyz.dtype)
        covariance = torch.zeros(3, 3, device=query_xyz.device, dtype=query_xyz.dtype)
    else:
        points = query_xyz.index_select(0, unique)
        centroid = points.mean(dim=0)
        centered = points - centroid
        covariance = centered.transpose(0, 1) @ centered / max(1, points.shape[0] - 1)
    return QueryMembership(
        unique_indices=unique,
        neighbor_count=neighbor_count,
        centroid_world=centroid,
        covariance_world=covariance,
        group_indices=tuple(group_indices),
    )


def query_membership_jaccard(first: QueryMembership, second: QueryMembership) -> float:
    first_indices = first.unique_indices
    second_indices = second.unique_indices.to(first_indices.device)
    if first_indices.numel() == 0 and second_indices.numel() == 0:
        return 1.0
    intersection = torch.isin(first_indices, second_indices).sum()
    union = first_indices.numel() + second_indices.numel() - int(intersection)
    return float(intersection) / float(max(1, union))


def query_membership_group_jaccard(
    first: QueryMembership, second: QueryMembership
) -> float:
    """Mean Jaccard over aligned scale × timestep × anchor neighborhoods."""

    if len(first.group_indices) != len(second.group_indices):
        raise ValueError("query memberships have different group counts")
    if not first.group_indices:
        return 1.0
    scores: list[float] = []
    for first_indices, second_indices in zip(
        first.group_indices, second.group_indices, strict=True
    ):
        second_indices = second_indices.to(first_indices.device)
        if first_indices.numel() == 0 and second_indices.numel() == 0:
            scores.append(1.0)
            continue
        intersection = torch.isin(first_indices, second_indices).sum()
        union = first_indices.numel() + second_indices.numel() - int(intersection)
        scores.append(float(intersection) / float(max(1, union)))
    return float(sum(scores) / len(scores))


def recovery_rate(error_before: float, error_after: float) -> float:
    """Unclipped recovery rate; zero-error controls are intentionally undefined."""

    before = float(error_before)
    if before <= 1e-8:
        return float("nan")
    return (before - float(error_after)) / before

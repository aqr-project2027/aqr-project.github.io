from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from contactflow.dp3.pointnext_relation.backbone import farthest_point_indices

from .config import PointCloudFrontEndConfig


GLOBAL_BANK_ID = 0
TCP_SWEPT_BANK_ID = 1
FILL_BANK_ID = 2
PADDING_BANK_ID = -1
_HASH_MODULUS = 2_147_483_647


class PointCloudContractError(ValueError):
    """Raised when the deployable point-cloud contract is violated."""


@dataclass(frozen=True)
class PointCloudBatch:
    """Fixed-shape encoder/query banks and their source-level provenance."""

    enc_points: torch.Tensor
    query_points: torch.Tensor
    enc_valid_mask: torch.Tensor
    query_valid_mask: torch.Tensor
    enc_robot_mask: torch.Tensor
    query_robot_mask: torch.Tensor
    enc_source_indices: torch.Tensor
    query_source_indices: torch.Tensor
    enc_bank_ids: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor]


def _fixed_take(
    order: torch.Tensor,
    eligible: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Take a deterministic ordered prefix and pad missing entries with ``-1``."""

    output = torch.full(
        (int(count),), -1, dtype=torch.long, device=eligible.device
    )
    take = min(int(count), int(order.numel()))
    if take:
        candidates = order[:take]
        output[:take] = torch.where(
            eligible[candidates],
            candidates,
            torch.full_like(candidates, -1),
        )
    return output


def _hashed_order(
    eligible: torch.Tensor,
    sample_key: torch.Tensor,
) -> torch.Tensor:
    """Return a stable device-local pseudo-random ordering of raw indices."""

    source = torch.arange(
        eligible.numel(), dtype=torch.long, device=eligible.device
    )
    key = torch.remainder(sample_key.to(dtype=torch.long), _HASH_MODULUS)
    quadratic = torch.remainder(source * source, _HASH_MODULUS)
    score = torch.remainder(
        quadratic * 40_692 + source * 48_271 + key,
        _HASH_MODULUS,
    )
    score = torch.where(
        eligible,
        score,
        torch.full_like(score, _HASH_MODULUS),
    )
    return torch.argsort(score, stable=True)


def _hashed_sample(
    eligible: torch.Tensor,
    count: int,
    sample_key: torch.Tensor,
) -> torch.Tensor:
    return _fixed_take(_hashed_order(eligible, sample_key), eligible, count)


def _farthest_point_sample(
    xyz: torch.Tensor,
    eligible: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Deterministic masked FPS that never transfers tensors to the CPU."""

    point_count = int(xyz.shape[0])
    output = torch.full(
        (int(count),), -1, dtype=torch.long, device=xyz.device
    )
    if point_count == 0:
        return output

    source = torch.arange(point_count, dtype=torch.long, device=xyz.device)
    available = eligible.clone()
    eligible_float = eligible.to(dtype=xyz.dtype)
    denominator = eligible_float.sum().clamp_min(1.0)
    centroid = (xyz * eligible_float.unsqueeze(-1)).sum(dim=0) / denominator
    centroid_distance = ((xyz - centroid) ** 2).sum(dim=-1)
    min_distance = torch.full_like(centroid_distance, torch.inf)

    for slot in range(int(count)):
        score = centroid_distance if slot == 0 else min_distance
        score = score.masked_fill(~available, -torch.inf)
        candidate = torch.argmax(score)
        has_candidate = available.any()
        output[slot] = torch.where(
            has_candidate,
            candidate,
            torch.full_like(candidate, -1),
        )
        candidate_distance = ((xyz - xyz[candidate]) ** 2).sum(dim=-1)
        min_distance = torch.where(
            has_candidate,
            torch.minimum(min_distance, candidate_distance),
            min_distance,
        )
        available = available & source.ne(candidate)
    return output


def _batched_farthest_point_sample(
    xyz: torch.Tensor,
    eligible: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the shared batched FPS backend when every source point is valid.

    Formal AQR datasets guarantee a full filtered Query4096 bank. That common
    path can therefore use the existing pointnet2/PyTorch3D CUDA backend (or
    its batch-vectorized PyTorch fallback) once for the whole batch instead of
    launching the scalar FPS loop ``B * count`` times. The exact masked
    implementation remains the fallback for ablations or padded inputs.
    """

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [B,N,3].")
    if tuple(eligible.shape) != tuple(xyz.shape[:2]):
        raise ValueError("eligible must have shape [B,N].")
    if int(count) <= 0:
        raise ValueError("FPS count must be positive.")

    full_valid_rows = eligible.all(dim=1)
    output = torch.full(
        (int(xyz.shape[0]), int(count)),
        -1,
        dtype=torch.long,
        device=xyz.device,
    )
    if bool(full_valid_rows.any().item()):
        output[full_valid_rows] = farthest_point_indices(
            xyz[full_valid_rows],
            int(count),
        )
    for batch_index in full_valid_rows.logical_not().nonzero(
        as_tuple=False
    ).flatten().tolist():
        output[batch_index] = _farthest_point_sample(
            xyz[batch_index],
            eligible[batch_index],
            int(count),
        )
    return output, full_valid_rows


def _mask_from_indices(indices: torch.Tensor, point_count: int) -> torch.Tensor:
    selected = torch.zeros(
        (int(point_count),), dtype=torch.long, device=indices.device
    )
    if point_count:
        safe = indices.clamp_min(0)
        selected.scatter_add_(0, safe, indices.ge(0).to(dtype=torch.long))
    return selected.gt(0)


def _tcp_swept_sample(
    xyz: torch.Tensor,
    eligible: torch.Tensor,
    excluded: torch.Tensor,
    tcp_xyz: torch.Tensor | None,
    tcp_valid: torch.Tensor | None,
    count: int,
    radius_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratify tube points over their nearest valid TCP-history slot."""

    if tcp_xyz is None or tcp_valid is None or int(tcp_xyz.shape[0]) == 0:
        return (
            torch.full(
                (int(count),), -1, dtype=torch.long, device=xyz.device
            ),
            torch.zeros((), dtype=torch.long, device=xyz.device),
        )

    safe_tcp = torch.where(tcp_valid.unsqueeze(-1), tcp_xyz, 0.0)
    distance = torch.linalg.vector_norm(
        xyz.unsqueeze(1) - safe_tcp.unsqueeze(0), dim=-1
    )
    distance = distance.masked_fill(~tcp_valid.unsqueeze(0), torch.inf)
    min_distance, assigned_slot = distance.min(dim=1)
    in_tube = (
        eligible
        & ~excluded
        & torch.isfinite(min_distance)
        & min_distance.lt(float(radius_m))
    )

    # Stable two-pass sorting yields (trajectory slot, distance, source index).
    distance_order = torch.argsort(min_distance, stable=True)
    group_order = torch.argsort(
        assigned_slot[distance_order], stable=True
    )
    grouped = distance_order[group_order]
    grouped_slot = assigned_slot[grouped]
    position = torch.arange(
        grouped.numel(), dtype=torch.long, device=xyz.device
    )
    new_group = torch.ones_like(position, dtype=torch.bool)
    if int(grouped.numel()) > 1:
        new_group[1:] = grouped_slot[1:].ne(grouped_slot[:-1])
    group_start = torch.cummax(
        torch.where(new_group, position, torch.zeros_like(position)), dim=0
    ).values
    local_rank_grouped = position - group_start
    local_rank = torch.empty_like(local_rank_grouped)
    local_rank[grouped] = local_rank_grouped

    # Round-robin ranks protect the full recent trajectory instead of letting a
    # dense single frame consume the entire TCP-Swept quota.
    priority = local_rank * (int(tcp_xyz.shape[0]) + 1) + assigned_slot
    priority = torch.where(
        in_tube,
        priority,
        torch.full_like(priority, torch.iinfo(torch.long).max),
    )
    order = torch.argsort(priority, stable=True)
    return _fixed_take(order, in_tube, count), in_tube.sum()


def _gather_points(
    points: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = indices.ge(0)
    safe = indices.clamp_min(0)
    gathered = torch.gather(
        points,
        1,
        safe.unsqueeze(-1).expand(-1, -1, points.shape[-1]),
    )
    gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0.0)
    return gathered, valid


def _gather_mask(mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    valid = indices.ge(0)
    return torch.gather(mask, 1, indices.clamp_min(0)) & valid


class PointCloudFrontEnd(nn.Module):
    """Build the deterministic protected encoder bank and raw query bank."""

    def __init__(self, config: PointCloudFrontEndConfig | None = None) -> None:
        super().__init__()
        self.config = (
            PointCloudFrontEndConfig() if config is None else config.validate()
        )
        crop_min = (
            torch.empty((0,), dtype=torch.float32)
            if self.config.workspace_crop_min is None
            else torch.tensor(self.config.workspace_crop_min, dtype=torch.float32)
        )
        crop_max = (
            torch.empty((0,), dtype=torch.float32)
            if self.config.workspace_crop_max is None
            else torch.tensor(self.config.workspace_crop_max, dtype=torch.float32)
        )
        self.register_buffer("_workspace_crop_min", crop_min, persistent=False)
        self.register_buffer("_workspace_crop_max", crop_max, persistent=False)

    def _sample_keys(
        self,
        sample_key: int | torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(sample_key, torch.Tensor):
            keys = sample_key.to(device=device, dtype=torch.long)
            if keys.numel() == 1:
                keys = keys.reshape(1).expand(batch_size)
            elif tuple(keys.shape) != (batch_size,):
                raise PointCloudContractError(
                    "sample_key tensor must be scalar or have shape [B]."
                )
        else:
            keys = torch.full(
                (batch_size,),
                int(sample_key),
                dtype=torch.long,
                device=device,
            )
        return torch.remainder(keys + int(self.config.seed), _HASH_MODULUS)

    @torch.no_grad()
    def forward(
        self,
        raw_points: torch.Tensor,
        tcp_history: torch.Tensor | None,
        raw_valid_mask: torch.Tensor | None = None,
        raw_robot_mask: torch.Tensor | None = None,
        sample_key: int | torch.Tensor = 0,
    ) -> PointCloudBatch:
        if not isinstance(raw_points, torch.Tensor):
            raise PointCloudContractError("raw_points must be a torch tensor.")
        if raw_points.ndim != 3 or raw_points.shape[-1] < 3:
            raise PointCloudContractError(
                "raw_points must have shape [B, N, C] with C >= 3."
            )
        if not raw_points.is_floating_point():
            raise PointCloudContractError("raw_points must be floating point.")
        batch_size, raw_count = (int(raw_points.shape[0]), int(raw_points.shape[1]))
        if batch_size <= 0 or raw_count <= 0:
            raise PointCloudContractError("raw_points cannot have an empty batch/bank.")

        finite = torch.isfinite(raw_points).all(dim=-1)
        if raw_valid_mask is None:
            input_valid = finite
        else:
            if tuple(raw_valid_mask.shape) != (batch_size, raw_count):
                raise PointCloudContractError(
                    "raw_valid_mask must have shape [B, N]."
                )
            input_valid = raw_valid_mask.to(
                device=raw_points.device, dtype=torch.bool
            ) & finite
        workspace_valid = torch.ones_like(input_valid)
        if self.config.workspace_filtering:
            xyz_raw = raw_points[..., :3]
            crop_min = self._workspace_crop_min.to(
                device=raw_points.device,
                dtype=raw_points.dtype,
            )
            crop_max = self._workspace_crop_max.to(
                device=raw_points.device,
                dtype=raw_points.dtype,
            )
            workspace_valid = (xyz_raw > crop_min).all(dim=-1)
            workspace_valid &= (xyz_raw < crop_max).all(dim=-1)
        workspace_rejected = input_valid & ~workspace_valid
        valid = input_valid & workspace_valid
        if raw_robot_mask is None:
            robot = torch.zeros_like(valid)
        else:
            if tuple(raw_robot_mask.shape) != (batch_size, raw_count):
                raise PointCloudContractError(
                    "raw_robot_mask must have shape [B, N]."
                )
            robot = raw_robot_mask.to(
                device=raw_points.device, dtype=torch.bool
            ) & valid
        safe_points = torch.nan_to_num(
            raw_points, nan=0.0, posinf=0.0, neginf=0.0
        )
        xyz = safe_points[..., :3]

        tcp_xyz: torch.Tensor | None
        tcp_valid: torch.Tensor | None
        if tcp_history is None:
            tcp_xyz = None
            tcp_valid = None
        else:
            if not isinstance(tcp_history, torch.Tensor):
                raise PointCloudContractError(
                    "tcp_history must be a torch tensor or None."
                )
            if (
                tcp_history.ndim < 3
                or int(tcp_history.shape[0]) != batch_size
                or int(tcp_history.shape[-1]) not in {3, 7}
            ):
                raise PointCloudContractError(
                    "tcp_history must have shape [B, ..., 3] or [B, ..., 7]."
                )
            tcp_xyz = tcp_history[..., :3].to(
                device=raw_points.device, dtype=raw_points.dtype
            )
            tcp_xyz = tcp_xyz.reshape(batch_size, -1, 3)
            tcp_valid = torch.isfinite(tcp_xyz).all(dim=-1)
            tcp_xyz = torch.nan_to_num(
                tcp_xyz, nan=0.0, posinf=0.0, neginf=0.0
            )

        keys = self._sample_keys(sample_key, batch_size, raw_points.device)
        fps_count = (
            self.config.enc_global_points
            if self.config.protected_sampling
            else self.config.enc_points
        )
        batched_fps_indices, used_batched_fps = _batched_farthest_point_sample(
            xyz,
            valid,
            fps_count,
        )
        enc_indices_per_batch: list[torch.Tensor] = []
        enc_bank_ids_per_batch: list[torch.Tensor] = []
        query_indices_per_batch: list[torch.Tensor] = []
        tube_count_per_batch: list[torch.Tensor] = []

        for batch_index in range(batch_size):
            sample_valid = valid[batch_index]
            if self.config.protected_sampling:
                global_indices = batched_fps_indices[batch_index]
                global_selected = _mask_from_indices(
                    global_indices, raw_count
                )
                swept_indices, tube_count = _tcp_swept_sample(
                    xyz[batch_index],
                    sample_valid,
                    global_selected,
                    None if tcp_xyz is None else tcp_xyz[batch_index],
                    None if tcp_valid is None else tcp_valid[batch_index],
                    self.config.enc_tcp_swept_points,
                    self.config.tcp_radius_m,
                )
                base_indices = torch.cat([global_indices, swept_indices])
                base_bank_ids = torch.cat(
                    [
                        torch.where(
                            global_indices.ge(0),
                            torch.full_like(global_indices, GLOBAL_BANK_ID),
                            torch.full_like(global_indices, PADDING_BANK_ID),
                        ),
                        torch.where(
                            swept_indices.ge(0),
                            torch.full_like(
                                swept_indices, TCP_SWEPT_BANK_ID
                            ),
                            torch.full_like(
                                swept_indices, PADDING_BANK_ID
                            ),
                        ),
                    ]
                )

                used = _mask_from_indices(base_indices, raw_count)
                fill_indices = _hashed_sample(
                    sample_valid & ~used,
                    self.config.enc_points,
                    keys[batch_index] + 17,
                )
                missing = base_indices.lt(0)
                missing_rank = missing.to(dtype=torch.long).cumsum(0) - 1
                replacement = fill_indices[
                    missing_rank.clamp(min=0, max=self.config.enc_points - 1)
                ]
                can_fill = missing & replacement.ge(0)
                base_indices = torch.where(
                    can_fill, replacement, base_indices
                )
                base_bank_ids = torch.where(
                    can_fill,
                    torch.full_like(base_bank_ids, FILL_BANK_ID),
                    base_bank_ids,
                )
                enc_indices = base_indices
                enc_bank_ids = base_bank_ids
            else:
                enc_indices = batched_fps_indices[batch_index]
                enc_bank_ids = torch.where(
                    enc_indices.ge(0),
                    torch.full_like(enc_indices, GLOBAL_BANK_ID),
                    torch.full_like(enc_indices, PADDING_BANK_ID),
                )
                tube_count = torch.zeros(
                    (), dtype=torch.long, device=raw_points.device
                )

            query_indices = _hashed_sample(
                sample_valid,
                self.config.query_points,
                keys[batch_index] + 1_000_003,
            )
            enc_indices_per_batch.append(enc_indices)
            enc_bank_ids_per_batch.append(enc_bank_ids)
            query_indices_per_batch.append(query_indices)
            tube_count_per_batch.append(tube_count)

        enc_source_indices = torch.stack(enc_indices_per_batch)
        enc_bank_ids = torch.stack(enc_bank_ids_per_batch)
        query_source_indices = torch.stack(query_indices_per_batch)
        enc_points, enc_valid_mask = _gather_points(
            safe_points, enc_source_indices
        )
        query_points, query_valid_mask = _gather_points(
            safe_points, query_source_indices
        )
        enc_robot_mask = _gather_mask(robot, enc_source_indices)
        query_robot_mask = _gather_mask(robot, query_source_indices)

        enc_denominator = enc_valid_mask.sum(dim=1).clamp_min(1)
        query_denominator = query_valid_mask.sum(dim=1).clamp_min(1)
        tube_count = torch.stack(tube_count_per_batch)
        diagnostics: dict[str, torch.Tensor] = {
            "raw_input_valid_count": input_valid.sum(dim=1),
            "raw_valid_count": valid.sum(dim=1),
            "raw_workspace_rejected_count": workspace_rejected.sum(dim=1),
            "enc_valid_count": enc_valid_mask.sum(dim=1),
            "query_valid_count": query_valid_mask.sum(dim=1),
            "enc_global_count": enc_bank_ids.eq(GLOBAL_BANK_ID).sum(dim=1),
            "enc_tcp_swept_count": enc_bank_ids.eq(
                TCP_SWEPT_BANK_ID
            ).sum(dim=1),
            "enc_fill_count": enc_bank_ids.eq(FILL_BANK_ID).sum(dim=1),
            "enc_padding_count": enc_valid_mask.logical_not().sum(dim=1),
            "query_padding_count": query_valid_mask.logical_not().sum(dim=1),
            "tcp_tube_count": tube_count,
            "tcp_swept_zero_coverage": (
                tube_count.eq(0)
                if self.config.protected_sampling
                else torch.zeros_like(tube_count, dtype=torch.bool)
            ),
            "enc_robot_ratio": enc_robot_mask.sum(dim=1).to(
                dtype=torch.float32
            )
            / enc_denominator.to(dtype=torch.float32),
            "query_robot_ratio": query_robot_mask.sum(dim=1).to(
                dtype=torch.float32
            )
            / query_denominator.to(dtype=torch.float32),
            "enc_duplicate_count": torch.zeros(
                batch_size, dtype=torch.long, device=raw_points.device
            ),
            "query_duplicate_count": torch.zeros(
                batch_size, dtype=torch.long, device=raw_points.device
            ),
            "encoder_fps_batched_fast_path": used_batched_fps,
        }
        return PointCloudBatch(
            enc_points=enc_points,
            query_points=query_points,
            enc_valid_mask=enc_valid_mask,
            query_valid_mask=query_valid_mask,
            enc_robot_mask=enc_robot_mask,
            query_robot_mask=query_robot_mask,
            enc_source_indices=enc_source_indices,
            query_source_indices=query_source_indices,
            enc_bank_ids=enc_bank_ids,
            diagnostics=diagnostics,
        )

from __future__ import annotations

import importlib
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

try:
    from pointnet2_ops.pointnet2_utils import (
        furthest_point_sample as _pointnet2_fps,
    )
except Exception:
    _pointnet2_fps = None

try:
    from pytorch3d.ops import sample_farthest_points as _pytorch3d_fps
    _pytorch3d_fps_module = importlib.import_module(
        "pytorch3d.ops.sample_farthest_points"
    )
    # The simplified PyTorch3D bundled by some DP3 environments exposes the
    # Python wrapper even when its compiled extension is unavailable.  In that
    # case importing sample_farthest_points succeeds, but every call fails with
    # ``AttributeError: 'NoneType' object has no attribute ...``.
    if getattr(_pytorch3d_fps_module, "_C", None) is None:
        _pytorch3d_fps = None
except Exception:
    _pytorch3d_fps = None


# This module is an adapted, pure-PyTorch subset of the PointNeXt design from
# guochengqian/PointNeXt and guochengqian/openpoints (MIT).  The upstream
# implementation uses compiled FPS/ball-query operators.  Here, deterministic
# FPS and KNN are used so the policy remains installable in the existing DP3
# environment.  See THIRD_PARTY_NOTICES.md and third_party/POINTNEXT_SOURCE.md.


def _batched_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B,N,C] values with [B,...] point indices."""

    if values.ndim != 3 or indices.ndim < 2:
        raise ValueError("batched gather expects values [B,N,C] and indices [B,...]")
    if values.shape[0] != indices.shape[0]:
        raise ValueError("batch dimensions disagree")
    batch_shape = [values.shape[0]] + [1] * (indices.ndim - 1)
    batch = torch.arange(values.shape[0], device=values.device).view(*batch_shape)
    return values[batch, indices]


def farthest_point_indices(xyz: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic FPS for the small point sets used by the policy."""

    global _pointnet2_fps, _pytorch3d_fps

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [B,N,3]")
    batch, points, _ = xyz.shape
    samples = min(max(1, int(count)), int(points))
    if samples == points:
        return torch.arange(points, device=xyz.device).expand(batch, -1)
    # Normalize the starting center across compiled and pure-PyTorch backends.
    # pointnet2/PyTorch3D normally start from array index zero, which would make
    # FPS depend on a harmless raw-point permutation.  Swapping the
    # centroid-farthest point into index zero preserves the deterministic
    # contract used by the fallback and by offline nested preordering.
    centroid = xyz.mean(dim=1, keepdim=True)
    first = (xyz - centroid).square().sum(dim=-1).argmax(dim=-1)
    permutation = torch.arange(
        points,
        device=xyz.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch, -1).clone()
    batch_index = torch.arange(batch, device=xyz.device)
    permutation[:, 0] = first
    permutation[batch_index, first] = 0
    backend_xyz = _batched_gather(xyz, permutation)
    if xyz.is_cuda and _pointnet2_fps is not None:
        try:
            backend_indices = _pointnet2_fps(
                backend_xyz.contiguous(),
                samples,
            ).long()
            return _batched_gather(
                permutation.unsqueeze(-1),
                backend_indices,
            ).squeeze(-1)
        except Exception:
            # Optional CUDA operators can import successfully while being
            # incompatible with the active torch/CUDA build. Disable a failed
            # backend once instead of paying for an exception every batch.
            _pointnet2_fps = None
    if _pytorch3d_fps is not None:
        try:
            _, indices = _pytorch3d_fps(
                backend_xyz,
                K=samples,
                random_start_point=False,
            )
            return _batched_gather(
                permutation.unsqueeze(-1),
                indices.long(),
            ).squeeze(-1)
        except Exception:
            _pytorch3d_fps = None
    selected = torch.empty(batch, samples, dtype=torch.long, device=xyz.device)
    minimum_distance = torch.full(
        (batch, points),
        float("inf"),
        dtype=xyz.dtype,
        device=xyz.device,
    )
    farthest = first
    for slot in range(samples):
        selected[:, slot] = farthest
        center = xyz[batch_index, farthest].unsqueeze(1)
        squared_distance = (xyz - center).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, squared_distance)
        farthest = minimum_distance.argmax(dim=-1)
    return selected


def fps_backend(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type == "cuda" and _pointnet2_fps is not None:
        return "pointnet2_ops"
    if _pytorch3d_fps is not None:
        return "pytorch3d"
    return "pure_torch"


def _knn_indices(
    query_xyz: torch.Tensor,
    support_xyz: torch.Tensor,
    neighbors: int,
) -> torch.Tensor:
    count = min(max(1, int(neighbors)), int(support_xyz.shape[1]))
    return torch.cdist(query_xyz, support_xyz).topk(
        k=count,
        dim=-1,
        largest=False,
        sorted=False,
    ).indices


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SetAbstraction(nn.Module):
    """PointNeXt-style downsampling and local feature aggregation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        stride: int,
        neighbors: int,
        sampling_mode: str = "fps",
    ):
        super().__init__()
        self.stride = int(stride)
        self.neighbors = int(neighbors)
        self.sampling_mode = str(sampling_mode).strip().lower()
        if self.sampling_mode not in {"fps", "preordered", "preordered_train"}:
            raise ValueError(
                "PointNeXt sampling_mode must be fps, preordered, or "
                "preordered_train"
            )
        local_input = input_dim * 2 + 3
        self.local = nn.Sequential(
            nn.Linear(local_input, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_count = max(1, int(xyz.shape[1]) // self.stride)
        use_preordered = self.sampling_mode == "preordered" or (
            self.sampling_mode == "preordered_train" and self.training
        )
        if use_preordered:
            center_indices = torch.arange(
                sample_count,
                device=xyz.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(xyz.shape[0], -1)
        else:
            center_indices = farthest_point_indices(xyz, sample_count)
        center_xyz = _batched_gather(xyz, center_indices)
        center_features = _batched_gather(features, center_indices)
        neighbor_indices = _knn_indices(center_xyz, xyz, self.neighbors)
        neighbor_xyz = _batched_gather(xyz, neighbor_indices)
        neighbor_features = _batched_gather(features, neighbor_indices)
        relative_xyz = neighbor_xyz - center_xyz.unsqueeze(2)
        relative_features = neighbor_features - center_features.unsqueeze(2)
        local = self.local(
            torch.cat(
                [neighbor_features, relative_features, relative_xyz],
                dim=-1,
            )
        ).amax(dim=2)
        output = self.output_norm(local + self.skip(center_features))
        return center_xyz, F.silu(output)


class InvResMLP(nn.Module):
    """Local aggregation followed by an inverted residual point MLP."""

    def __init__(
        self,
        channels: int,
        *,
        neighbors: int,
        expansion: int = 2,
    ):
        super().__init__()
        self.neighbors = int(neighbors)
        self.local = nn.Sequential(
            nn.Linear(channels + 3, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        hidden = int(channels * expansion)
        self.pointwise = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )
        self.output_norm = nn.LayerNorm(channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        neighbor_indices = _knn_indices(xyz, xyz, self.neighbors)
        neighbor_xyz = _batched_gather(xyz, neighbor_indices)
        neighbor_features = _batched_gather(features, neighbor_indices)
        relative_xyz = neighbor_xyz - xyz.unsqueeze(2)
        relative_features = neighbor_features - features.unsqueeze(2)
        aggregate = self.local(
            torch.cat([relative_features, relative_xyz], dim=-1)
        ).amax(dim=2)
        output = self.output_norm(features + self.pointwise(aggregate))
        return F.silu(output)


class PointNeXtStage(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        stride: int,
        neighbors: int,
        blocks: int,
        sampling_mode: str = "fps",
    ):
        super().__init__()
        self.abstraction = SetAbstraction(
            input_dim,
            output_dim,
            stride=stride,
            neighbors=neighbors,
            sampling_mode=sampling_mode,
        )
        self.blocks = nn.ModuleList(
            [
                InvResMLP(output_dim, neighbors=neighbors)
                for _ in range(max(0, int(blocks) - 1))
            ]
        )

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyz, features = self.abstraction(xyz, features)
        for block in self.blocks:
            features = block(xyz, features)
        return xyz, features


class DualScalePointNeXt(nn.Module):
    """Global approach context plus high-resolution local interaction tokens."""

    def __init__(
        self,
        *,
        input_dim: int = 7,
        width: int = 32,
        token_dim: int = 128,
        stage_blocks: Iterable[int] = (1, 2, 2),
        stage_strides: Iterable[int] = (4, 4, 2),
        neighbors: Iterable[int] = (16, 16, 16),
        sampling_mode: str = "fps",
    ):
        super().__init__()
        blocks = tuple(int(value) for value in stage_blocks)
        strides = tuple(int(value) for value in stage_strides)
        neighbor_counts = tuple(int(value) for value in neighbors)
        self.sampling_mode = str(sampling_mode).strip().lower()
        if self.sampling_mode not in {"fps", "preordered", "preordered_train"}:
            raise ValueError(
                "PointNeXt sampling_mode must be fps, preordered, or "
                "preordered_train"
            )
        if not (len(blocks) == len(strides) == len(neighbor_counts) == 3):
            raise ValueError("DualScalePointNeXt expects three stages")
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        channels = (width * 2, width * 4, width * 4)
        stages = []
        input_channels = width
        for output_channels, block_count, stride, neighbor_count in zip(
            channels,
            blocks,
            strides,
            neighbor_counts,
        ):
            stages.append(
                PointNeXtStage(
                    input_channels,
                    output_channels,
                    stride=stride,
                    neighbors=neighbor_count,
                    blocks=block_count,
                    sampling_mode=self.sampling_mode,
                )
            )
            input_channels = output_channels
        self.stages = nn.ModuleList(stages)
        self.local_projection = _mlp(channels[0] + 3, token_dim, token_dim)
        self.global_projection = _mlp(channels[-1] * 2, token_dim, token_dim)
        self.output_dim = int(token_dim)

    def forward(
        self,
        xyz: torch.Tensor,
        point_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape [B,N,3]")
        if point_features.ndim != 3 or point_features.shape[:2] != xyz.shape[:2]:
            raise ValueError("point_features must have shape [B,N,C]")
        features = self.stem(point_features)
        pyramid: list[tuple[torch.Tensor, torch.Tensor]] = []
        points = xyz
        for stage in self.stages:
            points, features = stage(points, features)
            pyramid.append((points, features))
        fine_xyz, fine_features = pyramid[0]
        local_tokens = self.local_projection(
            torch.cat([fine_features, fine_xyz], dim=-1)
        )
        global_token = self.global_projection(
            torch.cat(
                [features.mean(dim=1), features.amax(dim=1)],
                dim=-1,
            )
        )
        return {
            "global_token": global_token,
            "local_tokens": local_tokens,
            "local_xyz": fine_xyz,
            "coarse_xyz": points,
        }

    def extra_repr(self) -> str:
        return f"sampling_mode={self.sampling_mode}"

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from contactflow.dp3.pointnext_relation.backbone import DualScalePointNeXt


def _check_xyz(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [B,N,3], got {tuple(value.shape)}.")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite coordinates.")


def _check_color(
    value: torch.Tensor | None,
    xyz: torch.Tensor,
    *,
    color_dim: int,
    name: str,
) -> torch.Tensor | None:
    if int(color_dim) == 0:
        if value is not None:
            raise ValueError(f"{name} must be None for an XYZ-only point cloud.")
        return None
    if value is None or tuple(value.shape) != (*xyz.shape[:2], int(color_dim)):
        raise ValueError(
            f"{name} must have shape "
            f"{(*xyz.shape[:2], int(color_dim))}, got "
            f"{None if value is None else tuple(value.shape)}."
        )
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values.")
    return value.to(device=xyz.device, dtype=xyz.dtype)


def _check_mask(mask: torch.Tensor, xyz: torch.Tensor, name: str) -> torch.Tensor:
    if tuple(mask.shape) != tuple(xyz.shape[:2]):
        raise ValueError(
            f"{name} must have shape {tuple(xyz.shape[:2])}, got {tuple(mask.shape)}."
        )
    return mask.to(device=xyz.device, dtype=torch.bool)


def gather_points(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B,N,C]`` values with an arbitrary ``[B,...]`` index prefix."""

    if values.ndim != 3 or indices.ndim < 2:
        raise ValueError("gather_points expects values [B,N,C] and indices [B,...].")
    if values.shape[0] != indices.shape[0]:
        raise ValueError("gather_points batch dimensions disagree.")
    safe = indices.clamp(min=0, max=max(0, values.shape[1] - 1))
    batch_shape = [values.shape[0]] + [1] * (indices.ndim - 1)
    batch = torch.arange(values.shape[0], device=values.device).view(*batch_shape)
    gathered = values[batch, safe]
    return gathered.masked_fill((indices < 0).unsqueeze(-1), 0.0)


def inverse_distance_interpolate(
    query_xyz: torch.Tensor,
    support_xyz: torch.Tensor,
    support_features: torch.Tensor,
    *,
    support_valid_mask: torch.Tensor | None = None,
    neighbors: int = 3,
    chunk_size: int = 256,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate cached point features without materializing a full large cdist.

    Returns the interpolated features and a boolean query-valid mask. A query is
    invalid only when its batch item has no valid support point.
    """

    _check_xyz(query_xyz, "query_xyz")
    _check_xyz(support_xyz, "support_xyz")
    if support_features.ndim != 3:
        raise ValueError("support_features must have shape [B,N,D].")
    if support_features.shape[:2] != support_xyz.shape[:2]:
        raise ValueError("support features and support coordinates are misaligned.")
    if query_xyz.shape[0] != support_xyz.shape[0]:
        raise ValueError("query and support batch dimensions disagree.")
    if query_xyz.device != support_xyz.device or query_xyz.device != support_features.device:
        raise ValueError("interpolation tensors must share a device.")
    if query_xyz.dtype != support_xyz.dtype:
        raise ValueError("query and support coordinates must share a dtype.")
    if int(neighbors) <= 0 or int(chunk_size) <= 0:
        raise ValueError("neighbors and chunk_size must be positive.")

    if support_valid_mask is None:
        support_valid_mask = torch.ones(
            support_xyz.shape[:2], dtype=torch.bool, device=support_xyz.device
        )
    else:
        support_valid_mask = _check_mask(
            support_valid_mask, support_xyz, "support_valid_mask"
        )
    any_support = support_valid_mask.any(dim=1)
    k = min(int(neighbors), int(support_xyz.shape[1]))
    chunks: list[torch.Tensor] = []
    for start in range(0, int(query_xyz.shape[1]), int(chunk_size)):
        stop = min(start + int(chunk_size), int(query_xyz.shape[1]))
        distance = torch.cdist(query_xyz[:, start:stop], support_xyz)
        distance = distance.masked_fill(~support_valid_mask[:, None], torch.inf)
        nearest_distance, nearest_index = distance.topk(
            k=k, dim=-1, largest=False, sorted=False
        )
        finite = torch.isfinite(nearest_distance)
        safe_distance = nearest_distance.masked_fill(~finite, 0.0)
        weight = finite.to(query_xyz.dtype) / safe_distance.clamp_min(float(eps))
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        nearest_features = gather_points(support_features, nearest_index)
        chunks.append((nearest_features * weight.unsqueeze(-1)).sum(dim=-2))
    interpolated = torch.cat(chunks, dim=1)
    valid = any_support[:, None].expand(-1, query_xyz.shape[1])
    return interpolated.masked_fill(~valid.unsqueeze(-1), 0.0), valid


def _masked_centroid(xyz: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weight = valid_mask.to(device=xyz.device, dtype=xyz.dtype).unsqueeze(-1)
    return (xyz * weight).sum(dim=1, keepdim=True) / weight.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)


@dataclass(frozen=True)
class PointFeatureCache:
    enc_xyz_metric: torch.Tensor
    enc_valid_mask: torch.Tensor
    enc_robot_mask: torch.Tensor
    enc_features: torch.Tensor
    global_point_token: torch.Tensor
    global_condition: torch.Tensor
    local_xyz_metric: torch.Tensor
    local_features: torch.Tensor


class CachedPointNeXtEncoder(nn.Module):
    """One-pass PointNeXt encoder with point-aligned cached semantics."""

    def __init__(
        self,
        *,
        state_dim: int,
        obs_steps: int,
        color_dim: int = 0,
        token_dim: int = 128,
        state_feature_dim: int = 64,
        width: int = 32,
        stage_blocks: tuple[int, int, int] = (1, 2, 2),
        stage_strides: tuple[int, int, int] = (4, 4, 2),
        neighbors: tuple[int, int, int] = (16, 16, 16),
        interpolation_neighbors: int = 3,
        interpolation_chunk_size: int = 256,
    ):
        super().__init__()
        if int(state_dim) <= 0 or int(obs_steps) <= 0:
            raise ValueError("state_dim and obs_steps must be positive.")
        if int(token_dim) <= 0 or int(state_feature_dim) <= 0:
            raise ValueError("feature dimensions must be positive.")
        self.state_dim = int(state_dim)
        self.obs_steps = int(obs_steps)
        self.color_dim = int(color_dim)
        if self.color_dim not in {0, 3}:
            raise ValueError("color_dim must be 0 (XYZ) or 3 (XYZRGB).")
        self.token_dim = int(token_dim)
        self.state_feature_dim = int(state_feature_dim)
        self.interpolation_neighbors = int(interpolation_neighbors)
        self.interpolation_chunk_size = int(interpolation_chunk_size)
        # Metric geometry remains XYZ-only. RGB, when present, is appended as
        # a point feature and is never used by FPS, KNN, or radius distances.
        self.backbone = DualScalePointNeXt(
            input_dim=8 + self.color_dim,
            width=int(width),
            token_dim=self.token_dim,
            stage_blocks=stage_blocks,
            stage_strides=stage_strides,
            neighbors=neighbors,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.state_feature_dim),
            nn.LayerNorm(self.state_feature_dim),
            nn.SiLU(),
            nn.Linear(self.state_feature_dim, self.state_feature_dim),
        )

    @property
    def observation_feature_dim(self) -> int:
        return self.token_dim + self.state_feature_dim

    @property
    def global_condition_dim(self) -> int:
        return self.observation_feature_dim * self.obs_steps

    def forward(
        self,
        *,
        enc_xyz_metric: torch.Tensor,
        enc_xyz_normalized: torch.Tensor,
        enc_color: torch.Tensor | None,
        enc_valid_mask: torch.Tensor,
        enc_robot_mask: torch.Tensor,
        state_history_normalized: torch.Tensor,
    ) -> PointFeatureCache:
        _check_xyz(enc_xyz_metric, "enc_xyz_metric")
        _check_xyz(enc_xyz_normalized, "enc_xyz_normalized")
        if enc_xyz_metric.shape != enc_xyz_normalized.shape:
            raise ValueError("metric and normalized P_enc tensors must align.")
        enc_color = _check_color(
            enc_color,
            enc_xyz_metric,
            color_dim=self.color_dim,
            name="enc_color",
        )
        enc_valid_mask = _check_mask(
            enc_valid_mask, enc_xyz_normalized, "enc_valid_mask"
        )
        enc_robot_mask = _check_mask(
            enc_robot_mask, enc_xyz_normalized, "enc_robot_mask"
        )
        if not bool(enc_valid_mask.any(dim=1).all()):
            raise ValueError("every PointNeXt batch item needs at least one valid point.")
        expected_state = (
            enc_xyz_metric.shape[0],
            self.obs_steps,
            self.state_dim,
        )
        if tuple(state_history_normalized.shape) != expected_state:
            raise ValueError(
                "state_history_normalized must have shape "
                f"{expected_state}, got {tuple(state_history_normalized.shape)}."
            )

        centroid = _masked_centroid(enc_xyz_normalized, enc_valid_mask)
        centered = enc_xyz_normalized - centroid
        radius = centered.square().sum(dim=-1, keepdim=True).sqrt()
        feature_parts = [
            enc_xyz_normalized,
            centered,
            radius,
            enc_robot_mask.to(enc_xyz_normalized.dtype).unsqueeze(-1),
        ]
        if enc_color is not None:
            feature_parts.append(enc_color)
        point_features = torch.cat(feature_parts, dim=-1).masked_fill(
            ~enc_valid_mask.unsqueeze(-1), 0.0
        )
        output = self.backbone(enc_xyz_normalized, point_features)
        local_xyz_normalized = output["local_xyz"]
        local_features = output["local_tokens"]
        enc_features, _ = inverse_distance_interpolate(
            enc_xyz_normalized,
            local_xyz_normalized,
            local_features,
            neighbors=self.interpolation_neighbors,
            chunk_size=self.interpolation_chunk_size,
        )
        enc_features = enc_features.masked_fill(~enc_valid_mask.unsqueeze(-1), 0.0)

        # Map the sampled local coordinates back to metric positions by using
        # their nearest P_enc source. This keeps all AQR radii in metres while
        # PointNeXt still consumes normalized coordinates.
        local_metric, _ = inverse_distance_interpolate(
            local_xyz_normalized,
            enc_xyz_normalized,
            enc_xyz_metric,
            support_valid_mask=enc_valid_mask,
            neighbors=1,
            chunk_size=self.interpolation_chunk_size,
        )
        state_features = self.state_encoder(state_history_normalized)
        point_token = output["global_token"]
        per_observation = torch.cat(
            [
                point_token[:, None].expand(-1, self.obs_steps, -1),
                state_features,
            ],
            dim=-1,
        )
        global_condition = per_observation.reshape(enc_xyz_metric.shape[0], -1)
        return PointFeatureCache(
            enc_xyz_metric=enc_xyz_metric,
            enc_valid_mask=enc_valid_mask,
            enc_robot_mask=enc_robot_mask,
            enc_features=enc_features,
            global_point_token=point_token,
            global_condition=global_condition,
            local_xyz_metric=local_metric,
            local_features=local_features,
        )


class HighResolutionQueryStem(nn.Module):
    """Lightweight geometry stem plus interpolated cached PointNeXt semantics."""

    def __init__(
        self,
        *,
        semantic_dim: int,
        color_dim: int = 0,
        output_dim: int = 128,
        geometry_dim: int = 64,
        interpolation_neighbors: int = 3,
        interpolation_chunk_size: int = 256,
    ):
        super().__init__()
        if min(int(semantic_dim), int(output_dim), int(geometry_dim)) <= 0:
            raise ValueError("query stem dimensions must be positive.")
        self.semantic_dim = int(semantic_dim)
        self.color_dim = int(color_dim)
        if self.color_dim not in {0, 3}:
            raise ValueError("color_dim must be 0 (XYZ) or 3 (XYZRGB).")
        self.output_dim = int(output_dim)
        self.interpolation_neighbors = int(interpolation_neighbors)
        self.interpolation_chunk_size = int(interpolation_chunk_size)
        self.geometry_stem = nn.Sequential(
            nn.Linear(8 + self.color_dim, int(geometry_dim)),
            nn.LayerNorm(int(geometry_dim)),
            nn.SiLU(),
            nn.Linear(int(geometry_dim), int(geometry_dim)),
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(geometry_dim) + self.semantic_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(
        self,
        *,
        query_xyz_metric: torch.Tensor,
        query_xyz_normalized: torch.Tensor,
        query_color: torch.Tensor | None,
        query_valid_mask: torch.Tensor,
        query_robot_mask: torch.Tensor,
        cache: PointFeatureCache,
    ) -> torch.Tensor:
        _check_xyz(query_xyz_metric, "query_xyz_metric")
        _check_xyz(query_xyz_normalized, "query_xyz_normalized")
        if query_xyz_metric.shape != query_xyz_normalized.shape:
            raise ValueError("metric and normalized P_query tensors must align.")
        query_color = _check_color(
            query_color,
            query_xyz_metric,
            color_dim=self.color_dim,
            name="query_color",
        )
        query_valid_mask = _check_mask(
            query_valid_mask, query_xyz_normalized, "query_valid_mask"
        )
        query_robot_mask = _check_mask(
            query_robot_mask, query_xyz_normalized, "query_robot_mask"
        )
        centroid = _masked_centroid(query_xyz_normalized, query_valid_mask)
        centered = query_xyz_normalized - centroid
        radius = centered.square().sum(dim=-1, keepdim=True).sqrt()
        geometry_parts = [
            query_xyz_normalized,
            centered,
            radius,
            query_robot_mask.to(query_xyz_normalized.dtype).unsqueeze(-1),
        ]
        if query_color is not None:
            geometry_parts.append(query_color)
        geometry = self.geometry_stem(torch.cat(geometry_parts, dim=-1))
        semantic, semantic_valid = inverse_distance_interpolate(
            query_xyz_metric,
            cache.enc_xyz_metric,
            cache.enc_features,
            support_valid_mask=cache.enc_valid_mask,
            neighbors=self.interpolation_neighbors,
            chunk_size=self.interpolation_chunk_size,
        )
        valid = query_valid_mask & semantic_valid
        fused = self.fusion(torch.cat([geometry, semantic], dim=-1))
        return fused.masked_fill(~valid.unsqueeze(-1), 0.0)

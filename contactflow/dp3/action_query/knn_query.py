from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


RAW_QUERY_POINT_COUNT = 2048


class QueryContractError(ValueError):
    """Raised when the raw-point query contract is violated."""


@dataclass(frozen=True)
class RawPointKNNConfig:
    """Non-trainable raw-XYZ query contract used by AQR.

    ``source_points`` is deliberately fixed at 2048.  A coarse 64-token tensor
    is a different representation and must never be accepted by this operator.
    """

    source_points: int = RAW_QUERY_POINT_COUNT
    k: int = 16
    max_distance_m: float = 0.10

    def __post_init__(self) -> None:
        if int(self.source_points) != RAW_QUERY_POINT_COUNT:
            raise QueryContractError(
                "AQR raw-point KNN requires exactly 2048 source XYZ points; "
                f"got {int(self.source_points)}. A 64-token coarse query is forbidden."
            )
        if int(self.k) <= 0 or int(self.k) > RAW_QUERY_POINT_COUNT:
            raise QueryContractError(
                f"k must be in [1, {RAW_QUERY_POINT_COUNT}], got {int(self.k)}."
            )
        if not np.isfinite(self.max_distance_m) or float(self.max_distance_m) <= 0.0:
            raise QueryContractError("max_distance_m must be positive and finite.")


@dataclass(frozen=True)
class RawPointSample:
    """Exactly 2048 raw XYZ points plus padding validity and source indices."""

    xyz: np.ndarray
    valid_mask: np.ndarray
    source_indices: np.ndarray

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        indices = np.asarray(self.source_indices, dtype=np.int64)
        if xyz.shape != (RAW_QUERY_POINT_COUNT, 3):
            raise QueryContractError(
                "RawPointSample.xyz must have shape (2048, 3), "
                f"got {xyz.shape}."
            )
        if valid.shape != (RAW_QUERY_POINT_COUNT,):
            raise QueryContractError(
                "RawPointSample.valid_mask must have shape (2048,), "
                f"got {valid.shape}."
            )
        if indices.shape != (RAW_QUERY_POINT_COUNT,):
            raise QueryContractError(
                "RawPointSample.source_indices must have shape (2048,), "
                f"got {indices.shape}."
            )
        if not np.isfinite(xyz).all():
            raise QueryContractError("RawPointSample.xyz contains non-finite values.")
        if np.any(indices[valid] < 0) or np.any(indices[~valid] != -1):
            raise QueryContractError(
                "Valid samples need non-negative source indices and padding needs -1."
            )
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "source_indices", indices)


@dataclass(frozen=True)
class KNNQueryResult:
    """KNN tensors with the query prefix preserved.

    For queries shaped ``[B, S, A, 3]``, all outputs are shaped
    ``[B, S, A, K, ...]``.  Invalid neighbors are zero-filled and must be
    ignored using ``valid_mask``.
    """

    indices: torch.Tensor
    neighbor_xyz: torch.Tensor
    relative_xyz: torch.Tensor
    distance: torch.Tensor
    valid_mask: torch.Tensor

    @property
    def local_features(self) -> torch.Tensor:
        """Return ``[relative_xyz, distance]`` with four channels."""

        return torch.cat([self.relative_xyz, self.distance.unsqueeze(-1)], dim=-1)


def sample_raw_xyz(
    xyzw_or_xyz: np.ndarray,
    rng: np.random.Generator,
) -> RawPointSample:
    """Uniformly sample exactly 2048 sensor XYZ points without semantics.

    A fourth input channel, when present, is used only as the sensor's validity
    bit (ManiSkill stores padded zeros with ``w=0``).  It is never returned as a
    feature and is not interpreted as a semantic label.  If fewer than 2048
    valid points exist, the remainder is zero padded with ``valid_mask=False``.
    """

    array = np.asarray(xyzw_or_xyz)
    if array.ndim < 2 or array.shape[-1] not in (3, 4):
        raise QueryContractError(
            "Raw sensor points must end in XYZ or XYZW, "
            f"got shape {array.shape}."
        )
    flat = array.reshape(-1, array.shape[-1])
    xyz = np.asarray(flat[:, :3], dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) < 5.0).all(axis=1)
    if flat.shape[1] == 4:
        validity_channel = np.asarray(flat[:, 3])
        valid &= np.isfinite(validity_channel) & (validity_channel > 0)
    source_indices = np.flatnonzero(valid)
    valid_xyz = xyz[source_indices]

    if len(valid_xyz) >= RAW_QUERY_POINT_COUNT:
        selected = rng.choice(
            len(valid_xyz), size=RAW_QUERY_POINT_COUNT, replace=False
        )
        return RawPointSample(
            xyz=valid_xyz[selected],
            valid_mask=np.ones(RAW_QUERY_POINT_COUNT, dtype=bool),
            source_indices=source_indices[selected],
        )

    sampled_xyz = np.zeros((RAW_QUERY_POINT_COUNT, 3), dtype=np.float32)
    sampled_valid = np.zeros(RAW_QUERY_POINT_COUNT, dtype=bool)
    sampled_indices = np.full(RAW_QUERY_POINT_COUNT, -1, dtype=np.int64)
    if len(valid_xyz):
        permutation = rng.permutation(len(valid_xyz))
        count = len(valid_xyz)
        sampled_xyz[:count] = valid_xyz[permutation]
        sampled_valid[:count] = True
        sampled_indices[:count] = source_indices[permutation]
    return RawPointSample(
        xyz=sampled_xyz,
        valid_mask=sampled_valid,
        source_indices=sampled_indices,
    )


def query_raw_point_knn(
    source_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
    source_valid_mask: torch.Tensor | None = None,
    config: RawPointKNNConfig | None = None,
) -> KNNQueryResult:
    """Query per-slot neighborhoods from exactly 2048 raw XYZ points.

    ``source_xyz`` must be ``[B, 2048, 3]``. ``query_xyz`` may have any
    non-empty query prefix after the batch dimension, e.g. ``[B, S, 3]`` or
    ``[B, S, A, 3]``.  Relative coordinates use ``neighbor - query``.
    """

    config = RawPointKNNConfig() if config is None else config
    if not isinstance(source_xyz, torch.Tensor) or not isinstance(
        query_xyz, torch.Tensor
    ):
        raise QueryContractError("source_xyz and query_xyz must be torch tensors.")
    if source_xyz.ndim != 3 or source_xyz.shape[-1] != 3:
        raise QueryContractError(
            "source_xyz must have shape [B, 2048, 3], "
            f"got {tuple(source_xyz.shape)}."
        )
    if source_xyz.shape[1] != config.source_points:
        raise QueryContractError(
            "AQR raw-point KNN requires exactly 2048 source XYZ points at runtime; "
            f"got {int(source_xyz.shape[1])}. A 64-token coarse query is forbidden."
        )
    if query_xyz.ndim < 3 or query_xyz.shape[-1] != 3:
        raise QueryContractError(
            "query_xyz must have shape [B, ..., 3] with a non-empty query prefix, "
            f"got {tuple(query_xyz.shape)}."
        )
    if source_xyz.shape[0] != query_xyz.shape[0]:
        raise QueryContractError("source and query batch sizes disagree.")
    if not source_xyz.is_floating_point() or not query_xyz.is_floating_point():
        raise QueryContractError("source_xyz and query_xyz must be floating point.")
    if source_xyz.device != query_xyz.device:
        raise QueryContractError("source_xyz and query_xyz must share a device.")
    if source_xyz.dtype != query_xyz.dtype:
        raise QueryContractError("source_xyz and query_xyz must share a dtype.")
    if not bool(torch.isfinite(source_xyz).all()) or not bool(
        torch.isfinite(query_xyz).all()
    ):
        raise QueryContractError("source_xyz or query_xyz contains non-finite values.")

    batch = int(source_xyz.shape[0])
    query_prefix = tuple(int(x) for x in query_xyz.shape[1:-1])
    flat_queries = query_xyz.reshape(batch, -1, 3)
    if flat_queries.shape[1] == 0:
        raise QueryContractError("query_xyz contains no query positions.")

    if source_valid_mask is None:
        source_valid_mask = torch.ones(
            source_xyz.shape[:2], dtype=torch.bool, device=source_xyz.device
        )
    else:
        if not isinstance(source_valid_mask, torch.Tensor):
            raise QueryContractError("source_valid_mask must be a torch tensor.")
        if tuple(source_valid_mask.shape) != tuple(source_xyz.shape[:2]):
            raise QueryContractError(
                "source_valid_mask must have shape [B, 2048], "
                f"got {tuple(source_valid_mask.shape)}."
            )
        source_valid_mask = source_valid_mask.to(
            device=source_xyz.device, dtype=torch.bool
        )

    pairwise = torch.cdist(flat_queries, source_xyz)
    pairwise = pairwise.masked_fill(~source_valid_mask[:, None, :], torch.inf)
    distance, local_indices = torch.topk(
        pairwise, k=config.k, dim=-1, largest=False, sorted=True
    )
    gather_indices = local_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    expanded_source = source_xyz[:, None, :, :].expand(
        -1, flat_queries.shape[1], -1, -1
    )
    neighbor_xyz = torch.gather(expanded_source, 2, gather_indices)
    valid_mask = torch.isfinite(distance) & (
        distance <= float(config.max_distance_m)
    )
    output_indices = local_indices.masked_fill(~valid_mask, -1)
    neighbor_xyz = neighbor_xyz.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    relative_xyz = (neighbor_xyz - flat_queries.unsqueeze(-2)).masked_fill(
        ~valid_mask.unsqueeze(-1), 0.0
    )
    distance = distance.masked_fill(~valid_mask, 0.0)

    scalar_shape = (batch, *query_prefix, config.k)
    vector_shape = (*scalar_shape, 3)
    return KNNQueryResult(
        indices=output_indices.reshape(scalar_shape),
        neighbor_xyz=neighbor_xyz.reshape(vector_shape),
        relative_xyz=relative_xyz.reshape(vector_shape),
        distance=distance.reshape(scalar_shape),
        valid_mask=valid_mask.reshape(scalar_shape),
    )

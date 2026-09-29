"""Online sensor point sampling; no dataset conversion commands."""
from __future__ import annotations
import numpy as np

def _parse_crop_bound(value: str, name: str) -> tuple[float, float, float] | None:
    if not value.strip():
        return None
    parts = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated values")
    if not np.isfinite(np.asarray(parts, dtype=np.float64)).all():
        raise ValueError(f"{name} values must be finite")
    return parts

def _farthest_point_indices(xyz: np.ndarray, num_points: int) -> np.ndarray:
    """Deterministic dependency-free FPS over XYZ coordinates.

    The first point is the point farthest from the cloud centroid. This avoids
    introducing another random variable into train/eval preprocessing while
    retaining the spatial-coverage property used by DP3.
    """
    xyz64 = np.asarray(xyz, dtype=np.float64)
    if xyz64.ndim != 2 or xyz64.shape[1] != 3:
        raise ValueError(f"FPS expects [N,3] XYZ input, got {xyz64.shape}")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if len(xyz64) < num_points:
        raise ValueError("FPS requires at least num_points candidates")

    centroid = xyz64.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.sum((xyz64 - centroid) ** 2, axis=1)))
    selected = np.empty(num_points, dtype=np.int64)
    selected[0] = first
    min_sq_dist = np.sum((xyz64 - xyz64[first]) ** 2, axis=1)
    for index in range(1, num_points):
        farthest = int(np.argmax(min_sq_dist))
        selected[index] = farthest
        sq_dist = np.sum((xyz64 - xyz64[farthest]) ** 2, axis=1)
        np.minimum(min_sq_dist, sq_dist, out=min_sq_dist)
    return selected

def _sample_aqr_points(
    points: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    *,
    segmentation: np.ndarray,
    robot_seg_ids: tuple[int, ...],
    point_sampling_mode: str,
    crop_min: tuple[float, float, float] | None,
    crop_max: tuple[float, float, float] | None,
    rgb: np.ndarray | None = None,
    append_rgb: bool = False,
    return_sampled_segmentation: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Sample one dense frame with aligned validity and offline robot labels."""

    if point_sampling_mode not in {"pooled", "fps"}:
        raise ValueError("AQR dense preparation supports pooled or fps sampling.")
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] not in {3, 4}:
        raise ValueError("AQR source points must have shape [N,3] or [N,4].")
    xyz = np.asarray(array[:, :3], dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) < 5.0).all(axis=1)
    if array.shape[1] == 4:
        valid &= np.isfinite(array[:, 3]) & (array[:, 3] > 0)
    if crop_min is not None:
        valid &= np.all(xyz > np.asarray(crop_min, dtype=xyz.dtype), axis=1)
    if crop_max is not None:
        valid &= np.all(xyz < np.asarray(crop_max, dtype=xyz.dtype), axis=1)
    seg = np.asarray(segmentation).reshape(-1)
    if seg.shape != valid.shape:
        raise ValueError("AQR segmentation must align one-to-one with XYZ.")
    source_indices = np.flatnonzero(valid)
    valid_xyz = xyz[source_indices]
    valid_seg = seg[source_indices]

    valid_rgb = None
    if append_rgb:
        if rgb is None:
            raise ValueError("--append-rgb requires point-aligned source RGB.")
        source_rgb = np.asarray(rgb).reshape(-1, np.asarray(rgb).shape[-1])
        if source_rgb.shape[0] != len(valid) or source_rgb.shape[1] < 3:
            raise ValueError("AQR source RGB must align with XYZ.")
        source_rgb = source_rgb[:, :3].astype(np.float32)
        if source_rgb.size and float(np.nanmax(source_rgb)) > 1.0 + 1e-6:
            source_rgb = source_rgb / 255.0
        valid_rgb = np.clip(np.nan_to_num(source_rgb[source_indices]), 0.0, 1.0)

    take = min(int(num_points), len(valid_xyz))
    if take:
        if point_sampling_mode == "fps" and len(valid_xyz) > take:
            selected = _farthest_point_indices(valid_xyz, take)
        elif len(valid_xyz) > take:
            selected = rng.choice(len(valid_xyz), size=take, replace=False)
        else:
            selected = np.arange(take, dtype=np.int64)
    else:
        selected = np.zeros((0,), dtype=np.int64)

    channels = 3 + (3 if append_rgb else 0)
    sampled = np.zeros((int(num_points), channels), dtype=np.float32)
    sampled_valid = np.zeros((int(num_points),), dtype=bool)
    sampled_robot = np.zeros((int(num_points),), dtype=bool)
    sampled_segmentation = np.full((int(num_points),), -1, dtype=np.int32)
    if take:
        sampled[:take, :3] = valid_xyz[selected]
        if append_rgb:
            assert valid_rgb is not None
            sampled[:take, 3:6] = valid_rgb[selected]
        sampled_valid[:take] = True
        sampled_robot[:take] = np.isin(valid_seg[selected], robot_seg_ids)
        sampled_segmentation[:take] = valid_seg[selected].astype(
            np.int32, copy=False
        )
    if return_sampled_segmentation:
        return (
            sampled,
            sampled_robot,
            sampled_valid,
            sampled_segmentation,
        )
    return sampled, sampled_robot, sampled_valid

"""PointNeXt relation-conditioned DP3 components (lazy imports)."""

__all__ = ["DualScalePointNeXt", "farthest_point_indices"]


def __getattr__(name):
    if name in {"DualScalePointNeXt", "farthest_point_indices"}:
        from contactflow.dp3.pointnext_relation.backbone import (
            DualScalePointNeXt,
            farthest_point_indices,
        )
        return {"DualScalePointNeXt": DualScalePointNeXt,
                "farthest_point_indices": farthest_point_indices}[name]
    raise AttributeError(name)

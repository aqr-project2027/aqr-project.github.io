"""AQR-DP3 with the A6_POST one-shot post-diffusion geometry refiner.

The accompanying config selects convex relation fusion and anchor-preserving
queries. Experimental attention and gain branches are inactive in this variant.
"""

__all__ = [
    "AQRDP3",
    "AQRDP3Config",
    "AQRDP3Dataset",
    "EXPERIMENT_PRESETS",
    "PointCloudFrontEndConfig",
]


def __getattr__(name):
    if name == "AQRDP3":
        from contactflow.dp3.aqr_dp3.policy import AQRDP3
        return AQRDP3
    if name == "AQRDP3Dataset":
        from contactflow.dp3.aqr_dp3.dataset import AQRDP3Dataset
        return AQRDP3Dataset
    if name in {
        "AQRDP3Config",
        "EXPERIMENT_PRESETS",
        "PointCloudFrontEndConfig",
    }:
        from contactflow.dp3.aqr_dp3.config import (
            AQRDP3Config,
            EXPERIMENT_PRESETS,
            PointCloudFrontEndConfig,
        )
        return {
            "AQRDP3Config": AQRDP3Config,
            "EXPERIMENT_PRESETS": EXPERIMENT_PRESETS,
            "PointCloudFrontEndConfig": PointCloudFrontEndConfig,
        }[name]
    raise AttributeError(name)

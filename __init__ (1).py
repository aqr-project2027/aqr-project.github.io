"""ContactFlow's active DP3 extension surface.

Only the deployment-safe AQR-DP3 path is public here.  Historical experiment
modules remain importable by their explicit module paths so old checkpoints are
not destroyed, but adaptive commit, hazard, consequence, recovery, and coarse
policy prototypes cannot leak into new wildcard/public imports.
"""

__all__ = ["AQRDP3", "AQRDP3Config", "AQRDP3Dataset"]


def __getattr__(name):
    if name in __all__:
        from contactflow.dp3.aqr_dp3 import (
            AQRDP3,
            AQRDP3Config,
            AQRDP3Dataset,
        )

        return {
            "AQRDP3": AQRDP3,
            "AQRDP3Config": AQRDP3Config,
            "AQRDP3Dataset": AQRDP3Dataset,
        }[name]
    raise AttributeError(name)

from __future__ import annotations

import math

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.registration import register_env


def validate_clearance_m(value: float) -> float:
    clearance = float(value)
    if not math.isfinite(clearance) or not 0.00025 <= clearance <= 0.015:
        raise ValueError("clearance_m must be finite and lie in [0.25, 15] mm.")
    return clearance


@register_env("PegInsertionSidePrecision-v1", max_episode_steps=100)
class PegInsertionSidePrecisionEnv(PegInsertionSideEnv):
    """PegInsertionSide with an explicit per-side radial clearance.

    The upstream task fixes clearance as a class attribute. This local variant
    sets an instance value before ``BaseEnv`` builds collision geometry, so a
    difficulty sweep never edits the installed ManiSkill package.
    """

    def __init__(self, *args, clearance_m: float = 0.003, **kwargs):
        self._clearance = validate_clearance_m(clearance_m)
        super().__init__(*args, **kwargs)

    @property
    def precision_clearance_m(self) -> float:
        return float(self._clearance)

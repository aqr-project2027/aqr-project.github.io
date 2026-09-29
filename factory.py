from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AQRBCConfig
from .model import AQRBCPolicy
from .torch_trajectory import (
    PyTorchKinematicsFK,
    TorchCandidateTrajectoryLift,
    lift_kwargs_from_action_contract,
)


def load_action_contract(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_candidate_trajectory_lift(
    action_contract: dict[str, Any],
    *,
    urdf_path: str = "",
) -> TorchCandidateTrajectoryLift:
    """Build the verified P0 linear-response + exact-FK trajectory lift.

    Keeping this construction in one place prevents AQR-BC and downstream DP3
    policies from silently using different URDF or controller contracts.
    """

    import pytorch_kinematics as pk

    from contactflow.dp3.aqr_bc.urdf import (
        _resolve_urdf,
        _sanitized_urdf_text,
    )

    resolved_urdf = _resolve_urdf(urdf_path)
    end_link = str(action_contract["kinematics"]["end_link"])
    chain = pk.build_serial_chain_from_urdf(
        _sanitized_urdf_text(resolved_urdf), end_link
    )
    runtime_names = list(chain.get_joint_parameter_names())
    expected_names = list(action_contract["kinematics"]["joint_names"])
    if runtime_names != expected_names:
        raise RuntimeError(
            "Runtime URDF disagrees with the P0 action contract: "
            f"{runtime_names} != {expected_names}."
        )
    fk = PyTorchKinematicsFK(chain)
    return TorchCandidateTrajectoryLift(
        **lift_kwargs_from_action_contract(action_contract, fk)
    )


def build_aqr_bc_policy(
    cfg: AQRBCConfig,
    action_contract: dict[str, Any],
    *,
    urdf_path: str = "",
) -> AQRBCPolicy:
    cfg = cfg.validate()
    candidate_lift = None
    if str(cfg.variant).lower() == "a2":
        candidate_lift = build_candidate_trajectory_lift(
            action_contract,
            urdf_path=urdf_path,
        )
    return AQRBCPolicy(cfg, candidate_lift=candidate_lift)

"""AQR-BC non-diffusion baselines (lazy imports to avoid pulling in heavy deps)."""

__all__ = [
    "PyTorchKinematicsFK",
    "TorchCandidateTrajectory",
    "TorchCandidateTrajectoryLift",
    "lift_kwargs_from_action_contract",
    "pose_wxyz_to_matrix_torch",
    "build_aqr_bc_policy",
    "load_action_contract",
]


def __getattr__(name):
    if name in {"PyTorchKinematicsFK", "TorchCandidateTrajectory",
                "TorchCandidateTrajectoryLift", "lift_kwargs_from_action_contract",
                "pose_wxyz_to_matrix_torch"}:
        from contactflow.dp3.aqr_bc.torch_trajectory import (
            PyTorchKinematicsFK,
            TorchCandidateTrajectory,
            TorchCandidateTrajectoryLift,
            lift_kwargs_from_action_contract,
            pose_wxyz_to_matrix_torch,
        )
        return {
            "PyTorchKinematicsFK": PyTorchKinematicsFK,
            "TorchCandidateTrajectory": TorchCandidateTrajectory,
            "TorchCandidateTrajectoryLift": TorchCandidateTrajectoryLift,
            "lift_kwargs_from_action_contract": lift_kwargs_from_action_contract,
            "pose_wxyz_to_matrix_torch": pose_wxyz_to_matrix_torch,
        }[name]
    if name in {"build_aqr_bc_policy", "load_action_contract"}:
        from contactflow.dp3.aqr_bc.factory import (
            build_aqr_bc_policy,
            load_action_contract,
        )
        return {
            "build_aqr_bc_policy": build_aqr_bc_policy,
            "load_action_contract": load_action_contract,
        }[name]
    raise AttributeError(name)

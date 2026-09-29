from __future__ import annotations

from dataclasses import dataclass

from contactflow.dp3.action_query.knn_query import RAW_QUERY_POINT_COUNT


@dataclass(frozen=True)
class AQRBCConfig:
    """Locked A0/A1/A2 comparison contract before DP3 integration."""

    variant: str
    state_dim: int = 16
    action_dim: int = 8
    arm_dim: int = 7
    chunk_steps: int = 8
    query_steps: int = 4
    source_points: int = RAW_QUERY_POINT_COUNT
    point_dim: int = 3
    point_hidden_dim: int = 64
    token_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 2
    knn_k: int = 16
    knn_max_distance_m: float = 0.10
    tcp_pose_start: int = 9
    coarse_loss_weight: float = 0.25
    future_refined_loss_weight: float = 0.25

    def validate(self) -> "AQRBCConfig":
        variant = str(self.variant).lower()
        if variant not in {"a0", "a1", "a2"}:
            raise ValueError("variant must be one of a0, a1, a2.")
        if int(self.source_points) != RAW_QUERY_POINT_COUNT:
            raise ValueError("AQR-BC requires exactly 2048 raw XYZ source points.")
        if self.point_dim != 3:
            raise ValueError("AQR-BC policy input must use XYZ only.")
        if self.state_dim != 16:
            raise ValueError("AQR-BC deployable state must be 9-D qpos + 7-D TCP.")
        if self.action_dim <= self.arm_dim:
            raise ValueError("action_dim must contain arm and separate gripper dimensions.")
        if self.chunk_steps != 8:
            raise ValueError("A0/A1/A2 must predict one shared 8-action chunk.")
        if self.query_steps != 4:
            raise ValueError(
                "A2 queries exactly the first four actions needed by Exec4."
            )
        if self.query_steps > self.chunk_steps:
            raise ValueError("query_steps cannot exceed chunk_steps.")
        if self.token_dim <= 0 or self.point_hidden_dim <= 0:
            raise ValueError("feature dimensions must be positive.")
        if self.token_dim % self.attention_heads:
            raise ValueError("token_dim must be divisible by attention_heads.")
        if self.attention_layers <= 0:
            raise ValueError("attention_layers must be positive.")
        if self.knn_k <= 0 or self.knn_k > self.source_points:
            raise ValueError("knn_k must be in [1, source_points].")
        if self.knn_max_distance_m <= 0.0:
            raise ValueError("knn_max_distance_m must be positive.")
        if not 0 <= self.tcp_pose_start <= self.state_dim - 7:
            raise ValueError("tcp_pose_start does not fit state_dim.")
        if self.coarse_loss_weight < 0.0:
            raise ValueError("coarse_loss_weight must be non-negative.")
        if self.future_refined_loss_weight < 0.0:
            raise ValueError("future_refined_loss_weight must be non-negative.")
        return self

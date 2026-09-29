from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PointNeXtRelationConfig:
    """Runtime-observable configuration for the relation encoder."""

    observable_dim: int = 16
    teacher_state_dim: int = 44
    point_dim: int = 3
    output_dim: int = 128
    token_dim: int = 128
    width: int = 32
    stage_blocks: tuple[int, int, int] = (1, 2, 2)
    stage_strides: tuple[int, int, int] = (4, 4, 2)
    neighbors: tuple[int, int, int] = (16, 16, 16)
    attention_heads: int = 4
    attention_layers: int = 2
    tcp_position_start: int = 9
    use_uncertainty: bool = True
    use_local_refinement: bool = False
    local_point_count: int = 128
    local_neighbors: int = 16
    local_blocks: int = 2
    local_temperature: float = 0.35

    def validate(self) -> "PointNeXtRelationConfig":
        if self.observable_dim != 16:
            raise ValueError("the deployable policy must read exactly 16 observable dims")
        if self.teacher_state_dim != 44:
            raise ValueError("the frozen oracle teacher contract is 44-D")
        if self.point_dim != 3:
            raise ValueError("PointNeXt relation input must be XYZ only")
        if self.output_dim <= 0 or self.token_dim <= 0 or self.width <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.token_dim % self.attention_heads:
            raise ValueError("token_dim must be divisible by attention_heads")
        if self.attention_layers <= 0:
            raise ValueError("attention_layers must be positive")
        if not (
            len(self.stage_blocks)
            == len(self.stage_strides)
            == len(self.neighbors)
            == 3
        ):
            raise ValueError("the lightweight PointNeXt backbone has exactly three stages")
        if any(value <= 0 for value in self.stage_blocks):
            raise ValueError("stage block counts must be positive")
        if any(value <= 0 for value in self.stage_strides):
            raise ValueError("stage strides must be positive")
        if any(value <= 0 for value in self.neighbors):
            raise ValueError("neighbor counts must be positive")
        if not 0 <= self.tcp_position_start <= self.observable_dim - 3:
            raise ValueError("tcp_position_start does not fit the observable state")
        if self.local_point_count <= 0:
            raise ValueError("local_point_count must be positive")
        if self.local_neighbors <= 0:
            raise ValueError("local_neighbors must be positive")
        if self.local_blocks <= 0:
            raise ValueError("local_blocks must be positive")
        if self.local_temperature <= 0.0:
            raise ValueError("local_temperature must be positive")
        return self

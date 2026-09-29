from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from contactflow.dp3.action_query.knn_query import (
    RawPointKNNConfig,
    query_raw_point_knn,
)

from .config import AQRBCConfig
from .torch_trajectory import (
    TorchCandidateTrajectoryLift,
    pose_wxyz_to_matrix_torch,
)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _masked_pool(
    features: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = valid_mask.unsqueeze(-1)
    count = valid.sum(dim=1).clamp_min(1)
    mean = (features * valid).sum(dim=1) / count
    minimum = torch.finfo(features.dtype).min
    maximum = features.masked_fill(~valid, minimum).amax(dim=1)
    any_valid = valid_mask.any(dim=1, keepdim=True)
    maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
    return mean, maximum


def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe = indices.clamp_min(0)
    batch = torch.arange(tokens.shape[0], device=tokens.device).view(
        tokens.shape[0], *([1] * (indices.ndim - 1))
    )
    gathered = tokens[batch, safe]
    return gathered.masked_fill((indices < 0).unsqueeze(-1), 0.0)


class RelationBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_valid,
            need_weights=True,
            average_attn_weights=False,
        )
        query = query + attended
        query = query + self.feed_forward(query)
        return query, weights


class RelationRefiner(nn.Module):
    def __init__(self, cfg: AQRBCConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelationBlock(cfg.token_dim, cfg.attention_heads)
                for _ in range(cfg.attention_layers)
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or memory.ndim != 3 or memory_valid.ndim != 2:
            raise ValueError("relation tensors must be [B,Q,D], [B,N,D], [B,N].")
        row_valid = memory_valid.any(dim=1)
        if not bool(row_valid.all()):
            memory_valid = memory_valid.clone()
            memory = memory.clone()
            memory_valid[~row_valid, 0] = True
            memory[~row_valid, 0] = 0.0
        weights = torch.empty(0, device=query.device)
        for layer in self.layers:
            query, weights = layer(query, memory, memory_valid)
        query = query * row_valid[:, None, None]
        return query, weights


class AQRBCPolicy(nn.Module):
    """Shared A0/A1/A2 action-chunk BC implementation.

    A0 uses only global point pooling. A1 adds static current-TCP relation
    attention. A2 replaces the static memory with corrected candidate-trajectory
    KNN neighborhoods from the raw 2048 XYZ points.
    """

    def __init__(
        self,
        cfg: AQRBCConfig,
        candidate_lift: TorchCandidateTrajectoryLift | None = None,
    ):
        super().__init__()
        self.cfg = cfg.validate()
        self.variant = str(cfg.variant).lower()
        if self.variant == "a2" and candidate_lift is None:
            raise ValueError("A2 requires a corrected candidate trajectory lift.")
        self.candidate_lift = candidate_lift
        self.point_encoder = nn.Sequential(
            nn.Linear(6, cfg.point_hidden_dim),
            nn.LayerNorm(cfg.point_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.point_hidden_dim, cfg.token_dim),
            nn.LayerNorm(cfg.token_dim),
            nn.SiLU(),
        )
        self.global_projection = _mlp(
            cfg.token_dim * 2, cfg.token_dim, cfg.token_dim
        )
        self.state_encoder = _mlp(
            cfg.state_dim, cfg.token_dim, cfg.token_dim
        )
        self.coarse_head = nn.Sequential(
            _mlp(cfg.token_dim * 2, cfg.token_dim * 2, cfg.token_dim),
            nn.SiLU(),
            nn.Linear(cfg.token_dim, cfg.chunk_steps * cfg.action_dim),
            nn.Tanh(),
        )
        self.register_buffer("state_mean", torch.zeros(cfg.state_dim))
        self.register_buffer("state_std", torch.ones(cfg.state_dim))

        # All three variants instantiate exactly the same trainable modules.
        # Only the relation memory/query source changes, so a gain cannot be
        # dismissed as extra parameter capacity.
        self.slot_embedding = nn.Parameter(
            torch.zeros(cfg.query_steps, cfg.token_dim)
        )
        nn.init.normal_(self.slot_embedding, std=0.02)
        self.query_projection = _mlp(
            cfg.token_dim + cfg.action_dim + 3,
            cfg.token_dim,
            cfg.token_dim,
        )
        self.relative_projection = _mlp(4, cfg.token_dim, cfg.token_dim)
        self.relation = RelationRefiner(cfg)
        self.residual_head = nn.Sequential(
            nn.Linear(
                cfg.token_dim * 2 + cfg.action_dim,
                cfg.token_dim,
            ),
            nn.LayerNorm(cfg.token_dim),
            nn.SiLU(),
            nn.Linear(cfg.token_dim, cfg.action_dim),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.knn_config = RawPointKNNConfig(
            source_points=cfg.source_points,
            k=cfg.knn_k,
            max_distance_m=cfg.knn_max_distance_m,
        )

    def set_state_normalization(
        self, mean: torch.Tensor | Any, std: torch.Tensor | Any
    ) -> None:
        mean_tensor = torch.as_tensor(
            mean, dtype=self.state_mean.dtype, device=self.state_mean.device
        ).reshape(-1)
        std_tensor = torch.as_tensor(
            std, dtype=self.state_std.dtype, device=self.state_std.device
        ).reshape(-1)
        if mean_tensor.shape != self.state_mean.shape:
            raise ValueError("state mean shape disagrees with config.")
        if std_tensor.shape != self.state_std.shape or bool((std_tensor <= 0).any()):
            raise ValueError("state std must be positive and match config.")
        self.state_mean.copy_(mean_tensor)
        self.state_std.copy_(std_tensor)

    def _point_tokens(
        self,
        point_xyz: torch.Tensor,
        point_valid_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tcp_pose = state[:, self.cfg.tcp_pose_start : self.cfg.tcp_pose_start + 7]
        world_from_tcp = pose_wxyz_to_matrix_torch(tcp_pose)
        rotation = world_from_tcp[:, :3, :3]
        translation = world_from_tcp[:, :3, 3]
        xyz_tcp = (point_xyz - translation[:, None]) @ rotation
        point_input = torch.cat([point_xyz, xyz_tcp], dim=-1)
        tokens = self.point_encoder(point_input)
        tokens = tokens.masked_fill(~point_valid_mask.unsqueeze(-1), 0.0)
        return tokens, xyz_tcp, tcp_pose

    def forward(
        self,
        point_xyz: torch.Tensor,
        point_valid_mask: torch.Tensor,
        state: torch.Tensor,
        previous_arm_qpos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        if tuple(point_xyz.shape[1:]) != (cfg.source_points, 3):
            raise ValueError("point_xyz must have shape [B,2048,3].")
        if tuple(point_valid_mask.shape) != tuple(point_xyz.shape[:2]):
            raise ValueError("point_valid_mask must have shape [B,2048].")
        if tuple(state.shape) != (point_xyz.shape[0], cfg.state_dim):
            raise ValueError("state shape disagrees with config.")
        if tuple(previous_arm_qpos.shape) != (point_xyz.shape[0], cfg.arm_dim):
            raise ValueError("previous_arm_qpos shape disagrees with config.")

        point_tokens, xyz_tcp, tcp_pose = self._point_tokens(
            point_xyz, point_valid_mask, state
        )
        pooled_mean, pooled_max = _masked_pool(point_tokens, point_valid_mask)
        global_token = self.global_projection(
            torch.cat([pooled_mean, pooled_max], dim=-1)
        )
        normalized_state = (state - self.state_mean) / self.state_std
        state_token = self.state_encoder(normalized_state)
        coarse = self.coarse_head(
            torch.cat([global_token, state_token], dim=-1)
        ).reshape(-1, cfg.chunk_steps, cfg.action_dim)
        output: dict[str, torch.Tensor] = {
            "action": coarse,
            "coarse_action": coarse,
            "global_token": global_token,
        }
        slot = self.slot_embedding[None].expand(point_xyz.shape[0], -1, -1)
        zero_action = torch.zeros_like(coarse[:, : cfg.query_steps])
        zero_position = torch.zeros(
            (point_xyz.shape[0], cfg.query_steps, 3),
            dtype=point_xyz.dtype,
            device=point_xyz.device,
        )
        static_query = self.query_projection(
            torch.cat(
                [
                    state_token[:, None].expand(-1, cfg.query_steps, -1),
                    zero_action,
                    zero_position,
                ],
                dim=-1,
            )
        ) + slot
        if self.variant == "a0":
            memory = global_token[:, None]
            memory_valid = torch.ones(
                (point_xyz.shape[0], 1),
                dtype=torch.bool,
                device=point_xyz.device,
            )
            context, attention = self.relation(
                static_query, memory, memory_valid
            )
            relation_valid = memory_valid[:, None].expand(
                -1, cfg.query_steps, -1
            )
        elif self.variant == "a1":
            distance = xyz_tcp.norm(dim=-1, keepdim=True)
            memory = point_tokens + self.relative_projection(
                torch.cat([xyz_tcp, distance], dim=-1)
            )
            context, attention = self.relation(
                static_query, memory, point_valid_mask
            )
            relation_valid = point_valid_mask[:, None].expand(
                -1, cfg.query_steps, -1
            )
        else:
            assert self.candidate_lift is not None
            trajectory = self.candidate_lift(
                policy_actions=coarse[:, : cfg.query_steps],
                current_arm_qpos=state[:, : cfg.arm_dim],
                previous_arm_qpos=previous_arm_qpos,
                current_tcp_pose_wxyz=tcp_pose,
            )
            candidate_xyz = trajectory.tcp_xyz_world
            knn = query_raw_point_knn(
                source_xyz=point_xyz,
                query_xyz=candidate_xyz,
                source_valid_mask=point_valid_mask,
                config=self.knn_config,
            )
            neighbor_tokens = _gather_tokens(point_tokens, knn.indices)
            memory = neighbor_tokens + self.relative_projection(
                knn.local_features
            )
            query = self.query_projection(
                torch.cat(
                    [
                        state_token[:, None].expand(-1, cfg.query_steps, -1),
                        coarse[:, : cfg.query_steps],
                        candidate_xyz,
                    ],
                    dim=-1,
                )
            ) + slot
            batch = point_xyz.shape[0]
            context, attention = self.relation(
                query.reshape(batch * cfg.query_steps, 1, cfg.token_dim),
                memory.reshape(
                    batch * cfg.query_steps, cfg.knn_k, cfg.token_dim
                ),
                knn.valid_mask.reshape(batch * cfg.query_steps, cfg.knn_k),
            )
            context = context.reshape(batch, cfg.query_steps, cfg.token_dim)
            attention = attention.reshape(
                batch,
                cfg.query_steps,
                cfg.attention_heads,
                1,
                cfg.knn_k,
            )
            relation_valid = knn.valid_mask
            output["candidate_xyz"] = candidate_xyz
            output["knn_distance"] = knn.distance

        residual = self.residual_head(
            torch.cat(
                [
                    context,
                    state_token[:, None].expand(-1, cfg.query_steps, -1),
                    coarse[:, : cfg.query_steps],
                ],
                dim=-1,
            )
        )
        refined_prefix = torch.clamp(
            coarse[:, : cfg.query_steps] + residual, -1.0, 1.0
        )
        action = torch.cat(
            [refined_prefix, coarse[:, cfg.query_steps :]], dim=1
        )
        output.update(
            {
                "action": action,
                "residual_action": residual,
                "relation_valid_mask": relation_valid,
                "attention": attention,
            }
        )
        return output

    def loss(
        self, output: dict[str, torch.Tensor], target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        first_action = F.smooth_l1_loss(
            output["action"][:, 0],
            target[:, 0],
        )
        if self.cfg.query_steps > 1:
            future_refined = F.smooth_l1_loss(
                output["action"][:, 1 : self.cfg.query_steps],
                target[:, 1 : self.cfg.query_steps],
            )
        else:
            future_refined = first_action.new_zeros(())
        coarse = F.smooth_l1_loss(output["coarse_action"], target)
        total = (
            first_action
            + self.cfg.future_refined_loss_weight * future_refined
            + self.cfg.coarse_loss_weight * coarse
        )
        return {
            "loss": total,
            "refined_smooth_l1": first_action,
            "first_action_smooth_l1": first_action,
            "future_refined_smooth_l1": future_refined,
            "coarse_smooth_l1": coarse,
        }

    def checkpoint_config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

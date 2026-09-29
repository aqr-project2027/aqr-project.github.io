from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FutureAuxLoss:
    total: torch.Tensor
    latent: torch.Tensor
    action: torch.Tensor


class ActionConditionedFutureAuxiliary(nn.Module):
    """Optional training-only latent regularizer; it has no inference method."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        horizons: tuple[int, ...] = (1, 2, 4),
        hidden_dim: int = 256,
        margin: float = 0.2,
    ):
        super().__init__()
        ordered = tuple(sorted(set(int(value) for value in horizons)))
        if not ordered or any(value <= 0 for value in ordered):
            raise ValueError("future auxiliary horizons must be positive.")
        if min(int(latent_dim), int(action_dim), int(hidden_dim)) <= 0:
            raise ValueError("future auxiliary dimensions must be positive.")
        if float(margin) < 0.0:
            raise ValueError("future auxiliary margin must be non-negative.")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.horizons = ordered
        self.margin = float(margin)
        self.action_encoder = nn.GRU(
            input_size=self.action_dim,
            hidden_size=int(hidden_dim),
            batch_first=True,
        )
        self.horizon_embedding = nn.Embedding(
            max(self.horizons) + 1, int(hidden_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim + int(hidden_dim) * 2, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )

    def predict(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        if current_latent.ndim != 2 or current_latent.shape[-1] != self.latent_dim:
            raise ValueError("current_latent must have shape [B,D].")
        if (
            action_sequence.ndim != 3
            or action_sequence.shape[0] != current_latent.shape[0]
            or action_sequence.shape[-1] != self.action_dim
        ):
            raise ValueError("action_sequence must have shape [B,H,A].")
        if action_sequence.shape[1] < max(self.horizons):
            raise ValueError(
                "action_sequence is shorter than the largest future horizon."
            )
        predictions: list[torch.Tensor] = []
        for horizon in self.horizons:
            _, hidden = self.action_encoder(action_sequence[:, :horizon])
            horizon_index = torch.full(
                (current_latent.shape[0],),
                horizon,
                dtype=torch.long,
                device=current_latent.device,
            )
            delta = self.predictor(
                torch.cat(
                    [
                        current_latent,
                        hidden[-1],
                        self.horizon_embedding(horizon_index),
                    ],
                    dim=-1,
                )
            )
            predictions.append(current_latent + delta)
        return torch.stack(predictions, dim=1)

    def loss(
        self,
        *,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        target_latents: torch.Tensor,
        negative_action_sequence: torch.Tensor,
        latent_weight: float,
        action_weight: float,
    ) -> FutureAuxLoss:
        expected_target = (
            current_latent.shape[0],
            len(self.horizons),
            self.latent_dim,
        )
        if tuple(target_latents.shape) != expected_target:
            raise ValueError(
                f"target_latents must have shape {expected_target}, "
                f"got {tuple(target_latents.shape)}."
            )
        if negative_action_sequence.shape != action_sequence.shape:
            raise ValueError("negative actions must align with positive actions.")
        positive = self.predict(current_latent, action_sequence)
        negative = self.predict(current_latent, negative_action_sequence)
        target = target_latents.detach()
        latent_loss = (
            1.0
            - F.cosine_similarity(positive, target, dim=-1, eps=1e-6)
        ).mean()
        positive_distance = 1.0 - F.cosine_similarity(
            positive, target, dim=-1, eps=1e-6
        )
        negative_distance = 1.0 - F.cosine_similarity(
            negative, target, dim=-1, eps=1e-6
        )
        action_loss = F.relu(
            self.margin + positive_distance - negative_distance
        ).mean()
        total = float(latent_weight) * latent_loss + float(action_weight) * action_loss
        return FutureAuxLoss(total=total, latent=latent_loss, action=action_loss)


def hard_negative_actions(
    action_sequence: torch.Tensor,
    current_state: torch.Tensor,
) -> torch.Tensor:
    """Choose near-state, near-magnitude but directionally distinct batch actions."""

    if action_sequence.ndim != 3 or current_state.ndim != 2:
        raise ValueError("hard-negative inputs must be [B,H,A] and [B,S].")
    if action_sequence.shape[0] != current_state.shape[0]:
        raise ValueError("hard-negative batch dimensions disagree.")
    batch = action_sequence.shape[0]
    if batch == 1:
        # A same-magnitude opposite direction is the only non-trivial
        # counterfactual available in a singleton batch.
        return -action_sequence
    flat = action_sequence.reshape(batch, -1)
    state_distance = torch.cdist(current_state, current_state)
    magnitude = flat.norm(dim=-1, keepdim=True)
    magnitude_distance = torch.cdist(magnitude, magnitude)
    cosine = F.normalize(flat, dim=-1, eps=1e-6) @ F.normalize(
        flat, dim=-1, eps=1e-6
    ).transpose(0, 1)
    score = state_distance + 0.25 * magnitude_distance + 0.25 * cosine
    score.fill_diagonal_(torch.inf)
    index = score.argmin(dim=1)
    return action_sequence[index]

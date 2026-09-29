from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        if int(dimension) <= 0:
            raise ValueError("time embedding dimension must be positive.")
        self.dimension = int(dimension)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep[None]
        half = self.dimension // 2
        if half == 0:
            return timestep.to(torch.float32).unsqueeze(-1)
        exponent = -math.log(10000.0) * torch.arange(
            half, device=timestep.device, dtype=torch.float32
        ) / max(1, half - 1)
        angle = timestep.to(torch.float32).unsqueeze(-1) * exponent.exp()
        embedding = torch.cat([angle.sin(), angle.cos()], dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(
                embedding, (0, self.dimension - embedding.shape[-1])
            )
        return embedding


class SlotWiseLocalResidual(nn.Module):
    """Lightweight slot-preserving FiLM residual in diffusion output space.

    Layer 3: relation-conditioned gain modulation.
    When ``relation_stats`` is provided and ``relation_gain_scale > 0``,
    the hard ``clamp(max=1.0)`` on local_strength is replaced by a
    data-driven upper bound that scales with the geometric deviation signal.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        local_dim: int,
        global_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 128,
        zero_init: bool = True,
        relation_gain_scale: float = 0.0,
    ):
        super().__init__()
        if min(
            int(action_dim),
            int(local_dim),
            int(global_dim),
            int(hidden_dim),
            int(time_dim),
        ) <= 0:
            raise ValueError("local residual dimensions must be positive.")
        self.action_dim = int(action_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(int(time_dim)),
            nn.Linear(int(time_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.action_projection = nn.Linear(int(action_dim), int(hidden_dim))
        self.global_projection = nn.Sequential(
            nn.Linear(int(global_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.local_film = nn.Sequential(
            nn.Linear(int(local_dim), int(hidden_dim), bias=False),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim) * 2, bias=False),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )
        if zero_init:
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

        # Layer 3: relation-conditioned gain (initialised to 0 = off)
        self.register_buffer(
            "relation_gain_scale",
            torch.tensor(float(relation_gain_scale)),
        )

    def forward(
        self,
        *,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor | int,
        global_condition: torch.Tensor,
        local_tokens: torch.Tensor,
        relation_stats: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if noisy_action.ndim != 3 or noisy_action.shape[-1] != self.action_dim:
            raise ValueError("noisy_action must have shape [B,H,action_dim].")
        if local_tokens.ndim != 3 or local_tokens.shape[:2] != noisy_action.shape[:2]:
            raise ValueError("local_tokens must preserve the action [B,H] slots.")
        if global_condition.ndim != 2 or global_condition.shape[0] != noisy_action.shape[0]:
            raise ValueError("global_condition must have shape [B,D].")
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(
                [int(timestep)], dtype=torch.long, device=noisy_action.device
            )
        timestep = timestep.to(device=noisy_action.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.expand(noisy_action.shape[0])
        base = (
            self.action_projection(noisy_action)
            + self.time_embedding(timestep).to(noisy_action.dtype)[:, None]
            + self.global_projection(global_condition)[:, None]
        )
        gamma, beta = self.local_film(local_tokens).chunk(2, dim=-1)
        modulated = gamma * base + beta

        # ── Layer 3: relation-conditioned gain ──
        batch_size = noisy_action.shape[0]
        relation_gain = torch.ones(batch_size, device=noisy_action.device,
                                   dtype=noisy_action.dtype)
        if relation_stats is not None and self.relation_gain_scale > 0:
            r_face_neg = relation_stats.get("r_face_neg_fraction")
            nearest_abnormal = relation_stats.get("nearest_abnormal")
            if r_face_neg is not None and nearest_abnormal is not None:
                deviation = (r_face_neg + nearest_abnormal) / 2.0
                relation_gain = 1.0 + self.relation_gain_scale * deviation
            elif r_face_neg is not None:
                relation_gain = 1.0 + self.relation_gain_scale * r_face_neg
            # Ensure at least 1-D; broadcast to batch if scalar
            if relation_gain.ndim == 0:
                relation_gain = relation_gain.expand(batch_size)
        local_strength = local_tokens.norm(dim=-1, keepdim=True).clamp(
            max=relation_gain[:, None, None]
        )

        # The zero-local causal intervention must remove the AQR contribution
        # exactly.  Depending on a learned bias here would let the residual head
        # become a second global denoiser and invalidate Q6.
        bias_baseline = self.output(torch.zeros_like(modulated))
        delta = (self.output(modulated) - bias_baseline) * local_strength
        return delta, {
            "film_gamma": gamma,
            "film_beta": beta,
            "local_token_norm": local_tokens.norm(dim=-1),
            "delta_norm": delta.norm(dim=-1),
            "relation_gain": relation_gain,
        }


class PostDiffusionActionRefiner(nn.Module):
    """One-shot slot-wise correction of a completed diffusion horizon.

    Unlike :class:`SlotWiseLocalResidual`, this module is never called from a
    scheduler step. It consumes the final normalized DP3 horizon, global scene
    condition, slot identity, and optional slot-aligned local tokens, then
    predicts a residual and scalar confidence for every slot. Explicit input
    modes implement trainable action-only/global/current-TCP ablations without
    evaluation-time feature masking.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        local_dim: int,
        global_dim: int,
        hidden_dim: int = 256,
        confidence_initial_bias: float = -2.0,
        maximum_horizon: int = 64,
        input_mode: str = "full",
    ) -> None:
        super().__init__()
        if min(action_dim, local_dim, global_dim, hidden_dim, maximum_horizon) <= 0:
            raise ValueError("post-refiner dimensions must be positive.")
        self.action_dim = int(action_dim)
        input_mode = str(input_mode).strip().lower()
        if input_mode not in {
            "full",
            "action_only",
            "global",
            "current_tcp_local",
        }:
            raise ValueError(f"unsupported post-refiner input mode {input_mode!r}.")
        self.input_mode = input_mode
        self.uses_global_feature = input_mode != "action_only"
        self.uses_local_query = input_mode in {"full", "current_tcp_local"}
        self.action_projection = nn.Linear(int(action_dim), int(hidden_dim))
        self.slot_embedding = nn.Embedding(int(maximum_horizon), int(hidden_dim))
        self.global_projection = (
            nn.Sequential(
                nn.Linear(int(global_dim), int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            if self.uses_global_feature
            else None
        )
        self.local_projection = (
            nn.Sequential(
                nn.Linear(int(local_dim), int(hidden_dim), bias=False),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim), bias=False),
            )
            if self.uses_local_query
            else None
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.constant_(
            self.confidence_head[-1].bias, float(confidence_initial_bias)
        )

    def forward(
        self,
        *,
        base_action: torch.Tensor,
        global_condition: torch.Tensor,
        local_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if base_action.ndim != 3 or base_action.shape[-1] != self.action_dim:
            raise ValueError("base_action must have shape [B,H,action_dim].")
        if local_tokens.ndim != 3 or local_tokens.shape[:2] != base_action.shape[:2]:
            raise ValueError("local_tokens must preserve final action [B,H] slots.")
        if global_condition.ndim != 2 or global_condition.shape[0] != base_action.shape[0]:
            raise ValueError("global_condition must have shape [B,D].")
        if base_action.shape[1] > self.slot_embedding.num_embeddings:
            raise ValueError("action horizon exceeds post-refiner slot embeddings.")

        slot_ids = torch.arange(
            base_action.shape[1], device=base_action.device, dtype=torch.long
        )
        base_context = self.action_projection(base_action) + self.slot_embedding(
            slot_ids
        )[None]
        if self.global_projection is not None:
            base_context = base_context + self.global_projection(global_condition)[:, None]

        if self.local_projection is not None:
            context = base_context + self.local_projection(local_tokens)
            # Local-query variants retain the exact zero-local causal contract.
            delta = self.delta_head(context) - self.delta_head(base_context)
            local_present = local_tokens.norm(dim=-1, keepdim=True).gt(0.0)
            delta = delta * local_present.to(delta.dtype)
        else:
            context = base_context
            delta = self.delta_head(context)
            local_present = torch.ones(
                (*base_action.shape[:2], 1),
                dtype=torch.bool,
                device=base_action.device,
            )
        confidence = torch.sigmoid(self.confidence_head(context))
        if self.uses_local_query:
            confidence = confidence * local_present.to(confidence.dtype)
        return delta, confidence, {
            "post_raw_residual_norm": delta.norm(dim=-1),
            "post_confidence": confidence.squeeze(-1),
            "post_local_present": local_present.squeeze(-1),
            "post_global_feature_enabled": torch.full(
                base_action.shape[:2],
                self.uses_global_feature,
                dtype=torch.bool,
                device=base_action.device,
            ),
            "post_local_query_enabled": torch.full(
                base_action.shape[:2],
                self.uses_local_query,
                dtype=torch.bool,
                device=base_action.device,
            ),
        }


def bound_action_residual(
    delta: torch.Tensor,
    *,
    reference_action: torch.Tensor,
    arm_dim: int,
    maximum_norm: float,
    minimum_progress_ratio: float,
    protect_non_arm: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply a task-agnostic safety envelope to a clean-action residual.

    ``reference_action`` is the semantic/global policy prediction, not an A3
    synthetically offset candidate.  The envelope has three properties:

    1. the correction has a hard normalized-action L2 budget;
    2. protected actuator dimensions (for example the gripper) remain exactly
       under semantic-policy control; and
    3. the corrected arm action retains at least ``minimum_progress_ratio`` of
       the semantic action's projection, so a local correction cannot turn
       purposeful motion into a stall or reversal.
    """

    if delta.shape != reference_action.shape or delta.ndim < 2:
        raise ValueError("delta and reference_action must have the same action shape.")
    action_dim = int(delta.shape[-1])
    arm_dim = int(arm_dim)
    if not 0 < arm_dim <= action_dim:
        raise ValueError("arm_dim must be in [1, action_dim].")
    maximum_norm = float(maximum_norm)
    if maximum_norm <= 0.0:
        raise ValueError("maximum_norm must be positive.")
    minimum_progress_ratio = float(minimum_progress_ratio)
    if not 0.0 <= minimum_progress_ratio <= 1.0:
        raise ValueError("minimum_progress_ratio must be in [0,1].")

    protected_norm = delta[..., arm_dim:].norm(dim=-1)
    if arm_dim < action_dim:
        non_arm = delta[..., arm_dim:]
        if protect_non_arm:
            non_arm = torch.zeros_like(non_arm)
        bounded = torch.cat([delta[..., :arm_dim], non_arm], dim=-1)
    else:
        bounded = delta

    raw_norm = bounded.norm(dim=-1, keepdim=True)
    norm_scale = (maximum_norm / raw_norm.clamp_min(1e-8)).clamp(max=1.0)
    bounded = bounded * norm_scale
    norm_clipped = raw_norm.squeeze(-1) > maximum_norm

    reference_arm = reference_action[..., :arm_dim]
    bounded_arm = bounded[..., :arm_dim]
    reference_norm_sq = reference_arm.square().sum(dim=-1, keepdim=True)
    delta_projection = (bounded_arm * reference_arm).sum(dim=-1, keepdim=True)
    minimum_projection = -(1.0 - minimum_progress_ratio) * reference_norm_sq
    progress_deficit = (minimum_projection - delta_projection).clamp_min(0.0)
    progress_adjustment = (
        progress_deficit / reference_norm_sq.clamp_min(1e-8)
    ) * reference_arm
    has_reference_motion = reference_norm_sq > 1e-12
    progress_adjustment = torch.where(
        has_reference_motion,
        progress_adjustment,
        torch.zeros_like(progress_adjustment),
    )
    adjusted_arm = bounded_arm + progress_adjustment
    if arm_dim < action_dim:
        bounded = torch.cat([adjusted_arm, bounded[..., arm_dim:]], dim=-1)
    else:
        bounded = adjusted_arm
    progress_limited = progress_deficit.squeeze(-1) > 0.0

    # The projection adjustment removes an excessive opposing component and
    # should not increase the norm in exact arithmetic.  Reapply the hard bound
    # to make that guarantee explicit under finite precision.
    adjusted_norm = bounded.norm(dim=-1, keepdim=True)
    bounded = bounded * (
        maximum_norm / adjusted_norm.clamp_min(1e-8)
    ).clamp(max=1.0)

    corrected_arm = reference_arm + bounded[..., :arm_dim]
    corrected_progress = (corrected_arm * reference_arm).sum(
        dim=-1
    ) / reference_norm_sq.squeeze(-1).clamp_min(1e-8)
    corrected_progress = torch.where(
        reference_norm_sq.squeeze(-1) > 1e-12,
        corrected_progress,
        torch.ones_like(corrected_progress),
    )
    return bounded, {
        "raw_action_residual_norm": delta.norm(dim=-1),
        "bounded_action_residual_norm": bounded.norm(dim=-1),
        "action_residual_clip_fraction": norm_clipped,
        "action_residual_progress_limited_fraction": progress_limited,
        "action_residual_progress_ratio": corrected_progress,
        "protected_non_arm_residual_norm": protected_norm,
    }


def relative_action_improvement_loss(
    *,
    base_action: torch.Tensor,
    applied_residual: torch.Tensor,
    expert_action: torch.Tensor,
    active_gate: torch.Tensor,
    margin_ratio: float,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Require an applied AQR residual to improve on a detached base action.

    Inputs contain only the executable query slots and arm dimensions.  The
    base action is detached internally so this auxiliary objective cannot be
    satisfied by deliberately degrading the semantic policy.  The relative
    margin vanishes when the base action is already correct and scales with the
    temporal AQR gate so inactive diffusion steps contribute exactly zero.
    ``sample_mask`` optionally restricts the objective to eligible batch items;
    clean-only A3 training uses it to exclude synthetic-offset samples without
    removing them from the diffusion/BC objective.
    """

    if (
        base_action.shape != applied_residual.shape
        or base_action.shape != expert_action.shape
        or base_action.ndim < 2
    ):
        raise ValueError(
            "base_action, applied_residual, and expert_action must have the "
            "same [B,...,A] shape."
        )
    if active_gate.ndim != 1 or active_gate.shape[0] != base_action.shape[0]:
        raise ValueError("active_gate must have shape [B].")
    if sample_mask is not None and (
        sample_mask.ndim != 1 or sample_mask.shape[0] != base_action.shape[0]
    ):
        raise ValueError("sample_mask must have shape [B].")
    margin_ratio = float(margin_ratio)
    if not 0.0 <= margin_ratio <= 1.0:
        raise ValueError("margin_ratio must be in [0,1].")

    base = base_action.detach()
    expert = expert_action.detach()
    corrected = base + applied_residual
    reduce_dims = tuple(range(1, base.ndim))
    base_error = (base - expert).square().mean(dim=reduce_dims)
    corrected_error = (corrected - expert).square().mean(dim=reduce_dims)

    gate = active_gate.detach().to(
        device=base.device, dtype=base.dtype
    ).clamp(0.0, 1.0)
    eligible = (
        torch.ones_like(gate, dtype=torch.bool)
        if sample_mask is None
        else sample_mask.detach().to(device=base.device, dtype=torch.bool)
    )
    active = (gate > 0.0) & eligible
    per_sample = F.relu(
        corrected_error
        - base_error
        + margin_ratio * gate * base_error
    )
    active_float = active.to(base.dtype)
    active_count = active_float.sum()
    eligible_float = eligible.to(base.dtype)
    eligible_count = eligible_float.sum()
    loss = (per_sample * active_float).sum() / active_count.clamp_min(1.0)

    def active_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * active_float).sum() / active_count.clamp_min(1.0)

    tolerance = torch.finfo(base.dtype).eps * 16.0
    error_delta = corrected_error - base_error
    return loss, {
        "improvement_base_error": active_mean(base_error),
        "improvement_corrected_error": active_mean(corrected_error),
        "improvement_error_delta": active_mean(error_delta),
        "improvement_helpful_fraction": active_mean(
            (error_delta < -tolerance).to(base.dtype)
        ),
        "improvement_no_harm_violation_fraction": active_mean(
            (error_delta > tolerance).to(base.dtype)
        ),
        "improvement_margin_satisfied_fraction": active_mean(
            (per_sample <= tolerance).to(base.dtype)
        ),
        "improvement_active_fraction": active_float.mean(),
        "improvement_eligible_fraction": eligible_float.mean(),
        "improvement_active_among_eligible_fraction": (
            active_count / eligible_count.clamp_min(1.0)
        ),
    }


def predicted_clean_sample(
    *,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    scheduler,
) -> torch.Tensor:
    """Convert epsilon/sample/v prediction into the current clean-action estimate."""

    if not torch.is_tensor(timestep):
        timestep = torch.tensor([int(timestep)], device=sample.device)
    timestep = timestep.to(device=sample.device, dtype=torch.long)
    if timestep.ndim == 0:
        timestep = timestep[None]
    timestep = timestep.expand(sample.shape[0])
    prediction_type = str(scheduler.config.prediction_type)
    if prediction_type == "sample":
        clean = model_output
    else:
        alpha_bar = scheduler.alphas_cumprod.to(
            device=sample.device, dtype=sample.dtype
        )[timestep]
        alpha = alpha_bar.sqrt().view(-1, *([1] * (sample.ndim - 1)))
        sigma = (1.0 - alpha_bar).clamp_min(0.0).sqrt().view(
            -1, *([1] * (sample.ndim - 1))
        )
        if prediction_type == "epsilon":
            clean = (sample - sigma * model_output) / alpha.clamp_min(1e-8)
        elif prediction_type == "v_prediction":
            clean = alpha * sample - sigma * model_output
        else:
            raise ValueError(
                "AQR supports scheduler prediction_type epsilon, sample, or "
                f"v_prediction; got {prediction_type!r}."
            )
    clip_sample = bool(getattr(scheduler.config, "clip_sample", False))
    if clip_sample:
        clip_range = float(getattr(scheduler.config, "clip_sample_range", 1.0))
        clean = clean.clamp(-clip_range, clip_range)
    return clean


def denoising_progress(
    timestep: torch.Tensor | int,
    *,
    num_train_timesteps: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([int(timestep)], device=device)
    value = timestep.to(device=device, dtype=dtype)
    if value.ndim == 0:
        value = value[None]
    value = value.expand(int(batch_size))
    denominator = max(1, int(num_train_timesteps) - 1)
    return 1.0 - value / float(denominator)


def query_gate(
    progress: torch.Tensor,
    *,
    start_fraction: float,
    maximum: float,
) -> torch.Tensor:
    start = float(start_fraction)
    if not 0.0 <= start < 1.0:
        raise ValueError("AQR query start_fraction must be in [0,1).")
    if float(maximum) < 0.0:
        raise ValueError("AQR maximum gate must be non-negative.")
    return (
        ((progress - start) / max(1e-8, 1.0 - start)).clamp(0.0, 1.0)
        * float(maximum)
    )

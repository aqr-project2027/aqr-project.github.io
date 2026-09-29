from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


POINT_CLOUD_FRONT_END_MODES = frozenset(
    {"baseline_fps2048", "cache_only", "dual_bank"}
)
QUERY_SOURCES = frozenset(
    {
        "off",
        "static",
        "noisy",
        "predicted_clean",
        "final_base",
        "expert",
        "wrong",
    }
)
TASK_PROFILES = frozenset(
    {"peg_insertion", "stack_cube", "pull_cube_tool"}
)
LOCAL_INTERVENTIONS = frozenset({"normal", "shuffle_slots", "zero", "dropout"})
TOOL_KEYPOINT_MODES = frozenset({"tcp_only", "tcp_fingertips"})
GEOMETRY_MODES = frozenset(
    {"absolute", "translation_relative", "tool_frame_relative"}
)
RELATION_FUSION_MODES = frozenset({"convex", "semantic_residual"})
REFINEMENT_STAGES = frozenset({"denoising", "post_diffusion"})
POST_REFINER_INPUT_MODES = frozenset(
    {"full", "action_only", "global", "current_tcp_local"}
)
PROTECTED_ENCODER_QUOTAS = frozenset(
    {
        (1536, 512),
        (1280, 768),
        (1024, 1024),
    }
)

# Deployment-safe world-frame bounds for ManiSkill PegInsertionSide-v1.
# The generic point front end keeps these optional, while the task config and
# training static contract require the same explicit pair end to end.
PEG_INSERTION_WORKSPACE_CROP_MIN = (-0.5, -0.5, -0.05)
PEG_INSERTION_WORKSPACE_CROP_MAX = (0.5, 0.65, 0.5)
# StackCube uses the same deployment-observable tabletop volume, but keeps a
# distinct profile so task selection and action-contract provenance are never
# inferred from coincidentally equal numeric bounds.
STACK_CUBE_WORKSPACE_CROP_MIN = (-0.5, -0.5, -0.05)
STACK_CUBE_WORKSPACE_CROP_MAX = (0.5, 0.65, 0.5)
# PullCubeTool needs a slightly wider positive-X bound for the tool and cube
# trajectory.  The bound is still deployment-observable and is applied before
# both FPS2048 and Query4096.
PULL_CUBE_TOOL_WORKSPACE_CROP_MIN = (-0.5, -0.5, -0.05)
PULL_CUBE_TOOL_WORKSPACE_CROP_MAX = (0.6, 0.65, 0.5)


def validate_workspace_bounds(
    crop_min,
    crop_max,
    *,
    allow_none: bool = True,
) -> tuple[
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    """Normalize a paired XYZ workspace contract and reject ambiguous bounds."""

    if crop_min is None and crop_max is None:
        if allow_none:
            return None, None
        raise ValueError("workspace_crop_min/workspace_crop_max are required.")
    if crop_min is None or crop_max is None:
        raise ValueError(
            "workspace_crop_min and workspace_crop_max must be configured together."
        )
    try:
        lower = tuple(float(value) for value in crop_min)
        upper = tuple(float(value) for value in crop_max)
    except TypeError as exc:
        raise ValueError("workspace bounds must each contain three XYZ values.") from exc
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("workspace bounds must each contain three XYZ values.")
    if any(not math.isfinite(value) for value in lower + upper):
        raise ValueError("workspace bounds must contain only finite values.")
    if any(low >= high for low, high in zip(lower, upper)):
        raise ValueError(
            "workspace_crop_min must be strictly below workspace_crop_max."
        )
    return lower, upper

# These switches belonged to the superseded adaptive-commit / consequence
# branch.  Refusing them is safer than silently accepting a stale experiment
# override and accidentally presenting it as the new AQR mainline.
REMOVED_MAINLINE_KEYS = frozenset(
    {
        "adaptive_commit",
        "adaptive_horizon",
        "coarse_policy",
        "commit_head",
        "consequence_model",
        "event_goal",
        "future_goal",
        "hazard",
        "hazard_head",
        "motion_bank",
        "recovery",
        "recovery_policy",
    }
)


# Canonical, causal experiment identities from the design document.  Explicit
# values supplied by a YAML config override these defaults.
EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {
    "AQR_OFF": {
        "enabled": False,
        "front_end_mode": "baseline_fps2048",
        "cache_query_features": False,
        "query_source": "off",
    },
    "P0": {
        "enabled": False,
        "front_end_mode": "baseline_fps2048",
        "cache_query_features": False,
        "query_source": "off",
    },
    "P1": {
        "enabled": False,
        "front_end_mode": "dual_bank",
        "cache_query_features": False,
        "query_source": "off",
    },
    "P2": {
        "enabled": False,
        "front_end_mode": "cache_only",
        "cache_query_features": True,
        "query_source": "off",
    },
    "P3": {
        "enabled": True,
        "front_end_mode": "dual_bank",
        "cache_query_features": True,
        "query_source": "static",
    },
    "B0": {
        "enabled": False,
        "front_end_mode": "baseline_fps2048",
        "cache_query_features": False,
        "query_source": "off",
    },
    "B1": {
        "enabled": False,
        "front_end_mode": "dual_bank",
        "cache_query_features": False,
        "query_source": "off",
    },
    "S0": {
        "enabled": True,
        "front_end_mode": "baseline_fps2048",
        "cache_query_features": True,
        "query_source": "static",
    },
    "S1": {
        "enabled": True,
        "front_end_mode": "dual_bank",
        "cache_query_features": True,
        "query_source": "static",
    },
    "A0": {
        "enabled": True,
        "front_end_mode": "baseline_fps2048",
        "cache_query_features": True,
        "query_source": "predicted_clean",
    },
    "A1": {
        "enabled": True,
        "front_end_mode": "baseline_fps2048",
        "cache_query_features": True,
        "query_source": "predicted_clean",
    },
    "A1_PROTECTED": {
        "enabled": True,
        "front_end_mode": "dual_bank",
        "cache_query_features": True,
        "query_source": "predicted_clean",
    },
    "A1_QUERY3": {
        "enabled": True,
        "front_end_mode": "baseline_fps2048",
        "cache_query_features": True,
        "query_source": "predicted_clean",
        "query_update_fractions": (0.55, 0.75, 0.90),
    },
    "Q0": {"enabled": True, "query_source": "static"},
    "Q1": {"enabled": True, "query_source": "noisy"},
    "Q2": {"enabled": True, "query_source": "predicted_clean"},
    "Q3": {"enabled": True, "query_source": "expert"},
    "Q4": {"enabled": True, "query_source": "wrong"},
    "Q5": {
        "enabled": True,
        "query_source": "predicted_clean",
        "local_intervention": "shuffle_slots",
    },
    "Q6": {
        "enabled": True,
        "query_source": "predicted_clean",
        "local_intervention": "zero",
    },
    "Q7_TCP": {
        "enabled": True,
        "query_source": "predicted_clean",
        "tool_keypoints": "tcp_only",
    },
    "Q7_FINGERS": {
        "enabled": True,
        "query_source": "predicted_clean",
        "tool_keypoints": "tcp_fingertips",
    },
    # ─── Relation Query experiments ───
    # A2-A5 are the clean dual-branch ablation line.
    "A2": {
        "enabled": True,
        "front_end_mode": "dual_bank",
        "cache_query_features": True,
        "query_source": "predicted_clean",
        "relation_features": True,
        "relation_attention": False,
        "relation_gain_scale": 0.0,
        "offset_training_enabled": False,
    },
    "A3": {
        "enabled": True,
        "front_end_mode": "dual_bank",
        "cache_query_features": True,
        "query_source": "predicted_clean",
        "relation_features": True,
        "relation_attention": False,
        "relation_gain_scale": 0.0,
        "offset_training_enabled": True,
        # Normalized arm-action L2 radius.  The curriculum ramps to this
        # configured ceiling instead of using hard-coded Cartesian values.
        "offset_max_action_norm": 0.30,
        "offset_prob": 0.60,
    },
    "A4": {
        "enabled": True,
        "front_end_mode": "dual_bank",
        "cache_query_features": True,
        "query_source": "predicted_clean",
        "relation_features": True,
        "relation_attention": True,
        "relation_gain_scale": 0.0,
        "offset_training_enabled": True,
        "offset_max_action_norm": 0.30,
        "offset_prob": 0.60,
    },
    "A5": {
        "enabled": True,
        "front_end_mode": "dual_bank",
        "cache_query_features": True,
        "query_source": "predicted_clean",
        "relation_features": True,
        "relation_attention": True,
        "relation_gain_scale": 2.0,
        "offset_training_enabled": True,
        "offset_max_action_norm": 0.30,
        "offset_prob": 0.60,
    },
    # Pure DP3 diffusion followed by one final-action AQR correction.  The
    # denoising trajectory is never modified by local geometry in this mode.
    "A6_POST": {
        "enabled": True,
        "front_end_mode": "dual_bank",
        "cache_query_features": True,
        "query_source": "final_base",
        "relation_features": True,
        "relation_attention": False,
        "relation_gain_scale": 0.0,
        "offset_training_enabled": True,
        "offset_max_action_norm": 0.30,
        "offset_prob": 0.60,
        "refinement_stage": "post_diffusion",
        "local_action_slots": (1, 2, 3),
        "post_base_frozen": True,
        "anchor_preserving_query": True,
    },
}


@dataclass(frozen=True)
class PointCloudFrontEndConfig:
    """Validated point-budget contract for the AQR-DP3 front end.

    ``baseline_fps2048`` reproduces a single ordinary FPS encoder bank.
    ``cache_only`` builds the protected banks but leaves AQR disabled so the
    query-cache data path can be tested in isolation.  ``dual_bank`` enables
    the full protected encoder/query-bank path.
    """

    mode: str = "baseline_fps2048"
    enc_points: int = 2048
    enc_global_points: int = 1280
    enc_tcp_swept_points: int = 768
    query_points: int = 4096
    tcp_radius_m: float = 0.08
    seed: int = 0
    deterministic: bool = True
    workspace_crop_min: tuple[float, float, float] | None = None
    workspace_crop_max: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        object.__setattr__(self, "mode", mode)
        if mode not in POINT_CLOUD_FRONT_END_MODES:
            choices = ", ".join(sorted(POINT_CLOUD_FRONT_END_MODES))
            raise ValueError(f"mode must be one of {choices}; got {self.mode!r}.")
        if int(self.enc_points) != 2048:
            raise ValueError("AQR-DP3 fixes the PointNeXt encoder budget at 2048.")
        quota = (
            int(self.enc_global_points),
            int(self.enc_tcp_swept_points),
        )
        if quota not in PROTECTED_ENCODER_QUOTAS:
            choices = ", ".join(
                f"{global_points}+{tcp_points}"
                for global_points, tcp_points in sorted(
                    PROTECTED_ENCODER_QUOTAS, reverse=True
                )
            )
            raise ValueError(
                "Protected encoder quotas must be one of "
                f"{choices}; got {quota[0]}+{quota[1]}."
            )
        if int(self.enc_global_points) + int(self.enc_tcp_swept_points) != int(
            self.enc_points
        ):
            raise ValueError("Global and TCP-Swept budgets must sum to enc_points.")
        if int(self.query_points) not in {4096, 8192}:
            raise ValueError("query_points must be either 4096 or 8192.")
        if not (float(self.tcp_radius_m) > 0.0):
            raise ValueError("tcp_radius_m must be positive.")
        if not bool(self.deterministic):
            raise ValueError(
                "The staged AQR validation contract requires deterministic sampling."
            )
        crop_min, crop_max = validate_workspace_bounds(
            self.workspace_crop_min,
            self.workspace_crop_max,
        )
        object.__setattr__(self, "workspace_crop_min", crop_min)
        object.__setattr__(self, "workspace_crop_max", crop_max)

    def validate(self) -> "PointCloudFrontEndConfig":
        """Return ``self`` after construction-time validation."""

        return self

    @property
    def p_enc(self) -> int:
        return int(self.enc_points)

    @property
    def p_query(self) -> int:
        return int(self.query_points)

    @property
    def protected_sampling(self) -> bool:
        return self.mode in {"cache_only", "dual_bank"}

    @property
    def aqr_enabled(self) -> bool:
        return self.mode == "dual_bank"

    @property
    def is_cache_only(self) -> bool:
        return self.mode == "cache_only"

    @property
    def workspace_filtering(self) -> bool:
        return self.workspace_crop_min is not None


@dataclass(frozen=True)
class AQRDP3Config:
    """Validated switches for the deployment-safe AQR-DP3 mainline."""

    experiment: str = "A1"
    task_profile: str = "peg_insertion"
    enabled: bool = True
    cache_query_features: bool = True
    front_end: PointCloudFrontEndConfig = field(
        default_factory=PointCloudFrontEndConfig
    )
    query_source: str = "predicted_clean"
    refinement_stage: str = "denoising"
    local_intervention: str = "normal"
    local_dropout_probability: float = 0.0
    geometry_mode: str = "tool_frame_relative"
    tool_keypoints: str = "tcp_fingertips"
    query_radii_m: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)
    query_neighbors: tuple[int, ...] = (16, 64, 128, 256)
    query_start_fraction: float = 0.5
    query_update_fractions: tuple[float, ...] = (0.75,)
    query_gate_maximum: float = 1.0
    local_action_slots: tuple[int, ...] = (1,)
    slot1_sweep_enabled: bool = True
    current_tcp_fallback_enabled: bool = True
    slot1_sweep_samples: int = 4
    wrong_query_offset_m: tuple[float, float, float] = (0.15, 0.0, 0.0)
    use_robot_mask_feature: bool = False
    action_to_tool_mode: str = "joint_delta_fk"
    action_contract_path: str = ""
    urdf_path: str = ""
    arm_dim: int = 7
    tcp_pose_start: int = 9
    exec_steps: int = 1
    local_dim: int = 128
    local_hidden_dim: int = 256
    query_feature_dim: int = 128
    future_aux: bool = False
    future_horizons: tuple[int, ...] = (1, 2, 4)
    future_latent_weight: float = 0.05
    future_action_weight: float = 0.02
    future_ema_decay: float = 0.99

    # ─── Layer 1: relation features ───
    relation_features: bool = False
    relation_gate_initial_bias: float = -1.5
    # ``convex`` preserves the validated A3 checkpoint distribution.
    # ``semantic_residual`` remains an explicit architecture-migration
    # ablation and must not be enabled by merely reusing convex local-head
    # weights.
    relation_fusion_mode: str = "convex"
    relation_gate_maximum: float = 1.0
    surface_normal_k: int = 8
    surface_normal_confidence_threshold: float = 0.5

    # ─── Layer 2: action-conditioned attention ───
    relation_attention: bool = False
    relation_scale_gate: bool = True
    relation_scale_gate_geo_bias: bool = True
    relation_query_dim: int = 128
    relation_key_dim: int = 128

    # A4.1: retain tool-anchor identity through local pooling.  Legacy A2/A3
    # keep their exact mean/max aggregation unless this is explicitly enabled.
    anchor_preserving_query: bool = False

    # ─── Layer 3: relation-conditioned gain ───
    relation_gain_scale: float = 0.0

    # Multi-task safety envelope for the local action correction.  The bound is
    # expressed in normalized arm-action L2 units because the diffusion output
    # is a clean normalized action (prediction_type=sample).  Non-arm actuator
    # dimensions, such as the gripper, remain owned by the semantic policy.
    bounded_action_residual: bool = False
    action_residual_max_norm: float = 0.30
    action_residual_min_progress_ratio: float = 0.50
    action_residual_protect_non_arm: bool = True

    # Counterfactual supervision for the executable local correction.  The
    # margin is relative to the detached base-action error, so a correct base
    # action has a zero margin while an incorrect base action requires a real
    # fractional improvement from AQR.
    improvement_loss_enabled: bool = False
    improvement_loss_weight: float = 0.0
    improvement_margin_ratio: float = 0.05
    # When enabled, synthetically offset A3 samples still train through the
    # diffusion/BC objective but are excluded from the relative improvement
    # hinge.  This prevents easy artificial corrections from hiding deployment-
    # relevant performance on naturally predicted actions.
    improvement_loss_clean_only: bool = False

    # Final-action refiner. ``post_diffusion`` runs the global DP3 sampler to
    # completion, then queries geometry and corrects every configured
    # executable slot exactly once. The frozen-base default makes confidence=0
    # an exact return to the initialized pure-DP3 policy.
    post_base_frozen: bool = True
    # Structural post-refiner ablations. ``full`` is the deployed A6 path;
    # the other modes deliberately remove inputs instead of zeroing them only
    # at evaluation time.
    post_refiner_input_mode: str = "full"
    post_delta_loss_weight: float = 1.0
    post_corrected_loss_weight: float = 1.0
    post_confidence_loss_weight: float = 0.10
    post_confidence_initial_bias: float = -2.0

    # ─── Layer 4: offset training ───
    offset_training_enabled: bool = False
    offset_max_action_norm: float = 0.0
    offset_prob: float = 0.0

    def __post_init__(self) -> None:
        experiment = str(self.experiment).strip().upper()
        object.__setattr__(self, "experiment", experiment)
        task_profile = str(self.task_profile).strip().lower()
        object.__setattr__(self, "task_profile", task_profile)
        if task_profile not in TASK_PROFILES:
            choices = ", ".join(sorted(TASK_PROFILES))
            raise ValueError(
                f"task_profile must be one of {choices}; got {task_profile!r}."
            )
        if experiment not in EXPERIMENT_PRESETS:
            choices = ", ".join(EXPERIMENT_PRESETS)
            raise ValueError(
                f"Unknown AQR experiment {experiment!r}; expected one of {choices}."
            )
        query_source = str(self.query_source).strip().lower()
        refinement_stage = str(self.refinement_stage).strip().lower()
        post_refiner_input_mode = str(self.post_refiner_input_mode).strip().lower()
        intervention = str(self.local_intervention).strip().lower()
        geometry = str(self.geometry_mode).strip().lower()
        keypoints = str(self.tool_keypoints).strip().lower()
        action_mode = str(self.action_to_tool_mode).strip().lower()
        object.__setattr__(self, "query_source", query_source)
        object.__setattr__(self, "refinement_stage", refinement_stage)
        object.__setattr__(
            self, "post_refiner_input_mode", post_refiner_input_mode
        )
        object.__setattr__(self, "local_intervention", intervention)
        object.__setattr__(self, "geometry_mode", geometry)
        object.__setattr__(self, "tool_keypoints", keypoints)
        object.__setattr__(self, "action_to_tool_mode", action_mode)
        if query_source not in QUERY_SOURCES:
            raise ValueError(f"Unsupported AQR query_source {query_source!r}.")
        if refinement_stage not in REFINEMENT_STAGES:
            choices = ", ".join(sorted(REFINEMENT_STAGES))
            raise ValueError(
                f"refinement_stage must be one of {choices}; "
                f"got {refinement_stage!r}."
            )
        if post_refiner_input_mode not in POST_REFINER_INPUT_MODES:
            choices = ", ".join(sorted(POST_REFINER_INPUT_MODES))
            raise ValueError(
                f"post_refiner_input_mode must be one of {choices}; "
                f"got {post_refiner_input_mode!r}."
            )
        if (
            refinement_stage != "post_diffusion"
            and post_refiner_input_mode != "full"
        ):
            raise ValueError(
                "post_refiner_input_mode ablations require "
                "refinement_stage=post_diffusion."
            )
        if refinement_stage == "post_diffusion" and query_source != "final_base":
            raise ValueError(
                "post_diffusion refinement requires query_source=final_base."
            )
        if refinement_stage == "denoising" and query_source == "final_base":
            raise ValueError(
                "query_source=final_base is valid only for post_diffusion refinement."
            )
        if intervention not in LOCAL_INTERVENTIONS:
            raise ValueError(
                f"Unsupported AQR local_intervention {intervention!r}."
            )
        if geometry not in GEOMETRY_MODES:
            raise ValueError(f"Unsupported AQR geometry_mode {geometry!r}.")
        if keypoints not in TOOL_KEYPOINT_MODES:
            raise ValueError(f"Unsupported AQR tool_keypoints {keypoints!r}.")
        if action_mode not in {"joint_delta_fk", "tcp_delta_pose"}:
            raise ValueError(
                "action_to_tool_mode must be joint_delta_fk or tcp_delta_pose."
            )
        if not 1 <= int(self.exec_steps) <= 8:
            raise ValueError("AQR-DP3 deployment requires exec_steps in [1,8].")
        if bool(self.enabled) != (query_source != "off"):
            raise ValueError(
                "enabled and query_source disagree; disabled experiments must use "
                "query_source=off and enabled experiments must use a query source."
            )
        if (
            bool(self.enabled)
            and self.post_refiner_uses_local_query
            and not bool(self.cache_query_features)
        ):
            raise ValueError("Enabled AQR requires cache_query_features=true.")
        if not 0.0 <= float(self.local_dropout_probability) < 1.0:
            raise ValueError("local_dropout_probability must be in [0,1).")
        if intervention == "dropout" and float(self.local_dropout_probability) <= 0:
            raise ValueError(
                "dropout intervention needs local_dropout_probability > 0."
            )
        radii = tuple(float(value) for value in self.query_radii_m)
        neighbors = tuple(int(value) for value in self.query_neighbors)
        object.__setattr__(self, "query_radii_m", radii)
        object.__setattr__(self, "query_neighbors", neighbors)
        object.__setattr__(
            self,
            "wrong_query_offset_m",
            tuple(float(value) for value in self.wrong_query_offset_m),
        )
        if len(radii) == 0 or len(radii) != len(neighbors):
            raise ValueError("query_radii_m and query_neighbors must align.")
        if any(value <= 0 for value in radii):
            raise ValueError("query radii must be positive.")
        if any(value <= 0 for value in neighbors):
            raise ValueError("query neighbor counts must be positive.")
        if not 0.0 <= float(self.query_start_fraction) < 1.0:
            raise ValueError("query_start_fraction must be in [0,1).")
        updates = tuple(float(value) for value in self.query_update_fractions)
        object.__setattr__(self, "query_update_fractions", updates)
        if not 1 <= len(updates) <= 3:
            raise ValueError("AQR inference must use one to three query updates.")
        if tuple(sorted(set(updates))) != updates:
            raise ValueError(
                "query_update_fractions must be unique and increasing."
            )
        if any(
            value < float(self.query_start_fraction) or value > 1.0
            for value in updates
        ):
            raise ValueError(
                "query updates must lie in the enabled, later denoising region."
            )
        if float(self.query_gate_maximum) < 0:
            raise ValueError("query_gate_maximum must be non-negative.")
        local_slots = tuple(int(value) for value in self.local_action_slots)
        object.__setattr__(self, "local_action_slots", local_slots)
        if (
            not local_slots
            or any(value < 0 for value in local_slots)
            or tuple(sorted(set(local_slots))) != local_slots
        ):
            raise ValueError(
                "local_action_slots must be non-negative, unique, and increasing."
            )
        if refinement_stage == "denoising" and local_slots != (1,):
            raise ValueError(
                "Denoising-time AQR only permits local_action_slots=(1,)."
            )
        if not isinstance(self.slot1_sweep_enabled, bool):
            raise ValueError("slot1_sweep_enabled must be boolean.")
        if not isinstance(self.current_tcp_fallback_enabled, bool):
            raise ValueError("current_tcp_fallback_enabled must be boolean.")
        if bool(self.enabled) and self.post_refiner_uses_local_query and not (
            self.slot1_sweep_enabled or self.current_tcp_fallback_enabled
        ):
            raise ValueError(
                "Enabled AQR requires slot1 sweep and/or current-TCP fallback."
            )
        if int(self.slot1_sweep_samples) < 2:
            raise ValueError("slot1_sweep_samples must be at least 2.")
        if not isinstance(self.use_robot_mask_feature, bool):
            raise ValueError("use_robot_mask_feature must be boolean.")
        if len(self.wrong_query_offset_m) != 3:
            raise ValueError("wrong_query_offset_m must have three values.")
        if min(
            int(self.arm_dim),
            int(self.local_dim),
            int(self.local_hidden_dim),
            int(self.query_feature_dim),
        ) <= 0:
            raise ValueError("AQR dimensions must be positive.")
        horizons = tuple(sorted(set(int(value) for value in self.future_horizons)))
        object.__setattr__(self, "future_horizons", horizons)
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("future_horizons must contain positive offsets.")
        if not 0.0 <= float(self.future_ema_decay) < 1.0:
            raise ValueError("future_ema_decay must be in [0,1).")
        if min(
            float(self.future_latent_weight), float(self.future_action_weight)
        ) < 0:
            raise ValueError("future auxiliary weights must be non-negative.")
        # ─── Validate Layer 0–4 fields ───
        object.__setattr__(self, "relation_features", bool(self.relation_features))
        object.__setattr__(
            self, "relation_gate_initial_bias", float(self.relation_gate_initial_bias)
        )
        relation_fusion_mode = str(self.relation_fusion_mode).strip().lower()
        object.__setattr__(self, "relation_fusion_mode", relation_fusion_mode)
        if relation_fusion_mode not in RELATION_FUSION_MODES:
            choices = ", ".join(sorted(RELATION_FUSION_MODES))
            raise ValueError(
                f"relation_fusion_mode must be one of {choices}; "
                f"got {relation_fusion_mode!r}."
            )
        object.__setattr__(
            self, "relation_gate_maximum", float(self.relation_gate_maximum)
        )
        if not 0.0 <= self.relation_gate_maximum <= 1.0:
            raise ValueError("relation_gate_maximum must be in [0,1].")
        if relation_fusion_mode == "semantic_residual" and not self.relation_features:
            raise ValueError(
                "semantic_residual fusion requires relation_features=true."
            )
        object.__setattr__(
            self, "surface_normal_k", max(3, int(self.surface_normal_k))
        )
        object.__setattr__(
            self,
            "surface_normal_confidence_threshold",
            max(0.0, min(1.0, float(self.surface_normal_confidence_threshold))),
        )
        object.__setattr__(self, "relation_attention", bool(self.relation_attention))
        object.__setattr__(
            self, "anchor_preserving_query", bool(self.anchor_preserving_query)
        )
        object.__setattr__(self, "relation_scale_gate", bool(self.relation_scale_gate))
        object.__setattr__(
            self, "relation_scale_gate_geo_bias", bool(self.relation_scale_gate_geo_bias)
        )
        object.__setattr__(self, "relation_query_dim", int(self.relation_query_dim))
        object.__setattr__(self, "relation_key_dim", int(self.relation_key_dim))
        object.__setattr__(
            self, "relation_gain_scale", float(self.relation_gain_scale)
        )
        object.__setattr__(
            self, "bounded_action_residual", bool(self.bounded_action_residual)
        )
        object.__setattr__(
            self, "action_residual_max_norm", float(self.action_residual_max_norm)
        )
        object.__setattr__(
            self,
            "action_residual_min_progress_ratio",
            float(self.action_residual_min_progress_ratio),
        )
        object.__setattr__(
            self,
            "action_residual_protect_non_arm",
            bool(self.action_residual_protect_non_arm),
        )
        object.__setattr__(
            self, "improvement_loss_enabled", bool(self.improvement_loss_enabled)
        )
        object.__setattr__(
            self, "improvement_loss_weight", float(self.improvement_loss_weight)
        )
        object.__setattr__(
            self,
            "improvement_margin_ratio",
            float(self.improvement_margin_ratio),
        )
        object.__setattr__(
            self,
            "improvement_loss_clean_only",
            bool(self.improvement_loss_clean_only),
        )
        object.__setattr__(self, "post_base_frozen", bool(self.post_base_frozen))
        object.__setattr__(
            self, "post_delta_loss_weight", float(self.post_delta_loss_weight)
        )
        object.__setattr__(
            self,
            "post_corrected_loss_weight",
            float(self.post_corrected_loss_weight),
        )
        object.__setattr__(
            self,
            "post_confidence_loss_weight",
            float(self.post_confidence_loss_weight),
        )
        object.__setattr__(
            self,
            "post_confidence_initial_bias",
            float(self.post_confidence_initial_bias),
        )
        if min(
            self.post_delta_loss_weight,
            self.post_corrected_loss_weight,
            self.post_confidence_loss_weight,
        ) < 0.0:
            raise ValueError("post-diffusion loss weights must be non-negative.")
        if refinement_stage == "post_diffusion" and (
            self.post_delta_loss_weight + self.post_corrected_loss_weight <= 0.0
        ):
            raise ValueError(
                "post_diffusion refinement needs a positive delta or corrected loss."
            )
        if self.action_residual_max_norm <= 0.0:
            raise ValueError("action_residual_max_norm must be positive.")
        if not 0.0 <= self.action_residual_min_progress_ratio <= 1.0:
            raise ValueError(
                "action_residual_min_progress_ratio must be in [0,1]."
            )
        if self.bounded_action_residual and not self.enabled:
            raise ValueError("bounded_action_residual requires enabled AQR.")
        if self.improvement_loss_weight < 0.0:
            raise ValueError("improvement_loss_weight must be non-negative.")
        if not 0.0 <= self.improvement_margin_ratio <= 1.0:
            raise ValueError("improvement_margin_ratio must be in [0,1].")
        if self.improvement_loss_enabled:
            if not self.enabled:
                raise ValueError("improvement loss requires enabled AQR.")
            if self.improvement_loss_weight <= 0.0:
                raise ValueError(
                    "enabled improvement loss requires a positive weight."
                )
        object.__setattr__(
            self, "offset_training_enabled", bool(self.offset_training_enabled)
        )
        object.__setattr__(
            self, "offset_max_action_norm", float(self.offset_max_action_norm)
        )
        object.__setattr__(self, "offset_prob", float(self.offset_prob))
        if self.relation_attention and not self.relation_features:
            raise ValueError(
                "relation_attention requires relation_features=true for r_face/r_motion."
            )
        if not 0.0 <= self.offset_prob <= 1.0:
            raise ValueError("offset_prob must be in [0,1].")
        if self.offset_max_action_norm < 0.0:
            raise ValueError("offset_max_action_norm must be non-negative.")
        if self.offset_training_enabled and self.query_source not in {
            "predicted_clean",
            "final_base",
        }:
            raise ValueError(
                "offset training requires query_source=predicted_clean or final_base."
            )

    @property
    def post_refiner_uses_global_feature(self) -> bool:
        return self.post_refiner_input_mode in {
            "full",
            "global",
            "current_tcp_local",
        }

    @property
    def post_refiner_uses_local_query(self) -> bool:
        if self.refinement_stage != "post_diffusion":
            return bool(self.enabled)
        return self.post_refiner_input_mode in {"full", "current_tcp_local"}

    @classmethod
    def from_mapping(
        cls, value: "AQRDP3Config | Mapping[str, Any] | None"
    ) -> "AQRDP3Config":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(value, DictConfig):
                value = OmegaConf.to_container(value, resolve=True)
        except ImportError:
            pass
        if not isinstance(value, Mapping):
            raise TypeError("aqr_config must be a mapping or AQRDP3Config.")
        supplied = dict(value)
        removed = sorted(REMOVED_MAINLINE_KEYS.intersection(supplied))
        if removed:
            raise ValueError(
                "Removed legacy switches are not part of AQR-DP3: "
                + ", ".join(removed)
            )
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(supplied) - known - {"front_end_mode"})
        if unknown:
            raise ValueError("Unknown AQR-DP3 settings: " + ", ".join(unknown))
        experiment = str(supplied.get("experiment", "A1")).strip().upper()
        if experiment not in EXPERIMENT_PRESETS:
            choices = ", ".join(EXPERIMENT_PRESETS)
            raise ValueError(
                f"Unknown AQR experiment {experiment!r}; expected one of {choices}."
            )
        merged: dict[str, Any] = {
            "experiment": experiment,
            **EXPERIMENT_PRESETS[experiment],
            **supplied,
        }
        front_end_value = merged.pop("front_end", {})
        if isinstance(front_end_value, PointCloudFrontEndConfig):
            front_end_dict = {
                name: getattr(front_end_value, name)
                for name in PointCloudFrontEndConfig.__dataclass_fields__
            }
        else:
            if not isinstance(front_end_value, Mapping):
                raise TypeError("aqr_config.front_end must be a mapping.")
            front_end_dict = dict(front_end_value)
        preset_front_end_mode = merged.pop("front_end_mode", "dual_bank")
        front_end_mode = supplied.get(
            "front_end_mode",
            front_end_dict.get("mode", preset_front_end_mode),
        )
        front_end_dict["mode"] = front_end_mode
        try:
            merged["front_end"] = PointCloudFrontEndConfig(**front_end_dict)
        except TypeError as exc:
            raise ValueError(f"Invalid front_end settings: {exc}") from exc
        return cls(**merged)

    @property
    def dynamic_query(self) -> bool:
        return self.query_source in {
            "noisy",
            "predicted_clean",
            "final_base",
            "expert",
            "wrong",
        }

    @property
    def strict_dp3_off(self) -> bool:
        return self.experiment == "AQR_OFF"

    @property
    def keypoint_offsets_tcp(self) -> dict[str, tuple[float, float, float]]:
        if self.tool_keypoints == "tcp_only":
            return {"tcp": (0.0, 0.0, 0.0)}
        # Panda hand TCP is centered between the finger pads.  The ±3 cm
        # offsets approximate the pad centers for the fixed-width grasp used by
        # the peg demonstrations and remain explicit/ablatable via Q7.
        return {
            "tcp": (0.0, 0.0, 0.0),
            "left_fingertip": (0.0, 0.03, 0.0),
            "right_fingertip": (0.0, -0.03, 0.0),
        }

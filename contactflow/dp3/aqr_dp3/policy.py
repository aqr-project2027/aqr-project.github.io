from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F

from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3

from contactflow.dp3.action_query.trajectory import execution_slot_indices
from contactflow.dp3.aqr_bc.factory import (
    build_candidate_trajectory_lift,
    load_action_contract,
)
from contactflow.dp3.aqr_dp3.config import AQRDP3Config
from contactflow.dp3.aqr_dp3.features import (
    CachedPointNeXtEncoder,
    HighResolutionQueryStem,
    PointFeatureCache,
)
from contactflow.dp3.aqr_dp3.future_aux import (
    ActionConditionedFutureAuxiliary,
    hard_negative_actions,
)
from contactflow.dp3.aqr_dp3.geometry import (
    ActionToTool,
    RelativeMultiScaleQuery,
    ToolTrajectory,
    build_current_tcp_only_trajectory,
    build_slot1_sweep_trajectory,
    offset_tool_trajectory,
)
from contactflow.dp3.aqr_dp3.model import (
    PostDiffusionActionRefiner,
    SlotWiseLocalResidual,
    bound_action_residual,
    denoising_progress,
    predicted_clean_sample,
    query_gate,
    relative_action_improvement_loss,
)
from contactflow.dp3.aqr_dp3.pointcloud import (
    PointCloudBatch,
    PointCloudFrontEnd,
)


@dataclass(frozen=True)
class EncodedAQRObservation:
    raw_points: torch.Tensor
    raw_valid_mask: torch.Tensor
    raw_robot_mask: torch.Tensor
    state_history_raw: torch.Tensor
    state_history_normalized: torch.Tensor
    point_banks: PointCloudBatch
    point_cache: PointFeatureCache
    query_features: torch.Tensor | None
    timing_ms: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class LocalQueryResult:
    tokens: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor]
    trajectory: ToolTrajectory | None


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        value = value["shape"]
    return tuple(int(item) for item in value)


def _scalar_tensor(
    value: float, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.tensor(float(value), device=device, dtype=dtype)


class AQRDP3(SimpleDP3):
    """DP3 with a cached dual-bank point front end and denoising-time AQR.

    The global DP3 denoiser remains unchanged.  The local branch is a small,
    slot-preserving residual head whose input is rebuilt from the predicted
    clean action at one to three later denoising updates.
    """

    def __init__(
        self,
        *,
        shape_meta: dict,
        noise_scheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        aqr_config: AQRDP3Config | Mapping[str, Any] | None = None,
        num_inference_steps: int | None = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = "film",
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
        encoder_output_dim: int = 128,
        crop_shape=None,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
        pointcloud_encoder_cfg=None,
        **kwargs,
    ) -> None:
        settings = AQRDP3Config.from_mapping(aqr_config)
        if not bool(obs_as_global_cond):
            raise ValueError("AQR-DP3 supports global-condition DP3 only.")
        if "cross_attention" in str(condition_type).lower():
            raise ValueError(
                "Large cross-attention is intentionally absent from the AQR-DP3 "
                "mainline; use film/add/mlp_film."
            )
        obs_meta = shape_meta["obs"]
        if "point_cloud" not in obs_meta or "agent_pos" not in obs_meta:
            raise ValueError("shape_meta must include point_cloud and agent_pos.")
        point_shape = _shape_tuple(obs_meta["point_cloud"])
        state_shape = _shape_tuple(obs_meta["agent_pos"])
        if len(point_shape) != 2 or point_shape[-1] not in {3, 6}:
            raise ValueError(
                "AQR-DP3 expects dense point_cloud shape [N,3] or [N,6]."
            )
        point_channels = int(point_shape[-1])
        point_color_dim = point_channels - 3
        if bool(use_pc_color) != (point_color_dim == 3):
            raise ValueError(
                "use_pc_color must match the point-cloud contract: use false "
                "for XYZ and true for XYZRGB."
            )
        if (
            not settings.strict_dp3_off
            and point_shape[0] < settings.front_end.p_query
        ):
            raise ValueError(
                "shape_meta point budget must cover P_query: "
                f"{point_shape[0]} < {settings.front_end.p_query}."
            )
        if len(state_shape) != 1:
            raise ValueError("agent_pos must be a flat deployable state vector.")
        state_dim = int(state_shape[0])
        if settings.tcp_pose_start + 7 > state_dim:
            raise ValueError("tcp_pose_start does not fit agent_pos.")
        base_shape_meta = shape_meta
        if isinstance(shape_meta, dict):
            # SimpleDP3's legacy constructor distinguishes OmegaConf nodes from
            # nested Python dictionaries when extracting leaf shapes.
            from omegaconf import OmegaConf

            base_shape_meta = OmegaConf.create(shape_meta)

        # SimpleDP3 builds the unchanged global UNet with an output feature
        # contract of encoder_output_dim + its fixed 64-D state MLP.  Replacing
        # the temporary encoder after construction removes it from the module
        # tree while keeping the exact global-condition dimension.
        super().__init__(
            shape_meta=base_shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=int(horizon),
            n_action_steps=int(n_action_steps),
            n_obs_steps=int(n_obs_steps),
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=int(diffusion_step_embed_dim),
            down_dims=down_dims,
            kernel_size=int(kernel_size),
            n_groups=int(n_groups),
            condition_type=condition_type,
            use_down_condition=bool(use_down_condition),
            use_mid_condition=bool(use_mid_condition),
            use_up_condition=bool(use_up_condition),
            encoder_output_dim=int(encoder_output_dim),
            crop_shape=crop_shape,
            use_pc_color=bool(use_pc_color),
            pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            **kwargs,
        )
        self.aqr = settings
        self.state_dim = state_dim
        self.point_channels = point_channels
        self.point_color_dim = point_color_dim
        self._profile_timing = False
        self._evaluation_tool_diagnostics = False
        self._evaluation_mechanism_diagnostics = False
        self._disable_post_refiner_for_evaluation = False
        self._strict_dp3_off = settings.strict_dp3_off
        self.query_stem: HighResolutionQueryStem | None = None
        self.relative_query: RelativeMultiScaleQuery | None = None
        self.local_residual: SlotWiseLocalResidual | None = None
        self.post_refiner: PostDiffusionActionRefiner | None = None
        self.action_to_tool: ActionToTool | None = None
        self.future_auxiliary: ActionConditionedFutureAuxiliary | None = None
        self.future_target_encoder: CachedPointNeXtEncoder | None = None
        self._needs_local_query = bool(settings.post_refiner_uses_local_query)

        # ── Layer 4: mutable curriculum state (updated by trainer each epoch) ──
        self._offset_target_prob: float = float(settings.offset_prob)
        self._offset_target_max_action_norm: float = float(
            settings.offset_max_action_norm
        )
        self._relation_gain_target: float = float(settings.relation_gain_scale)
        self._offset_prob: float = self._offset_target_prob
        self._offset_max_action_norm: float = (
            self._offset_target_max_action_norm
        )
        self._offset_training_enabled: bool = bool(
            settings.offset_training_enabled
        )
        if (
            self._offset_training_enabled
            and self.noise_scheduler.config.prediction_type != "sample"
        ):
            raise ValueError(
                "A3 offset correction requires prediction_type='sample'."
            )
        if (
            settings.bounded_action_residual
            and self.noise_scheduler.config.prediction_type != "sample"
        ):
            raise ValueError(
                "Bounded action residuals require prediction_type='sample' "
                "so the safety envelope is applied in clean-action space."
            )
        if (
            settings.improvement_loss_enabled
            and self.noise_scheduler.config.prediction_type != "sample"
        ):
            raise ValueError(
                "Action improvement loss requires prediction_type='sample' "
                "so base, corrected, and expert actions share clean-action space."
            )

        if self._strict_dp3_off:
            # Preserve the exact original SimpleDP3 module tree and execution
            # path.  In particular, do not replace its encoder or delete its
            # scheduler/mask-generator state.
            self.exec_steps = int(self.n_action_steps)
            return

        # The AQR mainline uses action-only global conditioning, so the legacy
        # point-cloud diffusion scheduler and inpainting mask are dead state.
        if hasattr(self, "noise_scheduler_pc"):
            del self.noise_scheduler_pc
        if hasattr(self, "mask_generator"):
            del self.mask_generator
        self.obs_encoder = CachedPointNeXtEncoder(
            state_dim=state_dim,
            obs_steps=int(n_obs_steps),
            color_dim=self.point_color_dim,
            token_dim=int(encoder_output_dim),
            state_feature_dim=64,
        )
        if self.obs_encoder.observation_feature_dim != self.obs_feature_dim:
            raise RuntimeError(
                "Cached PointNeXt and the unchanged DP3 UNet disagree on the "
                "global-condition dimension."
            )
        self.point_frontend = PointCloudFrontEnd(settings.front_end)

        execution_slots = execution_slot_indices(
            horizon=int(horizon),
            n_obs_steps=int(n_obs_steps),
            n_action_steps=int(n_action_steps),
        )
        query_slots = tuple(int(value) for value in settings.local_action_slots)
        expected_query_slots = (
            tuple(int(value) for value in execution_slots)
            if settings.refinement_stage == "post_diffusion"
            else (int(execution_slots[0]),)
        )
        if len(execution_slots) == 0 or query_slots != expected_query_slots:
            raise ValueError(
                "AQR local geometry slots disagree with the refinement stage; "
                f"expected {expected_query_slots}, got {query_slots}."
            )
        self.register_buffer(
            "_query_slot_indices",
            torch.as_tensor(query_slots, dtype=torch.long),
            persistent=False,
        )
        self.exec_steps = int(settings.exec_steps)
        if self.exec_steps > len(execution_slots):
            raise ValueError("exec_steps exceeds the configured action window.")

        candidate_lift = None
        needs_candidate_trajectory = self._needs_local_query and not (
            settings.refinement_stage == "post_diffusion"
            and settings.post_refiner_input_mode == "current_tcp_local"
        )
        if (
            needs_candidate_trajectory
            and settings.action_to_tool_mode == "joint_delta_fk"
        ):
            if not str(settings.action_contract_path).strip():
                raise ValueError(
                    "Joint-space AQR requires action_contract_path from "
                    "the validated P0 Action-to-Tool stage."
                )
            contract_path = Path(settings.action_contract_path).expanduser()
            if not contract_path.is_file():
                raise FileNotFoundError(
                    f"AQR action contract does not exist: {contract_path}"
                )
            candidate_lift = build_candidate_trajectory_lift(
                load_action_contract(contract_path),
                urdf_path=settings.urdf_path,
            )
            if candidate_lift.action_dim != self.action_dim:
                raise ValueError(
                    "Action contract and policy action dimensions disagree: "
                    f"{candidate_lift.action_dim} != {self.action_dim}."
                )

        if settings.cache_query_features and self._needs_local_query:
            self.query_stem = HighResolutionQueryStem(
                semantic_dim=int(encoder_output_dim),
                color_dim=self.point_color_dim,
                output_dim=int(settings.query_feature_dim),
            )
        if self._needs_local_query:
            assert self.query_stem is not None
            self.action_to_tool = ActionToTool(
                mode=settings.action_to_tool_mode,
                horizon=len(query_slots),
                state_dim=state_dim,
                tcp_pose_start=int(settings.tcp_pose_start),
                arm_dim=int(settings.arm_dim),
                candidate_lift=candidate_lift,
                keypoint_offsets_tcp=settings.keypoint_offsets_tcp,
            )
            self.relative_query = RelativeMultiScaleQuery(
                query_feature_dim=int(settings.query_feature_dim),
                output_dim=int(settings.local_dim),
                hidden_dim=int(settings.local_dim),
                radii_m=settings.query_radii_m,
                neighbors=settings.query_neighbors,
                geometry_mode=settings.geometry_mode,
                # ── Layer 1+2 options ──
                relation_features=bool(settings.relation_features),
                relation_gate_initial_bias=float(
                    settings.relation_gate_initial_bias
                ),
                relation_fusion_mode=str(settings.relation_fusion_mode),
                relation_gate_maximum=float(settings.relation_gate_maximum),
                relation_attention=bool(settings.relation_attention),
                relation_scale_gate=bool(settings.relation_scale_gate),
                relation_scale_gate_geo_bias=bool(settings.relation_scale_gate_geo_bias),
                relation_query_dim=int(settings.relation_query_dim),
                relation_key_dim=int(settings.relation_key_dim),
                anchor_preserving_query=bool(settings.anchor_preserving_query),
                action_dim=self.action_dim,
                surface_normal_k=int(settings.surface_normal_k),
                surface_normal_confidence_threshold=float(
                    settings.surface_normal_confidence_threshold
                ),
            )
            # Keep the unused legacy residual module only for exact loading of
            # existing Full-A6 checkpoints. New structural post ablations do
            # not carry a dead denoising-time head.
            if (
                settings.refinement_stage == "denoising"
                or settings.post_refiner_input_mode == "full"
            ):
                self.local_residual = SlotWiseLocalResidual(
                    action_dim=self.action_dim,
                    local_dim=int(settings.local_dim),
                    global_dim=self.obs_encoder.global_condition_dim,
                    hidden_dim=int(settings.local_hidden_dim),
                    relation_gain_scale=float(settings.relation_gain_scale),
                )

        if settings.refinement_stage == "post_diffusion":
            self.post_refiner = PostDiffusionActionRefiner(
                action_dim=self.action_dim,
                local_dim=int(settings.local_dim),
                global_dim=self.obs_encoder.global_condition_dim,
                hidden_dim=int(settings.local_hidden_dim),
                confidence_initial_bias=float(
                    settings.post_confidence_initial_bias
                ),
                input_mode=str(settings.post_refiner_input_mode),
            )
            if settings.post_base_frozen:
                # All three ablations share one immutable DP3-H12 reference.
                self.model.requires_grad_(False)
                self.obs_encoder.requires_grad_(False)
                if self.local_residual is not None:
                    self.local_residual.requires_grad_(False)

        if settings.future_aux:
            if not settings.enabled:
                raise ValueError(
                    "future_aux is an AQR increment and cannot be enabled on a "
                    "disabled baseline."
                )
            self.future_auxiliary = ActionConditionedFutureAuxiliary(
                latent_dim=int(encoder_output_dim),
                action_dim=self.action_dim,
                horizons=settings.future_horizons,
            )
            self.future_target_encoder = copy.deepcopy(self.obs_encoder)
            self.future_target_encoder.requires_grad_(False)
            self.future_target_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if (
            getattr(self, "aqr", None) is not None
            and self.aqr.refinement_stage == "post_diffusion"
            and self.aqr.post_base_frozen
        ):
            self.model.eval()
            self.obs_encoder.eval()
            if self.local_residual is not None:
                self.local_residual.eval()
        if self.future_target_encoder is not None:
            self.future_target_encoder.eval()
        return self

    @staticmethod
    def _base_dp3_observation(
        obs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: obs[key]
            for key in ("point_cloud", "agent_pos")
            if key in obs
        }

    # ------------------------------------------------------------------
    # Observation and point-cache construction
    # ------------------------------------------------------------------
    def _validate_observation(
        self, obs: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if "point_cloud" not in obs or "agent_pos" not in obs:
            raise KeyError("AQR-DP3 observations need point_cloud and agent_pos.")
        points = obs["point_cloud"].to(device=self.device, non_blocking=True).to(device=self.device, non_blocking=True)
        state = obs["agent_pos"].to(device=self.device, non_blocking=True).to(device=self.device, non_blocking=True)
        if points.ndim != 4 or points.shape[-1] != self.point_channels:
            raise ValueError(
                "point_cloud must have shape "
                f"[B,T,N,{self.point_channels}]."
            )
        if state.ndim != 3 or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"agent_pos must have shape [B,T,{self.state_dim}]."
            )
        if points.shape[:2] != state.shape[:2]:
            raise ValueError("point and state observation histories must align.")
        if points.shape[1] < self.n_obs_steps:
            raise ValueError(
                f"AQR-DP3 needs at least {self.n_obs_steps} observation frames."
            )
        if points.shape[2] < self.aqr.front_end.p_query:
            raise ValueError("raw point bank is smaller than configured P_query.")
        valid_value = obs.get("point_valid_mask")
        if valid_value is None:
            valid = torch.isfinite(points).all(dim=-1)
        else:
            valid = valid_value
            if valid.ndim == 4 and valid.shape[-1] == 1:
                valid = valid[..., 0]
            if tuple(valid.shape) != tuple(points.shape[:3]):
                raise ValueError(
                    "point_valid_mask must have shape [B,T,N] (or trailing 1)."
                )
            valid = valid.to(device=points.device, dtype=torch.bool)
            valid = valid & torch.isfinite(points).all(dim=-1)
        robot_value = obs.get("robot_mask")
        if not self.aqr.use_robot_mask_feature:
            # Robot segmentation is privileged in ManiSkill.  The deployable
            # default therefore ignores any supplied mask and uses an all-zero
            # feature.  A real sensor/kinematics-derived mask may be enabled
            # explicitly once its deployment path has been validated.
            robot = torch.zeros_like(valid)
        elif robot_value is None:
            if self.aqr.front_end.protected_sampling or self.aqr.enabled:
                raise KeyError(
                    "use_robot_mask_feature=true requires a robot_mask produced "
                    "by a validated deployable source; simulator segmentation "
                    "must not be used as a deployment input."
                )
            robot = torch.zeros_like(valid)
        else:
            robot = robot_value
            if robot.ndim == 4 and robot.shape[-1] == 1:
                robot = robot[..., 0]
            if tuple(robot.shape) != tuple(points.shape[:3]):
                raise ValueError(
                    "robot_mask must have shape [B,T,N] (or trailing 1)."
                )
            robot = robot.to(device=points.device, dtype=torch.bool) & valid
        return points, state, valid, robot

    def _sample_key(
        self, obs: Mapping[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> int | torch.Tensor:
        value = obs.get("sample_key")
        if value is None:
            return 0
        if not torch.is_tensor(value):
            return int(value)
        value = value.to(device=device, dtype=torch.long)
        if value.ndim > 1:
            value = value.reshape(batch_size, -1)
            current = min(self.n_obs_steps - 1, value.shape[1] - 1)
            value = value[:, current]
        return value

    def _history_ending_at(
        self, values: torch.Tensor, index: int
    ) -> torch.Tensor:
        start = max(0, int(index) - self.n_obs_steps + 1)
        history = values[:, start : int(index) + 1]
        if history.shape[1] < self.n_obs_steps:
            padding = history[:, :1].expand(
                -1, self.n_obs_steps - history.shape[1], *history.shape[2:]
            )
            history = torch.cat([padding, history], dim=1)
        return history

    def _sync_timing(self, device: torch.device) -> None:
        # Training avoids synchronization stalls. In evaluation the returned
        # latency decomposition must describe completed CUDA work, not only
        # kernel-enqueue time.
        if self._profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _encode_frame(
        self,
        *,
        points: torch.Tensor,
        valid: torch.Tensor,
        robot: torch.Tensor,
        state_history_raw: torch.Tensor,
        sample_key: int | torch.Tensor,
        encoder: CachedPointNeXtEncoder,
        build_query_features: bool,
    ) -> tuple[
        PointCloudBatch,
        PointFeatureCache,
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        dtype = points.dtype
        device = points.device
        self._sync_timing(device)
        start = time.perf_counter()
        tcp_history = state_history_raw[
            ...,
            self.aqr.tcp_pose_start : self.aqr.tcp_pose_start + 7,
        ]
        banks = self.point_frontend(
            points,
            tcp_history,
            raw_valid_mask=valid,
            raw_robot_mask=robot,
            sample_key=sample_key,
        )
        if not bool(banks.enc_valid_mask.all()):
            minimum = int(banks.enc_valid_mask.sum(dim=1).min().item())
            raise ValueError(
                "PointNeXt requires a full 2048-point valid encoder bank; "
                f"smallest batch item has {minimum}. Fix/crop the dense source "
                "before training rather than treating padding as geometry."
            )
        self._sync_timing(device)
        after_frontend = time.perf_counter()
        enc_metric = banks.enc_points[..., :3]
        # The point normalizer is channel-wise. Normalize the complete row so
        # its fitted XYZRGB shape remains valid, then expose only XYZ to metric
        # geometry. Raw RGB stays in [0,1] and enters solely as a feature.
        enc_normalized = self.normalizer["point_cloud"].normalize(
            banks.enc_points
        )[..., :3]
        enc_color = (
            banks.enc_points[..., 3:6]
            if self.point_color_dim == 3
            else None
        )
        state_normalized = self.normalizer["agent_pos"].normalize(
            state_history_raw
        )
        point_cache = encoder(
            enc_xyz_metric=enc_metric,
            enc_xyz_normalized=enc_normalized,
            enc_color=enc_color,
            enc_valid_mask=banks.enc_valid_mask,
            enc_robot_mask=banks.enc_robot_mask,
            state_history_normalized=state_normalized,
        )
        self._sync_timing(device)
        after_encoder = time.perf_counter()
        query_features = None
        if build_query_features:
            if self.query_stem is None:
                raise RuntimeError("query feature caching was not constructed.")
            query_metric = banks.query_points[..., :3]
            query_normalized = self.normalizer["point_cloud"].normalize(
                banks.query_points
            )[..., :3]
            query_color = (
                banks.query_points[..., 3:6]
                if self.point_color_dim == 3
                else None
            )
            query_cache = point_cache
            if self.aqr.refinement_stage == "post_diffusion":
                # Keep post-refiner gradients out of the global observation
                # encoder. An unfrozen long-horizon base is trained only by its
                # pure diffusion objective, never by the correction head.
                query_cache = PointFeatureCache(
                    **{
                        name: getattr(point_cache, name).detach()
                        for name in PointFeatureCache.__dataclass_fields__
                    }
                )
            query_features = self.query_stem(
                query_xyz_metric=query_metric,
                query_xyz_normalized=query_normalized,
                query_color=query_color,
                query_valid_mask=banks.query_valid_mask,
                query_robot_mask=banks.query_robot_mask,
                cache=query_cache,
            )
        self._sync_timing(device)
        after_query = time.perf_counter()
        timing = {
            "point_frontend_ms": _scalar_tensor(
                1000.0 * (after_frontend - start), device=device, dtype=dtype
            ),
            "pointnext_ms": _scalar_tensor(
                1000.0 * (after_encoder - after_frontend),
                device=device,
                dtype=dtype,
            ),
            "query_stem_ms": _scalar_tensor(
                1000.0 * (after_query - after_encoder),
                device=device,
                dtype=dtype,
            ),
        }
        return banks, point_cache, query_features, timing

    def _encode_observation(
        self, obs: Mapping[str, torch.Tensor]
    ) -> EncodedAQRObservation:
        points, state, valid, robot = self._validate_observation(obs)
        current_index = self.n_obs_steps - 1
        state_history = state[:, : self.n_obs_steps]
        banks, cache, query_features, timing = self._encode_frame(
            points=points[:, current_index],
            valid=valid[:, current_index],
            robot=robot[:, current_index],
            state_history_raw=state_history,
            sample_key=self._sample_key(obs, points.shape[0], points.device),
            encoder=self.obs_encoder,
            build_query_features=(
                self.aqr.cache_query_features and self._needs_local_query
            ),
        )
        return EncodedAQRObservation(
            raw_points=points,
            raw_valid_mask=valid,
            raw_robot_mask=robot,
            state_history_raw=state_history,
            state_history_normalized=self.normalizer["agent_pos"].normalize(
                state_history
            ),
            point_banks=banks,
            point_cache=cache,
            query_features=query_features,
            timing_ms=timing,
        )

    # ------------------------------------------------------------------
    # Action-conditioned relative query
    # ------------------------------------------------------------------
    def _expert_query_actions(
        self,
        expert_actions_raw: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if expert_actions_raw is None:
            raise ValueError(
                "query_source=expert is evaluation-only and requires "
                "obs['expert_action'] in raw environment units."
            )
        expert = expert_actions_raw.to(device=device, dtype=dtype)
        if expert.ndim != 3 or expert.shape[0] != batch_size:
            raise ValueError("expert_action must have shape [B,S,A] or [B,H,A].")
        if expert.shape[-1] != self.action_dim:
            raise ValueError("expert_action has the wrong action dimension.")
        if expert.shape[1] == self.horizon:
            expert = expert.index_select(1, self._query_slot_indices)
        elif expert.shape[1] != self._query_slot_indices.numel():
            raise ValueError("expert_action does not cover the AQR query slots.")
        return self.normalizer["action"].normalize(expert)

    def _query_actions(
        self,
        *,
        noisy_action: torch.Tensor,
        global_model_output: torch.Tensor,
        timestep: torch.Tensor | int,
        expert_actions_raw: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        source = self.aqr.query_source
        clean = predicted_clean_sample(
            sample=noisy_action,
            model_output=global_model_output,
            timestep=timestep,
            scheduler=self.noise_scheduler,
        )
        if source == "static":
            return None, {
                "a0_min": clean.amin(dim=(1, 2)),
                "a0_max": clean.amax(dim=(1, 2)),
                "a0_boundary_fraction": clean.abs().ge(0.999).to(clean.dtype).mean(
                    dim=(1, 2)
                ),
            }
        if source == "noisy":
            query_actions = noisy_action
        elif source in {"predicted_clean", "wrong"}:
            query_actions = clean
        elif source == "expert":
            expert = self._expert_query_actions(
                expert_actions_raw,
                batch_size=noisy_action.shape[0],
                dtype=noisy_action.dtype,
                device=noisy_action.device,
            )
            return expert, {
                "a0_min": clean.amin(dim=(1, 2)),
                "a0_max": clean.amax(dim=(1, 2)),
                "a0_boundary_fraction": clean.abs().ge(0.999).to(clean.dtype).mean(
                    dim=(1, 2)
                ),
            }
        else:
            raise RuntimeError(f"Unexpected enabled query source {source!r}.")
        query_actions = query_actions.index_select(1, self._query_slot_indices)
        return query_actions, {
            "a0_min": clean.amin(dim=(1, 2)),
            "a0_max": clean.amax(dim=(1, 2)),
            "a0_boundary_fraction": clean.abs().ge(0.999).to(clean.dtype).mean(
                dim=(1, 2)
            ),
        }

    def _apply_intervention(self, tokens: torch.Tensor) -> torch.Tensor:
        intervention = self.aqr.local_intervention
        if intervention == "normal":
            return tokens
        if intervention == "shuffle_slots":
            if tokens.shape[1] < 2:
                raise ValueError(
                    "shuffle_slots needs at least two AQR action slots."
                )
            return torch.roll(tokens, shifts=1, dims=1)
        if intervention == "zero":
            return torch.zeros_like(tokens)
        if intervention == "dropout":
            keep = torch.rand(
                tokens.shape[:2],
                device=tokens.device,
                dtype=tokens.dtype,
            ) >= float(self.aqr.local_dropout_probability)
            return tokens * keep.unsqueeze(-1)
        raise RuntimeError(f"Unexpected local intervention {intervention!r}.")

    def _tool_from_normalized_query_actions(
        self,
        normalized_actions: torch.Tensor,
        encoded: EncodedAQRObservation,
    ) -> tuple[ToolTrajectory, torch.Tensor, torch.Tensor]:
        """Lift one normalized full-horizon action to the queried tool slot.

        This helper is used by inference diagnostics to compare the global
        predicted-clean candidate with the immediately AQR-corrected candidate
        at the same diffusion timestep.  It does not change the sampled action
        or any model input.
        """

        if self.action_to_tool is None:
            raise RuntimeError("Action-to-Tool diagnostics require enabled AQR.")
        selected = normalized_actions.index_select(
            1, self._query_slot_indices
        )
        environment_actions = self.normalizer["action"].unnormalize(selected)
        trajectory = self.action_to_tool(
            policy_actions=selected,
            environment_actions=environment_actions,
            state_history_raw=encoded.state_history_raw,
        )
        return trajectory, selected, environment_actions

    def _workspace_outside_fraction(
        self, trajectory: ToolTrajectory, encoded: EncodedAQRObservation
    ) -> torch.Tensor:
        points = encoded.point_banks.query_points[..., :3]
        valid = encoded.point_banks.query_valid_mask
        positive = torch.finfo(points.dtype).max
        negative = torch.finfo(points.dtype).min
        lower = points.masked_fill(~valid.unsqueeze(-1), positive).amin(dim=1)
        upper = points.masked_fill(~valid.unsqueeze(-1), negative).amax(dim=1)
        has_points = valid.any(dim=1)
        margin = max(self.aqr.query_radii_m)
        tool = trajectory.keypoints_world
        outside = ((tool < (lower[:, None, None] - margin)) | (
            tool > (upper[:, None, None] + margin)
        )).any(dim=-1)
        outside = torch.where(
            has_points[:, None, None], outside, torch.ones_like(outside)
        )
        return outside.to(points.dtype).mean(dim=(1, 2))

    def _build_local_query(
        self,
        *,
        noisy_action: torch.Tensor,
        global_model_output: torch.Tensor,
        timestep: torch.Tensor | int,
        encoded: EncodedAQRObservation,
        expert_actions_raw: torch.Tensor | None = None,
        clean_actions_override: torch.Tensor | None = None,
    ) -> LocalQueryResult:
        if (
            not self.aqr.enabled
            or self.action_to_tool is None
            or self.relative_query is None
            or encoded.query_features is None
        ):
            raise RuntimeError("AQR local query was requested in a disabled stage.")
        self._sync_timing(noisy_action.device)
        start = time.perf_counter()
        current_tcp_only = (
            self.aqr.refinement_stage == "post_diffusion"
            and self.aqr.post_refiner_input_mode == "current_tcp_local"
        )

        if clean_actions_override is not None:
            # ── Offset training path: skip _query_actions, use provided actions ──
            query_actions = clean_actions_override.index_select(
                1, self._query_slot_indices
            )
            action_diagnostics: dict[str, torch.Tensor] = {}
            candidate_trajectory = None
            if not current_tcp_only:
                environment_actions = self.normalizer["action"].unnormalize(
                    query_actions
                )
                candidate_trajectory = self.action_to_tool(
                    policy_actions=query_actions,
                    environment_actions=environment_actions,
                    state_history_raw=encoded.state_history_raw,
                )
        else:
            # ── Original path ──
            query_actions, action_diagnostics = self._query_actions(
                noisy_action=noisy_action,
                global_model_output=global_model_output,
                timestep=timestep,
                expert_actions_raw=expert_actions_raw,
            )
            if query_actions is None and not current_tcp_only:
                candidate_trajectory = current_trajectory = self.action_to_tool.static(
                    encoded.state_history_raw
                )
            elif current_tcp_only:
                candidate_trajectory = None
            else:
                environment_actions = self.normalizer["action"].unnormalize(
                    query_actions
                )
                candidate_trajectory = self.action_to_tool(
                    policy_actions=query_actions,
                    environment_actions=environment_actions,
                    state_history_raw=encoded.state_history_raw,
                )

        current_trajectory = self.action_to_tool.static(
            encoded.state_history_raw
        )
        if self.aqr.query_source == "wrong" and candidate_trajectory is not None:
            candidate_trajectory = offset_tool_trajectory(
                candidate_trajectory, self.aqr.wrong_query_offset_m
            )
        if current_tcp_only:
            trajectory = build_current_tcp_only_trajectory(
                current_trajectory,
                horizon=int(self._query_slot_indices.numel()),
            )
        else:
            assert candidate_trajectory is not None
            trajectory = build_slot1_sweep_trajectory(
                current=current_trajectory,
                candidate_slot1=candidate_trajectory,
                sweep_enabled=self.aqr.slot1_sweep_enabled,
                current_tcp_fallback_enabled=(
                    self.aqr.current_tcp_fallback_enabled
                ),
                sweep_samples=self.aqr.slot1_sweep_samples,
            )
        self._sync_timing(noisy_action.device)
        after_trajectory = time.perf_counter()

        # ── Layer 2: build action-conditioned query params ──
        action_slot: torch.Tensor | None = None
        tool_pose_slot: torch.Tensor | None = None
        action_direction: torch.Tensor | None = None
        if (
            not current_tcp_only
            and query_actions is not None
            and candidate_trajectory is not None
        ):
            tcp_pos = candidate_trajectory.tcp_matrices_world[:, :, :3, 3]
            current_tcp = current_trajectory.tcp_matrices_world[:, :1, :3, 3]
            segment_start_tcp = torch.cat(
                [current_tcp, tcp_pos[:, :-1]], dim=1
            )
            action_direction = tcp_pos - segment_start_tcp
            if self.aqr.relation_attention:
                if query_actions.shape[1] != 1:
                    raise RuntimeError(
                        "relation_attention has not been migrated to the "
                        "multi-slot post-diffusion path."
                    )
                action_slot = query_actions[:, 0]
                tcp_z = candidate_trajectory.tcp_matrices_world[:, 0, :3, 2]
                tool_pose_slot = torch.cat([tcp_pos[:, 0], tcp_z], dim=-1)

        slot_tokens, query_diagnostics = self.relative_query(
            query_xyz=encoded.point_banks.query_points[..., :3],
            query_features=encoded.query_features,
            query_valid_mask=encoded.point_banks.query_valid_mask,
            query_robot_mask=encoded.point_banks.query_robot_mask,
            trajectory=trajectory,
            action_slot=action_slot,
            tool_pose_slot=tool_pose_slot,
            timestep=(
                timestep.to(noisy_action.device)
                if torch.is_tensor(timestep)
                else torch.tensor(
                    [int(timestep)], device=noisy_action.device
                ).expand(noisy_action.shape[0])
            ),
            action_direction=action_direction,
        )
        slot_tokens = self._apply_intervention(slot_tokens)
        full_tokens = torch.zeros(
            noisy_action.shape[0],
            self.horizon,
            slot_tokens.shape[-1],
            dtype=slot_tokens.dtype,
            device=slot_tokens.device,
        )
        full_tokens.index_copy_(1, self._query_slot_indices, slot_tokens)
        self._sync_timing(noisy_action.device)
        after_query = time.perf_counter()
        diagnostics: dict[str, torch.Tensor] = {
            **action_diagnostics,
            "valid_neighbor_count": query_diagnostics.valid_neighbor_count,
            "empty_fraction": query_diagnostics.empty_fraction,
            "robot_point_fraction": query_diagnostics.robot_point_fraction,
            "nearest_distance_m": query_diagnostics.nearest_distance_m,
            "tool_outside_fraction": self._workspace_outside_fraction(
                trajectory, encoded
            ),
            "tool_keypoints_world": trajectory.keypoints_world,
            "local_token_norm": full_tokens.norm(dim=-1),
            "slot1_query_anchor_count": torch.tensor(
                trajectory.keypoints_world.shape[2],
                device=noisy_action.device,
                dtype=torch.long,
            ),
            "slot1_sweep_enabled": torch.tensor(
                self.aqr.slot1_sweep_enabled,
                device=noisy_action.device,
                dtype=torch.bool,
            ),
            "current_tcp_fallback_enabled": torch.tensor(
                self.aqr.current_tcp_fallback_enabled,
                device=noisy_action.device,
                dtype=torch.bool,
            ),
            "trajectory_ms": _scalar_tensor(
                1000.0 * (after_trajectory - start),
                device=noisy_action.device,
                dtype=noisy_action.dtype,
            ),
            "relative_query_ms": _scalar_tensor(
                1000.0 * (after_query - after_trajectory),
                device=noisy_action.device,
                dtype=noisy_action.dtype,
            ),
            # ── Layer 3: relation stats for gain modulation ──
            "relation_stats": {
                "r_face_neg_fraction": (
                    query_diagnostics.r_face_neg_fraction.mean(dim=-1)
                    if query_diagnostics.r_face_neg_fraction is not None
                    else torch.zeros(
                        noisy_action.shape[0],
                        device=noisy_action.device,
                        dtype=noisy_action.dtype,
                    )
                ),
                "nearest_abnormal": (
                    (query_diagnostics.nearest_distance_m > 0.05)
                    .to(noisy_action.dtype)
                    if hasattr(query_diagnostics, "nearest_distance_m")
                    and query_diagnostics.nearest_distance_m is not None
                    else torch.zeros(
                        noisy_action.shape[0],
                        device=noisy_action.device,
                        dtype=noisy_action.dtype,
                    )
                ),
            },
        }
        optional_relation_diagnostics = {
            "r_face_mean": query_diagnostics.r_face_mean,
            "r_face_neg_fraction": query_diagnostics.r_face_neg_fraction,
            "r_motion_mean": query_diagnostics.r_motion_mean,
            "normal_confidence_mean": query_diagnostics.normal_confidence_mean,
            "relation_gate_mean": query_diagnostics.relation_gate_mean,
            "attn_entropy": query_diagnostics.attn_entropy,
            "scale_gate_weights": query_diagnostics.scale_gate_weights,
            "anchor_attention_weights": (
                query_diagnostics.anchor_attention_weights
            ),
            "anchor_role_ids": query_diagnostics.anchor_role_ids,
            "anchor_phase_ids": query_diagnostics.anchor_phase_ids,
            "anchor_sweep_ids": query_diagnostics.anchor_sweep_ids,
        }
        diagnostics.update(
            {
                name: value
                for name, value in optional_relation_diagnostics.items()
                if value is not None
            }
        )
        if query_diagnostics.anchor_attention_weights is not None:
            anchor_weights = query_diagnostics.anchor_attention_weights.mean(
                dim=-1
            )
            role_ids = query_diagnostics.anchor_role_ids
            phase_ids = query_diagnostics.anchor_phase_ids
            assert role_ids is not None and phase_ids is not None

            def anchor_weight(mask: torch.Tensor) -> torch.Tensor:
                return (
                    anchor_weights
                    * mask.to(anchor_weights.dtype)[None, None, None]
                ).sum(dim=-1).mean(dim=(1, 2))

            diagnostics.update(
                {
                    "anchor_current_weight": anchor_weight(
                        (phase_ids == 1) | (phase_ids == 3)
                    ),
                    "anchor_candidate_weight": anchor_weight(phase_ids == 2),
                    "anchor_left_weight": anchor_weight(role_ids == 2),
                    "anchor_right_weight": anchor_weight(role_ids == 3),
                }
            )
        if self._evaluation_tool_diagnostics:
            diagnostics.update(
                {
                    "current_tool_keypoints_world": (
                        current_trajectory.keypoints_world
                    ),
                    "current_tool_rotations_world": (
                        current_trajectory.rotations_world_from_tool
                    ),
                    "query_source_indices": (
                        encoded.point_banks.query_source_indices
                    ),
                }
            )
            if candidate_trajectory is not None:
                diagnostics.update(
                    {
                        "base_candidate_tool_keypoints_world": (
                            candidate_trajectory.keypoints_world
                        ),
                        "base_candidate_tool_rotations_world": (
                            candidate_trajectory.rotations_world_from_tool
                        ),
                    }
                )
        return LocalQueryResult(
            tokens=full_tokens,
            diagnostics=diagnostics,
            trajectory=trajectory,
        )

    def _apply_local_residual(
        self,
        *,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor | int,
        global_condition: torch.Tensor,
        global_output: torch.Tensor,
        local_tokens: torch.Tensor,
        relation_stats: dict[str, torch.Tensor] | None = None,
        residual_reference_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.local_residual is None:
            return global_output, {}
        delta, diagnostics = self.local_residual(
            noisy_action=noisy_action,
            timestep=timestep,
            global_condition=global_condition,
            local_tokens=local_tokens,
            relation_stats=relation_stats,
        )
        slot_mask = torch.zeros(
            (1, self.horizon, 1),
            dtype=delta.dtype,
            device=delta.device,
        )
        slot_mask[:, self._query_slot_indices] = 1.0
        unmasked_delta = delta
        delta = delta * slot_mask
        bounded_diagnostics: dict[str, torch.Tensor] = {}
        if self.aqr.bounded_action_residual:
            reference = (
                global_output
                if residual_reference_output is None
                else residual_reference_output
            )
            delta, bounded_diagnostics = bound_action_residual(
                delta,
                reference_action=reference,
                arm_dim=int(self.aqr.arm_dim),
                maximum_norm=float(self.aqr.action_residual_max_norm),
                minimum_progress_ratio=float(
                    self.aqr.action_residual_min_progress_ratio
                ),
                protect_non_arm=bool(
                    self.aqr.action_residual_protect_non_arm
                ),
            )
        progress = denoising_progress(
            timestep,
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            batch_size=noisy_action.shape[0],
            device=noisy_action.device,
            dtype=noisy_action.dtype,
        )
        gate = query_gate(
            progress,
            start_fraction=self.aqr.query_start_fraction,
            maximum=self.aqr.query_gate_maximum,
        )
        output = global_output + gate[:, None, None] * delta
        return output, {
            **diagnostics,
            **bounded_diagnostics,
            "unmasked_delta_norm": unmasked_delta.norm(dim=-1),
            "delta_norm": delta.norm(dim=-1),
            "local_residual_slot_mask": slot_mask[0, :, 0].to(torch.bool),
            "query_gate": gate,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _conditional_sample_global_only(
        self,
        *,
        shape: tuple[int, int, int],
        global_condition: torch.Tensor,
        generator: torch.Generator | None = None,
        **scheduler_step_kwargs,
    ) -> torch.Tensor:
        """Run the unchanged global DP3 sampler without any AQR intervention."""

        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            shape,
            dtype=global_condition.dtype,
            device=global_condition.device,
            generator=generator,
        )
        scheduler.set_timesteps(self.num_inference_steps)
        for timestep in scheduler.timesteps:
            model_output = self.model(
                sample=trajectory,
                timestep=timestep,
                local_cond=None,
                global_cond=global_condition,
            )
            trajectory = scheduler.step(
                model_output,
                timestep,
                trajectory,
                **scheduler_step_kwargs,
            ).prev_sample
        return trajectory

    def _post_diffusion_refine(
        self,
        *,
        base_action: torch.Tensor,
        encoded: EncodedAQRObservation,
        query_action_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Query and correct a completed normalized action exactly once.

        ``base_action`` is always the proposal being corrected.  The optional
        ``query_action_override`` changes only the trajectory used to gather
        local geometry.  Keeping these roles separate is required by the
        proposal-shuffle negative control: a shuffled query must not silently
        replace the action whose residual is predicted or executed.
        """

        if self.post_refiner is None:
            raise RuntimeError("post_diffusion refinement head was not constructed.")
        query_action = base_action
        query_override_used = query_action_override is not None
        if query_action_override is not None:
            if query_action_override.shape != base_action.shape:
                raise ValueError(
                    "query_action_override must have the same [B,H,A] shape "
                    "as base_action."
                )
            if query_action_override.device != base_action.device:
                raise ValueError(
                    "query_action_override and base_action must share a device."
                )
            if query_action_override.dtype != base_action.dtype:
                raise ValueError(
                    "query_action_override and base_action must share a dtype."
                )
            query_action = query_action_override
        if self.aqr.post_refiner_uses_local_query:
            zero_timestep = torch.zeros(
                base_action.shape[0], dtype=torch.long, device=base_action.device
            )
            query_result = self._build_local_query(
                noisy_action=query_action,
                global_model_output=query_action,
                timestep=zero_timestep,
                encoded=encoded,
                clean_actions_override=query_action,
            )
        else:
            query_result = LocalQueryResult(
                tokens=torch.zeros(
                    base_action.shape[0],
                    self.horizon,
                    int(self.aqr.local_dim),
                    dtype=base_action.dtype,
                    device=base_action.device,
                ),
                diagnostics={
                    "post_local_query_skipped": torch.ones(
                        base_action.shape[0],
                        dtype=torch.bool,
                        device=base_action.device,
                    )
                },
                trajectory=None,
            )
        raw_delta, confidence, refiner_diagnostics = self.post_refiner(
            base_action=base_action,
            global_condition=encoded.point_cache.global_condition.detach(),
            local_tokens=query_result.tokens,
        )
        slot_mask = torch.zeros(
            (1, self.horizon, 1),
            dtype=raw_delta.dtype,
            device=raw_delta.device,
        )
        slot_mask[:, self._query_slot_indices] = 1.0
        raw_delta = raw_delta * slot_mask
        confidence = confidence * slot_mask

        bounded_diagnostics: dict[str, torch.Tensor] = {}
        bounded_delta = raw_delta
        if self.aqr.bounded_action_residual:
            bounded_delta, bounded_diagnostics = bound_action_residual(
                raw_delta,
                reference_action=base_action,
                arm_dim=int(self.aqr.arm_dim),
                maximum_norm=float(self.aqr.action_residual_max_norm),
                minimum_progress_ratio=float(
                    self.aqr.action_residual_min_progress_ratio
                ),
                protect_non_arm=bool(
                    self.aqr.action_residual_protect_non_arm
                ),
            )
        applied_delta = confidence * bounded_delta
        corrected = base_action + applied_delta
        diagnostics = {
            **query_result.diagnostics,
            **refiner_diagnostics,
            **bounded_diagnostics,
            "post_bounded_residual_norm": bounded_delta.norm(dim=-1),
            "post_applied_residual_norm": applied_delta.norm(dim=-1),
            "post_applied_residual": applied_delta,
            "post_bounded_residual": bounded_delta,
            "post_refiner_slot_mask": slot_mask[0, :, 0].to(torch.bool),
            "post_query_action": query_action,
            "post_query_action_override_used": torch.full(
                (base_action.shape[0],),
                query_override_used,
                device=base_action.device,
                dtype=torch.bool,
            ),
            "post_refine_count": torch.ones(
                base_action.shape[0],
                device=base_action.device,
                dtype=torch.long,
            ),
        }
        if getattr(self, "_evaluation_mechanism_diagnostics", False):
            diagnostics["post_local_tokens"] = query_result.tokens
        if (
            self._evaluation_tool_diagnostics
            and self.action_to_tool is not None
            and self.aqr.post_refiner_input_mode != "current_tcp_local"
        ):
            (
                base_candidate,
                base_actions_normalized,
                base_actions_environment,
            ) = self._tool_from_normalized_query_actions(base_action, encoded)
            (
                corrected_candidate,
                corrected_actions_normalized,
                corrected_actions_environment,
            ) = self._tool_from_normalized_query_actions(corrected, encoded)
            keypoint_shift = (
                corrected_candidate.keypoints_world
                - base_candidate.keypoints_world
            )
            diagnostics.update(
                {
                    "base_candidate_actions_normalized": base_actions_normalized,
                    "base_candidate_actions_environment": base_actions_environment,
                    "aqr_corrected_actions_normalized": (
                        corrected_actions_normalized
                    ),
                    "aqr_corrected_actions_environment": (
                        corrected_actions_environment
                    ),
                    "aqr_corrected_tool_keypoints_world": (
                        corrected_candidate.keypoints_world
                    ),
                    "aqr_corrected_tool_rotations_world": (
                        corrected_candidate.rotations_world_from_tool
                    ),
                    "aqr_tool_keypoint_shift_world": keypoint_shift,
                    "aqr_tool_keypoint_shift_m": keypoint_shift.norm(dim=-1),
                }
            )
            if query_override_used:
                query_candidate, query_actions_normalized, query_actions_environment = (
                    self._tool_from_normalized_query_actions(query_action, encoded)
                )
                diagnostics.update(
                    {
                        "query_candidate_actions_normalized": (
                            query_actions_normalized
                        ),
                        "query_candidate_actions_environment": (
                            query_actions_environment
                        ),
                        "query_candidate_tool_keypoints_world": (
                            query_candidate.keypoints_world
                        ),
                        "query_candidate_tool_rotations_world": (
                            query_candidate.rotations_world_from_tool
                        ),
                    }
                )
        return corrected, diagnostics

    def _update_indices(self) -> set[int]:
        timesteps = self.noise_scheduler.timesteps
        progress = [
            float(
                1.0
                - float(timestep)
                / max(
                    1,
                    int(self.noise_scheduler.config.num_train_timesteps) - 1,
                )
            )
            for timestep in timesteps
        ]
        available = [
            index
            for index, value in enumerate(progress)
            if value >= float(self.aqr.query_start_fraction)
        ]
        if not available:
            raise RuntimeError(
                "Inference schedule has no timestep in the enabled AQR region."
            )
        targets = self.aqr.query_update_fractions
        if self.aqr.query_source in {"static", "expert"}:
            targets = targets[:1]
        selected: set[int] = set()
        for target in targets:
            candidates = [index for index in available if index not in selected]
            if not candidates:
                break
            selected.add(min(candidates, key=lambda index: abs(progress[index] - target)))
        if len(selected) != len(targets):
            raise ValueError(
                "num_inference_steps is too small to realize every configured "
                "AQR update without duplication."
            )
        return selected

    def conditional_sample_aqr(
        self,
        *,
        shape: tuple[int, int, int],
        global_condition: torch.Tensor,
        encoded: EncodedAQRObservation,
        expert_actions_raw: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **scheduler_step_kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.aqr.refinement_stage == "post_diffusion":
            base_action = self._conditional_sample_global_only(
                shape=shape,
                global_condition=global_condition,
                generator=generator,
                **scheduler_step_kwargs,
            )
            if self._disable_post_refiner_for_evaluation:
                diagnostics = {
                    **encoded.point_banks.diagnostics,
                    **encoded.timing_ms,
                    "aqr_update_count": torch.zeros(
                        shape[0], device=base_action.device, dtype=torch.long
                    ),
                    "post_refiner_disabled": torch.ones(
                        shape[0], device=base_action.device, dtype=torch.bool
                    ),
                    "post_base_action": base_action,
                }
                return base_action, diagnostics
            corrected, post_diagnostics = self._post_diffusion_refine(
                base_action=base_action,
                encoded=encoded,
            )
            diagnostics = {
                **encoded.point_banks.diagnostics,
                **encoded.timing_ms,
                **post_diagnostics,
                "aqr_update_count": torch.ones(
                    shape[0], device=base_action.device, dtype=torch.long
                ),
                "post_base_action": base_action,
            }
            return corrected, diagnostics

        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            shape,
            dtype=global_condition.dtype,
            device=global_condition.device,
            generator=generator,
        )
        scheduler.set_timesteps(self.num_inference_steps)
        update_indices = self._update_indices() if self.aqr.enabled else set()
        local_tokens = torch.zeros(
            shape[0],
            shape[1],
            int(self.aqr.local_dim),
            dtype=trajectory.dtype,
            device=trajectory.device,
        )
        latest_query: Mapping[str, torch.Tensor] = {}
        latest_residual: Mapping[str, torch.Tensor] = {}
        previous_tool: torch.Tensor | None = None
        trajectory_change: list[torch.Tensor] = []
        residual_elapsed_ms = 0.0
        for index, timestep in enumerate(scheduler.timesteps):
            query_result_for_step: LocalQueryResult | None = None
            global_output = self.model(
                sample=trajectory,
                timestep=timestep,
                local_cond=None,
                global_cond=global_condition,
            )
            if index in update_indices:
                query_result = self._build_local_query(
                    noisy_action=trajectory,
                    global_model_output=global_output,
                    timestep=timestep,
                    encoded=encoded,
                    expert_actions_raw=expert_actions_raw,
                )
                query_result_for_step = query_result
                local_tokens = query_result.tokens
                latest_query = dict(query_result.diagnostics)
                if query_result.trajectory is not None:
                    current_tool = query_result.trajectory.keypoints_world
                    if previous_tool is not None:
                        trajectory_change.append(
                            (current_tool - previous_tool).norm(dim=-1).mean(
                                dim=(1, 2)
                            )
                        )
                    previous_tool = current_tool
            self._sync_timing(trajectory.device)
            residual_start = time.perf_counter()
            if self.aqr.enabled:
                model_output, latest_residual = self._apply_local_residual(
                    noisy_action=trajectory,
                    timestep=timestep,
                    global_condition=global_condition,
                    global_output=global_output,
                    local_tokens=local_tokens,
                    relation_stats=latest_query.get("relation_stats"),
                )
            else:
                model_output = global_output
            if (
                query_result_for_step is not None
                and self.action_to_tool is not None
                and self._evaluation_tool_diagnostics
            ):
                base_clean = predicted_clean_sample(
                    sample=trajectory,
                    model_output=global_output,
                    timestep=timestep,
                    scheduler=scheduler,
                )
                corrected_clean = predicted_clean_sample(
                    sample=trajectory,
                    model_output=model_output,
                    timestep=timestep,
                    scheduler=scheduler,
                )
                (
                    base_candidate,
                    base_actions_normalized,
                    base_actions_environment,
                ) = self._tool_from_normalized_query_actions(
                    base_clean,
                    encoded,
                )
                (
                    corrected_candidate,
                    corrected_actions_normalized,
                    corrected_actions_environment,
                ) = self._tool_from_normalized_query_actions(
                    corrected_clean,
                    encoded,
                )
                keypoint_shift = (
                    corrected_candidate.keypoints_world
                    - base_candidate.keypoints_world
                )
                latest_query.update(
                    {
                        "base_candidate_actions_normalized": (
                            base_actions_normalized
                        ),
                        "base_candidate_actions_environment": (
                            base_actions_environment
                        ),
                        "aqr_corrected_actions_normalized": (
                            corrected_actions_normalized
                        ),
                        "aqr_corrected_actions_environment": (
                            corrected_actions_environment
                        ),
                        "aqr_corrected_tool_keypoints_world": (
                            corrected_candidate.keypoints_world
                        ),
                        "aqr_corrected_tool_rotations_world": (
                            corrected_candidate.rotations_world_from_tool
                        ),
                        "aqr_tool_keypoint_shift_world": keypoint_shift,
                        "aqr_tool_keypoint_shift_m": keypoint_shift.norm(
                            dim=-1
                        ),
                    }
                )
            self._sync_timing(trajectory.device)
            residual_elapsed_ms += 1000.0 * (time.perf_counter() - residual_start)
            trajectory = scheduler.step(
                model_output,
                timestep,
                trajectory,
                **scheduler_step_kwargs,
            ).prev_sample
        diagnostics: dict[str, torch.Tensor] = {
            **encoded.point_banks.diagnostics,
            **encoded.timing_ms,
            **latest_query,
            **latest_residual,
            "aqr_update_count": torch.tensor(
                len(update_indices), device=trajectory.device, dtype=torch.long
            ),
            "local_residual_ms_total": _scalar_tensor(
                residual_elapsed_ms,
                device=trajectory.device,
                dtype=trajectory.dtype,
            ),
        }
        if trajectory_change:
            diagnostics["tool_trajectory_update_change_m"] = torch.stack(
                trajectory_change, dim=1
            )
        return trajectory, diagnostics

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self._strict_dp3_off:
            return super().predict_action(
                self._base_dp3_observation(obs_dict)
            )
        previous_profile = self._profile_timing
        self._profile_timing = True
        try:
            encoded = self._encode_observation(obs_dict)
            batch_size = encoded.state_history_raw.shape[0]
            sample, diagnostics = self.conditional_sample_aqr(
                shape=(batch_size, self.horizon, self.action_dim),
                global_condition=encoded.point_cache.global_condition,
                encoded=encoded,
                expert_actions_raw=obs_dict.get("expert_action"),
                generator=generator,
                **self.kwargs,
            )
        finally:
            self._profile_timing = previous_profile
        action_prediction = self.normalizer["action"].unnormalize(sample)
        start = self.n_obs_steps - 1
        action = action_prediction[:, start : start + self.exec_steps]
        return {
            "action": action,
            "action_pred": action_prediction,
            "aqr_diagnostics": diagnostics,
        }

    # ------------------------------------------------------------------
    # Optional future-latent target branch (training only)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_future_target_encoder(self) -> None:
        if self.future_target_encoder is None:
            return
        decay = float(self.aqr.future_ema_decay)
        for target, online in zip(
            self.future_target_encoder.parameters(), self.obs_encoder.parameters()
        ):
            target.lerp_(online.detach(), 1.0 - decay)
        for target, online in zip(
            self.future_target_encoder.buffers(), self.obs_encoder.buffers()
        ):
            if target.is_floating_point():
                target.lerp_(online.detach(), 1.0 - decay)
            else:
                target.copy_(online)

    def _future_auxiliary_loss(
        self,
        *,
        encoded: EncodedAQRObservation,
        normalized_actions: torch.Tensor,
        full_state_raw: torch.Tensor,
    ):
        if self.future_auxiliary is None or self.future_target_encoder is None:
            return None
        current_index = self.n_obs_steps - 1
        maximum = max(self.aqr.future_horizons)
        if current_index + maximum >= encoded.raw_points.shape[1]:
            raise ValueError(
                "future_aux needs observation sequences through current+k; "
                f"need index {current_index + maximum}, have "
                f"{encoded.raw_points.shape[1] - 1}. Increase dataset horizon."
            )
        if current_index + maximum > normalized_actions.shape[1]:
            raise ValueError("future_aux action prefixes exceed the sampled horizon.")
        self._update_future_target_encoder()
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for offset in self.aqr.future_horizons:
                index = current_index + int(offset)
                state_history = self._history_ending_at(full_state_raw, index)
                banks, cache, _, _ = self._encode_frame(
                    points=encoded.raw_points[:, index],
                    valid=encoded.raw_valid_mask[:, index],
                    robot=encoded.raw_robot_mask[:, index],
                    state_history_raw=state_history,
                    sample_key=index,
                    encoder=self.future_target_encoder,
                    build_query_features=False,
                )
                del banks
                targets.append(cache.global_point_token)
        target_latents = torch.stack(targets, dim=1)
        action_prefix = normalized_actions[
            :, current_index : current_index + maximum
        ]
        negative = hard_negative_actions(
            action_prefix,
            encoded.state_history_normalized[:, -1],
        )
        return self.future_auxiliary.loss(
            current_latent=encoded.point_cache.global_point_token,
            action_sequence=action_prefix,
            target_latents=target_latents,
            negative_action_sequence=negative,
            latent_weight=self.aqr.future_latent_weight,
            action_weight=self.aqr.future_action_weight,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def set_normalizer(self, normalizer) -> None:
        current_device = self.device
        super().set_normalizer(normalizer)
        self.normalizer.to(current_device)
        if self.action_to_tool is None:
            return
        lift = self.action_to_tool.candidate_lift
        if lift is None:
            return
        action_params = self.normalizer.params_dict["action"]
        scale = action_params["scale"].reshape(-1)
        offset = action_params["offset"].reshape(-1)
        if scale.numel() != lift.normalizer_scale.numel():
            raise ValueError("Dataset normalizer and P0 action contract disagree.")
        # The controller/FK contract is independent of dataset scaling.  Keep
        # the lift's policy->environment transform exactly synchronized with
        # the normalizer used by this training/inference policy.
        lift.normalizer_scale.copy_(scale.to(lift.normalizer_scale))
        lift.normalizer_offset.copy_(offset.to(lift.normalizer_offset))

    @staticmethod
    def _prediction_target(
        *,
        scheduler,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = str(scheduler.config.prediction_type)
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "sample":
            return clean
        if prediction_type == "v_prediction":
            alpha_bar = scheduler.alphas_cumprod.to(
                device=clean.device, dtype=clean.dtype
            )[timesteps]
            alpha = alpha_bar.sqrt().view(-1, 1, 1)
            sigma = (1.0 - alpha_bar).clamp_min(0.0).sqrt().view(-1, 1, 1)
            return alpha * noise - sigma * clean
        raise ValueError(
            "AQR-DP3 supports epsilon, sample, or v_prediction; got "
            f"{prediction_type!r}."
        )

    @staticmethod
    def _summary_loss_value(value: torch.Tensor) -> float:
        return float(value.detach().to(torch.float32).mean().cpu())

    def _compute_post_diffusion_loss(self, batch):
        """Train a one-shot refiner on completed pure-DP3 action samples.

        Natural candidates use the deployment DDIM loop. Synthetic A3 offsets
        are added only after that loop, so both candidate types enter the same
        final-action geometry query and correction path.
        """

        obs = batch["obs"]
        actions = batch["action"].to(device=self.device, non_blocking=True).to(device=self.device, non_blocking=True)
        if actions.ndim != 3 or actions.shape[1] != self.horizon:
            raise ValueError(
                f"training actions must have shape [B,{self.horizon},A]."
            )
        expert = self.normalizer["action"].normalize(actions)
        encoded = self._encode_observation(obs)
        batch_size = expert.shape[0]

        # Conventional diffusion loss remains visible as a diagnostic and is
        # optimized only in the explicit unfrozen-base ablation.
        noise = torch.randn_like(expert)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=expert.device,
            dtype=torch.long,
        )
        noisy = self.noise_scheduler.add_noise(expert, noise, timesteps)
        global_output = self.model(
            sample=noisy,
            timestep=timesteps,
            local_cond=None,
            global_cond=encoded.point_cache.global_condition,
        )
        diffusion_target = self._prediction_target(
            scheduler=self.noise_scheduler,
            clean=expert,
            noise=noise,
            timesteps=timesteps,
        )
        base_bc_loss = (global_output - diffusion_target).square().mean()

        # Do not backpropagate through the sampler. The refiner learns against
        # the fixed policy that will actually be deployed.
        with torch.no_grad():
            base_action = self._conditional_sample_global_only(
                shape=tuple(expert.shape),
                global_condition=encoded.point_cache.global_condition.detach(),
            )

        candidate = base_action.clone()
        use_offset = torch.zeros(
            batch_size, device=expert.device, dtype=torch.bool
        )
        offset_magnitude = torch.zeros(
            batch_size, device=expert.device, dtype=expert.dtype
        )
        if (
            self.training
            and self._offset_training_enabled
            and self._offset_prob > 0.0
            and self._offset_max_action_norm > 0.0
        ):
            use_offset = (
                torch.rand(batch_size, device=expert.device) < self._offset_prob
            )
            if use_offset.any():
                count = int(use_offset.sum().item())
                arm_dim = int(self.aqr.arm_dim)
                slots = self._query_slot_indices
                slot_count = int(slots.numel())
                direction = F.normalize(
                    torch.randn(
                        count,
                        slot_count,
                        arm_dim,
                        device=expert.device,
                        dtype=expert.dtype,
                    ),
                    dim=-1,
                )
                magnitude = (
                    torch.rand(
                        count,
                        slot_count,
                        device=expert.device,
                        dtype=expert.dtype,
                    )
                    * self._offset_max_action_norm
                )
                selected = candidate[use_offset].clone()
                before = selected.index_select(1, slots)[..., :arm_dim]
                after = (
                    before + direction * magnitude.unsqueeze(-1)
                ).clamp(-1.0, 1.0)
                selected[:, slots, :arm_dim] = after
                candidate[use_offset] = selected
                offset_magnitude[use_offset] = (
                    after - before
                ).norm(dim=-1).mean(dim=-1)

        corrected, diagnostics = self._post_diffusion_refine(
            base_action=candidate,
            encoded=encoded,
        )
        slots = self._query_slot_indices
        arm_dim = int(self.aqr.arm_dim)
        candidate_arm = candidate.index_select(1, slots)[..., :arm_dim]
        expert_arm = expert.index_select(1, slots)[..., :arm_dim]
        corrected_arm = corrected.index_select(1, slots)[..., :arm_dim]
        bounded_delta = diagnostics["post_bounded_residual"].index_select(
            1, slots
        )[..., :arm_dim]
        confidence = diagnostics["post_confidence"].index_select(
            1, slots
        )

        target_delta = expert_arm - candidate_arm
        target_norm = target_delta.norm(dim=-1, keepdim=True)
        target_scale = (
            float(self.aqr.action_residual_max_norm)
            / target_norm.clamp_min(1e-8)
        ).clamp(max=1.0)
        bounded_target = target_delta * target_scale
        post_delta_loss = F.smooth_l1_loss(bounded_delta, bounded_target)
        post_corrected_loss = F.smooth_l1_loss(corrected_arm, expert_arm)

        # Optimal least-squares execution fraction along the proposed residual.
        detached_delta = bounded_delta.detach()
        confidence_target = (
            (target_delta.detach() * detached_delta).sum(dim=-1)
            / detached_delta.square().sum(dim=-1).clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        confidence_active = detached_delta.norm(dim=-1) > 1e-5
        confidence_weight = confidence_active.to(expert.dtype)
        post_confidence_loss = (
            F.smooth_l1_loss(
                confidence, confidence_target, reduction="none"
            )
            * confidence_weight
        ).sum() / confidence_weight.sum().clamp_min(1.0)

        post_loss = (
            float(self.aqr.post_delta_loss_weight) * post_delta_loss
            + float(self.aqr.post_corrected_loss_weight) * post_corrected_loss
            + float(self.aqr.post_confidence_loss_weight) * post_confidence_loss
        )
        total_loss = post_loss
        if not self.aqr.post_base_frozen:
            total_loss = total_loss + base_bc_loss

        base_error = (candidate_arm - expert_arm).square().mean(dim=(-1, -2))
        corrected_error = (corrected_arm - expert_arm).square().mean(
            dim=(-1, -2)
        )
        error_delta = corrected_error - base_error

        def metric(name: str) -> float:
            value = diagnostics.get(name)
            return (
                0.0
                if value is None
                else self._summary_loss_value(value)
            )

        loss_dict: dict[str, float] = {
            "bc_loss": self._summary_loss_value(base_bc_loss),
            "aqr_total_loss": self._summary_loss_value(total_loss),
            "aqr_post_loss": self._summary_loss_value(post_loss),
            "aqr_post_delta_loss": self._summary_loss_value(post_delta_loss),
            "aqr_post_corrected_loss": self._summary_loss_value(
                post_corrected_loss
            ),
            "aqr_post_confidence_loss": self._summary_loss_value(
                post_confidence_loss
            ),
            "aqr_post_confidence_mean": metric("post_confidence"),
            "aqr_post_raw_residual_norm": metric("post_raw_residual_norm"),
            "aqr_post_bounded_residual_norm": metric(
                "post_bounded_residual_norm"
            ),
            "aqr_post_applied_residual_norm": metric(
                "post_applied_residual_norm"
            ),
            "aqr_post_base_error": self._summary_loss_value(base_error),
            "aqr_post_corrected_error": self._summary_loss_value(
                corrected_error
            ),
            "aqr_post_error_delta": self._summary_loss_value(error_delta),
            "aqr_post_helpful_fraction": self._summary_loss_value(
                (error_delta < 0.0).to(expert.dtype)
            ),
            "aqr_post_no_harm_violation_fraction": self._summary_loss_value(
                (error_delta > 0.0).to(expert.dtype)
            ),
            "aqr_post_confidence_target_mean": self._summary_loss_value(
                confidence_target
            ),
            "aqr_post_confidence_active_fraction": self._summary_loss_value(
                confidence_active.to(expert.dtype)
            ),
            "aqr_offset_fraction": self._summary_loss_value(
                use_offset.to(expert.dtype)
            ),
            "aqr_offset_magnitude_mean": self._summary_loss_value(
                offset_magnitude
            ),
            "aqr_nearest_distance_m": metric("nearest_distance_m"),
            "aqr_empty_fraction": metric("empty_fraction"),
            "aqr_relation_gate_mean": metric("relation_gate_mean"),
            "aqr_post_base_frozen": float(self.aqr.post_base_frozen),
        }
        return total_loss, loss_dict

    def compute_loss(self, batch):
        if self._strict_dp3_off:
            base_batch = dict(batch)
            base_batch["obs"] = self._base_dp3_observation(batch["obs"])
            return super().compute_loss(base_batch)
        if self.aqr.refinement_stage == "post_diffusion":
            return self._compute_post_diffusion_loss(batch)
        if self.aqr.query_source == "expert":
            raise RuntimeError(
                "Expert query is an evaluation-only upper bound. Regular AQR "
                "training must use predicted_clean (or an explicit ablation)."
            )
        obs = batch["obs"]
        actions = batch["action"]
        if actions.ndim != 3 or actions.shape[1] != self.horizon:
            raise ValueError(
                f"training actions must have shape [B,{self.horizon},A]."
            )
        normalized_actions = self.normalizer["action"].normalize(actions)
        encoded = self._encode_observation(obs)
        batch_size = normalized_actions.shape[0]
        noise = torch.randn_like(normalized_actions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=normalized_actions.device,
            dtype=torch.long,
        )
        noisy = self.noise_scheduler.add_noise(
            normalized_actions, noise, timesteps
        )
        global_output = self.model(
            sample=noisy,
            timestep=timesteps,
            local_cond=None,
            global_cond=encoded.point_cache.global_condition,
        )
        query_diagnostics: Mapping[str, torch.Tensor] = {}
        residual_diagnostics: Mapping[str, torch.Tensor] = {}
        improvement_diagnostics: Mapping[str, torch.Tensor] = {}
        improvement_loss: torch.Tensor | None = None
        applied_query_residual_log: torch.Tensor | None = None
        use_offset = torch.zeros(
            batch_size, device=normalized_actions.device, dtype=torch.bool
        )
        offset_magnitude_log = torch.zeros(
            batch_size, device=normalized_actions.device,
            dtype=normalized_actions.dtype,
        )
        if self.aqr.enabled:
            clean_override: torch.Tensor | None = None
            residual_base_output = global_output
            if (
                self.training
                and self._offset_training_enabled
                and self._offset_prob > 0.0
                and self._offset_max_action_norm > 0.0
            ):
                predicted_clean = predicted_clean_sample(
                    sample=noisy,
                    model_output=global_output,
                    timestep=timesteps,
                    scheduler=self.noise_scheduler,
                )
                use_offset = (
                    torch.rand(batch_size, device=noisy.device)
                    < self._offset_prob
                )
                if use_offset.any():
                    n_offset = int(use_offset.sum().item())
                    arm_offset_direction = F.normalize(
                        torch.randn(
                            n_offset,
                            int(self.aqr.arm_dim),
                            device=noisy.device,
                            dtype=noisy.dtype,
                        ),
                        dim=-1,
                    )
                    offset_magnitude = (
                        torch.rand(n_offset, device=noisy.device)
                        * self._offset_max_action_norm
                    )
                    arm_offset = (
                        arm_offset_direction * offset_magnitude[:, None]
                    )
                    clean_override = predicted_clean.clone()
                    query_slot = int(self._query_slot_indices[0].item())
                    clean_override[
                        use_offset, query_slot, : int(self.aqr.arm_dim)
                    ] = (
                        predicted_clean[
                            use_offset, query_slot, : int(self.aqr.arm_dim)
                        ].detach()
                        + arm_offset
                    )
                    residual_base_output = global_output.clone()
                    residual_base_output[use_offset, query_slot] = (
                        clean_override[use_offset, query_slot]
                    )
                    offset_magnitude_log[use_offset] = offset_magnitude

            query_result = self._build_local_query(
                noisy_action=noisy,
                global_model_output=global_output,
                timestep=timesteps,
                encoded=encoded,
                clean_actions_override=clean_override,
            )
            relation_stats = query_result.diagnostics.get("relation_stats")
            model_output, residual_diagnostics = self._apply_local_residual(
                noisy_action=noisy,
                timestep=timesteps,
                global_condition=encoded.point_cache.global_condition,
                global_output=residual_base_output,
                local_tokens=query_result.tokens,
                relation_stats=relation_stats,
                # During A3 offset training the applied base may contain a
                # synthetic perturbation.  Direction preservation must refer
                # to the unperturbed semantic prediction, matching inference.
                residual_reference_output=global_output,
            )
            query_diagnostics = query_result.diagnostics
            query_slots = self._query_slot_indices
            arm_dim = int(self.aqr.arm_dim)
            applied_query_residual_log = (
                model_output - residual_base_output
            ).index_select(1, query_slots)[..., :arm_dim]
            if self.aqr.improvement_loss_enabled:
                base_query = residual_base_output.index_select(
                    1, query_slots
                )[..., :arm_dim]
                expert_query = normalized_actions.index_select(
                    1, query_slots
                )[..., :arm_dim]
                # This subtraction cancels the direct semantic/base path.  The
                # auxiliary corrected action below therefore trains the AQR
                # correction against a detached base without rewarding a worse
                # base policy.
                improvement_loss, improvement_diagnostics = (
                    relative_action_improvement_loss(
                        base_action=base_query,
                        applied_residual=applied_query_residual_log,
                        expert_action=expert_query,
                        active_gate=residual_diagnostics["query_gate"],
                        margin_ratio=float(
                            self.aqr.improvement_margin_ratio
                        ),
                        sample_mask=(
                            ~use_offset
                            if self.aqr.improvement_loss_clean_only
                            else None
                        ),
                    )
                )
        else:
            model_output = global_output
        target = self._prediction_target(
            scheduler=self.noise_scheduler,
            clean=normalized_actions,
            noise=noise,
            timesteps=timesteps,
        )
        per_sample_diffusion_loss = (model_output - target).square().mean(
            dim=(1, 2)
        )
        diffusion_loss = per_sample_diffusion_loss.mean()

        def masked_sample_mean(
            value: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            if value.ndim != 1 or value.shape[0] != batch_size:
                raise ValueError("masked training metric must have shape [B].")
            weight = mask.to(device=value.device, dtype=value.dtype)
            return (value * weight).sum() / weight.sum().clamp_min(1.0)

        clean_sample_mask = ~use_offset
        clean_bc_loss = masked_sample_mean(
            per_sample_diffusion_loss, clean_sample_mask
        )
        offset_bc_loss = masked_sample_mean(
            per_sample_diffusion_loss, use_offset
        )
        if applied_query_residual_log is None:
            per_sample_residual_norm = torch.zeros_like(
                per_sample_diffusion_loss
            )
        else:
            per_sample_residual_norm = applied_query_residual_log.norm(
                dim=-1
            ).mean(dim=1)
        clean_residual_norm = masked_sample_mean(
            per_sample_residual_norm, clean_sample_mask
        )
        offset_residual_norm = masked_sample_mean(
            per_sample_residual_norm, use_offset
        )
        total_loss = diffusion_loss
        if improvement_loss is not None:
            total_loss = total_loss + (
                float(self.aqr.improvement_loss_weight) * improvement_loss
            )
        future_loss = self._future_auxiliary_loss(
            encoded=encoded,
            normalized_actions=normalized_actions,
            full_state_raw=obs["agent_pos"],
        )
        if future_loss is not None:
            total_loss = total_loss + future_loss.total
        loss_dict: dict[str, float] = {
                "bc_loss": self._summary_loss_value(diffusion_loss),
                "aqr_total_loss": self._summary_loss_value(total_loss),
                "aqr_improvement_loss": self._summary_loss_value(
                    improvement_loss
                    if improvement_loss is not None
                    else torch.zeros((), device=total_loss.device)
                ),
                "aqr_improvement_base_error": self._summary_loss_value(
                    improvement_diagnostics.get(
                        "improvement_base_error",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_improvement_corrected_error": self._summary_loss_value(
                    improvement_diagnostics.get(
                        "improvement_corrected_error",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_improvement_error_delta": self._summary_loss_value(
                    improvement_diagnostics.get(
                        "improvement_error_delta",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_improvement_helpful_fraction": self._summary_loss_value(
                    improvement_diagnostics.get(
                        "improvement_helpful_fraction",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_improvement_no_harm_violation_fraction": (
                    self._summary_loss_value(
                        improvement_diagnostics.get(
                            "improvement_no_harm_violation_fraction",
                            torch.zeros((), device=total_loss.device),
                        )
                    )
                ),
                "aqr_improvement_margin_satisfied_fraction": (
                    self._summary_loss_value(
                        improvement_diagnostics.get(
                            "improvement_margin_satisfied_fraction",
                            torch.zeros((), device=total_loss.device),
                        )
                    )
                ),
                "aqr_improvement_active_fraction": self._summary_loss_value(
                    improvement_diagnostics.get(
                        "improvement_active_fraction",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_improvement_eligible_fraction": self._summary_loss_value(
                    improvement_diagnostics.get(
                        "improvement_eligible_fraction",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_improvement_active_among_eligible_fraction": (
                    self._summary_loss_value(
                        improvement_diagnostics.get(
                            "improvement_active_among_eligible_fraction",
                            torch.zeros((), device=total_loss.device),
                        )
                    )
                ),
                "aqr_improvement_clean_only": float(
                    self.aqr.improvement_loss_clean_only
                ),
                "aqr_clean_fraction": self._summary_loss_value(
                    clean_sample_mask.to(dtype=total_loss.dtype)
                ),
                "aqr_offset_fraction": self._summary_loss_value(
                    use_offset.to(dtype=total_loss.dtype)
                ),
                "aqr_clean_bc_loss": self._summary_loss_value(clean_bc_loss),
                "aqr_offset_bc_loss": self._summary_loss_value(offset_bc_loss),
                "aqr_clean_residual_norm": self._summary_loss_value(
                    clean_residual_norm
                ),
                "aqr_offset_residual_norm": self._summary_loss_value(
                    offset_residual_norm
                ),
                "aqr_gate_mean": self._summary_loss_value(
                    residual_diagnostics.get(
                        "query_gate",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_delta_norm": self._summary_loss_value(
                    residual_diagnostics.get(
                        "delta_norm",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_raw_action_residual_norm": self._summary_loss_value(
                    residual_diagnostics.get(
                        "raw_action_residual_norm",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_bounded_action_residual_norm": self._summary_loss_value(
                    residual_diagnostics.get(
                        "bounded_action_residual_norm",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_action_residual_clip_fraction": self._summary_loss_value(
                    residual_diagnostics.get(
                        "action_residual_clip_fraction",
                        torch.zeros((), device=total_loss.device),
                    ).to(dtype=total_loss.dtype)
                ),
                "aqr_action_residual_progress_limited_fraction": (
                    self._summary_loss_value(
                        residual_diagnostics.get(
                            "action_residual_progress_limited_fraction",
                            torch.zeros((), device=total_loss.device),
                        ).to(dtype=total_loss.dtype)
                    )
                ),
                "aqr_action_residual_progress_ratio": self._summary_loss_value(
                    residual_diagnostics.get(
                        "action_residual_progress_ratio",
                        torch.ones((), device=total_loss.device),
                    )
                ),
                "aqr_protected_non_arm_residual_norm": self._summary_loss_value(
                    residual_diagnostics.get(
                        "protected_non_arm_residual_norm",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_film_gamma_norm": self._summary_loss_value(
                    residual_diagnostics.get(
                        "film_gamma",
                        torch.zeros((), device=total_loss.device),
                    ).norm(dim=-1)
                    if "film_gamma" in residual_diagnostics
                    else torch.zeros((), device=total_loss.device)
                ),
                "aqr_film_beta_norm": self._summary_loss_value(
                    residual_diagnostics.get(
                        "film_beta",
                        torch.zeros((), device=total_loss.device),
                    ).norm(dim=-1)
                    if "film_beta" in residual_diagnostics
                    else torch.zeros((), device=total_loss.device)
                ),
                "aqr_local_token_norm": self._summary_loss_value(
                    query_diagnostics.get(
                        "local_token_norm",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_empty_fraction": self._summary_loss_value(
                    query_diagnostics.get(
                        "empty_fraction",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_robot_point_fraction": self._summary_loss_value(
                    query_diagnostics.get(
                        "robot_point_fraction",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_tool_outside_fraction": self._summary_loss_value(
                    query_diagnostics.get(
                        "tool_outside_fraction",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_query_valid_count": self._summary_loss_value(
                    encoded.point_banks.diagnostics["query_valid_count"]
                ),
                "aqr_valid_neighbor_count": self._summary_loss_value(
                    query_diagnostics.get(
                        "valid_neighbor_count",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_nearest_distance_m": self._summary_loss_value(
                    query_diagnostics.get(
                        "nearest_distance_m",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_a0_min": self._summary_loss_value(
                    query_diagnostics.get(
                        "a0_min",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_a0_max": self._summary_loss_value(
                    query_diagnostics.get(
                        "a0_max",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_a0_boundary_fraction": self._summary_loss_value(
                    query_diagnostics.get(
                        "a0_boundary_fraction",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_tcp_swept_zero_coverage": self._summary_loss_value(
                    encoded.point_banks.diagnostics[
                        "tcp_swept_zero_coverage"
                    ]
                ),
                # ── Layer 1: relation features ──
                "aqr_r_face_mean": self._summary_loss_value(
                    query_diagnostics.get(
                        "r_face_mean",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_r_face_neg_fraction": self._summary_loss_value(
                    query_diagnostics.get(
                        "r_face_neg_fraction",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                "aqr_relation_gate_mean": self._summary_loss_value(
                    query_diagnostics.get(
                        "relation_gate_mean",
                        torch.zeros((), device=total_loss.device),
                    )
                ),
                # ── Layer 3: relation gain ──
                "aqr_relation_gain": self._summary_loss_value(
                    residual_diagnostics.get(
                        "relation_gain",
                        torch.ones((), device=total_loss.device),
                    )
                ),
                # ── Layer 4: offset training ──
                "aqr_offset_magnitude_mean": self._summary_loss_value(
                    offset_magnitude_log
                ),
            }
        if future_loss is not None:
            loss_dict.update(
                {
                    "future_aux_loss": self._summary_loss_value(
                        future_loss.total
                    ),
                    "future_latent_loss": self._summary_loss_value(
                        future_loss.latent
                    ),
                    "future_action_loss": self._summary_loss_value(
                        future_loss.action
                    ),
                }
            )
        return total_loss, loss_dict

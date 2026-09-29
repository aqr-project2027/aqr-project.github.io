from __future__ import annotations

import unittest

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3

from contactflow.dp3.aqr_dp3.policy import AQRDP3


def _scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=20,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="sample",
    )


def _shape_meta(point_channels: int = 3):
    return OmegaConf.create(
        {
            "obs": {
                "point_cloud": {
                    "shape": [64, point_channels],
                    "type": "point_cloud",
                },
                "agent_pos": {"shape": [16], "type": "low_dim"},
            },
            "action": {"shape": [8]},
        }
    )


def _model_kwargs(point_channels: int = 3):
    return {
        "shape_meta": _shape_meta(point_channels),
        "horizon": 4,
        "n_action_steps": 3,
        "n_obs_steps": 2,
        "num_inference_steps": 3,
        "obs_as_global_cond": True,
        "diffusion_step_embed_dim": 32,
        "down_dims": (32, 64),
        "kernel_size": 3,
        "n_groups": 8,
        "condition_type": "film",
        "encoder_output_dim": 16,
        "use_pc_color": point_channels == 6,
        "pointnet_type": "pointnet",
        "pointcloud_encoder_cfg": OmegaConf.create(
            {
                "in_channels": point_channels,
                "out_channels": 16,
                "use_layernorm": True,
                "final_norm": "layernorm",
                "normal_channel": False,
            }
        ),
    }


class AQRDP3StrictOffParityTest(unittest.TestCase):
    def _assert_parity(self, point_channels: int) -> None:
        torch.manual_seed(101)
        baseline = SimpleDP3(
            noise_scheduler=_scheduler(),
            **_model_kwargs(point_channels),
        )
        strict_off = AQRDP3(
            noise_scheduler=_scheduler(),
            aqr_config={"experiment": "AQR_OFF"},
            **_model_kwargs(point_channels),
        )
        strict_off.load_state_dict(baseline.state_dict(), strict=True)
        self.assertFalse(hasattr(strict_off, "point_frontend"))
        self.assertIs(type(strict_off.obs_encoder), type(baseline.obs_encoder))

        normalizer = LinearNormalizer()
        normalizer.fit(
            {
                "point_cloud": torch.linspace(
                    -1.0, 1.0, 64 * point_channels
                ).reshape(64, point_channels),
                "agent_pos": torch.linspace(-1.0, 1.0, 32).reshape(2, 16),
                "action": torch.linspace(-1.0, 1.0, 16).reshape(2, 8),
            },
            last_n_dims=1,
        )
        baseline.set_normalizer(normalizer)
        strict_off.set_normalizer(normalizer)
        baseline.eval()
        strict_off.eval()

        generator = torch.Generator().manual_seed(77)
        obs = {
            "point_cloud": torch.randn(
                (2, 2, 64, point_channels), generator=generator
            ),
            "agent_pos": torch.randn((2, 2, 16), generator=generator),
            "point_valid_mask": torch.ones((2, 2, 64), dtype=torch.bool),
            "robot_mask": torch.ones((2, 2, 64), dtype=torch.bool),
        }
        torch.manual_seed(909)
        expected = baseline.predict_action(
            {
                "point_cloud": obs["point_cloud"],
                "agent_pos": obs["agent_pos"],
            }
        )
        torch.manual_seed(909)
        actual = strict_off.predict_action(obs)
        torch.testing.assert_close(
            actual["action"], expected["action"], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            actual["action_pred"],
            expected["action_pred"],
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(tuple(actual["action"].shape), (2, 3, 8))
        self.assertNotIn("aqr_diagnostics", actual)

    def test_fixed_seed_xyz_output_matches_original_simple_dp3(self):
        self._assert_parity(3)

    def test_fixed_seed_xyzrgb_output_matches_original_simple_dp3(self):
        self._assert_parity(6)


if __name__ == "__main__":
    unittest.main()

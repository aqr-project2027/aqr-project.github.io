from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from contactflow.tools.train_aqr_dp3 import (
    DEFAULT_CONFIG,
    _apply_overrides,
    _initialize_model_from_checkpoint,
    _load_config,
    _training_sampler_static_contract,
    parse_args,
)


class AQRDP3TrainingProgressTest(unittest.TestCase):
    def test_progress_is_enabled_by_default(self):
        config = _load_config(DEFAULT_CONFIG)
        self.assertTrue(config["training"]["progress"])
        self.assertEqual(config["training"]["progress_update_steps"], 20)

    def test_cli_can_disable_or_retune_progress(self):
        args = parse_args(
            [
                "--progress",
                "false",
                "--progress-update-steps",
                "7",
            ]
        )
        config = _apply_overrides(_load_config(DEFAULT_CONFIG), args)
        self.assertFalse(config["training"]["progress"])
        self.assertEqual(config["training"]["progress_update_steps"], 7)

    def test_uniform_sampler_is_the_default(self):
        config = _load_config(DEFAULT_CONFIG)
        contract = _training_sampler_static_contract(config)
        self.assertEqual(contract["mode"], "uniform")
        self.assertFalse(contract["changes_default_dataloader"])

    def test_cli_can_select_phase_sampler_and_fresh_optimizer_init(self):
        args = parse_args(
            [
                "--sampling-mode",
                "phase_balanced",
                "--expert-events-csv",
                "outputs/events.csv",
                "--learning-rate",
                "0.00002",
                "--initialize-from",
                "outputs/epoch40.pt",
            ]
        )
        config = _apply_overrides(_load_config(DEFAULT_CONFIG), args)
        self.assertEqual(
            config["training_sampler"]["mode"],
            "phase_balanced",
        )
        self.assertEqual(
            config["training_sampler"]["events_csv"],
            "outputs/events.csv",
        )
        self.assertAlmostEqual(config["optimizer"]["lr"], 0.00002)
        self.assertEqual(args.initialize_from, "outputs/epoch40.pt")

    def test_seed_override_also_controls_phase_sampler(self):
        args = parse_args(["--seed", "43"])
        config = _apply_overrides(_load_config(DEFAULT_CONFIG), args)
        self.assertEqual(config["training"]["seed"], 43)
        self.assertEqual(config["dataset"]["seed"], 43)
        self.assertEqual(config["training_sampler"]["seed"], 43)

    def test_model_only_initialization_reapplies_policy_device(self):
        class FakeModel:
            def __init__(self):
                self.device = "cuda:0"
                self.loaded = None
                self.to_calls = []

            def load_state_dict(self, state_dict, strict):
                self.loaded = (state_dict, strict)
                # Mirrors LinearNormalizer's CPU ParameterDict reconstruction.
                self.normalizer_device = "cpu"

            def to(self, device):
                self.to_calls.append(str(device))
                self.normalizer_device = str(device)
                return self

        payload = {
            "kind": "aqr_dp3_training_checkpoint",
            "config": {},
            "model_state_dict": {"normalizer": "cpu_checkpoint_tensor"},
            "epoch": 40,
            "global_step": 48560,
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "epoch40.pt"
            checkpoint.touch()
            model = FakeModel()
            with mock.patch(
                "contactflow.tools.train_aqr_dp3._torch_load",
                return_value=payload,
            ):
                loaded, report, path = _initialize_model_from_checkpoint(
                    model,
                    {},
                    checkpoint,
                )

        self.assertIs(loaded, payload)
        self.assertEqual(path, checkpoint.resolve())
        self.assertEqual(model.loaded, (payload["model_state_dict"], True))
        self.assertEqual(model.to_calls, ["cuda:0"])
        self.assertEqual(model.normalizer_device, "cuda:0")
        self.assertFalse(report["optimizer_state_loaded"])
        self.assertEqual(
            report["model_device_reapplied_after_load"],
            "cuda:0",
        )

    def test_selective_initialization_keeps_base_and_resets_refiner(self):
        class FakeModel:
            def __init__(self):
                self.device = "cuda:0"
                self.loaded = None

            def load_state_dict(self, state_dict, strict):
                self.loaded = (dict(state_dict), strict)
                return type(
                    "Incompatible",
                    (),
                    {
                        "missing_keys": ["post_refiner.weight"],
                        "unexpected_keys": [],
                    },
                )()

            def to(self, _device):
                return self

        payload = {
            "kind": "aqr_dp3_training_checkpoint",
            "config": {},
            "model_state_dict": {
                "model.weight": "shared-base",
                "post_refiner.weight": "old-full-head",
            },
            "epoch": 77,
            "global_step": 116378,
        }
        config = {
            "training": {
                "initialize_exclude_prefixes": ["post_refiner."]
            }
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.touch()
            model = FakeModel()
            with mock.patch(
                "contactflow.tools.train_aqr_dp3._torch_load",
                return_value=payload,
            ):
                _, report, _ = _initialize_model_from_checkpoint(
                    model, config, checkpoint
                )

        self.assertEqual(model.loaded, ({"model.weight": "shared-base"}, False))
        self.assertEqual(report["architecture_extension"], "fresh_refiner_ablation")
        self.assertEqual(report["new_parameter_keys"], ["post_refiner.weight"])
        self.assertEqual(report["excluded_prefixes"], ["post_refiner."])


if __name__ == "__main__":
    unittest.main()

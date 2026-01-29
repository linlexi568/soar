import unittest

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch not installed, skipping meta-RL tests") from exc

from meta_rl.config import MetaRLConfig
from meta_rl.controller import MetaRLController
from meta_rl.hot_swap import build_args_proxy, maybe_apply_overrides, write_override_file


class MetaControllerTest(unittest.TestCase):
    def test_controller_generates_overrides(self):
        controller = MetaRLController(MetaRLConfig())
        metrics = {
            "iter_norm": 0.5,
            "reward_mean": -0.3,
            "reward_std": 0.1,
            "success_rate": 0.4,
            "zero_action_frac": 0.2,
            "entropy": 0.8,
            "ranking_blend": 0.1,
            "crash_ratio": 0.0,
        }
        output = controller.update(metrics)
        self.assertTrue(all(name in output.values for name in controller.config.output_names))

    def test_hot_swap_roundtrip(self):
        config = MetaRLConfig()
        overrides = {name: idx + 0.1 for idx, name in enumerate(config.output_names)}
        override_path = write_override_file(overrides, path=config.override_file)
        args = build_args_proxy(**{name: 0.0 for name in config.output_names})
        updated = maybe_apply_overrides(args, iter_idx=10, override_path=override_path)
        for key, value in overrides.items():
            self.assertAlmostEqual(getattr(updated, key), value)
        override_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

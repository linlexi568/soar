from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json

import torch


def _default_feature_names() -> List[str]:
    return [
        "iter_norm",
        "reward_mean",
        "reward_std",
        "success_rate",
        "zero_action_frac",
        "entropy",
        "ranking_blend",
        "crash_ratio",
    ]


@dataclass
class TelemetrySample:
    """Single meta-step telemetry packet consumed by the RNN policy."""

    features: torch.Tensor
    mask: torch.Tensor

    @staticmethod
    def from_metrics(metrics: Dict[str, float], feature_names: Iterable[str]) -> "TelemetrySample":
        vec = [float(metrics.get(name, 0.0)) for name in feature_names]
        tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask = torch.ones((1, 1, 1), dtype=torch.bool)
        return TelemetrySample(features=tensor, mask=mask)


@dataclass
class MetaRLConfig:
    feature_names: List[str] = field(default_factory=_default_feature_names)
    hidden_dim: int = 64
    num_layers: int = 1
    output_names: List[str] = field(default_factory=lambda: [
        "dirichlet_alpha",
        "dirichlet_eps",
        "policy_temperature",
        "zero_action_penalty",
        "replicas_per_program",
    ])
    override_file: Path = Path("meta_rl/configs/current_override.json")
    clamp_ranges: Dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "dirichlet_alpha": (0.1, 1.2),
        "dirichlet_eps": (0.05, 0.8),
        "policy_temperature": (0.4, 2.5),
        "zero_action_penalty": (0.0, 0.8),
        "replicas_per_program": (2.0, 8.0),
    })


@dataclass
class MetaRLOutput:
    values: Dict[str, float]

    def clamp(self, config: MetaRLConfig) -> "MetaRLOutput":
        clamped = {}
        for key, value in self.values.items():
            if key in config.clamp_ranges:
                lo, hi = config.clamp_ranges[key]
                clamped[key] = float(max(lo, min(hi, value)))
            else:
                clamped[key] = float(value)
        return MetaRLOutput(values=clamped)

    def to_json(self) -> str:
        return json.dumps(self.values, indent=2)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    def apply_to_args(self, args) -> None:
        for key, value in self.values.items():
            if hasattr(args, key):
                setattr(args, key, value)


def load_override_file(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return None


def save_config(path: Path, config: MetaRLConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

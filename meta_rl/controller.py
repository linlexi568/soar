from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

try:
    from .config import MetaRLConfig, MetaRLOutput, TelemetrySample
    from .rnn_meta_policy import MetaRNNPolicy
except ImportError:
    from config import MetaRLConfig, MetaRLOutput, TelemetrySample
    from rnn_meta_policy import MetaRNNPolicy


class MetaRLController:
    """Wraps the recurrent policy and exposes a friendly update/propose interface."""

    def __init__(
        self,
        config: Optional[MetaRLConfig] = None,
        policy: Optional[MetaRNNPolicy] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or MetaRLConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy or MetaRNNPolicy(self.config)
        self.policy.to(self.device)
        self.hidden: Optional[torch.Tensor] = None
        self.online_updates = False

    def load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state["policy_state"])
        self.hidden = state.get("hidden_state")

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "hidden_state": self.hidden,
                "config": self.config.__dict__,
            },
            path,
        )

    def update(self, metrics: Dict[str, float]) -> MetaRLOutput:
        sample = TelemetrySample.from_metrics(metrics, self.config.feature_names)
        telemetry = sample.features.to(self.device)
        self.hidden = self.hidden.to(self.device) if self.hidden is not None else None
        if self.online_updates:
            self.policy.train()
            logits, self.hidden = self.policy(telemetry, self.hidden)
            output = self.policy.decode_logits(logits)
        else:
            output, self.hidden = self.policy.predict(telemetry, self.hidden)
        return output.clamp(self.config)

    def reset_hidden(self) -> None:
        self.hidden = None

    def export_override(self, path: Optional[Path] = None) -> Path:
        path = path or self.config.override_file
        override = self.update({name: 0.0 for name in self.config.feature_names})
        override.write(path)
        return path

    def write_override_from_metrics(
        self, metrics: Dict[str, float], path: Optional[Path] = None
    ) -> Path:
        output = self.update(metrics)
        path = path or self.config.override_file
        output.write(path)
        return path

    def to_dict(self) -> Dict[str, float]:
        return {"hidden_norm": float(self.hidden.norm().item()) if self.hidden is not None else 0.0}

    def dump_hidden(self, path: Path) -> None:
        payload = {
            "hidden": self.hidden.detach().cpu().tolist() if self.hidden is not None else None,
            "feature_names": self.config.feature_names,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

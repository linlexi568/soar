from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from .config import MetaRLConfig, MetaRLOutput
except ImportError:
    from config import MetaRLConfig, MetaRLOutput


class MetaRNNPolicy(nn.Module):
    """Simple GRU-based controller that outputs bounded deltas for MCTS knobs."""

    def __init__(self, config: MetaRLConfig):
        super().__init__()
        self.config = config
        self.input_dim = len(config.feature_names)
        self.output_dim = len(config.output_names)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, self.output_dim),
            nn.Tanh(),
        )
        self.register_buffer("_output_scale", torch.tensor(1.0))

    def forward(
        self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: [batch, seq, features]
        out, hidden = self.gru(inputs, hidden)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits, hidden

    def decode_logits(self, logits: torch.Tensor) -> MetaRLOutput:
        logits = logits.squeeze(0)
        values = {}
        for idx, name in enumerate(self.config.output_names):
            lo, hi = self.config.clamp_ranges.get(name, (-1.0, 1.0))
            span = hi - lo
            mid = (hi + lo) / 2.0
            values[name] = float(mid + 0.5 * span * logits[idx].item())
        return MetaRLOutput(values=values)

    def predict(
        self,
        telemetry: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[MetaRLOutput, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(telemetry, hidden)
            output = self.decode_logits(logits)
            return output, hidden

    def loss_fn(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.view_as(pred[..., :1]).expand_as(pred)
        diff = (pred - target) * mask
        return (diff ** 2).mean()


def build_default_policy() -> MetaRNNPolicy:
    config = MetaRLConfig()
    return MetaRNNPolicy(config)

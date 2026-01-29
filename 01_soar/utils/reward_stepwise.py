"""Legacy-compatible wrapper around the exact SCG reward.

Historically Soar used a "stepwise" reward with many heuristic
components. The project now enforces the exact safe-control-gym cost
from the paper, so this module simply wraps :class:`SCGExactRewardCalculator`
while keeping the public API (``compute_step``, ``get_component_totals``)
unchanged for downstream scripts.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .reward_scg_exact import (
    SCGExactRewardCalculator,
    SCG_STATE_WEIGHTS,
    SCG_ACTION_WEIGHT,
)


class StepwiseRewardCalculator:
    """Compatibility shim that forwards to :class:`SCGExactRewardCalculator`."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ks: Optional[Dict[str, float]] = None,
        dt: float = 1.0 / 48.0,
        num_envs: int = 1,
        device: str = "cpu",
        state_weights: Optional[torch.Tensor] = None,
        action_weight: float = SCG_ACTION_WEIGHT,
    ) -> None:
        # 保留原始字段，便于旧代码读取/打印
        self.w = dict(weights or {})
        self.k = dict(ks or {})
        self.dt = float(dt)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        if state_weights is None:
            state_weights = SCG_STATE_WEIGHTS.clone()

        self._exact = SCGExactRewardCalculator(
            num_envs=self.num_envs,
            device=str(self.device),
            state_weights=state_weights.to(self.device),
            action_weight=action_weight,
        )

    def reset(self, num_envs: Optional[int] = None) -> None:
        self._exact.reset(num_envs=num_envs)
        if num_envs is not None:
            self.num_envs = int(num_envs)

    def compute_step(
        self,
        pos: torch.Tensor,
        target: torch.Tensor,
        vel: torch.Tensor,
        omega: torch.Tensor,
        actions: torch.Tensor,
        done_mask: Optional[torch.Tensor] = None,
        *,
        quat: Optional[torch.Tensor] = None,
        target_vel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the exact SCG reward for a single timestep."""
        if quat is None:
            raise ValueError("StepwiseRewardCalculator now requires quaternion data to compute SCG reward.")

        if target.dim() == 1:
            target_pos = target.view(1, 3).expand(pos.shape[0], -1)
        else:
            target_pos = target

        if actions.shape[1] >= 6:
            action_slice = actions[:, 2:6]
        elif actions.shape[1] >= 4:
            action_slice = actions[:, :4]
        else:
            raise ValueError("actions tensor must have at least 4 channels for thrust/torque commands")

        reward = self._exact.compute_step(
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            target_pos=target_pos,
            action=action_slice,
            target_vel=target_vel,
            done_mask=done_mask,
        )
        return reward

    def finalize(self) -> torch.Tensor:
        """Exact SCG reward has no terminal shaping; return zeros."""
        return torch.zeros(self.num_envs, device=self.device)

    def get_component_totals(self) -> Dict[str, torch.Tensor]:
        comps = self._exact.get_components()
        return {
            "state_cost": comps["state_cost"],
            "action_cost": comps["action_cost"],
            "total_cost": comps["total_cost"],
        }

    def reset_components(self, num_envs: Optional[int] = None) -> None:
        self.reset(num_envs=num_envs)


__all__ = ["StepwiseRewardCalculator"]

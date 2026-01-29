"""Reward weights for Safe-Control-Gym alignment only.

The user requested that Soar expose *only* the safe-control-gym reward
definition so downstream scripts cannot accidentally select legacy profiles.
This module therefore keeps a single profile that mirrors
``quadrotor_3D_track`` exactly and removes every other preset.
"""
from __future__ import annotations
from typing import Dict, Tuple

Weights = Dict[str, float]
Coeffs = Dict[str, float]


# Strict safe-control-gym quadrotor_3D_track reward:
#   r_t = -(x^T Q x + u^T R u)
# Only position_rmse (state error) and control_effort (action penalty) remain.
_safe_control_tracking_weights: Weights = {
    "position_rmse": 1.0,
    "settling_time": 0.0,
    "control_effort": 0.0001,
    "smoothness_jerk": 0.0,
    "gain_stability": 0.0,
    "saturation": 0.0,
    "peak_error": 0.0,
    "high_freq": 0.0,
    "finalize_bonus": 0.0,
}

_safe_control_tracking_ks: Coeffs = {
    "k_position": 1.0,
    "k_settle": 1.0,
    "k_effort": 1.0,
    "k_jerk": 1.0,
    "k_gain": 1.0,
    "k_sat": 1.0,
    "k_peak": 1.0,
    "k_high_freq": 1.0,
}


PROFILES: Dict[str, Tuple[Weights, Coeffs]] = {
    "safe_control_tracking": (
        _safe_control_tracking_weights,
        _safe_control_tracking_ks,
    ),
}


def list_profiles() -> Dict[str, Tuple[Weights, Coeffs]]:
    return PROFILES.copy()


def get_reward_profile(name: str) -> Tuple[Weights, Coeffs]:
    if name not in PROFILES:
        raise KeyError(f"Unknown reward profile '{name}'. Available: {list(PROFILES)}")
    weights, ks = PROFILES[name]
    # Return shallow copies to avoid accidental mutation.
    return dict(weights), dict(ks)


def describe_profile(name: str) -> str:
    weights, ks = get_reward_profile(name)
    lines = [f"Reward profile: {name}"]
    lines.append("Weights:")
    for k, v in weights.items():
        lines.append(f"  {k}: {v}")
    lines.append("Coefficients (k_*):")
    for k, v in ks.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

__all__ = [
    "Weights",
    "Coeffs",
    "list_profiles",
    "get_reward_profile",
    "describe_profile",
]
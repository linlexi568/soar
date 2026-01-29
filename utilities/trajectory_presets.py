"""Shared Safe-Control-Gym trajectory helpers.

This module centralizes the canonical definitions of the hover / figure8 /
circle / square / helix trajectories we use everywhere in Soar.  The goal
is to keep DSL training, PID/LQR baselines, SB3 policies, and PPO/SAC agents in
lockstep with the reference safe-control-gym quadrotor_3D_track tasks.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

SCG_TRAJECTORY_LIBRARY: Mapping[str, Mapping[str, float | Sequence[float]]] = {
    "hover": {
        "center": (0.0, 0.0, 1.0),
    },
    "figure8": {
        "center": (0.0, 0.0, 1.0),
        "plane": "xy",
        "period": 5.0,
        "scale": 0.8,
        "A": 0.8,
        "B": 0.8,
    },
    "circle": {
        "center": (0.0, 0.0, 1.0),
        "plane": "xy",
        "period": 5.0,
        "R": 0.9,
        "scale": 0.9,
    },
    "square": {
        "center": (0.0, 0.0, 1.0),
        "plane": "xy",
        "period": 5.0,
        "scale": 0.8,
    },
    "helix": {
        "center": (0.0, 0.0, 1.0),
        "plane": "xy",
        "period": 8.0,
        "R": 0.7,
        "scale": 0.7,
        "v_z": 0.1,
    },
}

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class TrajectoryConfig:
    """Resolved trajectory specification ready for consumption."""

    task: str
    params: Dict[str, float]
    center: Tuple[float, float, float]


def get_scg_trajectory_config(task: str, overrides: Mapping[str, float | int | str] | None = None) -> TrajectoryConfig:
    """Return a resolved trajectory config with optional overrides."""

    task_lc = task.lower()
    if task_lc not in SCG_TRAJECTORY_LIBRARY:
        raise ValueError(f"Unknown SCG trajectory '{task}'. Supported: {list(SCG_TRAJECTORY_LIBRARY)}")

    base = SCG_TRAJECTORY_LIBRARY[task_lc]
    center = tuple(float(x) for x in base.get("center", (0.0, 0.0, 1.0)))
    params: Dict[str, float] = {}
    for key, value in base.items():
        if key == "center":
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raise ValueError(f"Trajectory param '{key}' should be scalar, got sequence {value!r}")
        params[key] = _coerce_param_value(value, key)

    if overrides:
        for key, value in overrides.items():
            params[key] = _coerce_param_value(value, key)

    return TrajectoryConfig(task=task_lc, params=params, center=center)


def _coerce_param_value(value: float | int | str, key: str) -> float | str:
    """Normalize trajectory param scalars while preserving symbolic fields like 'plane'."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"Trajectory param '{key}' must be scalar numeric or string, got {value!r}")


def scg_position_velocity(
    task: str,
    t: float,
    *,
    params: Mapping[str, float] | None = None,
    center: Sequence[float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute position/velocity for the requested SCG trajectory at time ``t``."""

    cfg = get_scg_trajectory_config(task, overrides=params or {})
    base = np.array(center if center is not None else cfg.center, dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64)

    if cfg.task == "hover":
        return base, vel

    plane = str(cfg.params.get("plane", "xy")).lower()
    axis_a, axis_b = _resolve_plane_indices(plane)
    period = max(cfg.params.get("period", 5.0), 1e-6)
    omega = 2.0 * math.pi / period

    if cfg.task == "figure8":
        scale = cfg.params.get("scale", cfg.params.get("A", 0.8))
        A = cfg.params.get("A", scale)
        B = cfg.params.get("B", scale)
        coord_a = A * math.sin(omega * t)
        coord_b = B * math.sin(omega * t) * math.cos(omega * t)
        coord_a_dot = A * omega * math.cos(omega * t)
        coord_b_dot = B * omega * (math.cos(omega * t) ** 2 - math.sin(omega * t) ** 2)
        pos = base.copy()
        pos[axis_a] += coord_a
        pos[axis_b] += coord_b
        vel[axis_a] = coord_a_dot
        vel[axis_b] = coord_b_dot
        return pos, vel

    if cfg.task == "circle":
        radius = cfg.params.get("R", cfg.params.get("scale", 0.9))
        coord_a = radius * math.cos(omega * t)
        coord_b = radius * math.sin(omega * t)
        coord_a_dot = -radius * omega * math.sin(omega * t)
        coord_b_dot = radius * omega * math.cos(omega * t)
        pos = base.copy()
        pos[axis_a] += coord_a
        pos[axis_b] += coord_b
        vel[axis_a] = coord_a_dot
        vel[axis_b] = coord_b_dot
        return pos, vel

    if cfg.task == "helix":
        radius = cfg.params.get("R", cfg.params.get("scale", 0.7))
        vz = cfg.params.get("v_z", 0.1)
        coord_a = radius * math.cos(omega * t)
        coord_b = radius * math.sin(omega * t)
        coord_a_dot = -radius * omega * math.sin(omega * t)
        coord_b_dot = radius * omega * math.cos(omega * t)
        pos = base.copy()
        pos[axis_a] += coord_a
        pos[axis_b] += coord_b
        pos[2] += vz * t
        vel[axis_a] = coord_a_dot
        vel[axis_b] = coord_b_dot
        vel[2] = vz
        return pos, vel

    if cfg.task == "square":
        scale = cfg.params.get("scale", cfg.params.get("side", 0.8))
        segment_period = period / 4.0
        traverse_speed = scale / max(segment_period, 1e-6)
        cycle_time = t % period
        segment_index = int(math.floor(cycle_time / segment_period)) % 4
        segment_time = cycle_time - segment_index * segment_period
        segment_position = traverse_speed * segment_time
        coord_a = 0.0
        coord_b = 0.0
        coord_a_dot = 0.0
        coord_b_dot = 0.0
        if segment_index == 0:
            coord_b = segment_position
            coord_b_dot = traverse_speed
        elif segment_index == 1:
            coord_a = -segment_position
            coord_b = scale
            coord_a_dot = -traverse_speed
        elif segment_index == 2:
            coord_a = -scale
            coord_b = scale - segment_position
            coord_b_dot = -traverse_speed
        else:
            coord_a = -scale + segment_position
            coord_b = 0.0
            coord_a_dot = traverse_speed
        pos = base.copy()
        pos[axis_a] += coord_a
        pos[axis_b] += coord_b
        vel[axis_a] = coord_a_dot
        vel[axis_b] = coord_b_dot
        return pos, vel

    raise ValueError(f"Unsupported trajectory '{task}'.")


def scg_position(task: str, t: float, **kwargs) -> np.ndarray:
    """Convenience wrapper that returns only the position."""

    pos, _ = scg_position_velocity(task, t, **kwargs)
    return pos


def _resolve_plane_indices(plane: str) -> Tuple[int, int]:
    plane = (plane or "xy").lower()
    if len(plane) != 2 or plane[0] == plane[1]:
        return 0, 1
    return _AXIS_INDEX.get(plane[0], 0), _AXIS_INDEX.get(plane[1], 1)


__all__ = [
    "SCG_TRAJECTORY_LIBRARY",
    "TrajectoryConfig",
    "get_scg_trajectory_config",
    "scg_position_velocity",
    "scg_position",
]

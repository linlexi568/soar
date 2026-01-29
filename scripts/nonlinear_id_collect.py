#!/usr/bin/env python3
"""Collect SISO data for nonlinear-control analysis.

We focus on the lateral channel driven by body torque-x command (normalized `u_tx`).
The IsaacGymDroneEnv scales torques internally (u_tx * 0.002 -> N·m).

Outputs logged:
- y position, vy
- roll (phi), omega_x
- applied u_tx (normalized)

This is intentionally minimal and headless.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import sys

import numpy as np

# Ensure repo root import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "01_soar") not in sys.path:
    sys.path.insert(0, str(ROOT / "01_soar"))

from envs.isaac_gym_drone_env import IsaacGymDroneEnv


def _quat_to_euler_np(q: np.ndarray) -> np.ndarray:
    # q = [qx,qy,qz,qw]
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--mode", choices=["sine", "step"], default="sine")
    ap.add_argument("--amp", type=float, default=0.15, help="normalized u_tx amplitude")
    ap.add_argument("--freq", type=float, default=1.5, help="Hz for sine")
    ap.add_argument("--step_start", type=float, default=2.0, help="s")
    ap.add_argument("--step_dur", type=float, default=1.0, help="s")
    ap.add_argument("--out", default=str(ROOT / "results" / "id" / "u_tx_hover_sine.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = IsaacGymDroneEnv(num_envs=1, device=args.device, headless=True, duration_sec=args.duration)

    dt = 1.0 / float(env.control_freq)
    steps = int(args.duration / dt)
    warmup_steps = int(args.warmup / dt)

    # Action format expected by env: [N,6] = [fx,fy,fz,tx,ty,tz] in normalized units.
    # We'll keep hover thrust and excite only tx.
    u_hover_fz = 0.65

    time_s: list[float] = []
    u_tx: list[float] = []
    y_pos: list[float] = []
    y_vel: list[float] = []
    roll: list[float] = []
    omega_x: list[float] = []

    env.reset()

    for k in range(steps):
        t = k * dt

        if k < warmup_steps:
            tx = 0.0
        else:
            if args.mode == "sine":
                tx = float(args.amp * math.sin(2.0 * math.pi * args.freq * (t - args.warmup)))
            else:
                in_window = (t >= args.step_start) and (t < args.step_start + args.step_dur)
                tx = float(args.amp if in_window else 0.0)

        act = np.zeros((1, 6), dtype=np.float32)
        act[0, 2] = u_hover_fz
        act[0, 3] = tx

        obs, _, _, _ = env.step(_to_torch(act, device=str(env.device)))

        # obs values are numpy arrays
        pos = np.asarray(obs["position"][0], dtype=np.float32)
        vel = np.asarray(obs["velocity"][0], dtype=np.float32)
        quat = np.asarray(obs["orientation"][0], dtype=np.float32)
        omg = np.asarray(obs["angular_velocity"][0], dtype=np.float32)

        euler = _quat_to_euler_np(quat)

        time_s.append(float(t))
        u_tx.append(float(tx))
        y_pos.append(float(pos[1]))
        y_vel.append(float(vel[1]))
        roll.append(float(euler[0]))
        omega_x.append(float(omg[0]))

    env.close()

    payload = {
        "meta": {
            "mode": args.mode,
            "amp": args.amp,
            "freq": args.freq,
            "duration": args.duration,
            "warmup": args.warmup,
            "dt": dt,
            "hover_fz": u_hover_fz,
        },
        "t": time_s,
        "u_tx": u_tx,
        "y": y_pos,
        "vy": y_vel,
        "roll": roll,
        "omega_x": omega_x,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


def _to_torch(x: np.ndarray, device: str):
    # Local import to keep Isaac Gym import order safe (env already imported torch internally)
    import torch

    return torch.tensor(x, device=device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate LQR/PDFF Controllers using saved parameters from results/pdff/*.json
Only reproduction, NO tuning.

Usage:
    python baselines/eval_lqr.py --task circle --episodes 3
    python baselines/eval_lqr.py --task all
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any
import importlib.util

# Isaac Gym 路径优先配置 (same as eval_ppo.py)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ISAAC_GYM_PY = ROOT / "isaacgym" / "python"
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

_ISAAC_BINDINGS = _ISAAC_GYM_PY / "isaacgym" / "_bindings" / "linux-x86_64"
if _ISAAC_BINDINGS.exists():
    os.environ.setdefault("LD_LIBRARY_PATH", str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))

try:
    from isaacgym import gymapi  # type: ignore
except Exception:
    pass

import numpy as np
import torch
import math

def _load_class_from_file(file_path: Path, module_name: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

IsaacGymDroneEnv = _load_class_from_file(
    ROOT / '01_soar' / 'envs' / 'isaac_gym_drone_env.py',
    'isaac_gym_drone_env',
    'IsaacGymDroneEnv',
)

SCGExactRewardCalculator = _load_class_from_file(
    ROOT / '01_soar' / 'utils' / 'reward_scg_exact.py',
    'reward_scg_exact',
    'SCGExactRewardCalculator',
)

from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity

BENCHMARK_DIR = Path(__file__).parent.parent
RESULTS_DIR = BENCHMARK_DIR / "results" / "pdff"


def quat_to_euler_np(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


class IsaacLQRController:
    """LQR/PDFF controller - outputs physical values matching env's 6-dim input format."""
    def __init__(self, k_pos=4.0, k_vel=4.0, k_pos_z=None, k_vel_z=None,
                 k_att=12.0, k_omega=3.0, k_yaw=4.0, k_yaw_rate=0.8, att_scale=0.2,
                 thrust_clip=2.0, torque_clip=0.1, mass=0.027, g=9.81):
        self.k_pos = float(k_pos)
        self.k_vel = float(k_vel)
        self.k_pos_z = float(k_pos_z) if k_pos_z is not None else self.k_pos
        self.k_vel_z = float(k_vel_z) if k_vel_z is not None else self.k_vel
        self.k_att = float(k_att)
        self.k_omega = float(k_omega)
        self.k_yaw = float(k_yaw)
        self.k_yaw_rate = float(k_yaw_rate)
        self.att_scale = float(att_scale)
        self.thrust_clip = float(thrust_clip)
        self.torque_clip = float(torque_clip)
        self.mass = float(mass)
        self.g = float(g)
        self.hover_thrust = self.mass * self.g
        self.dt = 1.0 / 48.0

    def set_dt(self, dt):
        self.dt = float(dt)

    def reset(self):
        pass

    def compute(self, pos, vel, quat, omega, target_pos, target_vel=None, target_acc=None):
        if target_vel is None:
            target_vel = np.zeros(3)
        if target_acc is None:
            target_acc = np.zeros(3)
        pos = np.asarray(pos); vel = np.asarray(vel); omega = np.asarray(omega)
        target_pos = np.asarray(target_pos); target_vel = np.asarray(target_vel)
        target_acc = np.asarray(target_acc)
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        acc_des = np.array([
            self.k_pos * pos_err[0] + self.k_vel * vel_err[0],
            self.k_pos * pos_err[1] + self.k_vel * vel_err[1],
            self.k_pos_z * pos_err[2] + self.k_vel_z * vel_err[2]
        ])
        acc_des += target_acc
        acc_des[2] += self.g
        
        # Physical thrust (N)
        fz = self.mass * acc_des[2]
        fz = float(np.clip(fz, 0.0, self.thrust_clip * self.hover_thrust))
        
        roll_des = -acc_des[1] / self.g * self.att_scale
        pitch_des = acc_des[0] / self.g * self.att_scale
        rpy_des = np.array([roll_des, pitch_des, 0.0])
        rpy_cur = quat_to_euler_np(quat)
        rpy_err = rpy_des - rpy_cur
        yaw_err = -rpy_cur[2]
        
        # Physical torques (Nm)
        torque = np.array([
            self.k_att * rpy_err[0] - self.k_omega * omega[0],
            self.k_att * rpy_err[1] - self.k_omega * omega[1],
            self.k_yaw * yaw_err - self.k_yaw_rate * omega[2],
        ], dtype=np.float32)
        torque = np.clip(torque, -self.torque_clip, self.torque_clip)
        
        # Output physical values - env will scale them
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float32)


def compute_target_acc(task: str, t: float, cfg, center: np.ndarray) -> np.ndarray:
    acc = np.zeros(3, dtype=np.float32)
    period = cfg.params.get("period", 5.0)
    omega_val = 2.0 * math.pi / max(period, 1e-6)

    if task == 'figure8':
        scale = cfg.params.get("scale", cfg.params.get("A", 0.8))
        A = cfg.params.get("A", scale)
        B = cfg.params.get("B", scale)
        acc_a = -A * (omega_val**2) * math.sin(omega_val * t)
        acc_b = -2.0 * B * (omega_val**2) * math.sin(2.0 * omega_val * t)
        plane = str(cfg.params.get("plane", "xy")).lower()
        axis_a = 0 if plane[0] == 'x' else (1 if plane[0] == 'y' else 2)
        axis_b = 1 if plane[1] == 'y' else (2 if plane[1] == 'z' else 0)
        acc[axis_a] = acc_a
        acc[axis_b] = acc_b
    elif task == 'circle':
        pos, _ = scg_position_velocity(task, t, params=cfg.params, center=center)
        pos_rel = pos - center
        acc = -(omega_val**2) * pos_rel
    return acc


def load_params(task: str) -> dict:
    path = RESULTS_DIR / f"pdff_{task}.json"
    if not path.exists():
        print(f"[WARN] LQR/PDFF params not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params', {})


def evaluate_lqr(task: str, duration: float, episodes: int, num_envs: int, device: str = 'cuda') -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Evaluating LQR/PDFF on {task.upper()} trajectory")
    print(f"Duration: {duration}s, Episodes: {episodes}, Envs: {num_envs}")
    print(f"{'='*60}\n")

    params = load_params(task)
    if params is None:
        return None
    
    print(f"Loaded params: {params}")
    controller = IsaacLQRController(**params)

    env = IsaacGymDroneEnv(num_envs=num_envs, device=device, headless=True, duration_sec=duration)
    try:
        ctrl_freq = getattr(env, 'control_freq', 48.0)
        dt = 1.0 / float(ctrl_freq)
        reward_calc = SCGExactRewardCalculator(num_envs=num_envs, device=device)
        cfg = get_scg_trajectory_config(task)
        center = np.array(cfg.center, dtype=np.float32)

        def make_targets(t: float):
            pos, vel = scg_position_velocity(task, t, params=cfg.params, center=center)
            acc = compute_target_acc(task, t, cfg, center)
            return np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32), np.asarray(acc, dtype=np.float32)

        all_rewards = []
        all_rmses = []

        for ep in range(episodes):
            print(f"  Running episode {ep+1}/{episodes}...")
            t = 0.0
            tgt_pos0, tgt_vel0, _ = make_targets(t)
            env.reset(initial_pos=torch.tensor(tgt_pos0, device=device))
            reward_calc.reset(num_envs)
            controller.set_dt(dt)
            controller.reset()
            steps = int(duration * ctrl_freq)
            pos_errs = []

            for s in range(steps):
                obs = env.get_obs()
                tgt_pos, tgt_vel, tgt_acc = make_targets(t)
                forces = torch.zeros(num_envs, 6, device=device)
                for i in range(num_envs):
                    pos = np.asarray(obs['position'][i], dtype=np.float32)
                    vel = np.asarray(obs['velocity'][i], dtype=np.float32)
                    quat = np.asarray(obs['orientation'][i], dtype=np.float32)
                    omega_obs = np.asarray(obs['angular_velocity'][i], dtype=np.float32)
                    action4 = controller.compute(pos, vel, quat, omega_obs, tgt_pos, tgt_vel, tgt_acc)
                    forces[i, 2] = float(action4[0])
                    forces[i, 3] = float(action4[1])
                    forces[i, 4] = float(action4[2])
                    forces[i, 5] = float(action4[3])

                obs_next, _, done, _ = env.step(forces)
                pos_t = torch.tensor(obs_next['position'], device=device)
                vel_t = torch.tensor(obs_next['velocity'], device=device)
                quat_t = torch.tensor(obs_next['orientation'], device=device)
                omega_t = torch.tensor(obs_next['angular_velocity'], device=device)
                target_pos_t = torch.tensor(np.tile(tgt_pos, (num_envs, 1)), device=device)
                reward_calc.compute_step(pos_t, vel_t, quat_t, omega_t, target_pos_t, forces)
                pos_err = np.linalg.norm(obs_next['position'] - tgt_pos, axis=1)
                pos_errs.append(pos_err)
                t += dt

            comps = reward_calc.get_components()
            total_costs = comps['total_cost'].cpu().numpy()
            true_rewards = -total_costs
            all_rewards.extend(true_rewards.tolist())
            pos_errs_array = np.array(pos_errs)
            rmses = np.sqrt(np.mean(pos_errs_array**2, axis=0))
            all_rmses.extend(rmses.tolist())

        metrics = {
            'mean_true_reward': float(np.mean(all_rewards)),
            'std_true_reward': float(np.std(all_rewards)),
            'rmse_pos': float(np.mean(all_rmses)),
            'std_rmse_pos': float(np.std(all_rmses)),
        }
        
        print(f"\n{'='*60}")
        print(f"Results for {task.upper()}:")
        print(f"  Mean Reward: {metrics['mean_true_reward']:.3f} ± {metrics['std_true_reward']:.3f}")
        print(f"  RMSE (pos):  {metrics['rmse_pos']:.4f} ± {metrics['std_rmse_pos']:.4f}")
        print(f"{'='*60}")
        
        return metrics
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LQR/PDFF (reproduction only)")
    parser.add_argument('--task', type=str, default='all', choices=['circle', 'figure8', 'square', 'all'])
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--duration', type=float, default=5.0)
    args = parser.parse_args()

    tasks = ['circle', 'figure8', 'square'] if args.task == 'all' else [args.task]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for task in tasks:
        metrics = evaluate_lqr(task, args.duration, args.episodes, args.num_envs, device)
        if metrics:
            all_results[task] = metrics

    print("\n" + "=" * 80)
    print("LQR/PDFF EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Trajectory':<15} {'Mean Reward':<15} {'RMSE (pos)':<15}")
    print("-" * 50)
    for task, m in all_results.items():
        print(f"{task:<15} {m['mean_true_reward']:>12.3f}   {m['rmse_pos']:>12.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()

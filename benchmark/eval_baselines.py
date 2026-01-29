#!/usr/bin/env python3
"""
Reproduce benchmark results for PID/LQR/SoarControllers.
Uses saved parameters from results/ folder, no tuning.

Usage:
    python eval_baselines.py                     # Eval all controllers & tasks
    python eval_baselines.py --controller pid    # Only PID
    python eval_baselines.py --task circle       # Only circle trajectory
"""
import os
import sys
from pathlib import Path
import json
import argparse

# Isaac Gym 路径优先配置 (same pattern as eval_ppo.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
import importlib.util

# -------------------------------------------------------------------------
# Dynamic import helpers (avoid numeric folder naming issues)
# -------------------------------------------------------------------------
def _load_class_from_file(file_path: Path, module_name: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    cls = getattr(module, class_name)
    return cls

# Environment & Reward
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

# Trajectory utilities
from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity

# -------------------------------------------------------------------------
# Controller classes (copied from controllers.py for standalone)
# -------------------------------------------------------------------------
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


class IsaacPIDController:
    def __init__(self, kp_xy=8.0, kd_xy=4.0, ki_xy=0.0, kp_z=14.0, kd_z=6.0, ki_z=0.0,
                 kp_att=12.0, kd_att=2.0, kp_yaw=4.0, kd_yaw=0.8, att_scale=0.2,
                 thrust_clip=2.0, torque_clip=0.1, mass=0.027, g=9.81):
        self.kp_xy = float(kp_xy)
        self.kd_xy = float(kd_xy)
        self.ki_xy = float(ki_xy)
        self.kp_z = float(kp_z)
        self.kd_z = float(kd_z)
        self.ki_z = float(ki_z)
        self.kp_att = float(kp_att)
        self.kd_att = float(kd_att)
        self.kp_yaw = float(kp_yaw)
        self.kd_yaw = float(kd_yaw)
        self.att_scale = float(att_scale)
        self.thrust_clip = float(thrust_clip)
        self.torque_clip = float(torque_clip)
        self.mass = float(mass)
        self.g = float(g)
        self.hover_thrust = self.mass * self.g
        self.dt = 1.0 / 48.0
        self._int_xy = np.zeros(2, dtype=np.float32)
        self._int_z = 0.0

    def set_dt(self, dt):
        self.dt = float(dt)

    def reset(self):
        self._int_xy[...] = 0.0
        self._int_z = 0.0

    def compute(self, pos, vel, quat, omega, target_pos, target_vel=None, target_acc=None):
        if target_vel is None:
            target_vel = np.zeros(3)
        pos = np.asarray(pos); vel = np.asarray(vel); omega = np.asarray(omega)
        target_pos = np.asarray(target_pos); target_vel = np.asarray(target_vel)
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        acc_des = np.array([
            self.kp_xy * pos_err[0] + self.kd_xy * vel_err[0],
            self.kp_xy * pos_err[1] + self.kd_xy * vel_err[1],
            self.kp_z * pos_err[2] + self.kd_z * vel_err[2] + self.g,
        ])
        self._int_xy += pos_err[:2] * self.dt
        self._int_xy = np.clip(self._int_xy, -0.5, 0.5)
        acc_des[0] += self.ki_xy * self._int_xy[0]
        acc_des[1] += self.ki_xy * self._int_xy[1]
        self._int_z += pos_err[2] * self.dt
        self._int_z = float(np.clip(self._int_z, -0.5, 0.5))
        acc_des[2] += self.ki_z * self._int_z
        fz = self.mass * acc_des[2]
        fz = float(np.clip(fz, 0.0, self.thrust_clip * self.hover_thrust))
        roll_des = -acc_des[1] / self.g * self.att_scale
        pitch_des = acc_des[0] / self.g * self.att_scale
        rpy_des = np.array([roll_des, pitch_des, 0.0])
        rpy_cur = quat_to_euler_np(quat)
        rpy_err = rpy_des - rpy_cur
        yaw_err = -rpy_cur[2]
        torque = np.array([
            self.kp_att * rpy_err[0] - self.kd_att * omega[0],
            self.kp_att * rpy_err[1] - self.kd_att * omega[1],
            self.kp_yaw * yaw_err - self.kd_yaw * omega[2],
        ], dtype=np.float32)
        torque = np.clip(torque, -self.torque_clip, self.torque_clip)
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float32)


class IsaacLQRController:
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
        fz = self.mass * acc_des[2]
        fz = float(np.clip(fz, 0.0, self.thrust_clip * self.hover_thrust))
        roll_des = -acc_des[1] / self.g * self.att_scale
        pitch_des = acc_des[0] / self.g * self.att_scale
        rpy_des = np.array([roll_des, pitch_des, 0.0])
        rpy_cur = quat_to_euler_np(quat)
        rpy_err = rpy_des - rpy_cur
        yaw_err = -rpy_cur[2]
        torque = np.array([
            self.k_att * rpy_err[0] - self.k_omega * omega[0],
            self.k_att * rpy_err[1] - self.k_omega * omega[1],
            self.k_yaw * yaw_err - self.k_yaw_rate * omega[2],
        ], dtype=np.float32)
        torque = np.clip(torque, -self.torque_clip, self.torque_clip)
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float32)


# -------------------------------------------------------------------------
# Soar Controller (loads params from soar_controller.py)
# -------------------------------------------------------------------------
SoarController = _load_class_from_file(
    Path(__file__).parent / 'baselines' / 'soar_controller.py',
    'soar_controller',
    'SoarController',
)

# -------------------------------------------------------------------------
# Evaluation logic
# -------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"


def compute_target_acc(task: str, t: float, cfg, center: np.ndarray) -> np.ndarray:
    """Compute target acceleration for feedforward."""
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
    elif task == 'circle' or task == 'helix':
        pos, _ = scg_position_velocity(task, t, params=cfg.params, center=center)
        pos_rel = pos - center
        acc = -(omega_val**2) * pos_rel
        if task == 'helix':
            acc[2] = 0.0
    return acc


def evaluate_controller(controller, task: str, duration: float, episodes: int, num_envs: int, device: str = 'cuda'):
    """Evaluate a controller and return metrics."""
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
            t = 0.0
            tgt_pos0, tgt_vel0, _ = make_targets(t)
            env.reset(initial_pos=tgt_pos0)
            reward_calc.reset(num_envs)
            if hasattr(controller, 'set_dt'):
                controller.set_dt(dt)
            if hasattr(controller, 'reset'):
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
            total_costs = (comps['total_cost']).cpu().numpy()
            true_rewards = -total_costs
            all_rewards.extend(true_rewards.tolist())
            pos_errs_array = np.array(pos_errs)
            rmses = np.sqrt(np.mean(pos_errs_array**2, axis=0))
            all_rmses.extend(rmses.tolist())

        return {
            'mean_true_reward': float(np.mean(all_rewards)),
            'std_true_reward': float(np.std(all_rewards)),
            'rmse_pos': float(np.mean(all_rmses)),
            'std_rmse_pos': float(np.std(all_rmses)),
        }
    finally:
        env.close()


def load_pid_params(task: str) -> dict:
    """Load PID params from results/pid/*.json"""
    path = RESULTS_DIR / "pid" / f"pid_{task}.json"
    if not path.exists():
        print(f"[WARN] PID params not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params', {})


def load_lqr_params(task: str) -> dict:
    """Load LQR (stored as pdff) params from results/pdff/*.json"""
    path = RESULTS_DIR / "pdff" / f"pdff_{task}.json"
    if not path.exists():
        print(f"[WARN] LQR/PDFF params not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params', {})


def load_soar_params(task: str) -> dict:
    """Load Soar params from results/soar/*.json
    
    Note: SoarController is initialized with trajectory name, not individual params.
    This function returns the stored params for reference/verification only.
    """
    path = RESULTS_DIR / "soar" / f"{task}_soar.json"
    if not path.exists():
        print(f"[WARN] Soar params not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('final_params', {})


def main():
    parser = argparse.ArgumentParser(description="Reproduce benchmark results (no tuning)")
    parser.add_argument('--controller', type=str, default='all', choices=['pid', 'lqr', 'soar', 'all'])
    parser.add_argument('--task', type=str, default='all', choices=['circle', 'figure8', 'square', 'all'])
    parser.add_argument('--episodes', type=int, default=3, help='Evaluation episodes')
    parser.add_argument('--num-envs', type=int, default=64, help='Parallel environments')
    parser.add_argument('--duration', type=float, default=5.0, help='Episode duration (sec)')
    args = parser.parse_args()

    tasks = ['circle', 'figure8', 'square'] if args.task == 'all' else [args.task]
    controllers = ['pid', 'lqr', 'soar'] if args.controller == 'all' else [args.controller]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("Benchmark Reproduction (No Tuning)")
    print(f"Tasks: {tasks}")
    print(f"Controllers: {controllers}")
    print(f"Episodes: {args.episodes}, Envs: {args.num_envs}, Duration: {args.duration}s")
    print(f"Device: {device}")
    print("=" * 70)

    results_summary = []

    for task in tasks:
        for ctrl_name in controllers:
            print(f"\n🔄 Evaluating {ctrl_name.upper()} on {task.upper()}...")

            if ctrl_name == 'pid':
                params = load_pid_params(task)
                if params is None:
                    continue
                controller = IsaacPIDController(**params)
            elif ctrl_name == 'lqr':
                params = load_lqr_params(task)
                if params is None:
                    continue
                controller = IsaacLQRController(**params)
            elif ctrl_name == 'soar':
                params = load_soar_params(task)
                if params is None:
                    continue
                # SoarController uses trajectory name, not individual params
                controller = SoarController(trajectory=task)
            else:
                continue

            metrics = evaluate_controller(
                controller,
                task,
                args.duration,
                args.episodes,
                args.num_envs,
                device,
            )

            print(f"   ✅ {ctrl_name.upper()} | {task}")
            print(f"      Mean Reward: {metrics['mean_true_reward']:.4f} ± {metrics['std_true_reward']:.4f}")
            print(f"      RMSE Pos:    {metrics['rmse_pos']:.4f} ± {metrics['std_rmse_pos']:.4f} m")

            results_summary.append({
                'controller': ctrl_name,
                'task': task,
                **metrics,
            })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Controller':<12} {'Task':<10} {'Mean Reward':<15} {'RMSE (m)':<12}")
    print("-" * 50)
    for r in results_summary:
        print(f"{r['controller']:<12} {r['task']:<10} {r['mean_true_reward']:>12.4f}   {r['rmse_pos']:>10.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()

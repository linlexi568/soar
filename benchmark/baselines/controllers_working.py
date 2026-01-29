#!/usr/bin/env python3
"""
Tune PID/LQR baselines directly on Soar's Isaac Gym environment,
using the same SCG reward as training.

Strategy: start from reasonable initial gains, then do local random search
with annealed perturbation. Report best parameters and metrics.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple

# Import by file path to avoid numeric-leading package name issues
import sys
import importlib.util
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Note: Isaac Gym should be imported by the calling script BEFORE this module
# to avoid torch import order issues

try:
    import numpy as np
    import torch
except ModuleNotFoundError:
    np = None  # type: ignore
    torch = None  # type: ignore

def _load_class_from_file(file_path: Path, module_name: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    cls = getattr(module, class_name)
    return cls

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

from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity  # type: ignore


class IsaacPIDController:
    def __init__(self,
                 kp_xy: float = 8.0,
                 kd_xy: float = 4.0,
                 ki_xy: float = 0.0,
                 kp_z: float = 14.0,
                 kd_z: float = 6.0,
                 ki_z: float = 0.0,
                 kp_att: float = 12.0,
                 kd_att: float = 2.0,
                 kp_yaw: float = 4.0,
                 kd_yaw: float = 0.8,
                 att_scale: float = 0.2,
                 thrust_clip: float = 2.0,
                 torque_clip: float = 0.5,  # 归一化单位（Isaac Gym 期望）
                 mass: float = 0.027,
                 g: float = 9.81,
                 output_normalized: bool = True,
                 use_feedforward: bool = True):  # True: 输出归一化值
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
        self.output_normalized = output_normalized
        self.use_feedforward = use_feedforward
        # Isaac Gym 缩放因子
        self.FZ_SCALE = 0.027 * 9.81 / 0.65  # ≈ 0.408 N/unit
        self.TORQUE_SCALE = 0.002  # N·m/unit
        self.dt = 1.0 / 48.0
        self._int_xy = np.zeros(2, dtype=np.float32)
        self._int_z = 0.0

    def set_dt(self, dt: float):
        self.dt = float(dt)

    def reset(self):
        self._int_xy[...] = 0.0
        self._int_z = 0.0

    def compute(self, pos, vel, quat, omega, target_pos, target_vel=None):
        if target_vel is None or not self.use_feedforward:
            target_vel = np.zeros(3)
        pos = np.asarray(pos); vel = np.asarray(vel); omega = np.asarray(omega)
        target_pos = np.asarray(target_pos); target_vel = np.asarray(target_vel)

        pos_err = target_pos - pos
        vel_err = target_vel - vel
        # desired acceleration
        acc_des = np.array([
            self.kp_xy * pos_err[0] + self.kd_xy * vel_err[0],
            self.kp_xy * pos_err[1] + self.kd_xy * vel_err[1],
            self.kp_z * pos_err[2] + self.kd_z * vel_err[2] + self.g,
        ])
        # add integral terms with simple anti-windup
        self._int_xy += pos_err[:2] * self.dt
        self._int_xy = np.clip(self._int_xy, -0.5, 0.5)
        acc_des[0] += self.ki_xy * self._int_xy[0]
        acc_des[1] += self.ki_xy * self._int_xy[1]
        self._int_z += pos_err[2] * self.dt
        self._int_z = float(np.clip(self._int_z, -0.5, 0.5))
        acc_des[2] += self.ki_z * self._int_z
        # thrust z (body)
        fz = self.mass * acc_des[2]
        fz = float(np.clip(fz, 0.0, self.thrust_clip * self.hover_thrust))
        # desired roll/pitch from lateral acc
        roll_des = -acc_des[1] / self.g * self.att_scale
        pitch_des = acc_des[0] / self.g * self.att_scale
        rpy_des = np.array([roll_des, pitch_des, 0.0])
        # current euler from quat
        rpy_cur = quat_to_euler_np(quat)
        rpy_err = rpy_des - rpy_cur
        # yaw control towards 0
        yaw_err = -rpy_cur[2]
        torque = np.array([
            self.kp_att * rpy_err[0] - self.kd_att * omega[0],
            self.kp_att * rpy_err[1] - self.kd_att * omega[1],
            self.kp_yaw * yaw_err      - self.kd_yaw * omega[2],
        ], dtype=np.float32)
        
        if self.output_normalized:
            # 转换为归一化单位（Isaac Gym 期望）
            fz_norm = fz / self.FZ_SCALE
            torque_norm = torque / self.TORQUE_SCALE
            fz_norm = float(np.clip(fz_norm, 0.0, 1.3))
            torque_norm = np.clip(torque_norm, -self.torque_clip, self.torque_clip)
            return np.array([fz_norm, torque_norm[0], torque_norm[1], torque_norm[2]], dtype=np.float32)
        else:
            # 物理单位（用于其他环境）
            torque = np.clip(torque, -self.torque_clip * self.TORQUE_SCALE, self.torque_clip * self.TORQUE_SCALE)
            return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float32)


class IsaacLQRController:
    def __init__(self,
                 k_pos: float = 4.0,
                 k_vel: float = 4.0,
                 k_att: float = 12.0,
                 k_omega: float = 3.0,
                 k_yaw: float = 4.0,
                 k_yaw_rate: float = 0.8,
                 att_scale: float = 0.2,
                 thrust_clip: float = 2.0,
                 torque_clip: float = 0.1,
                 mass: float = 0.027,
                 g: float = 9.81):
        self.k_pos = float(k_pos)
        self.k_vel = float(k_vel)
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
        self._int_z = 0.0

    def set_dt(self, dt: float):
        self.dt = float(dt)

    def reset(self):
        self._int_z = 0.0

    def compute(self, pos, vel, quat, omega, target_pos, target_vel=None):
        if target_vel is None:
            target_vel = np.zeros(3)
        pos = np.asarray(pos); vel = np.asarray(vel); omega = np.asarray(omega)
        target_pos = np.asarray(target_pos); target_vel = np.asarray(target_vel)

        pos_err = target_pos - pos
        vel_err = target_vel - vel
        acc_des = self.k_pos * pos_err + self.k_vel * vel_err
        acc_des[2] += self.g
        # small integral on z to remove bias
        self._int_z += pos_err[2] * self.dt
        self._int_z = float(np.clip(self._int_z, -0.5, 0.5))
        acc_des[2] += 0.5 * self._int_z
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
            self.k_yaw * yaw_err     - self.k_yaw_rate * omega[2],
        ], dtype=np.float32)
        torque = np.clip(torque, -self.torque_clip, self.torque_clip)
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float32)


def quat_to_euler_np(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    # Roll
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # Yaw
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def evaluate_params(controller, task: str, duration: float, episodes: int = 3, num_envs: int = 1) -> Dict[str, Any]:
    if np is None or torch is None:
        raise RuntimeError("Required packages numpy/torch are not installed. Activate venv and pip install -r requirements.txt.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = IsaacGymDroneEnv(num_envs=num_envs, device=device, headless=True, duration_sec=duration)
    try:
        ctrl_freq = getattr(env, 'control_freq', 48.0)
        dt = 1.0 / float(ctrl_freq)

        reward_calc = SCGExactRewardCalculator(num_envs=num_envs, device=device)

        cfg = get_scg_trajectory_config(task)
        center = np.array(cfg.center, dtype=np.float32)

        def make_targets(t: float) -> Tuple[np.ndarray, np.ndarray]:
            pos, vel = scg_position_velocity(task, t, params=cfg.params, center=center)
            return np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32)

        # 批量评估：每个episode在所有环境中并行运行
        all_rewards = []
        all_rmses = []
        
        for ep in range(episodes):
            env.reset()
            reward_calc.reset(num_envs)
            if hasattr(controller, 'set_dt'):
                controller.set_dt(dt)
            if hasattr(controller, 'reset'):
                controller.reset()
            
            t = 0.0
            steps = int(duration * ctrl_freq)
            pos_errs = []
            
            for s in range(steps):
                obs = env.get_obs()
                tgt_pos, tgt_vel = make_targets(t)
                
                # 为每个环境计算控制动作（已经是归一化单位）
                forces = torch.zeros(num_envs, 6, device=device)
                normalized_actions = torch.zeros(num_envs, 4, device=device)
                
                for i in range(num_envs):
                    pos = np.asarray(obs['position'][i], dtype=np.float32)
                    vel = np.asarray(obs['velocity'][i], dtype=np.float32)
                    quat = np.asarray(obs['orientation'][i], dtype=np.float32)
                    omega = np.asarray(obs['angular_velocity'][i], dtype=np.float32)
                    
                    # PID 输出归一化单位（Isaac Gym 格式）
                    action4 = controller.compute(pos, vel, quat, omega, tgt_pos, tgt_vel)
                    
                    forces[i, 2] = float(action4[0])
                    forces[i, 3] = float(action4[1])
                    forces[i, 4] = float(action4[2])
                    forces[i, 5] = float(action4[3])
                    
                    normalized_actions[i, 0] = float(action4[0])
                    normalized_actions[i, 1] = float(action4[1])
                    normalized_actions[i, 2] = float(action4[2])
                    normalized_actions[i, 3] = float(action4[3])
                    normalized_actions[i, 3] = float(action4[3])
                
                obs_next, _, done, _ = env.step(forces)
                
                # 计算奖励（使用已归一化的 actions）
                pos_t = torch.tensor(obs_next['position'], device=device)
                vel_t = torch.tensor(obs_next['velocity'], device=device)
                quat_t = torch.tensor(obs_next['orientation'], device=device)
                omega_t = torch.tensor(obs_next['angular_velocity'], device=device)
                target_pos_t = torch.tensor(np.tile(tgt_pos, (num_envs, 1)), device=device)
                reward = reward_calc.compute_step(pos_t, vel_t, quat_t, omega_t, target_pos_t, normalized_actions)
                
                # 累积位置误差
                pos_err = np.linalg.norm(obs_next['position'] - tgt_pos, axis=1)
                pos_errs.append(pos_err)
                
                t += dt
            
            # Episode结束，收集所有环境的指标
            comps = reward_calc.get_components()
            total_costs = (comps['total_cost']).cpu().numpy()
            true_rewards = -total_costs
            all_rewards.extend(true_rewards.tolist())
            
            pos_errs_array = np.array(pos_errs)  # [steps, num_envs]
            rmses = np.sqrt(np.mean(pos_errs_array**2, axis=0))  # [num_envs]
            all_rmses.extend(rmses.tolist())
        
        return {
            'mean_true_reward': float(np.mean(all_rewards)),
            'std_true_reward': float(np.std(all_rewards)),
            'rmse_pos': float(np.mean(all_rmses)),
        }
    finally:
        env.close()


def local_random_search(base: Dict[str, float],
                        bounds: Dict[str, Tuple[float, float]],
                        trials: int,
                        eval_fn,
                        task: str,
                        duration: float,
                        episodes_per_eval: int = 3,
                        anneal: bool = True) -> Tuple[Dict[str, float], Dict[str, Any]]:
    best_params = dict(base)
    best_metrics = eval_fn(best_params, task, duration, episodes_per_eval)
    best_reward = best_metrics['mean_true_reward']
    print(f"[Init] Base reward={best_reward:.3f} rmse={best_metrics['rmse_pos']:.4f}")

    for i in range(trials):
        ratio = 1.0
        if anneal:
            # linearly shrink exploration
            ratio = max(0.2, 1.0 - i / trials)
        proposal = {}
        for k, v in base.items():
            lo, hi = bounds[k]
            span = (hi - lo) * 0.5 * ratio
            center = v
            proposal[k] = float(np.clip(center + np.random.uniform(-span, span), lo, hi))
        metrics = eval_fn(proposal, task, duration, episodes_per_eval)
        current_reward = metrics['mean_true_reward']
        
        # 每10次迭代显示一次当前进度
        if (i + 1) % 10 == 0:
            print(f"[Trial {i+1}/{trials}] Current={current_reward:.3f} Best={best_reward:.3f} (ratio={ratio:.2f})")
        
        if current_reward > best_reward:
            best_reward = current_reward
            best_params = proposal
            best_metrics = metrics
            print(f"  ✓ [IMPROVED] reward={best_reward:.3f} rmse={best_metrics['rmse_pos']:.4f}")
    
    print(f"[Final] Best reward={best_reward:.3f} after {trials} trials")
    return best_params, best_metrics


def build_controller_eval(algo: str, pid_mode: str = 'cascade', lqr_mode: str = 'pure', num_envs: int = 1):
    if algo == 'pid':
        def eval_pid(params: Dict[str, float], task: str, duration: float, episodes: int):
            ctrl = IsaacPIDController(**params)
            # normal mode: disable integrators and yaw PD, reduce attitude coupling
            if pid_mode == 'normal':
                ctrl.ki_xy = 0.0
                ctrl.ki_z = 0.0
                ctrl.kp_yaw = 0.0
                ctrl.kd_yaw = 0.0
                ctrl.att_scale = params.get('att_scale', ctrl.att_scale) * 0.1
            return evaluate_params(ctrl, task, duration, episodes, num_envs=num_envs)
        return eval_pid
    elif algo == 'lqr':
        def eval_lqr(params: Dict[str, float], task: str, duration: float, episodes: int):
            ctrl = IsaacLQRController(**params)
            if lqr_mode == 'pure':
                # remove z integral bias entirely
                ctrl._int_z = 0.0
                # ensure compute does not add integral: override by setting k to 0 via monkey patch
                original_compute = ctrl.compute
                def pure_compute(pos, vel, quat, omega, target_pos, target_vel=None):
                    if target_vel is None:
                        target_vel = np.zeros(3)
                    pos = np.asarray(pos); vel = np.asarray(vel); omega = np.asarray(omega)
                    target_pos = np.asarray(target_pos); target_vel = np.asarray(target_vel)
                    pos_err = target_pos - pos
                    vel_err = target_vel - vel
                    acc_des = ctrl.k_pos * pos_err + ctrl.k_vel * vel_err
                    acc_des[2] += ctrl.g
                    fz = ctrl.mass * acc_des[2]
                    fz = float(np.clip(fz, 0.0, ctrl.thrust_clip * ctrl.hover_thrust))
                    roll_des = -acc_des[1] / ctrl.g * ctrl.att_scale
                    pitch_des = acc_des[0] / ctrl.g * ctrl.att_scale
                    rpy_des = np.array([roll_des, pitch_des, 0.0])
                    rpy_cur = quat_to_euler_np(quat)
                    rpy_err = rpy_des - rpy_cur
                    yaw_err = -rpy_cur[2]
                    torque = np.array([
                        ctrl.k_att * rpy_err[0] - ctrl.k_omega * omega[0],
                        ctrl.k_att * rpy_err[1] - ctrl.k_omega * omega[1],
                        ctrl.k_yaw * yaw_err     - ctrl.k_yaw_rate * omega[2],
                    ], dtype=np.float32)
                    torque = np.clip(torque, -ctrl.torque_clip, ctrl.torque_clip)
                    return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float32)
                ctrl.compute = pure_compute  # type: ignore
            return evaluate_params(ctrl, task, duration, episodes, num_envs=num_envs)
        return eval_lqr
    else:
        raise ValueError('algo must be pid or lqr')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=str, default='square', choices=['hover','figure8','circle','square','helix'])
    ap.add_argument('--duration', type=float, default=5.0)
    ap.add_argument('--algo', type=str, default='both', choices=['pid','lqr','both'])
    ap.add_argument('--pid-mode', type=str, default='cascade', choices=['cascade','normal'])
    ap.add_argument('--lqr-mode', type=str, default='pure', choices=['pure','lqi'])
    ap.add_argument('--episodes-per-eval', type=int, default=3)
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--output', type=str, default='results/baseline_isaac/{task}_pid_lqr_scg_exact.json')
    args = ap.parse_args()

    out_path = Path(args.output.format(task=args.task))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {'task': args.task, 'duration': args.duration}

    if args.algo in ('pid','both'):
        base_pid = {
            'kp_xy': 8.0, 'kd_xy': 4.0,
            'kp_z': 14.0, 'kd_z': 6.0,
            'kp_att': 12.0, 'kd_att': 2.0,
            'att_scale': 0.2,
        }
        bounds_pid = {
            'kp_xy': (4.0, 20.0), 'kd_xy': (2.0, 8.0),
            'kp_z': (10.0, 25.0), 'kd_z': (4.0, 10.0),
            'kp_att': (8.0, 25.0), 'kd_att': (1.0, 4.0),
            'att_scale': (0.1, 0.3),
        }
        # normal PID: narrower att_scale and no integrators
        if args.pid_mode == 'normal':
            base_pid['att_scale'] = 0.1
            bounds_pid['att_scale'] = (0.05, 0.15)
        eval_fn = build_controller_eval('pid', pid_mode=args.pid_mode)
        best_pid, met_pid = local_random_search(base_pid, bounds_pid, args.trials, eval_fn,
                                                args.task, args.duration, args.episodes_per_eval)
        results[f"best_pid_{args.pid_mode}"] = {'params': best_pid, 'metrics': met_pid}
        print(f"[PID-{args.pid_mode}] Best reward={met_pid['mean_true_reward']:.3f}, rmse={met_pid['rmse_pos']:.3f}")

    if args.algo in ('lqr','both'):
        base_lqr = {
            'k_pos': 4.0, 'k_vel': 4.0,
            'k_att': 12.0, 'k_omega': 3.0,
            'att_scale': 0.2,
        }
        bounds_lqr = {
            'k_pos': (2.0, 10.0), 'k_vel': (2.0, 10.0),
            'k_att': (8.0, 25.0), 'k_omega': (2.0, 6.0),
            'att_scale': (0.1, 0.3),
        }
        eval_fn = build_controller_eval('lqr', lqr_mode=args.lqr_mode)
        best_lqr, met_lqr = local_random_search(base_lqr, bounds_lqr, args.trials, eval_fn,
                                                args.task, args.duration, args.episodes_per_eval)
        results[f"best_lqr_{args.lqr_mode}"] = {'params': best_lqr, 'metrics': met_lqr}
        print(f"[LQR-{args.lqr_mode}] Best reward={met_lqr['mean_true_reward']:.3f}, rmse={met_lqr['rmse_pos']:.3f}")

    with out_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved: {out_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Plot SCG exact cost components for SOAR vs PID vs LQR.

Outputs a 1x3 figure (figure8, square, circle) like nonlinear_analysis_plots_v2.png.

Components (SCG exact LQ cost):
- pos:   x,y,z
- vel:   vx,vy,vz
- att:   roll,pitch,yaw
- omega: wx,wy,wz
- action: u^T R u  (matched to reward calculator behavior)

Note: To stay consistent with existing benchmark evaluation code, we compute action-cost
on the same action tensor passed into SCGExactRewardCalculator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 字体设置（对标 scripts/plot_nonlinear_analysis.py） ---
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    print(f"Warning: Font file not found at {font_path}, trying generic family.")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# Ensure repo root is on sys.path
BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent
sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, str(REPO_ROOT))

# Import env by file path (keeps consistent with existing benchmark baselines)
import importlib.util


def _load_class_from_file(file_path: Path, module_name: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    cls = getattr(module, class_name)
    return cls


IsaacGymDroneEnv = _load_class_from_file(
    REPO_ROOT / '01_soar' / 'envs' / 'isaac_gym_drone_env.py',
    'isaac_gym_drone_env',
    'IsaacGymDroneEnv',
)

# Isaac Gym 必须先于 torch 导入：上面的 IsaacGymDroneEnv 模块内部已完成 isaacgym 导入。
import torch

from benchmark.envs.reward_calculator import SCG_ACTION_WEIGHT, SCG_STATE_WEIGHTS, SCGExactRewardCalculator
from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity  # type: ignore

from baselines.controllers import IsaacPIDController, IsaacLQRController
from baselines.soar_controller import SoarController


SCG_STATE_WEIGHTS = SCG_STATE_WEIGHTS.to(dtype=torch.float32)


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """[N,4] (qx,qy,qz,qw) -> [N,3] (roll,pitch,yaw)"""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=1)


def make_targets(task: str, cfg, center: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match acceleration feedforward logic used by benchmark baselines."""
    pos, vel = scg_position_velocity(task, t, params=cfg.params, center=center)
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

    elif task in ('circle', 'helix'):
        pos_arr = np.asarray(pos, dtype=np.float32)
        pos_rel = pos_arr - center
        acc = -(omega_val**2) * pos_rel
        if task == 'helix':
            acc[2] = 0.0

    return np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32), np.asarray(acc, dtype=np.float32)


def compute_cost_breakdown(
    pos: torch.Tensor,
    vel: torch.Tensor,
    quat: torch.Tensor,
    omega: torch.Tensor,
    target_pos: torch.Tensor,
    target_vel: torch.Tensor,
    action: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return per-env instantaneous cost components (not accumulated)."""
    N = pos.shape[0]
    Q = SCG_STATE_WEIGHTS.to(pos.device)

    euler = quat_to_euler(quat)

    pos_err = pos - target_pos
    # Important: match existing benchmark evaluation code path which uses
    # SCGExactRewardCalculator(target_vel=None) -> target_vel = 0.
    # Controllers can still use target_vel/feedforward; cost breakdown stays aligned.
    _ = target_vel
    vel_err = vel
    euler_err = euler  # ref euler = 0
    omega_err = omega  # ref omega = 0

    # indices: [x,vx,y,vy,z,vz,roll,pitch,yaw,wx,wy,wz]
    # pos: 0,2,4  vel:1,3,5  att:6,7,8  omega:9,10,11
    pos_cost = Q[0] * pos_err[:, 0] ** 2 + Q[2] * pos_err[:, 1] ** 2 + Q[4] * pos_err[:, 2] ** 2
    vel_cost = Q[1] * vel_err[:, 0] ** 2 + Q[3] * vel_err[:, 1] ** 2 + Q[5] * vel_err[:, 2] ** 2
    att_cost = Q[6] * euler_err[:, 0] ** 2 + Q[7] * euler_err[:, 1] ** 2 + Q[8] * euler_err[:, 2] ** 2
    omega_cost = Q[9] * omega_err[:, 0] ** 2 + Q[10] * omega_err[:, 1] ** 2 + Q[11] * omega_err[:, 2] ** 2

    # Match SCGExactRewardCalculator: use first 4 dims of action
    u = action[:, :4] if action.shape[1] >= 4 else action
    action_cost = float(SCG_ACTION_WEIGHT) * (u ** 2).sum(dim=1)

    total = pos_cost + vel_cost + att_cost + omega_cost + action_cost

    return {
        'pos': pos_cost,
        'vel': vel_cost,
        'att': att_cost,
        'omega': omega_cost,
        'action': action_cost,
        'total': total,
        'N': torch.full((N,), 1.0, device=pos.device),
    }


def load_best_params(results_dir: Path, algo: str, task: str) -> Dict[str, float]:
    path = results_dir / algo / f"{algo}_{task}.json"
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return {k: float(v) for k, v in data['best_params'].items()}


def evaluate_controller_components(
    controller_name: str,
    task: str,
    duration: float,
    episodes: int,
    device: str,
) -> Dict[str, float]:
    """Run episodes and return mean total cost per component."""

    # Instantiate controller
    results_dir = BENCH_ROOT / 'results'
    if controller_name == 'pid':
        params = load_best_params(results_dir, 'pid', task)
        controller = IsaacPIDController(**params)
    elif controller_name == 'lqr':
        params = load_best_params(results_dir, 'lqr', task)
        controller = IsaacLQRController(**params)
    elif controller_name == 'soar':
        controller = SoarController(trajectory=task)
    else:
        raise ValueError(controller_name)

    # Isaac Gym 这套 env 的 root state tensor 实际在 GPU pipeline 上。
    # 传 cpu 会导致 env.self.device=cpu 但内部张量在 cuda，从而 reset/赋值报错。
    # 因此这里强制：有 CUDA 就用 CUDA。
    sim_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != sim_device:
        print(f"[Info] Override device: requested={device} -> using={sim_device} (to match Isaac Gym tensors)")

    env = IsaacGymDroneEnv(num_envs=1, device=sim_device, headless=True, duration_sec=duration)
    reward_calc = SCGExactRewardCalculator(num_envs=1, device=sim_device)

    try:
        ctrl_freq = getattr(env, 'control_freq', 48.0)
        dt = 1.0 / float(ctrl_freq)

        cfg = get_scg_trajectory_config(task)
        center = np.array(cfg.center, dtype=np.float32)

        totals = {k: 0.0 for k in ('pos', 'vel', 'att', 'omega', 'action', 'total')}

        for ep in range(episodes):
            t = 0.0
            tgt_pos0, tgt_vel0, tgt_acc0 = make_targets(task, cfg, center, t)
            init_pos = torch.as_tensor(tgt_pos0, dtype=torch.float32, device=sim_device)
            env.reset(initial_pos=init_pos)

            reward_calc.reset(1)
            if hasattr(controller, 'set_dt'):
                controller.set_dt(dt)
            if hasattr(controller, 'reset'):
                controller.reset()

            steps = int(duration * ctrl_freq)

            for _ in range(steps):
                obs = env.get_obs()
                tgt_pos, tgt_vel, tgt_acc = make_targets(task, cfg, center, t)

                forces = torch.zeros(1, 6, device=sim_device)

                pos_np = np.asarray(obs['position'][0], dtype=np.float32)
                vel_np = np.asarray(obs['velocity'][0], dtype=np.float32)
                quat_np = np.asarray(obs['orientation'][0], dtype=np.float32)
                omega_np = np.asarray(obs['angular_velocity'][0], dtype=np.float32)

                action4 = controller.compute(pos_np, vel_np, quat_np, omega_np, tgt_pos, tgt_vel, tgt_acc)

                # Env expects normalized fz (u_fz where 0.65 ~= hover). PID/LQR controllers
                # compute fz in Newtons; convert only thrust to normalized scale.
                if controller_name in ('pid', 'lqr'):
                    hover_u = 0.65
                    fz_scale = float(getattr(controller, 'mass', 0.027) * getattr(controller, 'g', 9.81) / hover_u)
                    u_fz = float(action4[0]) / max(fz_scale, 1e-9)
                else:
                    u_fz = float(action4[0])

                forces[0, 2] = float(u_fz)
                forces[0, 3] = float(action4[1])
                forces[0, 4] = float(action4[2])
                forces[0, 5] = float(action4[3])

                obs_next, _, done, _ = env.step(forces)

                # Reward step (keeps consistent behavior)
                pos_t = torch.as_tensor(obs_next['position'], dtype=torch.float32, device=sim_device)
                vel_t = torch.as_tensor(obs_next['velocity'], dtype=torch.float32, device=sim_device)
                quat_t = torch.as_tensor(obs_next['orientation'], dtype=torch.float32, device=sim_device)
                omega_t = torch.as_tensor(obs_next['angular_velocity'], dtype=torch.float32, device=sim_device)
                target_pos_t = torch.as_tensor(np.tile(tgt_pos, (1, 1)), dtype=torch.float32, device=sim_device)
                target_vel_t = torch.as_tensor(np.tile(tgt_vel, (1, 1)), dtype=torch.float32, device=sim_device)

                reward_calc.compute_step(pos_t, vel_t, quat_t, omega_t, target_pos_t, forces)

                comps = compute_cost_breakdown(pos_t, vel_t, quat_t, omega_t, target_pos_t, target_vel_t, forces)
                for k in totals.keys():
                    totals[k] += float(comps[k][0].item())

                t += dt

        # mean per-episode totals
        for k in totals.keys():
            totals[k] /= float(episodes)

        return totals
    finally:
        env.close()


def plot_cost_components(out_path: Path, data: Dict[str, Dict[str, Dict[str, float]]]):
    """data[task][controller][component]"""

    tasks = ['figure8', 'square', 'circle']
    controllers = ['pid', 'lqr', 'soar']
    controller_labels = {'pid': 'PID', 'lqr': 'LQR', 'soar': 'SOAR'}

    components = ['pos', 'vel', 'att', 'omega', 'action']
    comp_labels = {
        'pos': 'Position',
        'vel': 'Velocity',
        'att': 'Attitude',
        'omega': 'Angular Rate',
        'action': 'Action',
    }
    comp_colors = {
        'pos': '#4C72B0',
        'vel': '#55A868',
        'att': '#C44E52',
        'omega': '#8172B2',
        'action': '#CCB974',
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for ax_i, task in enumerate(tasks):
        ax = axs[ax_i]
        x = np.arange(len(controllers))
        bottoms = np.zeros(len(controllers), dtype=np.float64)

        for comp in components:
            vals = [data[task][c][comp] for c in controllers]
            ax.bar(x, vals, bottom=bottoms, color=comp_colors[comp], label=comp_labels[comp])
            bottoms += np.array(vals, dtype=np.float64)

        ax.set_xticks(x)
        ax.set_xticklabels([controller_labels[c] for c in controllers], fontsize=18)
        ax.set_title(task.upper(), fontsize=20)
        ax.set_ylabel('Total Cost (per episode)', fontsize=20)
        ax.grid(True, axis='y', alpha=0.3)

        labels = ['(a)', '(b)', '(c)']
        ax.text(0.5, -0.25, labels[ax_i], transform=ax.transAxes,
                fontsize=20, fontweight='normal', ha='center', va='center')

    # Shared legend (like reference style)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, framealpha=0.9, fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=5.0)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--output', type=str, default='cost_components_plots.png')
    args = ap.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task in ('figure8', 'square', 'circle'):
        data[task] = {}
        for ctrl in ('pid', 'lqr', 'soar'):
            print(f"[Run] task={task} controller={ctrl} episodes={args.episodes} duration={args.duration}s")
            data[task][ctrl] = evaluate_controller_components(ctrl, task, args.duration, args.episodes, device)

    plot_cost_components(out_path, data)


if __name__ == '__main__':
    main()

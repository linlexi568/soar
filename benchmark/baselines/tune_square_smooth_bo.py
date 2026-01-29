#!/usr/bin/env python3
"""
Bayesian Optimization for Square trajectory using smooth approximation of sign

Usage:
    python baselines/tune_square_smooth_bo.py --trials 100 --episodes 3

This script tunes the parameters for:
    u_tx = -k_p * smooth(pos_err_y, s) + k_d * vel_y - k_w * ang_vel_x
    
where smooth(e, s) = s * tanh(e/s) with small s approximating bang-bang behavior.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
    import optuna
except ImportError:
    print("Error: Required packages not found.")
    print("Please activate your virtual environment and ensure optuna, torch, numpy are installed.")
    sys.exit(1)

# Import environment and reward calculator
import importlib.util

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

# Global environment (reused across trials for efficiency)
env = None
reward_calc = None


def smooth(val: float, s: float) -> float:
    """Smooth operator: s * tanh(val/s)
    
    When s → 0, this approximates sign(val), but remains Lipschitz continuous.
    """
    import math
    return s * math.tanh(val / s)


def evaluate_controller(k_p: float, k_s: float, k_d: float, k_w: float,
                        trajectory: str = 'square',
                        duration: float = 5.0,
                        episodes: int = 3) -> float:
    """Evaluate controller with given parameters.
    
    Returns:
        Mean cost (lower is better)
    """
    global env, reward_calc
    
    if env is None or reward_calc is None:
        raise RuntimeError("Environment not initialized. Call setup_environment() first.")
    
    cfg = get_scg_trajectory_config(trajectory)
    center = np.array(cfg.center, dtype=np.float32)
    
    def get_target(t: float):
        pos, vel = scg_position_velocity(trajectory, t, params=cfg.params, center=center)
        return np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32)
    
    ctrl_freq = getattr(env, 'control_freq', 48.0)
    dt = 1.0 / float(ctrl_freq)
    
    all_costs = []
    
    for ep in range(episodes):
        # Reset at trajectory start
        t = 0.0
        tgt_pos0, _ = get_target(t)
        env.reset(initial_pos=tgt_pos0)
        reward_calc.reset(1)
        
        steps = int(duration * ctrl_freq)
        
        for s in range(steps):
            obs = env.get_obs()
            tgt_pos, tgt_vel = get_target(t)
            
            # Get state
            pos = obs['position'][0].cpu().numpy()
            vel = obs['velocity'][0].cpu().numpy()
            quat = obs['orientation'][0].cpu().numpy()
            omega = obs['angular_velocity'][0].cpu().numpy()
            
            # Compute errors
            pos_err = tgt_pos - pos
            
            # Yaw control (same for all)
            import math
            qx, qy, qz, qw = quat
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            yaw_err = 0.0 - yaw
            while yaw_err > math.pi: yaw_err -= 2*math.pi
            while yaw_err < -math.pi: yaw_err += 2*math.pi
            
            # Control law with smooth
            u_tx = -k_p * smooth(pos_err[1], k_s) + k_d * vel[1] - k_w * omega[0]
            u_ty = k_p * smooth(pos_err[0], k_s) - k_d * vel[0] - k_w * omega[1]
            u_tz = 4.0 * yaw_err - 0.8 * omega[2]
            u_fz = 1.0 * pos_err[2] - 0.5 * vel[2] + 0.65
            
            # Clip
            u_tx = np.clip(u_tx, -0.5, 0.5)
            u_ty = np.clip(u_ty, -0.5, 0.5)
            u_tz = np.clip(u_tz, -0.5, 0.5)
            u_fz = np.clip(u_fz, 0.0, 1.3)
            
            # Step
            forces = torch.zeros(1, 6, device=env.device)
            forces[0, 2] = u_fz
            forces[0, 3] = u_tx
            forces[0, 4] = u_ty
            forces[0, 5] = u_tz
            
            obs_next, _, _, _ = env.step(forces)
            
            # Compute reward
            pos_t = torch.tensor(obs_next['position'], device=env.device)
            vel_t = torch.tensor(obs_next['velocity'], device=env.device)
            quat_t = torch.tensor(obs_next['orientation'], device=env.device)
            omega_t = torch.tensor(obs_next['angular_velocity'], device=env.device)
            target_pos_t = torch.tensor([tgt_pos], device=env.device)
            reward_calc.compute_step(pos_t, vel_t, quat_t, omega_t, target_pos_t, forces)
            
            t += dt
        
        # Get episode cost
        comps = reward_calc.get_components()
        cost = comps['total_cost'][0].item()
        all_costs.append(cost)
    
    return float(np.mean(all_costs))


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function."""
    # Sample parameters
    k_p = trial.suggest_float('k_p', 0.5, 8.0)
    k_s = trial.suggest_float('k_s', 0.05, 0.3)  # small s for bang-bang approximation
    k_d = trial.suggest_float('k_d', 0.5, 3.0)
    k_w = trial.suggest_float('k_w', 0.3, 1.5)
    
    try:
        cost = evaluate_controller(k_p, k_s, k_d, k_w,
                                   trajectory='square',
                                   duration=5.0,
                                   episodes=3)
        return cost
    except Exception as e:
        print(f"Trial failed: {e}")
        return 1e6  # Large penalty for failed trials


def setup_environment(device: str = 'cuda:0'):
    """Initialize global environment."""
    global env, reward_calc
    
    print("Initializing Isaac Gym environment...")
    env = IsaacGymDroneEnv(
        num_envs=1,
        device=device,
        headless=True,
        duration_sec=5.0
    )
    reward_calc = SCGExactRewardCalculator(num_envs=1, device=device)
    print("✓ Environment ready")


def main():
    parser = argparse.ArgumentParser(description='BO tuning for Square trajectory with smooth')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of BO trials')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Episodes per evaluation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Compute device')
    parser.add_argument('--output', type=str, 
                        default='results/soar/square_smooth_bo_tuning.json',
                        help='Output path for results')
    args = parser.parse_args()
    
    print("="*70)
    print("Bayesian Optimization: Square Trajectory with Smooth (Lipschitz)")
    print("="*70)
    print(f"Trials: {args.trials}")
    print(f"Episodes per trial: {args.episodes}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Setup
    setup_environment(args.device)
    
    # Create study
    study = optuna.create_study(direction='minimize')
    
    print("\nStarting optimization...")
    study.optimize(objective, n_trials=args.trials)
    
    # Results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Best cost: {study.best_value:.3f}")
    print(f"Best parameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val:.6f}")
    
    # Save results
    results = {
        'trajectory': 'square',
        'method': 'smooth_approximation',
        'best_cost': float(study.best_value),
        'best_params': study.best_params,
        'n_trials': args.trials,
        'note': 'Using smooth(e, s) instead of sign(e) for Lipschitz continuity'
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("="*70)
    
    # Cleanup
    if env is not None:
        env.close()


if __name__ == '__main__':
    main()

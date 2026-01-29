#!/usr/bin/env python3
"""
Evaluate Soar DSL Controllers in Benchmark Environment

This script evaluates the optimal Soar control rules from manual.md
on the same Isaac Gym environment and SCG reward used for PPO/PID/LQR benchmarks.

Usage:
    python baselines/eval_soar.py --task circle --duration 5.0 --episodes 10
    python baselines/eval_soar.py --task all  # Test all trajectories
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

def _load_class_from_file(file_path: Path, module_name: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    cls = getattr(module, class_name)
    return cls

# Load environment and reward calculator
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

# Load trajectory utilities
from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity  # type: ignore

# Load Soar controller (relative import within benchmark)
soar_controller_path = Path(__file__).parent / 'soar_controller.py'
SoarController = _load_class_from_file(
    soar_controller_path,
    'soar_controller',
    'SoarController',
)


def evaluate_soar(trajectory: str, duration: float = 5.0, episodes: int = 10, num_envs: int = 1, device: str = 'cuda') -> Dict[str, Any]:
    """Evaluate Soar controller on a specific trajectory.
    
    Args:
        trajectory: Trajectory type ('figure8', 'square', 'circle')
        duration: Episode duration (seconds)
        episodes: Number of evaluation episodes
        num_envs: Number of parallel environments
        device: Compute device ('cuda' or 'cpu')
    
    Returns:
        Dictionary with evaluation metrics:
        - mean_true_reward: Mean SCG reward (negative cost)
        - std_true_reward: Standard deviation of reward
        - rmse_pos: Root mean squared position error
        - mean_cost: Mean SCG cost
        - final_params: Soar parameters used
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Soar on {trajectory.upper()} trajectory")
    print(f"Duration: {duration}s, Episodes: {episodes}, Envs per episode: {num_envs}")
    print(f"{'='*60}\n")
    
    # Initialize environment
    env = IsaacGymDroneEnv(
        num_envs=num_envs,
        device=device,
        headless=True,
        duration_sec=duration
    )
    
    try:
        ctrl_freq = getattr(env, 'control_freq', 48.0)
        dt = 1.0 / float(ctrl_freq)
        
        # Initialize reward calculator
        reward_calc = SCGExactRewardCalculator(num_envs=num_envs, device=device)
        
        # Initialize Soar controller
        controller = SoarController(trajectory=trajectory)
        controller.set_dt(dt)
        
        # Get trajectory configuration
        cfg = get_scg_trajectory_config(trajectory)
        center = np.array(cfg.center, dtype=np.float32)
        
        # Trajectory target function
        def get_target(t: float):
            pos, vel = scg_position_velocity(trajectory, t, params=cfg.params, center=center)
            return np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32)
        
        # Run evaluation episodes
        all_rewards = []
        all_costs = []
        all_rmses = []
        
        for ep in range(episodes):
            # Reset environment at trajectory start (t=0)
            t = 0.0
            tgt_pos0, tgt_vel0 = get_target(t)
            # Convert to torch tensor for Isaac Gym
            tgt_pos0_torch = torch.as_tensor(tgt_pos0, dtype=torch.float32, device=device)
            env.reset(initial_pos=tgt_pos0_torch)
            reward_calc.reset(num_envs)
            controller.reset()
            
            steps = int(duration * ctrl_freq)
            pos_errs = []
            
            if (ep + 1) % max(1, episodes // 10) == 0:
                print(f"  Running episode {ep+1}/{episodes}...")
            
            for s in range(steps):
                obs = env.get_obs()
                tgt_pos, tgt_vel = get_target(t)
                
                # Compute control for each environment
                forces = torch.zeros(num_envs, 6, device=device)
                for i in range(num_envs):
                    # obs is already numpy arrays from get_obs()
                    pos = np.asarray(obs['position'][i], dtype=np.float32)
                    vel = np.asarray(obs['velocity'][i], dtype=np.float32)
                    quat = np.asarray(obs['orientation'][i], dtype=np.float32)
                    omega = np.asarray(obs['angular_velocity'][i], dtype=np.float32)
                    
                    # Soar control
                    action4 = controller.compute(pos, vel, quat, omega, tgt_pos, tgt_vel)
                    
                    # Map to Isaac Gym action format [0, 0, fz, tx, ty, tz]
                    forces[i, 2] = float(action4[0])  # fz
                    forces[i, 3] = float(action4[1])  # tx
                    forces[i, 4] = float(action4[2])  # ty
                    forces[i, 5] = float(action4[3])  # tz
                
                # Step environment
                obs_next, _, done, _ = env.step(forces)
                
                # Compute reward - convert obs to tensors for reward calculator
                pos_t = torch.as_tensor(obs_next['position'], dtype=torch.float32, device=device)
                vel_t = torch.as_tensor(obs_next['velocity'], dtype=torch.float32, device=device)
                quat_t = torch.as_tensor(obs_next['orientation'], dtype=torch.float32, device=device)
                omega_t = torch.as_tensor(obs_next['angular_velocity'], dtype=torch.float32, device=device)
                target_pos_t = torch.as_tensor(np.tile(tgt_pos, (num_envs, 1)), dtype=torch.float32, device=device)
                reward = reward_calc.compute_step(pos_t, vel_t, quat_t, omega_t, target_pos_t, forces)
                
                # Accumulate position error
                pos_err = np.linalg.norm(obs_next['position'] - tgt_pos, axis=1)
                pos_errs.append(pos_err)
                
                t += dt
            
            # Episode finished - collect metrics
            comps = reward_calc.get_components()
            total_costs = comps['total_cost'].cpu().numpy()
            true_rewards = -total_costs
            
            all_rewards.extend(true_rewards.tolist())
            all_costs.extend(total_costs.tolist())
            
            pos_errs_array = np.array(pos_errs)  # [steps, num_envs]
            rmses = np.sqrt(np.mean(pos_errs_array**2, axis=0))  # [num_envs]
            all_rmses.extend(rmses.tolist())
        
        # Compute final statistics
        results = {
            'trajectory': trajectory,
            'duration': duration,
            'episodes': episodes,
            'num_envs': num_envs,
            'mean_true_reward': float(np.mean(all_rewards)),
            'std_true_reward': float(np.std(all_rewards)),
            'mean_cost': float(np.mean(all_costs)),
            'std_cost': float(np.std(all_costs)),
            'rmse_pos': float(np.mean(all_rmses)),
            'std_rmse_pos': float(np.std(all_rmses)),
            'final_params': {
                'k_p': controller.k_p,
                'k_d': controller.k_d,
                'k_w': controller.k_w,
                'k_s': controller.k_s,
                'nonlinear': controller.nonlinear,
                'reference_cost': controller.params['cost']  # From manual.md
            }
        }
        
        print(f"\n{'='*60}")
        print(f"Results for {trajectory.upper()}:")
        print(f"  Mean Reward: {results['mean_true_reward']:.3f} ± {results['std_true_reward']:.3f}")
        print(f"  Mean Cost:   {results['mean_cost']:.3f} ± {results['std_cost']:.3f}")
        print(f"  RMSE (pos):  {results['rmse_pos']:.4f} ± {results['std_rmse_pos']:.4f}")
        print(f"  Reference Cost (manual.md): {results['final_params']['reference_cost']:.2f}")
        print(f"{'='*60}\n")
        
        return results
        
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Soar DSL controllers')
    parser.add_argument('--task', type=str, default='circle',
                        choices=['figure8', 'square', 'circle', 'all'],
                        help='Trajectory to evaluate (or "all" for all trajectories)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Episode duration in seconds')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of parallel environments per episode')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Compute device')
    parser.add_argument('--output', type=str, default='results/soar/{task}_soar.json',
                        help='Output path for results (use {task} for task name)')
    args = parser.parse_args()
    
    # Determine which tasks to run
    if args.task == 'all':
        tasks = ['figure8', 'square', 'circle']
    else:
        tasks = [args.task]
    
    # Run evaluations
    all_results = {}
    
    for task in tasks:
        results = evaluate_soar(
            trajectory=task,
            duration=args.duration,
            episodes=args.episodes,
            num_envs=args.num_envs,
            device=args.device
        )
        
        all_results[task] = results
        
        # Save individual result
        out_path = Path(args.output.format(task=task))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {out_path}")
    
    # Save combined results if multiple tasks
    if len(tasks) > 1:
        combined_path = Path(args.output.format(task='all'))
        with combined_path.open('w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved combined results: {combined_path}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SOAR EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Trajectory':<15} {'Mean Reward':<15} {'Mean Cost':<15} {'RMSE (pos)':<15}")
    print(f"{'-'*80}")
    for task, res in all_results.items():
        print(f"{task:<15} {res['mean_true_reward']:>14.3f} {res['mean_cost']:>14.3f} {res['rmse_pos']:>14.4f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

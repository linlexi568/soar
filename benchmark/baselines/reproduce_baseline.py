#!/usr/bin/env python3
"""
Reproduce PID/LQR results from benchmark/results/*.json
Uses the exact same evaluation logic as the original tuning scripts.

Usage:
    python baselines/reproduce_baseline.py --controller pid --task circle
    python baselines/reproduce_baseline.py --controller all --task all
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# === Isaac Gym 路径配置 (必须在导入 torch 之前) ===
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ISAAC_GYM_PY = ROOT / "isaacgym" / "python"
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

# Try alternate location
_ALT_ISAAC = Path("/home/linlexi/桌面/soar/isaacgym/python")
if _ALT_ISAAC.exists() and str(_ALT_ISAAC) not in sys.path:
    sys.path.insert(0, str(_ALT_ISAAC))

# Set LD_LIBRARY_PATH
for isaac_path in [_ISAAC_GYM_PY, _ALT_ISAAC]:
    _bindings = isaac_path / "isaacgym" / "_bindings" / "linux-x86_64"
    if _bindings.exists():
        os.environ["LD_LIBRARY_PATH"] = str(_bindings) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        break

# Import isaacgym BEFORE torch
try:
    from isaacgym import gymapi
except Exception:
    pass

# Now safe to import everything else
import argparse
import json

# Add benchmark to path for imports
BENCHMARK_DIR = Path(__file__).parent.parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from baselines.controllers import (
    IsaacPIDController,
    IsaacLQRController,
    evaluate_params,
)
# Alias for compatibility
IsaacPDFFController = IsaacLQRController

BENCHMARK_DIR = Path(__file__).parent.parent
RESULTS_DIR = BENCHMARK_DIR / "results"


def load_pid_params(task: str) -> dict:
    path = RESULTS_DIR / "pid" / f"pid_{task}.json"
    if not path.exists():
        print(f"[WARN] PID params not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params', {}), data.get('metrics', {})


def load_lqr_params(task: str) -> dict:
    """Load LQR/PDFF params from results/pdff/*.json"""
    path = RESULTS_DIR / "pdff" / f"pdff_{task}.json"
    if not path.exists():
        print(f"[WARN] LQR/PDFF params not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params', {}), data.get('metrics', {})


def main():
    parser = argparse.ArgumentParser(description="Reproduce PID/LQR results")
    parser.add_argument('--controller', type=str, default='all', choices=['pid', 'lqr', 'all'])
    parser.add_argument('--task', type=str, default='all', choices=['circle', 'figure8', 'square', 'all'])
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--duration', type=float, default=5.0)
    args = parser.parse_args()

    tasks = ['circle', 'figure8', 'square'] if args.task == 'all' else [args.task]
    controllers = ['pid', 'lqr'] if args.controller == 'all' else [args.controller]

    print("=" * 80)
    print("Baseline Reproduction (NO TUNING)")
    print(f"Tasks: {tasks}")
    print(f"Controllers: {controllers}")
    print(f"Episodes: {args.episodes}, Envs: {args.num_envs}, Duration: {args.duration}s")
    print("=" * 80)

    all_results = []

    for ctrl_name in controllers:
        for task in tasks:
            print(f"\n🔄 Reproducing {ctrl_name.upper()} on {task.upper()}...")

            if ctrl_name == 'pid':
                result = load_pid_params(task)
                if result is None:
                    continue
                params, stored_metrics = result
                print(f"   Stored reward: {stored_metrics.get('mean_true_reward', 'N/A')}")
                controller = IsaacPIDController(**params)
            elif ctrl_name == 'lqr':
                result = load_lqr_params(task)
                if result is None:
                    continue
                params, stored_metrics = result
                print(f"   Stored reward: {stored_metrics.get('mean_true_reward', 'N/A')}")
                controller = IsaacPDFFController(**params)

            metrics = evaluate_params(
                controller,
                task,
                args.duration,
                args.episodes,
                args.num_envs,
            )

            print(f"   ✅ {ctrl_name.upper()} | {task}")
            print(f"      Reproduced Reward: {metrics['mean_true_reward']:.4f}")
            print(f"      Stored Reward:     {stored_metrics.get('mean_true_reward', 'N/A')}")
            print(f"      RMSE Pos:          {metrics['rmse_pos']:.4f} m")

            all_results.append({
                'controller': ctrl_name,
                'task': task,
                'reproduced': metrics['mean_true_reward'],
                'stored': stored_metrics.get('mean_true_reward'),
                'rmse': metrics['rmse_pos'],
            })

    print("\n" + "=" * 80)
    print("REPRODUCTION SUMMARY")
    print("=" * 80)
    print(f"{'Controller':<12} {'Task':<12} {'Reproduced':<15} {'Stored':<15} {'Match?':<10}")
    print("-" * 70)
    for r in all_results:
        stored = r['stored']
        reproduced = r['reproduced']
        if stored is not None:
            diff_pct = abs(reproduced - stored) / abs(stored) * 100 if stored != 0 else float('inf')
            match = "✅" if diff_pct < 20 else "❌"
        else:
            match = "N/A"
        print(f"{r['controller']:<12} {r['task']:<12} {reproduced:>12.4f}   {stored if stored else 'N/A':>12}   {match}")
    print("=" * 80)


if __name__ == '__main__':
    main()

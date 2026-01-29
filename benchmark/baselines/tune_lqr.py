#!/usr/bin/env python3
"""LQR Parameter Tuning for Benchmark (Classic LQR based on small-angle linearized model)"""
import os
import sys
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaacgym" / "python"))

_ISAAC_BINDINGS = ROOT / "isaacgym" / "python" / "isaacgym" / "_bindings" / "linux-x86_64"
if _ISAAC_BINDINGS.exists():
    os.environ.setdefault("LD_LIBRARY_PATH", str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))

try:
    from isaacgym import gymapi
except Exception:
    pass

import numpy as np

BENCHMARK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from baselines.controllers import build_controller_eval, local_random_search


# LQR 基准参数 (基于小角度平动双积分模型)
BASE_PARAMS = {
    'Q_pos': 10.0,      # Position error weight
    'Q_vel': 1.0,       # Velocity error weight
    'Q_att': 5.0,       # Attitude error weight
    'Q_omega': 0.5,     # Angular velocity weight
    'R_thrust': 0.1,    # Thrust control cost
    'R_torque': 0.1,    # Torque control cost
}

# 搜索范围
BOUNDS = {
    'Q_pos': (1.0, 50.0),
    'Q_vel': (0.1, 10.0),
    'Q_att': (1.0, 20.0),
    'Q_omega': (0.1, 5.0),
    'R_thrust': (0.01, 1.0),
    'R_torque': (0.01, 1.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        help='Trajectory type: hover/circle/figure8/square/helix')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of random search trials')
    parser.add_argument('--num-envs', type=int, default=1024,
                        help='Number of parallel environments')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Evaluation duration in seconds')
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 创建结果目录
    results_dir = BENCHMARK_DIR / "results" / "lqr"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"lqr_{args.task}.json"

    print(f"🎯 Tuning LQR (classic) for {args.task} task")
    print(f"   Trials: {args.trials}")
    print(f"   Num envs: {args.num_envs}")

    # 构建评估函数
    eval_fn = build_controller_eval('lqr', lqr_mode='pure', num_envs=args.num_envs)

    # 随机搜索
    best_params, metrics = local_random_search(
        BASE_PARAMS,
        BOUNDS,
        args.trials,
        eval_fn,
        args.task,
        args.duration,
        episodes_per_eval=1,
    )

    # 保存结果
    result = {
        'task': args.task,
        'duration_sec': args.duration,
        'controller': 'lqr',
        'seed': args.seed,
        'trials': args.trials,
        'episodes_per_eval': 1,
        'best_params': best_params,
        'metrics': metrics,
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("LQR Tuning Complete")
    print(f" Task: {args.task}, Duration: {args.duration}s")
    print(f" Best mean reward: {metrics['mean_true_reward']:.2f}")
    print(f" Position RMSE: {metrics['rmse_pos']:.4f} m")
    print(" Best params:")
    for k, v in best_params.items():
        print(f"   {k}: {v:.4f}")
    print(f"\n✅ Results saved to {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

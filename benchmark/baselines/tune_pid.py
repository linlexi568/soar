#!/usr/bin/env python3
"""PID Parameter Tuning for Benchmark"""
import os
import sys
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Isaac Gym 路径 - 使用上层目录的 soar
_ISAAC_GYM_PATH = Path(__file__).resolve().parents[3] / "soar" / "isaacgym" / "python"
if _ISAAC_GYM_PATH.exists():
    sys.path.insert(0, str(_ISAAC_GYM_PATH))
    _ISAAC_BINDINGS = _ISAAC_GYM_PATH / "isaacgym" / "_bindings" / "linux-x86_64"
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


# PID 基准参数
BASE_PARAMS = {
    'kp_xy': 8.0,
    'kd_xy': 4.0,
    'ki_xy': 0.02,
    'kp_z': 14.0,
    'kd_z': 6.0,
    'ki_z': 0.05,
    'kp_att': 12.0,
    'kd_att': 2.0,
    'kp_yaw': 4.0,
    'kd_yaw': 0.8,
    'att_scale': 0.2,
}

# 搜索范围
BOUNDS = {
    'kp_xy': (4.0, 20.0),
    'kd_xy': (2.0, 8.0),
    'ki_xy': (0.0, 0.2),
    'kp_z': (10.0, 25.0),
    'kd_z': (4.0, 10.0),
    'ki_z': (0.0, 0.2),
    'kp_att': (8.0, 25.0),
    'kd_att': (1.0, 4.0),
    'kp_yaw': (0.0, 8.0),
    'kd_yaw': (0.0, 2.0),
    'att_scale': (0.05, 0.4),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        help='Trajectory type: hover/circle/figure8/square/helix')
    parser.add_argument('--trials', type=int, default=15,
                        help='Number of random search trials')
    parser.add_argument('--num-envs', type=int, default=1024,
                        help='Number of parallel environments')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Evaluation duration in seconds')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 创建结果目录
    results_dir = BENCHMARK_DIR / "results" / "pid"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"pid_{args.task}.json"

    print(f"🎯 Tuning PID for {args.task} task")
    print(f"   Trials: {args.trials}")
    print(f"   Num envs: {args.num_envs}")

    # 构建评估函数
    eval_fn = build_controller_eval('pid', pid_mode='cascade', num_envs=args.num_envs)

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
        'controller': 'pid',
        'seed': args.seed,
        'trials': args.trials,
        'episodes_per_eval': 1,
        'best_params': best_params,
        'metrics': metrics,
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("PID Tuning Complete")
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

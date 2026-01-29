#!/usr/bin/env python3
"""Cascaded PID (CPID) tuning with SCG reward alignment."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ⚠️ CRITICAL: Setup Isaac Gym paths BEFORE any imports that use torch
_ISAAC_GYM_PY = ROOT / 'isaacgym' / 'python'
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

_ISAAC_BINDINGS = _ISAAC_GYM_PY / 'isaacgym' / '_bindings' / 'linux-x86_64'
if _ISAAC_BINDINGS.exists():
    os.environ['LD_LIBRARY_PATH'] = str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

# Import Isaac Gym BEFORE any torch-dependent modules
try:
    from isaacgym import gymapi  # type: ignore
except Exception:
    pass

import numpy as np

from scripts.baselines.tune_pid_lqr_isaac import (  # type: ignore
    build_controller_eval,
    local_random_search,
)

# 任务配置
TASK = 'figure8'              # 轨迹类型: hover/figure8/circle/square/helix
DURATION_SEC = 5.0           # 每次评估飞行时长(秒)
NUM_ENVS = 1024               # 并行环境数(加速~250倍), 显存够用可设512
TRIALS = 10                 # 随机搜索次数(CPID参数多,建议>=250)
EPISODES_PER_EVAL = 1        # 每组参数评估的episode数(取平均)
OUTPUT_PATH = Path(__file__).parent / 'results' / 'cpid_train' / f'cpid_{TASK}.json'
SEED = 1

# 级联PID控制器基准参数(cascade模式: 带积分项, 消除稳态误差)
BASE_PARAMS = {
    'kp_xy': 10.0,     # 水平位置比例增益(级联模式通常更大)
    'kd_xy': 5.0,      # 水平速度阻尼增益
    'ki_xy': 0.05,     # 水平积分增益(消除xy偏差,避免过大)
    'kp_z': 18.0,      # 垂直位置比例增益(高度控制需更强)
    'kd_z': 8.0,       # 垂直速度阻尼增益
    'ki_z': 0.1,       # 垂直积分增益(消除重力误差)
    'kp_att': 15.0,    # 姿态角比例增益(级联模式更激进)
    'kd_att': 2.5,     # 姿态角速度阻尼
    'kp_yaw': 5.0,     # 偏航角比例增益(级联模式启用)
    'kd_yaw': 1.0,     # 偏航角速度阻尼(级联模式启用)
    'att_scale': 0.25, # 姿态耦合系数(级联模式更大)
}

# 参数搜索范围(级联PID范围更宽,允许更激进调优)
BOUNDS = {
    'kp_xy': (6.0, 24.0),      # 水平P增益范围(更宽)
    'kd_xy': (3.0, 10.0),      # 水平D增益范围
    'ki_xy': (0.0, 0.4),       # 水平I增益范围(启用但不宜过大)
    'kp_z': (12.0, 30.0),      # 垂直P增益范围(更高上限)
    'kd_z': (5.0, 12.0),       # 垂直D增益范围
    'ki_z': (0.0, 0.6),        # 垂直I增益范围(消除高度偏差)
    'kp_att': (10.0, 30.0),    # 姿态P增益范围(激进调优)
    'kd_att': (1.0, 5.0),      # 姿态D增益范围
    'kp_yaw': (2.0, 8.0),      # 偏航P增益范围(启用)
    'kd_yaw': (0.2, 2.5),      # 偏航D增益范围(启用)
    'att_scale': (0.1, 0.4),   # 姿态耦合范围(不缩小)
}


def main() -> None:
    np.random.seed(SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    eval_fn = build_controller_eval('pid', pid_mode='cascade', num_envs=NUM_ENVS)
    best_params, metrics = local_random_search(
        BASE_PARAMS,
        BOUNDS,
        TRIALS,
        eval_fn,
        TASK,
        DURATION_SEC,
        episodes_per_eval=EPISODES_PER_EVAL,
    )

    print("=" * 70)
    print("CPID tuning finished (SCG reward)")
    print(f" Task={TASK}, duration={DURATION_SEC}s")
    print(f" Best mean reward: {metrics['mean_true_reward']:.2f}")
    print(f" Position RMSE: {metrics['rmse_pos']:.3f} m")
    print(" Best params:")
    for k, v in best_params.items():
        print(f"  - {k}: {v:.4f}")

    payload = {
        'task': TASK,
        'duration_sec': DURATION_SEC,
        'controller': 'cpid',
        'seed': SEED,
        'trials': TRIALS,
        'episodes_per_eval': EPISODES_PER_EVAL,
        'best_params': best_params,
        'metrics': metrics,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

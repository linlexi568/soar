#!/usr/bin/env python3
"""LQR gain search with exact SCG reward alignment."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
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
TASK = 'square'              # 轨迹类型: hover/figure8/circle/square/helix
DURATION_SEC = 5.0           # 每次评估飞行时长(秒)
NUM_ENVS = 1024               # 并行环境数(加速~250倍), 显存够用可设512
TRIALS = 10                 # 随机搜索次数(LQR参数少,200次足够)
EPISODES_PER_EVAL = 1        # 每组参数评估的episode数(取平均)
OUTPUT_PATH = Path(f'results/aligned_baselines/lqr_{TASK}.json')
SEED = 2

# LQR控制器基准参数(pure模式: 无积分项, 纯线性二次型调节器)
BASE_PARAMS = {
    'k_pos': 4.0,      # 位置反馈增益(对应LQR的Q矩阵位置权重)
    'k_vel': 4.0,      # 速度反馈增益(对应LQR的Q矩阵速度权重)
    'k_att': 12.0,     # 姿态反馈增益(姿态角误差权重)
    'k_omega': 3.0,    # 角速度反馈增益(角速度阻尼)
    'k_yaw': 0.0,      # 偏航角增益(pure模式通常为0)
    'k_yaw_rate': 0.0, # 偏航角速度增益(pure模式通常为0)
    'att_scale': 0.2,  # 姿态耦合系数(位置误差→期望倾角)
}

# 参数搜索范围(LQR增益通常比PID更温和)
BOUNDS = {
    'k_pos': (2.0, 10.0),      # 位置增益范围(对应Q矩阵)
    'k_vel': (2.0, 10.0),      # 速度增益范围(对应Q矩阵)
    'k_att': (8.0, 25.0),      # 姿态增益范围(姿态响应)
    'k_omega': (2.0, 6.0),     # 角速度增益范围(角速度阻尼)
    'k_yaw': (0.0, 8.0),       # 偏航增益范围(pure模式忽略)
    'k_yaw_rate': (0.0, 2.0),  # 偏航速度增益范围(pure模式忽略)
    'att_scale': (0.1, 0.3),   # 姿态耦合范围(较保守)
}


def main() -> None:
    np.random.seed(SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    eval_fn = build_controller_eval('lqr', lqr_mode='pure', num_envs=NUM_ENVS)
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
    print("LQR tuning finished (SCG reward)")
    print(f" Task={TASK}, duration={DURATION_SEC}s")
    print(f" Best mean reward: {metrics['mean_true_reward']:.2f}")
    print(f" Position RMSE: {metrics['rmse_pos']:.3f} m")
    print(" Best params:")
    for k, v in best_params.items():
        print(f"  - {k}: {v:.4f}")

    payload = {
        'task': TASK,
        'duration_sec': DURATION_SEC,
        'controller': 'lqr',
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

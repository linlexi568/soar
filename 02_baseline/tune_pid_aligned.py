#!/usr/bin/env python3
"""SCG-aligned PID tuning with all knobs embedded in the script."""
from __future__ import annotations

# ⚠️ CRITICAL Step 1: Setup paths BEFORE ANY imports
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ⚠️ CRITICAL Step 2: Setup Isaac Gym BEFORE importing anything else
# Prioritize the working installation with compiled .so files
_ISAAC_GYM_PY = None
for _candidate in [Path('/home/linlexi/桌面/soar/isaacgym/python'), ROOT / 'isaacgym' / 'python']:
    if _candidate.exists():
        # Verify it has the compiled bindings
        _test_so = _candidate / 'isaacgym' / '_bindings' / 'linux-x86_64' / 'gym_38.so'
        if _test_so.exists():
            _ISAAC_GYM_PY = _candidate
            break

if _ISAAC_GYM_PY and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))
    _ISAAC_BINDINGS = _ISAAC_GYM_PY / 'isaacgym' / '_bindings' / 'linux-x86_64'
    if _ISAAC_BINDINGS.exists():
        os.environ['LD_LIBRARY_PATH'] = str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

# ⚠️ CRITICAL Step 3: Import Isaac Gym BEFORE torch
try:
    import isaacgym
except ImportError:
    pass

# ⚠️ Step 4: Now safe to import other modules
import json
import numpy as np

from scripts.baselines.tune_pid_lqr_isaac import (  # type: ignore
    build_controller_eval,
    local_random_search,
)

# =============================================================================
# Config (edit below)
# =============================================================================
# 任务配置
TASK = 'circle'              # 轨迹类型: hover/figure8/circle/square/helix
DURATION_SEC = 5.0           # 每次评估飞行时长(秒)
NUM_ENVS = 1024               # 并行环境数(加速~250倍), 显存够用可设512
TRIALS = 20                 # 随机搜索次数(越大越精细但越慢)
EPISODES_PER_EVAL = 1        # 每组参数评估的episode数(取平均)
OUTPUT_PATH = Path(__file__).parent / 'results' / 'pid_train' / f'pid_{TASK}.json'
SEED = 0

# 标准 PID 控制器基准参数(启用积分与偏航)
BASE_PARAMS = {
    'kp_xy': 8.0,      # 水平位置比例增益(越大响应越快但易振荡)
    'kd_xy': 4.0,      # 水平速度阻尼增益(抑制振荡)
    'ki_xy': 0.02,     # 水平积分增益(消除稳态误差, 建议小)
    'kp_z': 14.0,      # 垂直位置比例增益
    'kd_z': 6.0,       # 垂直速度阻尼增益
    'ki_z': 0.05,      # 垂直积分增益(补偿重力/偏差)
    'kp_att': 12.0,    # 姿态角比例增益(控制倾斜响应)
    'kd_att': 2.0,     # 姿态角速度阻尼(稳定姿态)
    'kp_yaw': 4.0,     # 偏航角比例增益(启用)
    'kd_yaw': 0.8,     # 偏航角速度阻尼(启用)
    'att_scale': 0.2,  # 姿态耦合系数(位置误差→倾角)
    'use_feedforward': 0.0, # 禁用前馈 (0.0=False)
}

# 参数搜索范围(格式: (最小值, 最大值))
BOUNDS = {
    'kp_xy': (4.0, 20.0),      # 水平P增益范围
    'kd_xy': (2.0, 8.0),       # 水平D增益范围
    'ki_xy': (0.0, 0.2),       # 水平I增益范围(标准PID, 建议较小)
    'kp_z': (10.0, 25.0),      # 垂直P增益范围
    'kd_z': (4.0, 10.0),       # 垂直D增益范围
    'ki_z': (0.0, 0.2),        # 垂直I增益范围(标准PID, 建议较小)
    'kp_att': (8.0, 25.0),     # 姿态P增益范围
    'kd_att': (1.0, 4.0),      # 姿态D增益范围
    'kp_yaw': (0.0, 8.0),      # 偏航P增益范围(启用)
    'kd_yaw': (0.0, 2.0),      # 偏航D增益范围(启用)
    'att_scale': (0.05, 0.4),  # 姿态耦合范围
    'use_feedforward': (0.0, 0.0), # 强制为False
}


def main() -> None:
    np.random.seed(SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 标准 PID: 启用积分与偏航。使用 cascade 分支但保持温和的参数/范围。
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
    print("PID tuning finished (SCG reward)")
    print(f" Task={TASK}, duration={DURATION_SEC}s")
    print(f" Best mean reward: {metrics['mean_true_reward']:.2f}")
    print(f" Position RMSE: {metrics['rmse_pos']:.3f} m")
    print(" Best params:")
    for k, v in best_params.items():
        print(f"  - {k}: {v:.4f}")

    payload = {
        'task': TASK,
        'duration_sec': DURATION_SEC,
        'controller': 'pid',
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

#!/usr/bin/env python3
"""
PID/CPID/LQR 基线评估脚本 - 使用 SCG 精确奖励（与 Soar 完全对齐）
所有参数写在脚本顶部，直接修改即可

奖励函数: r_t = -(x_err^T Q x_err + u^T R u)
  Q = diag([1, 0.01, 1, 0.01, 1, 0.01, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
  R = 0.0001
"""

import sys
from pathlib import Path
import os

# ============================================================================
# 路径设置 (Isaac Gym 必须在 torch 之前导入)
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ISAAC_GYM_PY = ROOT / 'isaacgym' / 'python'
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

_ISAAC_BINDINGS = _ISAAC_GYM_PY / 'isaacgym' / '_bindings' / 'linux-x86_64'
if _ISAAC_BINDINGS.exists():
    os.environ['LD_LIBRARY_PATH'] = str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

try:
    from isaacgym import gymapi  # type: ignore
except Exception:
    pass

# ============================================================================
#                    ★★★ 评估参数配置 (修改这里) ★★★
# ============================================================================

# --------------------- 任务配置 ---------------------
TASK = "figure8"          # 选择: circle, square, helix, figure8, hover
DURATION = 5.0            # 每个episode时长(秒)

# --------------------- 评估配置 ---------------------
NUM_ENVS = 1024           # 并行环境数 (与 Soar 评估对齐)
EPISODES = 10             # 评估episode数
DEVICE = "cuda:0"         # 设备

# --------------------- 控制器选择 ---------------------
CONTROLLER = "cpid"       # 选择: pid, cpid, lqr

# --------------------- 从调参结果加载参数 ---------------------
# 如果为 None，则使用默认参数；否则从 JSON 文件加载
PARAMS_FILE = Path(__file__).parent / 'results' / f'{CONTROLLER}_train' / f'{CONTROLLER}_{TASK}.json'
# 如果文件不存在，将使用下面的默认参数

# --------------------- 输出配置 ---------------------
OUTPUT_PATH = Path(__file__).parent / 'results' / f'{CONTROLLER}_test' / f'eval_{CONTROLLER}_{TASK}.json'

# --------------------- 默认控制器参数 ---------------------
# PID/CPID 默认参数 (cascade 模式)
DEFAULT_PID_PARAMS = {
    'kp_xy': 10.0,
    'kd_xy': 5.0,
    'ki_xy': 0.05,
    'kp_z': 18.0,
    'kd_z': 8.0,
    'ki_z': 0.1,
    'kp_att': 15.0,
    'kd_att': 2.5,
    'kp_yaw': 5.0,
    'kd_yaw': 1.0,
    'att_scale': 0.25,
}

# LQR 默认参数 (pure 模式)
DEFAULT_LQR_PARAMS = {
    'k_pos': 4.0,
    'k_vel': 4.0,
    'k_att': 12.0,
    'k_omega': 3.0,
    'k_yaw': 0.0,
    'k_yaw_rate': 0.0,
    'att_scale': 0.2,
}

# ============================================================================
#                         评估代码 (不需要修改)
# ============================================================================

def main():
    import json
    from datetime import datetime
    import numpy as np
    import torch
    
    from scripts.baselines.tune_pid_lqr_isaac import (
        IsaacPIDController,
        IsaacLQRController,
        evaluate_params,
    )
    
    print("=" * 70)
    print(f"{CONTROLLER.upper()} 基线评估 - 任务: {TASK}")
    print("=" * 70)
    print(f"  并行环境: {NUM_ENVS}")
    print(f"  评估episodes: {EPISODES}")
    print(f"  时长: {DURATION}s")
    print()
    
    # 加载参数
    params = None
    if PARAMS_FILE.exists():
        print(f"从调参结果加载参数: {PARAMS_FILE}")
        with open(PARAMS_FILE, 'r') as f:
            data = json.load(f)
        params = data.get('best_params', None)
        if params:
            print(f"  ✓ 加载成功")
    
    if params is None:
        print(f"使用默认参数")
        if CONTROLLER in ('pid', 'cpid'):
            params = DEFAULT_PID_PARAMS.copy()
        else:
            params = DEFAULT_LQR_PARAMS.copy()
    
    print("\n控制器参数:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")
    print()
    
    # 创建控制器
    if CONTROLLER in ('pid', 'cpid'):
        controller = IsaacPIDController(**params)
    else:
        controller = IsaacLQRController(**params)
    
    # 评估
    print("开始评估...")
    metrics = evaluate_params(controller, TASK, DURATION, episodes=EPISODES, num_envs=NUM_ENVS)
    
    print()
    print("=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"  平均奖励 (SCG): {metrics['mean_true_reward']:.2f} ± {metrics['std_true_reward']:.2f}")
    print(f"  位置 RMSE: {metrics['rmse_pos']:.4f} m")
    print(f"  Episodes: {EPISODES}")
    print(f"  环境数: {NUM_ENVS}")
    print()
    
    # 保存结果
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "algorithm": CONTROLLER.upper(),
        "task": TASK,
        "duration_sec": DURATION,
        "num_envs": NUM_ENVS,
        "episodes": EPISODES,
        "params": params,
        "mean_reward": metrics['mean_true_reward'],
        "std_reward": metrics['std_true_reward'],
        "rmse_pos": metrics['rmse_pos'],
        "reward_type": "scg_exact",
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

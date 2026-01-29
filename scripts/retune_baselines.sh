#!/bin/bash
#
# 重新调参传统基线 (PID/CPID/LQR)
# 使用统一的 SCG 精确奖励
#

set -e

cd "$(dirname "$0")/.."

# 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "错误: 未找到虚拟环境 .venv"
    exit 1
fi

# ============================================================================
#                    配置参数 (修改这里)
# ============================================================================

# 任务列表
TASKS="circle square helix figure8"

# 调参配置
TRIALS=30          # 随机搜索次数
EPISODES=5         # 每次评估的episode数
DURATION=5         # episode时长(秒)

# 输出
OUTPUT_DIR="results/baselines_unified"
mkdir -p "$OUTPUT_DIR"

# ============================================================================

echo "======================================================================"
echo "传统基线调参 (统一SCG奖励)"
echo "======================================================================"
echo "  任务: $TASKS"
echo "  调参轮数: $TRIALS"
echo "  评估episodes: $EPISODES"
echo "  时长: ${DURATION}s"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 运行调参
python3 << PYEOF
import sys
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '01_soar'))

# 导入必要模块
from scripts.baselines.tune_pid_lqr_isaac import (
    IsaacPIDController, IsaacLQRController, IsaacCascadePIDController,
    evaluate_params, local_random_search
)

# 配置
TASKS = "$TASKS".split()
TRIALS = $TRIALS
EPISODES = $EPISODES
DURATION = $DURATION
OUTPUT_DIR = Path("$OUTPUT_DIR")

# PID参数范围
PID_BOUNDS = {
    'kp_xy': (1.0, 50.0),
    'kd_xy': (0.5, 30.0),
    'kp_z': (1.0, 50.0),
    'kd_z': (0.5, 20.0),
    'kp_att': (5.0, 80.0),
    'kd_att': (0.1, 10.0),
    'att_scale': (0.05, 0.5),
}

PID_INIT = {
    'kp_xy': 8.0, 'kd_xy': 4.0,
    'kp_z': 14.0, 'kd_z': 6.0,
    'kp_att': 12.0, 'kd_att': 2.0,
    'att_scale': 0.2,
}

# LQR参数范围
LQR_BOUNDS = {
    'q_pos': (0.1, 100.0),
    'q_vel': (0.1, 50.0),
    'q_att': (0.1, 100.0),
    'q_omega': (0.1, 20.0),
    'r_fz': (0.001, 10.0),
    'r_torque': (0.001, 10.0),
}

LQR_INIT = {
    'q_pos': 10.0, 'q_vel': 2.0,
    'q_att': 20.0, 'q_omega': 5.0,
    'r_fz': 1.0, 'r_torque': 1.0,
}

# CPID参数范围
CPID_BOUNDS = {
    'kp_pos_xy': (0.5, 20.0),
    'kp_pos_z': (0.5, 10.0),
    'kd_pos_xy': (0.1, 10.0),
    'kd_pos_z': (0.1, 5.0),
    'kp_vel_xy': (0.5, 20.0),
    'kp_vel_z': (0.5, 20.0),
    'kp_att': (5.0, 80.0),
    'kd_att': (0.1, 10.0),
    'att_scale': (0.05, 0.5),
}

CPID_INIT = {
    'kp_pos_xy': 5.0, 'kp_pos_z': 2.0,
    'kd_pos_xy': 2.0, 'kd_pos_z': 1.0,
    'kp_vel_xy': 8.0, 'kp_vel_z': 4.0,
    'kp_att': 20.0, 'kd_att': 2.0,
    'att_scale': 0.2,
}


def make_pid_eval_fn(base_params, task, duration, episodes):
    ctrl = IsaacPIDController(**base_params)
    return evaluate_params(ctrl, task, duration, episodes)


def make_lqr_eval_fn(base_params, task, duration, episodes):
    ctrl = IsaacLQRController(**base_params)
    return evaluate_params(ctrl, task, duration, episodes)


def make_cpid_eval_fn(base_params, task, duration, episodes):
    ctrl = IsaacCascadePIDController(**base_params)
    return evaluate_params(ctrl, task, duration, episodes)


results = {}

for task in TASKS:
    print(f"\n{'='*70}")
    print(f"调参任务: {task}")
    print(f"{'='*70}")
    
    results[task] = {}
    
    # 1. PID
    print(f"\n[PID] 开始调参...")
    def pid_eval(params, t, d, e):
        ctrl = IsaacPIDController(**params)
        return evaluate_params(ctrl, t, d, e)
    
    best_pid, pid_metrics = local_random_search(
        PID_INIT, PID_BOUNDS, TRIALS, pid_eval, task, DURATION, EPISODES
    )
    results[task]['PID'] = {
        'params': best_pid,
        'reward': pid_metrics['mean_true_reward'],
        'std': pid_metrics['std_true_reward'],
        'rmse': pid_metrics['rmse_pos'],
        'trials': TRIALS,
    }
    print(f"[PID] 最佳奖励: {pid_metrics['mean_true_reward']:.2f}")
    
    # 2. CPID
    print(f"\n[CPID] 开始调参...")
    def cpid_eval(params, t, d, e):
        ctrl = IsaacCascadePIDController(**params)
        return evaluate_params(ctrl, t, d, e)
    
    best_cpid, cpid_metrics = local_random_search(
        CPID_INIT, CPID_BOUNDS, TRIALS, cpid_eval, task, DURATION, EPISODES
    )
    results[task]['CPID'] = {
        'params': best_cpid,
        'reward': cpid_metrics['mean_true_reward'],
        'std': cpid_metrics['std_true_reward'],
        'rmse': cpid_metrics['rmse_pos'],
        'trials': TRIALS,
    }
    print(f"[CPID] 最佳奖励: {cpid_metrics['mean_true_reward']:.2f}")
    
    # 3. LQR
    print(f"\n[LQR] 开始调参...")
    def lqr_eval(params, t, d, e):
        ctrl = IsaacLQRController(**params)
        return evaluate_params(ctrl, t, d, e)
    
    best_lqr, lqr_metrics = local_random_search(
        LQR_INIT, LQR_BOUNDS, TRIALS, lqr_eval, task, DURATION, EPISODES
    )
    results[task]['LQR'] = {
        'params': best_lqr,
        'reward': lqr_metrics['mean_true_reward'],
        'std': lqr_metrics['std_true_reward'],
        'rmse': lqr_metrics['rmse_pos'],
        'trials': TRIALS,
    }
    print(f"[LQR] 最佳奖励: {lqr_metrics['mean_true_reward']:.2f}")
    
    # 保存中间结果
    with open(OUTPUT_DIR / f'{task}_baselines.json', 'w') as f:
        json.dump({task: results[task]}, f, indent=2)

# 保存汇总结果
output_file = OUTPUT_DIR / 'baselines_unified.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("调参完成!")
print(f"{'='*70}")

# 打印汇总表格
print("\n| 方法 | circle | square | helix | figure8 |")
print("|------|--------|--------|-------|---------|")
for method in ['PID', 'CPID', 'LQR']:
    row = [method]
    for task in ['circle', 'square', 'helix', 'figure8']:
        if task in results and method in results[task]:
            r = results[task][method]['reward']
            row.append(f"{r:.2f}")
        else:
            row.append("N/A")
    print("| " + " | ".join(row) + " |")

print(f"\n结果已保存到: {output_file}")
PYEOF

echo ""
echo "======================================================================"
echo "调参完成!"
echo "======================================================================"

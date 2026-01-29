#!/usr/bin/env python3
"""
简化版基线调参 - 只调 square 任务，快速验证
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 必须先导入Isaac Gym
sys.path.insert(0, str(ROOT / 'isaacgym' / 'python'))
from isaacgym import gymapi

# 再导入其他
import json
import numpy as np
import torch

sys.path.insert(0, str(ROOT / '01_soar'))
from scripts.baselines.tune_pid_lqr_isaac import IsaacPIDController, IsaacLQRController, evaluate_params

print("="*70)
print("快速基线调参 - square 任务")
print("="*70)

TASK = 'square'
DURATION = 5.0
EPISODES = 5
TRIALS = 20

# PID初始参数
PID_INIT = {
    'kp_xy': 8.0, 'kd_xy': 4.0,
    'kp_z': 14.0, 'kd_z': 6.0,
    'kp_att': 12.0, 'kd_att': 2.0,
    'att_scale': 0.2,
}

# LQR初始参数
LQR_INIT = {
    'q_pos': 10.0, 'q_vel': 2.0,
    'q_att': 20.0, 'q_omega': 5.0,
    'r_fz': 1.0, 'r_torque': 1.0,
}

results = {}

# 1. 评估初始PID
print("\n【1. PID 初始参数】")
ctrl_pid = IsaacPIDController(**PID_INIT)
metrics_pid = evaluate_params(ctrl_pid, TASK, DURATION, EPISODES)
print(f"  初始奖励: {metrics_pid['mean_true_reward']:.2f} ± {metrics_pid['std_true_reward']:.2f}")

# 手动随机搜索PID（避免环境重建问题）
print(f"\n【2. PID 随机搜索】({TRIALS}轮)")
best_pid_params = dict(PID_INIT)
best_pid_reward = metrics_pid['mean_true_reward']

np.random.seed(42)
for i in range(TRIALS):
    # 生成随机扰动
    ratio = max(0.2, 1.0 - i / TRIALS)  # 退火
    trial_params = {}
    for k, v in PID_INIT.items():
        if k == 'att_scale':
            span = 0.15 * ratio
            trial_params[k] = np.clip(v + np.random.uniform(-span, span), 0.05, 0.5)
        elif k in ['kp_xy', 'kp_z']:
            span = 15.0 * ratio
            trial_params[k] = np.clip(v + np.random.uniform(-span, span), 1.0, 50.0)
        elif k in ['kd_xy', 'kd_z']:
            span = 8.0 * ratio
            trial_params[k] = np.clip(v + np.random.uniform(-span, span), 0.5, 30.0)
        elif k == 'kp_att':
            span = 20.0 * ratio
            trial_params[k] = np.clip(v + np.random.uniform(-span, span), 5.0, 80.0)
        elif k == 'kd_att':
            span = 3.0 * ratio
            trial_params[k] = np.clip(v + np.random.uniform(-span, span), 0.1, 10.0)
    
    # 评估
    try:
        ctrl = IsaacPIDController(**trial_params)
        metrics = evaluate_params(ctrl, TASK, DURATION, EPISODES)
        reward = metrics['mean_true_reward']
        
        if reward > best_pid_reward:
            best_pid_reward = reward
            best_pid_params = trial_params
            print(f"  [{i+1}/{TRIALS}] 新最佳: {reward:.2f}")
    except Exception as e:
        print(f"  [{i+1}/{TRIALS}] 失败: {e}")

print(f"\n  最佳PID奖励: {best_pid_reward:.2f}")
results['PID'] = {
    'params': best_pid_params,
    'reward': float(best_pid_reward),
    'trials': TRIALS,
}

# 3. 评估初始LQR
print("\n【3. LQR 初始参数】")
ctrl_lqr = IsaacLQRController(**LQR_INIT)
metrics_lqr = evaluate_params(ctrl_lqr, TASK, DURATION, EPISODES)
print(f"  初始奖励: {metrics_lqr['mean_true_reward']:.2f} ± {metrics_lqr['std_true_reward']:.2f}")

# 手动随机搜索LQR
print(f"\n【4. LQR 随机搜索】({TRIALS}轮)")
best_lqr_params = dict(LQR_INIT)
best_lqr_reward = metrics_lqr['mean_true_reward']

for i in range(TRIALS):
    ratio = max(0.2, 1.0 - i / TRIALS)
    trial_params = {}
    for k, v in LQR_INIT.items():
        if k in ['q_pos', 'q_att']:
            span = 30.0 * ratio
            trial_params[k] = np.clip(v + np.random.uniform(-span, span), 0.1, 100.0)
        elif k in ['q_vel', 'q_omega']:
            span = 10.0 * ratio
            trial_params[k] = np.clip(v + np.random.uniform(-span, span), 0.1, 50.0)
        elif k in ['r_fz', 'r_torque']:
            span = 3.0 * ratio
            trial_params[k] = np.clip(v + np.random.uniform(-span, span), 0.001, 10.0)
    
    try:
        ctrl = IsaacLQRController(**trial_params)
        metrics = evaluate_params(ctrl, TASK, DURATION, EPISODES)
        reward = metrics['mean_true_reward']
        
        if reward > best_lqr_reward:
            best_lqr_reward = reward
            best_lqr_params = trial_params
            print(f"  [{i+1}/{TRIALS}] 新最佳: {reward:.2f}")
    except Exception as e:
        print(f"  [{i+1}/{TRIALS}] 失败: {e}")

print(f"\n  最佳LQR奖励: {best_lqr_reward:.2f}")
results['LQR'] = {
    'params': best_lqr_params,
    'reward': float(best_lqr_reward),
    'trials': TRIALS,
}

# 保存结果
output_dir = ROOT / 'results' / 'baselines_unified'
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / 'square_quick.json'

with open(output_file, 'w') as f:
    json.dump({'square': results}, f, indent=2)

print("\n" + "="*70)
print("调参完成!")
print("="*70)
print(f"\nPID: {results['PID']['reward']:.2f}")
print(f"LQR: {results['LQR']['reward']:.2f}")
print(f"\n结果已保存: {output_file}")

#!/usr/bin/env python3
"""
紧急诊断：用统一评估环境重新测试 Soar 和 PID
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '01_soar'))

print("=" * 70)
print("🚨 紧急诊断：Soar 奖励计算问题")
print("=" * 70)

# 1. 读取Soar best程序
soar_file = ROOT / 'results/soar_train/square_safe_control_tracking_best.json'
with open(soar_file) as f:
    soar_data = json.load(f)

print("\n【Soar训练结果】")
print(f"  迭代: {soar_data['meta']['iteration']}")
print(f"  训练奖励: {soar_data['meta']['reward']:.2f}")
print(f"  state_cost: {soar_data['meta']['reward_components']['state_cost']:.2f}")
print(f"  action_cost: {soar_data['meta']['reward_components']['action_cost']:.6f}")
print(f"  isaac_num_envs: {soar_data['meta']['isaac_num_envs']}")

# 2. 检查配置
print("\n【问题分析】")
print(f"  如果用1个环境评估: state_cost ≈ {soar_data['meta']['reward_components']['state_cost']:.2f}")
print(f"  如果用8196个环境但只统计1个: 正常")
print(f"  如果被8196个环境平均: {soar_data['meta']['reward_components']['state_cost'] * 8196:.2f}")

# 3. 对比PID
print("\n【对比 PID】")
pid_reward = -520.05
soar_reward = soar_data['meta']['reward']
ratio = soar_reward / pid_reward

print(f"  PID (baselines_retune.json): {pid_reward:.2f}")
print(f"  Soar (训练结果): {soar_reward:.2f}")
print(f"  比例: {ratio:.3f} ({abs(1/ratio):.1f}x 差异)")

print("\n【建议】")
print("  1. 用统一脚本重新评估 Soar best 程序")
print("  2. 确保评估配置:")
print("     - isaac_num_envs=1024 (足够统计)")
print("     - replicas_per_program=5")
print("     - reward_reduction='sum'")
print("     - 240步 (5s @ 48Hz)")
print("  3. 同时重新评估 PID确保一致性")

print("\n" + "=" * 70)
print("需要立即行动：用utilities中的统一评估脚本重新测试!")
print("=" * 70)

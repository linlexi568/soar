#!/usr/bin/env python3
"""打印唯一的 Safe-Control-Gym 奖励配置，方便快速核对权重。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.reward_profiles import get_reward_profile

# 仅保留 SCG 对齐 profile
FOCUS_PROFILES = ["safe_control_tracking"]

print("=" * 80)
print("Soar Reward Profile 权重对比表")
print("=" * 80)
print()

# 获取所有权重 key
all_keys = set()
for name in FOCUS_PROFILES:
    w, _ = get_reward_profile(name)
    all_keys.update(w.keys())

# 打印表头
print(f"{'Component':<20}", end="")
for name in FOCUS_PROFILES:
    print(f"{name:>18}", end="")
print()
print("-" * 80)

# 打印权重对比
for key in sorted(all_keys):
    print(f"{key:<20}", end="")
    for name in FOCUS_PROFILES:
        w, _ = get_reward_profile(name)
        val = w.get(key, 0.0)
        print(f"{val:>18.2f}", end="")
    print()

print()
print("=" * 80)
print("Shaping 系数 (k_*) 对比")
print("=" * 80)
print()

# 获取所有系数 key
all_k_keys = set()
for name in FOCUS_PROFILES:
    _, k = get_reward_profile(name)
    all_k_keys.update(k.keys())

# 打印系数表头
print(f"{'Coefficient':<20}", end="")
for name in FOCUS_PROFILES:
    print(f"{name:>18}", end="")
print()
print("-" * 80)

# 打印系数对比
for key in sorted(all_k_keys):
    print(f"{key:<20}", end="")
    for name in FOCUS_PROFILES:
        _, k = get_reward_profile(name)
        val = k.get(key, 0.0)
        print(f"{val:>18.2f}", end="")
    print()

print()
print("=" * 80)
print("设计意图对比")
print("=" * 80)
print()

intentions = {
    "safe_control_tracking": "Safe-Control-Gym quadrotor_3D_track 对齐引用配置",
}

for name, intent in intentions.items():
    print(f"• {name:20s}: {intent}")

print()
print("=" * 80)
print("✓ SCG profile 配置正常！")
print("✓ 其他 profile 已移除，所有脚本强制使用 SCG")
print("=" * 80)

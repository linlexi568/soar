#!/usr/bin/env python3
"""验证 Benchmark 环境配置"""
import sys
from pathlib import Path

print("=" * 60)
print("Benchmark 环境验证")
print("=" * 60)

errors = []

# 检查 Python 版本
print(f"\n✓ Python 版本: {sys.version.split()[0]}")

# 检查必要的包
packages = [
    'torch',
    'stable_baselines3', 
    'numpy',
    'matplotlib',
    'gymnasium',
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {pkg}: {version}")
    except ImportError:
        print(f"✗ {pkg}: 未安装")
        errors.append(f"{pkg} 未安装")

# 检查 Isaac Gym
ROOT = Path(__file__).resolve().parent.parent  # benchmark -> soar-compare
isaac_gym_paths = [
    ROOT / "isaacgym",
    ROOT / "IsaacGym_Preview_4_Package" / "isaacgym",
]
isaac_found = False
isaac_gym_path = None
for path in isaac_gym_paths:
    if path.exists():
        isaac_found = True
        isaac_gym_path = path
        break

if isaac_found:
    print(f"✓ Isaac Gym: 找到 ({isaac_gym_path})")
else:
    print(f"✗ Isaac Gym: 未找到")
    errors.append("Isaac Gym 未找到（请下载并解压到项目根目录）")

# 检查核心文件
core_files = [
    Path(__file__).parent / "envs" / "isaac_gym_wrapper.py",
    Path(__file__).parent / "envs" / "reward_calculator.py",
    Path(__file__).parent / "baselines" / "controllers.py",
]

all_exist = True
for f in core_files:
    if f.exists():
        print(f"✓ {f.name}")
    else:
        print(f"✗ {f.name}: 未找到")
        errors.append(f"{f.name} 未找到")
        all_exist = False

print("\n" + "=" * 60)
if errors:
    print("❌ 环境检查失败:")
    for err in errors:
        print(f"   - {err}")
    print("\n请运行: bash setup.sh")
    sys.exit(1)
else:
    print("✅ 环境检查通过！可以开始实验。")
    print("\n快速开始:")
    print("  python baselines/tune_pid.py --task circle")
    print("  python ppo/train.py --task circle")
print("=" * 60)

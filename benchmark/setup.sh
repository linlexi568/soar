#!/bin/bash
# 环境安装脚本

set -e

echo "=========================================="
echo "Benchmark 环境安装"
echo "=========================================="

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

# 检查 Isaac Gym
ISAAC_GYM_PATH="../isaacgym"
if [ ! -d "$ISAAC_GYM_PATH" ]; then
    echo "错误: 未找到 Isaac Gym ($ISAAC_GYM_PATH)"
    echo "请先下载并解压 Isaac Gym Preview 4 到项目根目录"
    exit 1
fi

echo "✓ 找到 Isaac Gym: $ISAAC_GYM_PATH"

# 安装 Python 依赖
echo ""
echo "安装 Python 依赖..."
pip install -r requirements.txt

# 验证安装
echo ""
echo "验证安装..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo ""
echo "=========================================="
echo "✓ 环境安装完成！"
echo "=========================================="
echo ""
echo "运行示例:"
echo "  python ppo/train.py --task circle"
echo "  python baselines/tune_pid.py --task circle"

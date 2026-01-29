#!/bin/bash
# 一键运行所有实验（论文复现）

set -e

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_DIR"

echo "=========================================="
echo "Quadrotor Control Benchmark Suite"
echo "=========================================="
echo ""

# 任务列表
TASKS=("circle" "figure8" "hover")

# ============================================
# 阶段 1: 调优传统控制器
# ============================================
echo "📊 阶段 1/3: 调优 PID 和 LQR 控制器"
echo "预计时间: 约 30-60 分钟"
echo ""

for task in "${TASKS[@]}"; do
    echo "--- Tuning PID for $task ---"
    python baselines/tune_pid.py --task "$task" --trials 15
    
    echo ""
    echo "--- Tuning LQR for $task ---"
    python baselines/tune_lqr.py --task "$task" --trials 20
    echo ""
done

echo "✅ 阶段 1 完成：传统控制器调优完成"
echo ""

# ============================================
# 阶段 2: 训练 PPO 强化学习
# ============================================
echo "=========================================="
echo "🤖 阶段 2/3: 训练 PPO 强化学习控制器"
echo "预计时间: 约 6-12 小时（取决于 GPU）"
echo ""
echo "⚠️  注意: PPO 训练时间较长，建议使用 tmux/screen"
echo "   或将此脚本修改为后台运行"
echo ""

read -p "是否继续训练 PPO? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for task in "${TASKS[@]}"; do
        echo "--- Training PPO for $task ---"
        python ppo/train.py --task "$task" --max-steps 500000000
        echo ""
    done
    echo "✅ 阶段 2 完成：PPO 训练完成"
else
    echo "⏭️  跳过 PPO 训练（可稍后手动运行）"
fi

echo ""

# ============================================
# 阶段 3: 评估所有方法
# ============================================
echo "=========================================="
echo "📈 阶段 3/3: 评估所有控制器"
echo ""

for task in "${TASKS[@]}"; do
    echo "--- Evaluating $task ---"
    
    # 评估 PPO（如果模型存在）
    if [ -f "results/ppo/$task/best_model.zip" ]; then
        echo "Evaluating PPO..."
        python ppo/eval.py --task "$task" --use-best --episodes 20
    else
        echo "⚠️  PPO model not found, skipping"
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "✅ 所有实验完成！"
echo "=========================================="
echo ""
echo "结果保存在: $BENCHMARK_DIR/results/"
echo ""
echo "查看结果:"
echo "  - PID:  results/pid/pid_<task>.json"
echo "  - LQR:  results/lqr/lqr_<task>.json"
echo "  - PPO:  results/ppo/<task>/"
echo ""

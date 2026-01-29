#!/usr/bin/env python3
"""
汇总所有方法的评估结果
从各个评估脚本的输出 JSON 文件中读取并汇总

所有参数写在脚本顶部，直接修改即可
"""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

# ============================================================================
#                    ★★★ 配置 (修改这里) ★★★
# ============================================================================

TASK = "figure8"          # 选择: circle, square, helix, figure8, hover

# 各方法的结果文件路径
RESULT_FILES = {
    'Soar': ROOT / "results" / "soar" / f"eval_{TASK}.json",
    'CPID': ROOT / "results" / "aligned_baselines" / f"eval_cpid_{TASK}.json",
    'PID': ROOT / "results" / "aligned_baselines" / f"eval_pid_{TASK}.json",
    'LQR': ROOT / "results" / "aligned_baselines" / f"eval_lqr_{TASK}.json",
    'SAC': ROOT / "03_SAC" / "results" / f"eval_{TASK}.json",
    'TD3': ROOT / "04_TD3" / "results" / f"eval_{TASK}.json",
}

# ============================================================================
#                         汇总代码 (不需要修改)
# ============================================================================

def main():
    print("=" * 70)
    print(f"评估结果汇总 - 任务: {TASK}")
    print("=" * 70)
    print()
    
    results = {}
    
    for method, path in RESULT_FILES.items():
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            
            # 提取奖励
            if 'mean_reward_per_env' in data:
                reward = data['mean_reward_per_env']
                std = data.get('std_reward_per_env', 0)
            elif 'mean_reward' in data:
                reward = data['mean_reward']
                std = data.get('std_reward', 0)
            elif 'reward_true' in data:
                reward = data['reward_true']
                std = 0
            else:
                continue
            
            results[method] = {
                'reward': reward,
                'std': std,
                'rmse': data.get('rmse_pos'),
                'state_cost': data.get('state_cost'),
            }
    
    if not results:
        print("❌ 没有找到任何评估结果文件")
        print("请先运行各个评估脚本:")
        print("  .venv/bin/python 01_soar/eval_soar.py")
        print("  .venv/bin/python scripts/aligned/eval_baseline.py  # 修改 CONTROLLER")
        print("  .venv/bin/python 03_SAC/eval_sac.py")
        print("  .venv/bin/python 04_TD3/eval_td3.py")
        return
    
    # 按奖励排序 (奖励越大越好，即负数绝对值越小越好)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['reward'], reverse=True)
    
    print(f"{'排名':<6} {'方法':<15} {'奖励 (↑ 越大越好)':<25} {'备注':<20}")
    print("-" * 70)
    
    for rank, (method, data) in enumerate(sorted_results, 1):
        reward = data['reward']
        std = data['std']
        rmse = data.get('rmse')
        
        if std > 0:
            reward_str = f"{reward:.2f} ± {std:.2f}"
        else:
            reward_str = f"{reward:.2f}"
        
        note = f"RMSE={rmse:.3f}m" if rmse else ""
        print(f"{rank:<6} {method:<15} {reward_str:<25} {note:<20}")
    
    print("=" * 70)
    
    # 计算相对提升
    if 'Soar' in results and len(results) > 1:
        pf_reward = results['Soar']['reward']
        print("\nSoar 相对于其他方法的提升:")
        for method, data in sorted_results:
            if method != 'Soar':
                other_reward = data['reward']
                # 奖励都是负数，提升 = (other - pf) / |other| * 100
                improvement = (pf_reward - other_reward) / abs(other_reward) * 100
                print(f"  vs {method}: {improvement:+.1f}% 奖励提升")


if __name__ == "__main__":
    main()

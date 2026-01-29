#!/usr/bin/env python3
"""
快速分析训练实验结果
支持从JSON文件或日志文件解析数据
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

def parse_log_file(log_path):
    """从日志文件解析训练数据"""
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            # 匹配: [Iter 1] 完成 | 奖励: 5.5776 | 耗时: 51.1s | Buffer: 1
            match = re.search(r'\[Iter (\d+)\].*奖励:\s*([\d.]+)', line)
            if match:
                iter_num = int(match.group(1))
                reward = float(match.group(2))
                data.append({'iteration': iter_num, 'mean_reward': reward, 'best_reward': reward})
    return data

def analyze_results(path='results/sanity_check.json'):
    """分析训练结果并生成可视化"""
    
    path = Path(path)
    
    # 尝试从JSON加载
    if path.suffix == '.json' and path.exists():
        with open(path, 'r') as f:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                print(f"❌ JSON格式不支持,尝试从日志文件读取")
                return
    # 尝试从日志加载
    elif path.suffix == '.log' and path.exists():
        print(f"📄 从日志文件解析: {path}")
        data = parse_log_file(path)
        if not data:
            print("❌ 未找到训练数据")
            return
    else:
        print(f"❌ 文件不存在: {path}")
        return

    
    print("\n" + "="*80)
    print("📊 Sanity Check 实验结果分析")
    print("="*80)
    
    # 提取数据
    iterations = [d['iteration'] for d in data]
    mean_rewards = [d.get('mean_reward', 0) for d in data]
    best_rewards = [d.get('best_reward', 0) for d in data]
    
    # 统计分析
    print(f"\n总迭代数: {len(iterations)}")
    print(f"初始平均reward: {mean_rewards[0]:.4f}")
    print(f"最终平均reward: {mean_rewards[-1]:.4f}")
    print(f"最佳reward: {max(best_rewards):.4f}")
    print(f"Reward提升: {mean_rewards[-1] - mean_rewards[0]:.4f}")
    
    # 判断学习趋势
    if len(mean_rewards) >= 10:
        early_mean = np.mean(mean_rewards[:5])
        late_mean = np.mean(mean_rewards[-5:])
        improvement = late_mean - early_mean
        
        print(f"\n前5次平均: {early_mean:.4f}")
        print(f"后5次平均: {late_mean:.4f}")
        print(f"改进幅度: {improvement:.4f}")
        
        if improvement > 0.5:
            print("\n✅ 结论: 方法显示明显的学习进展!")
            print("   建议: 继续完整训练(2000 iterations)")
        elif improvement > 0.1:
            print("\n🔄 结论: 方法显示轻微改进")
            print("   建议: 增加MCTS模拟次数或调整奖励权重")
        elif improvement > -0.1:
            print("\n⚠️  结论: 基本无改进,reward停滞")
            print("   建议: 检查是否所有程序产生相同行为")
        else:
            print("\n❌ 结论: Reward下降,可能存在问题")
            print("   建议: 检查奖励函数设计或降级到增益调制模式")
    
    # 检查reward方差
    reward_std = np.std(mean_rewards)
    print(f"\nReward标准差: {reward_std:.4f}")
    if reward_std < 0.01:
        print("⚠️  警告: Reward方差极小,可能所有程序行为相同")
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 子图1: Reward曲线
    axes[0].plot(iterations, mean_rewards, 'b-o', label='Mean Reward', linewidth=2)
    axes[0].plot(iterations, best_rewards, 'g--s', label='Best Reward', linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle=':', alpha=0.5, label='Zero line')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Reward', fontsize=12)
    axes[0].set_title('Sanity Check: Reward Evolution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: Reward变化率
    if len(mean_rewards) > 1:
        reward_delta = np.diff(mean_rewards)
        axes[1].bar(iterations[1:], reward_delta, color='steelblue', alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='-', linewidth=1)
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Reward Change', fontsize=12)
        axes[1].set_title('Iteration-to-Iteration Reward Change', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = 'sanity_check_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 图表已保存: {output_path}")
    
    # 检查程序复杂度(如果有的话)
    if 'best_program' in data[-1]:
        best_prog = data[-1]['best_program']
        print(f"\n🔍 最佳程序: {best_prog}")
        # TODO: 分析程序AST复杂度
    
    print("\n" + "="*80)

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/sanity_check.json'
    
    # 如果指定的是日志文件,尝试.log
    if not Path(path).exists() and not path.endswith('.log'):
        log_path = path.replace('.json', '.log').replace('results/', 'logs/')
        if Path(log_path).exists():
            print(f"ℹ️  未找到JSON,尝试日志文件: {log_path}")
            path = log_path
    
    analyze_results(path)

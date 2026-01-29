#!/usr/bin/env python3
"""运行短测试并打印生成程序的详细内容，检查是否使用姿态分量"""
from __future__ import annotations
import argparse, json, random, sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / '01_soar'
for _p in (ROOT, PKG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_online
OnlineTrainer = train_online.OnlineTrainer
from argparse import Namespace

def analyze_program_variables(program, dsl_variables):
    """分析程序中使用的变量"""
    used_vars = set()
    
    def extract_vars_from_node(node):
        """递归提取节点中的变量"""
        if node is None:
            return
        node_str = str(node)
        for var in dsl_variables:
            if var in node_str:
                used_vars.add(var)
    
    for rule in program:
        if 'node' in rule and rule['node'] is not None:
            extract_vars_from_node(rule['node'])
    
    return used_vars

def run_and_analyze(prior_level: int):
    print(f"\n{'='*70}")
    print(f"  Prior-level {prior_level} 程序内容分析")
    print(f"{'='*70}\n")
    
    # 配置
    args = Namespace(
        total_iters=3,  # 只跑3次迭代
        mcts_simulations=100,  # 减少模拟数加快速度
        update_freq=100,
        train_steps_per_update=5,
        batch_size=128,
        replay_capacity=20000,
        use_gnn=True,
        prior_level=prior_level,
        nn_hidden=256,
        learning_rate=1e-3,
        value_loss_weight=0.5,
        exploration_weight=1.4,
        puct_c=1.5,
        max_depth=20,
        real_sim_frac=0.8,
        traj='hover',
        duration=8,
        isaac_num_envs=32,
        eval_replicas_per_program=1,
        min_steps_frac=0.0,
        reward_reduction='sum',
        use_fast_path=True,
        save_path=str(PKG / f"results/check_program_prior{prior_level}.json"),
        checkpoint_freq=10**9,
        warm_start=None,
    )
    
    np.random.seed(42)
    random.seed(42)
    
    # 获取DSL变量列表
    if prior_level == 3:
        dsl_variables = [
            'pos_err_x', 'pos_err_y', 'pos_err_z',
            'vel_x', 'vel_y', 'vel_z',
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z'
        ]
    elif prior_level == 2:
        dsl_variables = [
            'pos_err_x', 'pos_err_y', 'pos_err_z',
            'vel_x', 'vel_y', 'vel_z',
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
            'err_p_roll', 'err_p_pitch', 'err_p_yaw'
        ]
    else:
        dsl_variables = []
    
    print(f"可用DSL变量 ({len(dsl_variables)}个):")
    for i, v in enumerate(dsl_variables, 1):
        marker = "⭐" if 'err_p' in v else "  "
        print(f"  {marker} {i:2d}. {v}")
    print()
    
    trainer = OnlineTrainer(args)
    
    print("运行MCTS搜索...")
    best_reward = -1e9
    best_program = None
    all_programs = []
    
    for iter_idx in range(args.total_iters):
        print(f"  迭代 {iter_idx+1}/{args.total_iters}...", end=' ')
        children, visit_counts = trainer.mcts_search(
            trainer._generate_random_program(), 
            args.mcts_simulations
        )
        
        if not children:
            print("无子节点")
            continue
        
        # 收集所有生成的程序
        for child in children:
            reward = trainer.evaluator.evaluate_single(child.program)
            all_programs.append({
                'program': child.program,
                'reward': reward,
                'visits': 0
            })
        
        # 找最优
        idx = int(np.argmax(visit_counts))
        prog = children[idx].program
        reward = trainer.evaluator.evaluate_single(prog)
        
        if reward > best_reward:
            best_reward = reward
            best_program = prog
        
        print(f"reward={reward:.4f}, best={best_reward:.4f}")
    
    # 资源清理
    try:
        if hasattr(trainer, 'evaluator') and hasattr(trainer.evaluator, 'close'):
            trainer.evaluator.close()
    except Exception:
        pass
    
    # 分析最优程序
    print(f"\n{'─'*70}")
    print(f"  最优程序分析 (奖励: {best_reward:.6f})")
    print(f"{'─'*70}\n")
    
    if best_program:
        print(f"程序长度: {len(best_program)} 条规则\n")
        
        if len(best_program) == 0:
            print("⚠️  空程序！(零动作)")
        else:
            print("规则详情:")
            for i, rule in enumerate(best_program, 1):
                name = rule.get('name', 'unknown')
                mult = rule.get('multiplier', [1.0, 1.0, 1.0])
                node = rule.get('node', None)
                print(f"  {i}. 目标: {name}")
                print(f"     倍数: {mult}")
                if node:
                    print(f"     表达式: {node}")
                else:
                    print(f"     表达式: (空)")
                print()
        
        # 变量使用统计
        used_vars = analyze_program_variables(best_program, dsl_variables)
        print(f"使用的变量 ({len(used_vars)}个):")
        if used_vars:
            for var in sorted(used_vars):
                marker = "⭐" if 'err_p' in var else "  "
                print(f"  {marker} {var}")
        else:
            print("  (无变量使用)")
        
        # 姿态分量检查
        attitude_vars = {'err_p_roll', 'err_p_pitch', 'err_p_yaw'}
        used_attitude = used_vars & attitude_vars
        
        print(f"\n姿态分量使用情况:")
        if used_attitude:
            print(f"  ✅ 使用了姿态分量: {used_attitude}")
        else:
            print(f"  ❌ 未使用任何姿态分量")
            if prior_level == 2:
                print(f"     → Prior-2 应该能够使用姿态分量，但MCTS没有探索到")
                print(f"     → 说明搜索深度不足或零动作基线太强")
    else:
        print("未找到有效程序")
    
    print(f"\n{'='*70}\n")
    
    return {
        'prior_level': prior_level,
        'best_reward': float(best_reward),
        'program_length': len(best_program) if best_program else 0,
        'used_variables': list(used_vars) if best_program else [],
        'used_attitude_vars': list(used_attitude) if best_program else []
    }


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║            程序内容详细分析 - 检查姿态分量使用情况                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    # 测试Prior-2和Prior-3
    results = []
    for level in [2, 3]:
        result = run_and_analyze(level)
        results.append(result)
    
    # 对比分析
    print("\n" + "="*70)
    print("  对比分析")
    print("="*70 + "\n")
    
    for r in results:
        print(f"Prior-level {r['prior_level']}:")
        print(f"  最优奖励: {r['best_reward']:.6f}")
        print(f"  程序长度: {r['program_length']}")
        print(f"  使用变量: {len(r['used_variables'])}个")
        print(f"  姿态分量: {r['used_attitude_vars']}")
        print()
    
    # 结论
    print("【结论】")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    prior2_attitude = results[0]['used_attitude_vars']
    prior3_attitude = results[1]['used_attitude_vars']
    
    if not prior2_attitude and not prior3_attitude:
        print("⚠️  两个level都未使用姿态分量")
        print("   → MCTS搜索深度不足")
        print("   → 或零动作基线太强，不需要姿态控制")
        print("   → 建议: 提升MCTS参数或测试更难轨迹")
    elif prior2_attitude and not prior3_attitude:
        print("✅ Prior-2使用了姿态分量，Prior-3未使用")
        print("   → 符合设计预期")
    else:
        print(f"❓ 异常情况: P2={prior2_attitude}, P3={prior3_attitude}")
    
    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

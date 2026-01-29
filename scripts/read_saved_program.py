#!/usr/bin/env python3
"""直接读取保存的程序文件，不运行MCTS（避免GPU显存冲突）"""
import torch
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '01_soar'))

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

def analyze_checkpoint(ckpt_path, prior_level):
    """分析checkpoint中的程序"""
    print(f"\n{'='*70}")
    print(f"  分析文件: {ckpt_path}")
    print(f"  Prior Level: {prior_level}")
    print(f"{'='*70}\n")
    
    # 加载checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except FileNotFoundError:
        print(f"❌ 文件不存在: {ckpt_path}")
        return
    
    # 获取程序
    if 'best_program' not in ckpt:
        print(f"❌ checkpoint中没有'best_program'字段")
        print(f"可用字段: {list(ckpt.keys())}")
        return
    
    program = ckpt['best_program']
    best_reward = ckpt.get('best_reward', 'N/A')
    iteration = ckpt.get('iteration', 'N/A')
    
    print(f"迭代: {iteration}")
    print(f"最佳奖励: {best_reward}")
    print(f"\n程序长度: {len(program)} 条规则\n")
    
    # DSL变量定义
    if prior_level == 2:
        dsl_variables = [
            'pos_err_x', 'pos_err_y', 'pos_err_z',
            'vel_x', 'vel_y', 'vel_z',
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
            'err_p_roll', 'err_p_pitch', 'err_p_yaw'
        ]
        attitude_vars = ['err_p_roll', 'err_p_pitch', 'err_p_yaw']
    else:
        dsl_variables = [
            'pos_err_x', 'pos_err_y', 'pos_err_z',
            'vel_x', 'vel_y', 'vel_z',
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z'
        ]
        attitude_vars = []
    
    # 分析每条规则
    print("规则详情:")
    for i, rule in enumerate(program, 1):
        target = rule.get('target', 'unknown')
        multiplier = rule.get('multiplier', [1.0, 1.0, 1.0])
        node = rule.get('node', None)
        
        print(f"  {i}. 目标: {target}")
        print(f"     倍数: {multiplier}")
        if node is None:
            print(f"     表达式: (空)")
        else:
            print(f"     表达式: {node}")
        print()
    
    # 变量使用分析
    used_vars = analyze_program_variables(program, dsl_variables)
    
    print(f"使用的变量 ({len(used_vars)}个):")
    if used_vars:
        for var in sorted(used_vars):
            marker = "⭐" if var in attitude_vars else "  "
            print(f"  {marker} {var}")
    else:
        print(f"  (无变量使用)")
    
    # 姿态分量检查
    used_attitude = [v for v in used_vars if v in attitude_vars]
    print(f"\n姿态分量使用情况:")
    if used_attitude:
        print(f"  ✅ 使用了 {len(used_attitude)} 个姿态分量: {used_attitude}")
    else:
        print(f"  ❌ 未使用任何姿态分量")
        if attitude_vars:
            print(f"     → Prior-{prior_level} 应该能够使用姿态分量，但MCTS没有探索到")
    
    print(f"\n{'='*70}\n")
    
    return {
        'prior_level': prior_level,
        'best_reward': float(best_reward) if best_reward != 'N/A' else None,
        'program_length': len(program),
        'used_variables': list(used_vars),
        'used_attitude_vars': used_attitude
    }


if __name__ == '__main__':
    results_dir = ROOT / '01_soar' / 'results'
    
    # 检查Prior-2的最新checkpoint
    print("\n" + "="*70)
    print("  检查Circle训练生成的程序内容")
    print("="*70)
    
    # 查找最新的checkpoint
    checkpoints = sorted(results_dir.glob('online_best_program_nn_iter_*.pt'))
    
    if not checkpoints:
        print("\n❌ 没有找到任何checkpoint文件")
        print(f"查找路径: {results_dir}")
        sys.exit(1)
    
    latest_ckpt = checkpoints[-1]
    print(f"\n找到 {len(checkpoints)} 个checkpoint文件")
    print(f"分析最新的: {latest_ckpt.name}\n")
    
    result = analyze_checkpoint(latest_ckpt, prior_level=2)
    
    print("\n" + "="*70)
    print("  总结")
    print("="*70)
    print(json.dumps(result, indent=2, ensure_ascii=False))

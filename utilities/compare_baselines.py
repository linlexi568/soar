#!/usr/bin/env python3
"""
基线控制器对比工具
对比 PID/CPID/LQR 调参结果 vs Soar DSL 程序
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "01_soar"))

def load_program(path: str):
    """加载 DSL 程序"""
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'rules' in data:
        return data['rules'], data.get('meta', {})
    return data, {}

def evaluate_with_batch_evaluator(program, traj: str, duration: float, num_envs: int):
    """使用 BatchEvaluator 评估程序"""
    from utils.batch_evaluation import BatchEvaluator
    
    evaluator = BatchEvaluator(
        isaac_num_envs=num_envs,
        reward_profile='safe_control_tracking',
        duration=duration,
        device='cuda:0',
        use_fast_path=True
    )
    
    results = evaluator.evaluate_batch(
        batch_programs=[program],
        task_name=traj,
        device='cuda:0'
    )
    
    return {
        'reward': float(results['rewards'][0]),
        'state_cost': float(results.get('state_costs', [0])[0]),
        'action_cost': float(results.get('action_costs', [0])[0])
    }

def run_baseline_tuning(controller_type: str, traj: str, duration: float, num_envs: int):
    """运行基线控制器调参"""
    script_map = {
        'pid': 'scripts/aligned/tune_pid_aligned.py',
        'cpid': 'scripts/aligned/tune_cpid_aligned.py',
        'lqr': 'scripts/aligned/tune_lqr_aligned.py'
    }
    
    if controller_type not in script_map:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    script_path = REPO_ROOT / script_map[controller_type]
    if not script_path.exists():
        raise FileNotFoundError(f"Baseline script not found: {script_path}")
    
    print(f"运行 {controller_type.upper()} 调参...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True
    )
    
    # 从输出中提取最佳奖励
    for line in result.stdout.split('\n'):
        if 'Best reward' in line or '最佳奖励' in line:
            try:
                reward = float(line.split(':')[-1].strip())
                return {'reward': reward, 'controller': controller_type}
            except:
                pass
    
    return {'reward': None, 'controller': controller_type}

def main():
    parser = argparse.ArgumentParser(description='对比基线控制器 vs Soar 程序')
    parser.add_argument('--soar-program', type=str, help='Soar 程序路径')
    parser.add_argument('--baseline', type=str, choices=['pid', 'cpid', 'lqr', 'all'], 
                        default='all', help='要测试的基线控制器')
    parser.add_argument('--traj', type=str, default='square')
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--num-envs', type=int, default=1024)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("基线控制器对比")
    print("=" * 80)
    print()
    
    results = {}
    
    # 评估 Soar 程序
    if args.soar_program:
        print(f"[1] 评估 Soar 程序: {args.soar_program}")
        program, meta = load_program(args.soar_program)
        pf_results = evaluate_with_batch_evaluator(program, args.traj, args.duration, args.num_envs)
        results['soar'] = pf_results
        print(f"    ✓ Reward: {pf_results['reward']:.4f}")
        print()
    
    # 评估基线控制器
    baselines = ['pid', 'cpid', 'lqr'] if args.baseline == 'all' else [args.baseline]
    
    for idx, controller in enumerate(baselines, start=2):
        print(f"[{idx}] 评估 {controller.upper()} 基线")
        baseline_results = run_baseline_tuning(controller, args.traj, args.duration, args.num_envs)
        results[controller] = baseline_results
        if baseline_results['reward'] is not None:
            print(f"    ✓ Reward: {baseline_results['reward']:.4f}")
        else:
            print(f"    ✗ 调参失败或未找到结果")
        print()
    
    # 汇总对比
    print("=" * 80)
    print("对比结果汇总")
    print("=" * 80)
    print(f"{'方法':<20} {'奖励 (Reward)':<20} {'相对 Soar':<20}")
    print("-" * 80)
    
    pf_reward = results.get('soar', {}).get('reward')
    
    for method, res in results.items():
        reward = res.get('reward')
        if reward is not None:
            if pf_reward and method != 'soar':
                ratio = (reward / pf_reward) * 100
                rel_str = f"{ratio:.1f}%"
            else:
                rel_str = "基准"
            print(f"{method.upper():<20} {reward:<20.4f} {rel_str:<20}")
        else:
            print(f"{method.upper():<20} {'N/A':<20} {'N/A':<20}")
    
    print("=" * 80)

if __name__ == '__main__':
    main()

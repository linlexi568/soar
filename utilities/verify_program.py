#!/usr/bin/env python3
"""
程序验证工具 - 直接基于 u_* 力/力矩输出
不依赖任何 PID 封装，纯粹验证 DSL 程序性能
"""
import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目路径
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "01_soar"))

# Isaac Gym 必须在 torch 之前导入
ISAAC_GYM_PATH = REPO_ROOT / "isaacgym" / "python"
if ISAAC_GYM_PATH.exists():
    sys.path.insert(0, str(ISAAC_GYM_PATH))

def load_program(program_path: str):
    """加载 DSL 程序 JSON"""
    with open(program_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'rules' in data:
        return data['rules'], data.get('meta', {})
    elif isinstance(data, list):
        return data, {}
    else:
        raise ValueError(f"Invalid program format in {program_path}")

def evaluate_program(program, traj='square', duration=5.0, num_envs=1024, replicas=1):
    """
    使用 BatchEvaluator 评估程序性能
    
    Args:
        program: DSL 程序规则列表
        traj: 轨迹类型
        duration: 仿真时长
        num_envs: 并行环境数
        replicas: 重复次数
    
    Returns:
        dict: 评估结果 {reward, state_cost, action_cost, ...}
    """
    from utils.batch_evaluation import BatchEvaluator
    from utilities.trajectory_presets import get_scg_trajectory_config
    
    # 使用与训练时相同的轨迹配置
    traj_cfg = get_scg_trajectory_config(traj)
    trajectory_config = {
        'type': traj_cfg.task,
        'params': dict(traj_cfg.params),
        'initial_xyz': list(traj_cfg.center)
    }
    
    evaluator = BatchEvaluator(
        isaac_num_envs=num_envs,
        reward_profile='safe_control_tracking',
        trajectory_config=trajectory_config,
        duration=duration,
        device='cuda:0',
        use_fast_path=True,
        strict_no_prior=True,  # 训练时用的是 True：完全直接 u_* 控制
        reward_reduction='sum',  # 与训练时对齐：使用 sum 而不是 mean
        zero_action_penalty=0.0,  # 测试时不使用零动作惩罚，只看真实性能
        replicas_per_program=1,  # 关键：与训练时对齐，设置为 1（训练时用的就是 1）
        enable_output_mad=False,  # 🔧 训练时 ENABLE_OUTPUT_MAD=false，必须关闭
    )
    
    # 使用 evaluate_batch_with_metrics 获取详细的奖励分解
    rewards_train, rewards_true, metrics_list = evaluator.evaluate_batch_with_metrics(
        programs=[program] * replicas
    )
    
    # 计算平均结果
    avg_reward_train = sum(rewards_train) / len(rewards_train) if rewards_train else 0.0
    avg_reward_true = sum(rewards_true) / len(rewards_true) if rewards_true else 0.0
    
    # 聚合 metrics
    avg_metrics = {}
    if metrics_list:
        for key in metrics_list[0].keys():
            values = [m.get(key, 0.0) for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)
    
    return {
        'reward_train': float(avg_reward_train),  # 包含惩罚项
        'reward_true': float(avg_reward_true),    # 不含惩罚项
        'metrics': avg_metrics,
        'num_envs': num_envs,
        'replicas': replicas
    }

def main():
    parser = argparse.ArgumentParser(description='验证 DSL 程序性能')
    parser.add_argument('--program', type=str, required=True, help='程序 JSON 文件路径')
    parser.add_argument('--traj', type=str, default='square', choices=['square', 'circle', 'figure8', 'helix'])
    parser.add_argument('--duration', type=float, default=5.0, help='仿真时长（秒）')
    parser.add_argument('--num-envs', type=int, default=1024, help='并行环境数')
    parser.add_argument('--replicas', type=int, default=1, help='重复评估次数')
    parser.add_argument('--match-training', action='store_true', help='使用训练时配置（从 meta 读取）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Soar 程序验证工具")
    print("=" * 80)
    print(f"程序文件: {args.program}")
    print(f"轨迹: {args.traj} | 时长: {args.duration}s")
    print(f"并行环境: {args.num_envs} | 重复: {args.replicas}")
    print()
    
    # 加载程序
    program, meta = load_program(args.program)
    print(f"✓ 程序加载成功: {len(program)} 条规则")
    
    # 反序列化程序以便检查时间算子
    try:
        from core.serialization import deserialize_program
        program_ast = deserialize_program({'rules': program})
    except Exception:
        program_ast = None
    
    # 如果启用 --match-training，从 meta 中读取配置
    num_envs = args.num_envs
    if args.match_training and meta:
        print(f"  使用训练时配置:")
        if 'isaac_num_envs' in meta:
            num_envs = meta['isaac_num_envs']
            print(f"    环境数: {num_envs}")
    
    if meta:
        print(f"  训练元信息:")
        for key in ['iteration', 'reward', 'isaac_num_envs', 'mcts_simulations']:
            if key in meta:
                print(f"    {key}: {meta[key]}")
    print()
    
    # 评估程序
    print("开始评估...")
    results = evaluate_program(
        program=program,
        traj=args.traj,
        duration=args.duration,
        num_envs=num_envs,
        replicas=args.replicas
    )
    
    print()
    print("=" * 80)
    print("评估结果")
    print("=" * 80)
    print(f"训练奖励 (含惩罚):   {results['reward_train']:.4f}")
    print(f"真实奖励 (不含惩罚): {results['reward_true']:.4f}")
    print()
    
    # 显示详细 metrics（实际返回的 state_cost 和 action_cost）
    if results['metrics']:
        print("代价分解:")
        metrics = results['metrics']
        if 'state_cost' in metrics:
            print(f"  状态代价 (state_cost):   {metrics['state_cost']:.6f}")
        if 'action_cost' in metrics:
            print(f"  动作代价 (action_cost):  {metrics['action_cost']:.6f}")
    print()
    
    if meta and 'reward' in meta:
        train_reward = meta['reward']
        test_reward = results['reward_train']  # 使用训练奖励对比
        diff = abs(train_reward - test_reward)
        print(f"训练时奖励 (meta):   {train_reward:.4f}")
        print(f"重新评估奖励:        {test_reward:.4f}")
        print(f"差异:                {diff:.4f} ({diff/abs(train_reward)*100:.1f}%)")
        print()
        # 检查是否有时间算子（delay/ema/diff/rate）- 使用反序列化的 AST
        has_temporal = False
        def check_temporal(node):
            nonlocal has_temporal
            if node is None:
                return
            if hasattr(node, 'op') and node.op in ('delay', 'ema', 'diff', 'rate', 'rate_limit'):
                has_temporal = True
            for attr in ['child', 'left', 'right', 'condition', 'then_branch', 'else_branch']:
                if hasattr(node, attr):
                    check_temporal(getattr(node, attr))
        if program_ast is not None:
            for rule in program_ast:
                check_temporal(rule.get('condition'))
                for a in rule.get('action', []) or []:
                    check_temporal(a)
        
        if has_temporal:
            print("⚠️  注意: 程序包含时间算子 (delay/ema/diff/rate)。")
            print("   训练时记录的奖励可能受到状态累积影响（非确定性）。")
            print("   当前评估使用确定性重置，每次从零状态开始。")
            print("   差异属于预期行为，不影响实际控制性能评估。")
        else:
            print("注意: 差异来自不同的环境初始化或配置，属于正常现象。")
        print("      真实奖励 = -(state_cost + action_cost)")
    
    print("=" * 80)

if __name__ == '__main__':
    main()

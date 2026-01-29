#!/usr/bin/env python3
"""
统一评估脚本：用完全相同的配置重新评估 Soar 和传统控制器
确保奖励计算方式一致
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '01_soar'))
sys.path.insert(0, str(ROOT))

print("=" * 70)
print("🔬 统一评估：Soar vs 传统控制器")
print("=" * 70)

# 统一配置
CONFIG = {
    'task': 'square',
    'duration': 5.0,
    'isaac_num_envs': 1024,  # 足够统计
    'replicas_per_program': 1,  # 不需要额外replicas
    'reward_reduction': 'sum',  # 累加，不取平均
    'reward_profile': 'safe_control_tracking',
    'device': 'cuda:0',
}

print(f"\n【统一评估配置】")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# 导入评估器
try:
    from utils.batch_evaluation import BatchEvaluator
    
    # 创建评估器
    evaluator = BatchEvaluator(
        trajectory_config={'type': CONFIG['task']},
        duration=int(CONFIG['duration']),
        isaac_num_envs=CONFIG['isaac_num_envs'],
        device=CONFIG['device'],
        replicas_per_program=CONFIG['replicas_per_program'],
        reward_reduction=CONFIG['reward_reduction'],
        reward_profile=CONFIG['reward_profile'],
        use_scg_exact_reward=True,
        strict_no_prior=True,
        zero_action_penalty=0.0,  # 不惩罚零动作
        use_fast_path=True,
        use_gpu_expression_executor=True,
    )
    
    print("\n✓ 评估器初始化成功")
    print(f"  reward_reduction: {evaluator.reward_reduction}")
    print(f"  isaac_num_envs: {evaluator.isaac_num_envs}")
    
    # 1. 评估Soar
    print("\n" + "=" * 70)
    print("【1. 评估 Soar best 程序】")
    print("=" * 70)
    
    soar_file = ROOT / 'results/soar_train/square_safe_control_tracking_best.json'
    with open(soar_file) as f:
        soar_data = json.load(f)
    
    program = soar_data['rules']
    print(f"\n  程序规则数: {len(program)}")
    print(f"  训练时奖励: {soar_data['meta']['reward']:.2f}")
    
    print("\n  开始评估...")
    reward_train, reward_true, components = evaluator.evaluate_single_with_metrics(program)
    
    print(f"\n  重新评估结果:")
    print(f"    reward_train: {reward_train:.2f}")
    print(f"    reward_true: {reward_true:.2f}")
    print(f"    state_cost: {components.get('state_cost', 0):.2f}")
    print(f"    action_cost: {components.get('action_cost', 0):.6f}")
    
    print(f"\n  对比:")
    print(f"    训练时: {soar_data['meta']['reward']:.2f}")
    print(f"    重新评估: {reward_true:.2f}")
    print(f"    差异: {abs(reward_true - soar_data['meta']['reward']):.2f}")
    
except Exception as e:
    print(f"\n❌ 评估失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("说明：")
print("  如果重新评估结果接近PID (-520)，说明训练时的奖励计算有bug")
print("  如果重新评估结果仍然很小 (-73)，说明程序本身性能不佳")
print("=" * 70)

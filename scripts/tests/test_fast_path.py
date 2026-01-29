#!/usr/bin/env python3
"""
快速测试快速路径优化效果
"""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_soar'))

def test_fast_path():
    """测试快速路径效果"""
    from batch_evaluation import BatchEvaluator
    
    print("="*80)
    print("测试快速路径优化 (程序预编译 + 向量化)")
    print("="*80)
    
    # 创建评估器
    evaluator = BatchEvaluator(
        trajectory_config={'type': 'hover', 'params': {}},
        duration=12,
        isaac_num_envs=8192,
        device='cuda:0',
        replicas_per_program=4,
        min_steps_frac=0.3,
        reward_reduction='mean',
        strict_no_prior=True,
        use_fast_path=True  # ✅ 启用快速路径
    )
    
    # 简单常量程序
    test_program = [
        {'op': 'set', 'var': 'u_fz', 'expr': {'type': 'const', 'value': 0.5}},
        {'op': 'set', 'var': 'u_tx', 'expr': {'type': 'const', 'value': 0.0}},
        {'op': 'set', 'var': 'u_ty', 'expr': {'type': 'const', 'value': 0.0}},
        {'op': 'set', 'var': 'u_tz', 'expr': {'type': 'const', 'value': 0.0}},
    ]
    
    print("\n测试1: 800程序 (快速路径启用)")
    print("-"*80)
    t0 = time.time()
    programs = [test_program] * 800
    rewards = evaluator.evaluate_batch(programs)
    t1 = time.time()
    fast_time = t1 - t0
    print(f"✅ 快速路径: {fast_time:.2f}秒 ({fast_time/800*1000:.1f}ms/程序)")
    print(f"   缓存大小: {len(evaluator._program_cache)} 个不同程序")
    
    # 关闭快速路径对比
    print("\n测试2: 800程序 (快速路径关闭)")
    print("-"*80)
    evaluator.use_fast_path = False
    evaluator._program_cache.clear()
    
    t0 = time.time()
    rewards2 = evaluator.evaluate_batch(programs)
    t1 = time.time()
    slow_time = t1 - t0
    print(f"✅ 慢速路径: {slow_time:.2f}秒 ({slow_time/800*1000:.1f}ms/程序)")
    
    # 对比
    print("\n" + "="*80)
    print("性能对比")
    print("="*80)
    speedup = slow_time / fast_time
    print(f"快速路径: {fast_time:.2f}秒")
    print(f"慢速路径: {slow_time:.2f}秒")
    print(f"🚀 加速比: {speedup:.2f}×")
    print(f"⏱️ 节省: {slow_time - fast_time:.2f}秒 ({(1 - 1/speedup)*100:.1f}%)")

if __name__ == '__main__':
    test_fast_path()

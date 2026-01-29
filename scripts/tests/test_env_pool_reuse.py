#!/usr/bin/env python3
"""
测试环境池持久化优化效果
对比reset vs reuse性能差异
"""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_soar'))

# 启用调试日志
os.environ['DEBUG_ENV_POOL'] = '1'

def test_env_pool_optimization():
    """测试环境池持久化优化"""
    from batch_evaluation import BatchEvaluator  # type: ignore
    
    print("="*80)
    print("测试环境池持久化优化 - 对比 Reset vs Reuse")
    print("="*80)
    
    # 创建评估器 (使用8192环境,足够800×4=3200使用)
    evaluator = BatchEvaluator(
        trajectory_config={'type': 'hover', 'params': {}},
        duration=12,
        isaac_num_envs=8192,
        device='cuda:0',
        replicas_per_program=4,
        min_steps_frac=0.3,
        reward_reduction='mean',
        strict_no_prior=True
    )
    
    # 生成测试程序
    test_program = [
        {'op': 'set', 'var': 'u_fz', 'expr': {'type': 'const', 'value': 0.5}},
        {'op': 'set', 'var': 'u_tx', 'expr': {'type': 'const', 'value': 0.0}},
        {'op': 'set', 'var': 'u_ty', 'expr': {'type': 'const', 'value': 0.0}},
        {'op': 'set', 'var': 'u_tz', 'expr': {'type': 'const', 'value': 0.0}},
    ]
    
    print("\n" + "="*80)
    print("第1轮: 800程序批量评估 (首次 - 会触发reset)")
    print("="*80)
    t0 = time.time()
    programs_800 = [test_program] * 800
    rewards_800 = evaluator.evaluate_batch(programs_800)
    t1 = time.time()
    elapsed_800 = t1 - t0
    print(f"✅ 完成: {elapsed_800:.2f}秒 ({elapsed_800/800*1000:.1f}ms/程序)")
    
    print("\n" + "="*80)
    print("第2轮: 4程序评估 (复用环境池 - 应该很快!)")
    print("="*80)
    t0 = time.time()
    programs_4 = [test_program] * 4
    rewards_4 = evaluator.evaluate_batch(programs_4)
    t1 = time.time()
    elapsed_4 = t1 - t0
    print(f"✅ 完成: {elapsed_4:.2f}秒 ({elapsed_4/4*1000:.1f}ms/程序)")
    
    print("\n" + "="*80)
    print("第3轮: 再次800程序 (应该复用环境池)")
    print("="*80)
    t0 = time.time()
    rewards_800_2 = evaluator.evaluate_batch(programs_800)
    t1 = time.time()
    elapsed_800_2 = t1 - t0
    print(f"✅ 完成: {elapsed_800_2:.2f}秒 ({elapsed_800_2/800*1000:.1f}ms/程序)")
    
    print("\n" + "="*80)
    print("性能对比总结")
    print("="*80)
    print(f"800程序 (首次reset):   {elapsed_800:.2f}秒  ({elapsed_800/800*1000:.1f}ms/程序)")
    print(f"4程序   (复用环境):     {elapsed_4:.2f}秒  ({elapsed_4/4*1000:.1f}ms/程序)")
    print(f"800程序 (复用环境):     {elapsed_800_2:.2f}秒  ({elapsed_800_2/800*1000:.1f}ms/程序)")
    
    speedup_4 = 7000.0 / (elapsed_4/4*1000)  # 假设旧版7000ms/程序
    print(f"\n🚀 4程序评估加速比: {speedup_4:.1f}× (从7000ms → {elapsed_4/4*1000:.1f}ms)")
    
    if speedup_4 > 20:
        print("✅ 优化成功! 环境池复用大幅提升小批次评估速度")
    else:
        print("⚠️ 加速效果未达预期,可能需要进一步调试")
    
    print("\n💡 提示: 如看到很多 '♻️ 复用环境池' 说明优化生效!")

if __name__ == '__main__':
    test_env_pool_optimization()

#!/usr/bin/env python3
"""网格搜索调参 - 优化 u_tx 程序在圆形轨迹上的性能

目标：在 circle 轨迹上达到 reward > -100（超越PID基线）
"""
import sys
sys.path.insert(0, '01_soar')

from core.dsl import BinaryOpNode, TerminalNode, ConstantNode, UnaryOpNode
from utils.batch_evaluation import BatchEvaluator
import numpy as np
import json
from datetime import datetime

def build_utx_program(k_py, k_dy, k_ry, k_wy, k_cf, 
                      smooth_sy=0.5, smooth_sroll=0.4,
                      diff_ky=2.0, ema_alpha_x=0.3,
                      dz_vel_y=0.05, smoothstep_sy=0.4,
                      clamp_lo=-0.65, clamp_hi=0.65):
    """构建参数化的 u_tx 控制律"""
    k_py_node = ConstantNode(k_py, name='k_py')
    k_dy_node = ConstantNode(k_dy, name='k_dy')
    k_ry_node = ConstantNode(k_ry, name='k_ry')
    k_wy_node = ConstantNode(k_wy, name='k_wy')
    k_cf_node = ConstantNode(k_cf, name='k_cf')
    
    # 位置环
    pos_p = BinaryOpNode('*', k_py_node, 
        UnaryOpNode('smooth', TerminalNode('pos_err_y'), 
            {'s': ConstantNode(smooth_sy, name='smooth_sy')}))
    pos_d = BinaryOpNode('*', k_dy_node,
        UnaryOpNode('diff', TerminalNode('pos_err_y'),
            {'k': ConstantNode(diff_ky, name='diff_ky')}))
    pos_loop = BinaryOpNode('+', pos_p, pos_d)
    
    # 姿态环
    att_p = BinaryOpNode('*', k_ry_node,
        UnaryOpNode('smooth', TerminalNode('err_p_roll'),
            {'s': ConstantNode(smooth_sroll, name='smooth_sroll')}))
    att_d = BinaryOpNode('*', k_wy_node,
        UnaryOpNode('ema', TerminalNode('ang_vel_x'),
            {'alpha': ConstantNode(ema_alpha_x, name='ema_alpha_x')}))
    att_loop = BinaryOpNode('+', att_p, att_d)
    
    # 前馈补偿
    vel_dz = UnaryOpNode('deadzone', TerminalNode('vel_y'),
        {'eps': ConstantNode(dz_vel_y, name='dz_vel_y')})
    vel_sign = BinaryOpNode('*', UnaryOpNode('sign', TerminalNode('vel_y')), vel_dz)
    ff_term = BinaryOpNode('*', k_cf_node,
        UnaryOpNode('smoothstep', vel_sign,
            {'s': ConstantNode(smoothstep_sy, name='smoothstep_sy')}))
    
    # 总输出
    inner = BinaryOpNode('+', att_loop, ff_term)
    total = BinaryOpNode('+', pos_loop, inner)
    expr = UnaryOpNode('clamp', total, {
        'lo': ConstantNode(clamp_lo, name='clamp_lo_tx'),
        'hi': ConstantNode(clamp_hi, name='clamp_hi_tx')
    })
    
    return [{'condition': None, 'action': [BinaryOpNode('set', TerminalNode('u_tx'), expr)]}]


def grid_search():
    """网格搜索调参"""
    print("="*70)
    print("网格搜索调参 - u_tx 圆形轨迹优化")
    print("="*70)
    print(f"目标: reward_true > -100 (超越PID基线)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 初始化评估器
    be = BatchEvaluator(
        trajectory_config={'type': 'circle', 'params': {'R': 0.9, 'period': 10.0}, 
                          'initial_xyz': [0.0, 0.0, 1.0]},
        duration=8,
        isaac_num_envs=256,
        device='cuda:0',
        replicas_per_program=1,
        reward_profile='safe_control_tracking',
        enable_bayesian_tuning=False,
        use_fast_path=False,
        use_gpu_expression_executor=True,
    )
    
    # 粗网格搜索：主要增益参数
    k_py_grid = [0.4, 0.7, 1.0, 1.3]      # 位置比例增益
    k_dy_grid = [0.2, 0.35, 0.5]          # 位置微分增益
    k_ry_grid = [-0.6, -0.45, -0.3]       # 姿态比例增益（负反馈）
    k_wy_grid = [-0.25, -0.15, -0.08]     # 姿态微分增益（负反馈）
    k_cf_grid = [0.15, 0.25, 0.35]        # 前馈补偿增益
    
    total_tests = len(k_py_grid) * len(k_dy_grid) * len(k_ry_grid) * len(k_wy_grid) * len(k_cf_grid)
    print(f"\n总测试组合: {total_tests}")
    print(f"预计耗时: ~{total_tests * 3:.0f}秒 (每个3秒)")
    print("\n开始搜索...\n")
    
    best_reward = -float('inf')
    best_params = None
    best_metrics = None
    test_count = 0
    
    results = []
    
    for k_py in k_py_grid:
        for k_dy in k_dy_grid:
            for k_ry in k_ry_grid:
                for k_wy in k_wy_grid:
                    for k_cf in k_cf_grid:
                        test_count += 1
                        
                        # 构建并镜像程序
                        program = build_utx_program(k_py, k_dy, k_ry, k_wy, k_cf)
                        mirrored = be._mirror_expand_single_axis_program(program)
                        
                        # 评估
                        try:
                            r_train, r_true, metrics = be.evaluate_single_with_metrics(mirrored)
                            state_c = metrics.get('state_cost', 0)
                            
                            results.append({
                                'k_py': k_py, 'k_dy': k_dy, 'k_ry': k_ry,
                                'k_wy': k_wy, 'k_cf': k_cf,
                                'reward': r_true, 'state_cost': state_c
                            })
                            
                            # 更新最优
                            if r_true > best_reward:
                                best_reward = r_true
                                best_params = (k_py, k_dy, k_ry, k_wy, k_cf)
                                best_metrics = metrics
                                print(f"[{test_count}/{total_tests}] ✨ 新最优! reward={r_true:.2f}, state_cost={state_c:.1f}")
                                print(f"             参数: k_py={k_py:.2f}, k_dy={k_dy:.2f}, k_ry={k_ry:.2f}, k_wy={k_wy:.2f}, k_cf={k_cf:.2f}")
                            else:
                                if test_count % 10 == 0:
                                    print(f"[{test_count}/{total_tests}] reward={r_true:.2f}, state_cost={state_c:.1f} | 当前最优={best_reward:.2f}")
                        
                        except Exception as e:
                            print(f"[{test_count}/{total_tests}] ❌ 评估失败: {e}")
                            results.append({
                                'k_py': k_py, 'k_dy': k_dy, 'k_ry': k_ry,
                                'k_wy': k_wy, 'k_cf': k_cf,
                                'reward': -1e6, 'state_cost': 1e9
                            })
    
    # 输出结果
    print("\n" + "="*70)
    print("网格搜索完成！")
    print("="*70)
    
    if best_params:
        k_py, k_dy, k_ry, k_wy, k_cf = best_params
        print(f"\n🏆 最优参数:")
        print(f"  k_py (位置P)  = {k_py:.3f}")
        print(f"  k_dy (位置D)  = {k_dy:.3f}")
        print(f"  k_ry (姿态P)  = {k_ry:.3f}")
        print(f"  k_wy (姿态D)  = {k_wy:.3f}")
        print(f"  k_cf (前馈)   = {k_cf:.3f}")
        
        print(f"\n📊 最优性能:")
        print(f"  reward_true  = {best_reward:.4f}")
        print(f"  state_cost   = {best_metrics.get('state_cost', 0):.2f}")
        print(f"  action_cost  = {best_metrics.get('action_cost', 0):.2e}")
        
        if best_reward > -100:
            print(f"\n✅✅✅ 成功！超越PID基线 (reward > -100)")
        elif best_reward > -500:
            print(f"\n✅ 良好性能 (reward > -500)")
        else:
            print(f"\n⚠️  需要进一步优化或细化网格")
        
        # 保存最优程序
        print(f"\n💾 保存最优程序...")
        best_program = build_utx_program(k_py, k_dy, k_ry, k_wy, k_cf)
        mirrored_best = be._mirror_expand_single_axis_program(best_program)
        
        # 转换为可序列化格式并保存
        from core.serialization import serialize_program
        serialized = serialize_program(mirrored_best)
        
        output_path = 'results/grid_tuned_circle_utx_best.json'
        with open(output_path, 'w') as f:
            json.dump(serialized, f, indent=2)
        print(f"  ✓ 已保存到: {output_path}")
        
        # 保存所有结果
        results_sorted = sorted(results, key=lambda x: x['reward'], reverse=True)
        results_path = 'results/grid_search_circle_utx_all.json'
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'best_reward': best_reward,
                'best_params': {
                    'k_py': k_py, 'k_dy': k_dy, 'k_ry': k_ry,
                    'k_wy': k_wy, 'k_cf': k_cf
                },
                'all_results': results_sorted[:50]  # 保存前50个
            }, f, indent=2)
        print(f"  ✓ 完整结果已保存到: {results_path}")
    
    else:
        print("\n❌ 未找到有效参数")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    grid_search()

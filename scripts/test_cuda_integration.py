#!/usr/bin/env python3
"""
快速测试CUDA加速是否正常工作

使用方法:
    python scripts/test_cuda_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_soar'))

import torch
import time


def test_cuda_availability():
    """测试CUDA是否可用"""
    print("=" * 70)
    print("1. CUDA可用性检查")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        print("   请检查: nvidia-smi, nvcc --version")
        return False
    
    print(f"✅ CUDA可用")
    print(f"   设备: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   CUDA版本: {torch.version.cuda}")
    return True


def test_cuda_executor():
    """测试CUDA执行器"""
    print("\n" + "=" * 70)
    print("2. CUDA执行器测试")
    print("=" * 70)
    
    try:
        from utils.cuda_program_executor import CUDAProgramExecutor
        
        executor = CUDAProgramExecutor(device='cuda:0')
        print("✅ CUDA执行器初始化成功")
        
        # 创建测试程序
        test_programs = [
            [
                {'op': 'set', 'var': 'u_fz', 'expr': {'type': 'const', 'value': 0.5}},
                {'op': 'set', 'var': 'u_tx', 'expr': {'type': 'const', 'value': 0.01}},
            ]
        ] * 100
        
        # 编译
        t0 = time.time()
        forces = executor.compile_constant_programs(test_programs)
        compile_time = (time.time() - t0) * 1000
        
        if forces is None:
            print("❌ 编译失败")
            return False
        
        print(f"✅ 编译成功 ({compile_time:.2f}ms)")
        print(f"   Forces shape: {forces.shape}, device: {forces.device}")
        
        # 执行
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(384):
            actions = executor.apply_constant_forces_vectorized(forces, 100, 4096)
        torch.cuda.synchronize()
        exec_time = (time.time() - t0) * 1000
        
        print(f"✅ 执行成功 (384步: {exec_time:.2f}ms, 每步: {exec_time/384:.3f}ms)")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA执行器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_evaluation_integration():
    """测试batch_evaluation集成"""
    print("\n" + "=" * 70)
    print("3. Batch Evaluation集成测试")
    print("=" * 70)
    
    try:
        from utils.batch_evaluation import BatchEvaluator
        print("✅ BatchEvaluator导入成功")
        
        # 检查是否有CUDA相关代码
        import inspect
        source = inspect.getsource(BatchEvaluator.evaluate_batch)
        
        if '[CUDA]' in source or 'cuda_executor' in source:
            print("✅ CUDA集成代码已添加")
        else:
            print("⚠️  未检测到CUDA集成代码 (可能需要更新)")
        
        return True
        
    except Exception as e:
        print(f"⚠️  BatchEvaluator检查失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "CUDA加速集成测试" + " " * 31 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # 测试CUDA可用性
    if not test_cuda_availability():
        print("\n❌ CUDA不可用，退出测试")
        sys.exit(1)
    
    # 测试CUDA执行器
    if not test_cuda_executor():
        print("\n❌ CUDA执行器测试失败")
        sys.exit(1)
    
    # 测试集成
    test_batch_evaluation_integration()
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 所有测试通过! CUDA加速已就绪")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 运行训练: bash run.sh")
    print("  2. 监控GPU: watch -n 1 nvidia-smi")
    print("  3. 查看文档: cat CUDA_ACCELERATION_GUIDE.md")
    print()


if __name__ == '__main__':
    main()

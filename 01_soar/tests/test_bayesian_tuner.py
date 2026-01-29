"""Test script for Bayesian Optimization Tuner

测试 BayesianTuner 在一个简单的合成优化问题上的表现。
"""
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import sys
sys.path.insert(0, str(_ROOT / '01_soar'))
from utils.bayesian_tuner import BayesianTuner, ParameterSpec


def test_simple_optimization():
    """测试 BO 在简单的 2D 函数优化上的表现
    
    目标函数: f(x, y) = -(x - 2)^2 - (y + 1)^2
    最优解: (2, -1)，最大值: 0
    """
    print("=" * 70)
    print("Test 1: Simple 2D Optimization")
    print("=" * 70)
    
    # 定义参数空间
    param_specs = [
        ParameterSpec(name='x', low=-5.0, high=5.0),
        ParameterSpec(name='y', low=-5.0, high=5.0),
    ]
    
    # 定义目标函数
    def eval_fn(X):
        """X: [batch_size, 2]"""
        x, y = X[:, 0], X[:, 1]
        return -(x - 2.0)**2 - (y + 1.0)**2
    
    # 创建 BO 调参器
    tuner = BayesianTuner(
        param_specs=param_specs,
        batch_size=10,
        n_iterations=5,
        ucb_kappa=2.0,
        random_seed=42
    )
    
    # 运行优化
    best_params, best_reward = tuner.optimize(eval_fn, verbose=True)
    
    print(f"\n✅ Optimization finished!")
    print(f"   Best params: x={best_params[0]:.4f}, y={best_params[1]:.4f}")
    print(f"   Best reward: {best_reward:.4f}")
    print(f"   Target: x=2.0, y=-1.0, reward=0.0")
    print(f"   Error: |x-2|={abs(best_params[0]-2.0):.4f}, |y+1|={abs(best_params[1]+1.0):.4f}")
    
    # 验证结果
    assert abs(best_params[0] - 2.0) < 0.5, "x 参数偏差过大"
    assert abs(best_params[1] + 1.0) < 0.5, "y 参数偏差过大"
    print("\n✅ Test passed: BO 找到了接近最优解的参数\n")


def test_noisy_optimization():
    """测试 BO 在噪声环境下的鲁棒性"""
    print("=" * 70)
    print("Test 2: Optimization with Noisy Observations")
    print("=" * 70)
    
    param_specs = [
        ParameterSpec(name='kp', low=0.1, high=10.0, log_scale=True),
        ParameterSpec(name='ki', low=0.01, high=5.0, log_scale=True),
    ]
    
    def noisy_eval_fn(X):
        """模拟带噪声的控制器性能评估
        f(kp, ki) = -|kp - 1.5| - |ki - 0.5| + noise
        """
        kp, ki = X[:, 0], X[:, 1]
        rewards = -(np.abs(kp - 1.5) + np.abs(ki - 0.5))
        # 加入 20% 的噪声
        noise = np.random.randn(len(rewards)) * 0.2
        return rewards + noise
    
    tuner = BayesianTuner(
        param_specs=param_specs,
        batch_size=15,
        n_iterations=4,
        ucb_kappa=2.5,  # 更高的探索系数应对噪声
        random_seed=123
    )
    
    best_params, best_reward = tuner.optimize(noisy_eval_fn, verbose=True)
    
    print(f"\n✅ Optimization finished!")
    print(f"   Best params: kp={best_params[0]:.4f}, ki={best_params[1]:.4f}")
    print(f"   Best reward: {best_reward:.4f}")
    print(f"   Target: kp=1.5, ki=0.5")
    
    # 噪声环境下，要求更宽松
    assert abs(np.log10(best_params[0]) - np.log10(1.5)) < 1.0, "kp 偏差过大"
    assert abs(np.log10(best_params[1]) - np.log10(0.5)) < 1.0, "ki 偏差过大"
    print("\n✅ Test passed: BO 在噪声下依然有效\n")


def test_program_extraction():
    """测试从程序中提取可调参数"""
    print("=" * 70)
    print("Test 3: Extract Tunable Parameters from Program")
    print("=" * 70)
    
    # 构造一个简单的程序
    try:
        from core.dsl import TerminalNode, BinaryOpNode
    except ImportError:
        sys.path.insert(0, str(_ROOT / '01_soar' / 'core'))
        from dsl import TerminalNode, BinaryOpNode
    
    from utils.bayesian_tuner import extract_tunable_params, inject_tuned_params
    
    # 程序: u_fz = pos_err * 1.5 + vel_z * 0.8
    prog = [{
        'condition': TerminalNode(1.0),  # always true
        'action': [
            BinaryOpNode('set', TerminalNode('u_fz'),
                BinaryOpNode('+',
                    BinaryOpNode('*', TerminalNode('pos_err_z'), TerminalNode(1.5)),
                    BinaryOpNode('*', TerminalNode('vel_z'), TerminalNode(0.8))
                )
            )
        ]
    }]
    
    # 提取参数
    params = extract_tunable_params(prog)
    print(f"提取到 {len(params)} 个可调参数:")
    for path, value in params:
        print(f"   {path} = {value}")
    
    assert len(params) == 3, f"应提取 3 个参数（1.0, 1.5, 0.8），实际提取了 {len(params)} 个"
    
    # 注入新参数
    tuned_vals = {
        'rule_0_action_0_right_left_right': 2.0,  # 替换 1.5
        'rule_0_action_0_right_right_right': 1.2,  # 替换 0.8
    }
    inject_tuned_params(prog, tuned_vals)
    
    # 验证注入是否成功
    new_params = extract_tunable_params(prog)
    print(f"\n注入后的参数:")
    for path, value in new_params:
        print(f"   {path} = {value}")
    
    # 检查值是否更新
    param_dict = {path: val for path, val in new_params}
    assert param_dict['rule_0_action_0_right_left_right'] == 2.0
    assert param_dict['rule_0_action_0_right_right_right'] == 1.2
    
    print("\n✅ Test passed: 参数提取和注入功能正常\n")


if __name__ == '__main__':
    print("\n🚀 Starting Bayesian Optimization Tuner Tests\n")
    
    test_simple_optimization()
    test_noisy_optimization()
    test_program_extraction()
    
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)

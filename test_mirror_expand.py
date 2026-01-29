#!/usr/bin/env python3
"""测试镜像扩展功能"""
import sys
sys.path.insert(0, '01_soar')

from core.dsl import TerminalNode, ConstantNode, BinaryOpNode, UnaryOpNode
from utils.batch_evaluation import BatchEvaluator

# 创建一个简单的单轴程序
def create_simple_u_tx_program():
    """创建一个只设置 u_tx 的简单程序"""
    # u_tx = err_p_roll * 0.5
    expr = BinaryOpNode('mul', TerminalNode('err_p_roll'), ConstantNode(0.5))
    action = BinaryOpNode('set', TerminalNode('u_tx'), expr)
    rule = {
        'condition': ConstantNode(True),
        'action': [action]
    }
    return [rule]

# 测试镜像扩展
if __name__ == '__main__':
    program = create_simple_u_tx_program()
    print("原始程序:")
    print(f"  Rules: {len(program)}")
    for i, rule in enumerate(program):
        print(f"  Rule {i}: {rule}")
    
    # 初始化 BatchEvaluator (不需要真正初始化Isaac Gym)
    evaluator = BatchEvaluator(
        trajectory_config={'type': 'hover', 'params': {}},
        duration=5,
        isaac_num_envs=96,
        device='cuda:0'
    )
    
    # 执行镜像扩展
    expanded = evaluator._mirror_expand_single_axis_program(program)
    
    print("\n镜像扩展后:")
    print(f"  Rules: {len(expanded)}")
    for i, rule in enumerate(expanded):
        print(f"  Rule {i}:")
        actions = rule.get('action', [])
        print(f"    Actions: {len(actions)}")
        for j, act in enumerate(actions):
            if isinstance(act, BinaryOpNode):
                left_val = act.left.value if isinstance(act.left, TerminalNode) else '?'
                print(f"      Action {j}: BinaryOpNode(set, {left_val}, ...)")
            elif isinstance(act, dict):
                print(f"      Action {j}: dict({act.get('op')}, {act.get('left', {}).get('value')}, ...)")
    
    # 测试编译
    from utils.gpu_program_executor import ProgramCompiler
    try:
        compiler = ProgramCompiler()
        compiled = compiler.compile(expanded)
        print("\n编译成功!")
        print(f"  Outputs: {compiled.outputs}")
        print(f"  Instructions: {len(compiled.instructions)}")
    except Exception as e:
        print(f"\n编译失败: {e}")
        import traceback
        traceback.print_exc()

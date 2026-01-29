#!/usr/bin/env python3
"""测试镜像后的程序是否能正常执行"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_soar'))

from core.dsl import BinaryOpNode, ConstantNode, TerminalNode, UnaryOpNode

# 模拟 batch_evaluation 中的镜像逻辑
def mirror_expand_program(program):
    """完全复制 _mirror_expand_single_axis_program 的逻辑"""
    
    # 检查是否只有单个输出通道
    output_vars = set()
    for rule in program:
        if rule.get('type') == 'Assignment':
            output_vars.add(rule.get('target'))
    
    if len(output_vars) != 1:
        print(f"程序有 {len(output_vars)} 个输出，跳过镜像")
        return program
    
    single_channel = list(output_vars)[0]
    print(f"检测到单通道输出: {single_channel}")
    
    # 定义镜像映射
    if single_channel == 'u_tx':
        # roll → pitch
        tx_to_ty = {
            'err_p_roll': 'err_p_pitch',
            'err_d_roll': 'err_d_pitch',
            'err_i_roll': 'err_i_pitch',
            'ang_vel_x': 'ang_vel_y',
        }
        
        # 创建 u_ty 规则
        u_ty_rule = None
        for rule in program:
            if rule.get('target') == 'u_tx':
                u_ty_rule = _map_assignment(rule, tx_to_ty, 'u_ty')
                break
        
        # 创建固定 PID
        yaw_rule = create_fixed_yaw_pid()
        thrust_rule = create_fixed_thrust_pid()
        
        result = program + [u_ty_rule, yaw_rule, thrust_rule]
        print(f"镜像完成: {len(program)} → {len(result)} 规则")
        return result
    else:
        print(f"不支持的单通道: {single_channel}")
        return program

def _map_assignment(rule, var_mapping, new_target):
    """映射赋值规则到新变量"""
    import copy
    new_rule = copy.deepcopy(rule)
    new_rule['target'] = new_target
    new_rule['expression'] = _map_expr(rule['expression'], var_mapping)
    return new_rule

def _map_expr(expr, var_mapping):
    """递归映射表达式中的变量"""
    import copy
    
    if isinstance(expr, dict):
        etype = expr.get('type')
        if etype == 'TerminalNode':
            val = expr.get('value')
            if val in var_mapping:
                return {'type': 'TerminalNode', 'value': var_mapping[val]}
            return copy.deepcopy(expr)
        elif etype == 'ConstantNode':
            return copy.deepcopy(expr)
        elif etype == 'UnaryOpNode' or etype == 'Unary':
            return {
                'type': 'UnaryOpNode',
                'op': expr.get('op'),
                'child': _map_expr(expr.get('child'), var_mapping),
                'params': expr.get('params')
            }
        elif etype == 'BinaryOpNode' or etype == 'Binary':
            return {
                'type': 'BinaryOpNode',
                'op': expr.get('op'),
                'left': _map_expr(expr.get('left'), var_mapping),
                'right': _map_expr(expr.get('right'), var_mapping)
            }
    return copy.deepcopy(expr)

def create_fixed_yaw_pid():
    """创建固定的偏航 PID"""
    # 4.0 * err_p_yaw - 0.8 * ang_vel_z
    p_term = {
        'type': 'BinaryOpNode',
        'op': '*',
        'left': {'type': 'ConstantNode', 'value': 4.0, 'name': 'c_yaw_p', 'min_val': 4.0, 'max_val': 4.0},
        'right': {'type': 'TerminalNode', 'value': 'err_p_yaw'}
    }
    d_term = {
        'type': 'BinaryOpNode',
        'op': '*',
        'left': {'type': 'ConstantNode', 'value': 0.8, 'name': 'c_yaw_d', 'min_val': 0.8, 'max_val': 0.8},
        'right': {'type': 'TerminalNode', 'value': 'ang_vel_z'}
    }
    expr = {
        'type': 'BinaryOpNode',
        'op': '-',
        'left': p_term,
        'right': d_term
    }
    return {'type': 'Assignment', 'target': 'u_tz', 'expression': expr}

def create_fixed_thrust_pid():
    """创建固定的推力 PID"""
    # 14.0*pos_err_z - 6.0*vel_z + 0.05*err_i_z + 0.6
    p_term = {
        'type': 'BinaryOpNode',
        'op': '*',
        'left': {'type': 'ConstantNode', 'value': 14.0, 'name': 'c_thrust_p', 'min_val': 14.0, 'max_val': 14.0},
        'right': {'type': 'TerminalNode', 'value': 'pos_err_z'}
    }
    d_term = {
        'type': 'BinaryOpNode',
        'op': '*',
        'left': {'type': 'ConstantNode', 'value': 6.0, 'name': 'c_thrust_d', 'min_val': 6.0, 'max_val': 6.0},
        'right': {'type': 'TerminalNode', 'value': 'vel_z'}
    }
    i_term = {
        'type': 'BinaryOpNode',
        'op': '*',
        'left': {'type': 'ConstantNode', 'value': 0.05, 'name': 'c_thrust_i', 'min_val': 0.05, 'max_val': 0.05},
        'right': {'type': 'TerminalNode', 'value': 'err_i_z'}
    }
    ff_term = {'type': 'ConstantNode', 'value': 0.6, 'name': 'c_thrust_ff', 'min_val': 0.6, 'max_val': 0.6}
    
    # p - d
    pd = {'type': 'BinaryOpNode', 'op': '-', 'left': p_term, 'right': d_term}
    # i + ff
    i_ff = {'type': 'BinaryOpNode', 'op': '+', 'left': i_term, 'right': ff_term}
    # (p - d) + (i + ff)
    expr = {'type': 'BinaryOpNode', 'op': '+', 'left': pd, 'right': i_ff}
    
    return {'type': 'Assignment', 'target': 'u_fz', 'expression': expr}

def ast_from_dict(node_dict):
    """从字典构建 AST 节点（用于测试执行）"""
    if not isinstance(node_dict, dict):
        return node_dict
    
    ntype = node_dict.get('type')
    if ntype == 'TerminalNode':
        return TerminalNode(node_dict['value'])
    elif ntype == 'ConstantNode':
        return ConstantNode(
            node_dict['value'],
            name=node_dict.get('name'),
            min_val=node_dict.get('min_val'),
            max_val=node_dict.get('max_val')
        )
    elif ntype == 'UnaryOpNode' or ntype == 'Unary':
        return UnaryOpNode(
            node_dict['op'],
            ast_from_dict(node_dict['child']),
            node_dict.get('params', {})
        )
    elif ntype == 'BinaryOpNode' or ntype == 'Binary':
        return BinaryOpNode(
            node_dict['op'],
            ast_from_dict(node_dict['left']),
            ast_from_dict(node_dict['right'])
        )
    return None

def test_execution(program):
    """测试程序执行"""
    print("\n=== 测试执行 ===")
    
    # 模拟状态
    state = {
        'err_p_roll': 0.1,
        'err_d_roll': 0.05,
        'err_i_roll': 0.01,
        'ang_vel_x': -0.02,
        'err_p_pitch': 0.08,
        'err_d_pitch': 0.03,
        'err_i_pitch': 0.005,
        'ang_vel_y': -0.015,
        'err_p_yaw': 0.02,
        'ang_vel_z': 0.01,
        'pos_err_z': -0.5,
        'vel_z': 0.1,
        'err_i_z': -0.2,
    }
    
    outputs = {}
    for rule in program:
        if rule.get('type') == 'Assignment':
            target = rule['target']
            expr_dict = rule['expression']
            
            # 转换为 AST 并执行
            expr_ast = ast_from_dict(expr_dict)
            try:
                value = expr_ast.evaluate(state)
                outputs[target] = value
                print(f"  {target} = {value:.6f}")
            except Exception as e:
                print(f"  {target} = ERROR: {e}")
                outputs[target] = 0.0
    
    return outputs

# 主测试
if __name__ == '__main__':
    # 原始单通道程序 (来自日志)
    original_program = [
        {
            'type': 'Assignment',
            'target': 'u_tx',
            'expression': {
                'type': 'BinaryOpNode',
                'op': '-',
                'left': {'type': 'TerminalNode', 'value': 'err_d_roll'},
                'right': {
                    'type': 'BinaryOpNode',
                    'op': '*',
                    'left': {'type': 'ConstantNode', 'value': 0.845, 'name': 'c_94a3db', 'min_val': 0.0, 'max_val': 1.5},
                    'right': {
                        'type': 'UnaryOpNode',
                        'op': 'clamp',
                        'child': {'type': 'TerminalNode', 'value': 'err_p_roll'},
                        'params': {}
                    }
                }
            }
        }
    ]
    
    print("原始程序:")
    for rule in original_program:
        print(f"  {rule['target']} = ...")
    
    # 执行镜像
    mirrored_program = mirror_expand_program(original_program)
    
    print("\n镜像后程序:")
    for rule in mirrored_program:
        print(f"  {rule['target']} = ...")
    
    # 测试执行
    outputs = test_execution(mirrored_program)
    
    print("\n=== 结果诊断 ===")
    if all(abs(v) < 1e-9 for v in outputs.values()):
        print("❌ 所有输出都是 0！")
    else:
        print("✅ 有非零输出:")
        for k, v in outputs.items():
            if abs(v) > 1e-9:
                print(f"  {k} = {v:.6f}")

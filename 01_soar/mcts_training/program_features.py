"""Program structure featurization for neural network input.

将DSL程序（规则列表）转换为固定维度向量（64维），用于策略-价值网络输入。
不注入控制律知识，只提取纯结构化特征。
"""
from typing import List, Dict, Any
import numpy as np
import torch

# 导入DSL节点类型
try:
    from core.dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode
except Exception:
    # 添加路径以支持直接运行
    import sys, pathlib
    _parent = pathlib.Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from core.dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode


def count_nodes_recursive(node: Any, counts: Dict[str, int]) -> None:
    """递归统计节点类型和操作符"""
    if node is None:
        return
    
    # 统计节点类型
    node_type = type(node).__name__
    counts[node_type] = counts.get(node_type, 0) + 1
    
    # 统计具体操作符
    if isinstance(node, BinaryOpNode):
        counts[f'op_{node.op}'] = counts.get(f'op_{node.op}', 0) + 1
        count_nodes_recursive(node.left, counts)
        count_nodes_recursive(node.right, counts)
    elif isinstance(node, UnaryOpNode):
        counts[f'op_{node.op}'] = counts.get(f'op_{node.op}', 0) + 1
        count_nodes_recursive(node.child, counts)
    elif isinstance(node, IfNode):
        count_nodes_recursive(node.condition, counts)
        count_nodes_recursive(node.then_branch, counts)
        count_nodes_recursive(node.else_branch, counts)
    elif isinstance(node, TerminalNode):
        if isinstance(node.value, str):
            # 统计变量使用
            counts[f'var_{node.value}'] = counts.get(f'var_{node.value}', 0) + 1
        else:
            counts['const'] = counts.get('const', 0) + 1


def compute_max_depth(node: Any, current_depth: int = 0) -> int:
    """计算树的最大深度"""
    if node is None:
        return current_depth
    
    if isinstance(node, BinaryOpNode):
        left_depth = compute_max_depth(node.left, current_depth + 1)
        right_depth = compute_max_depth(node.right, current_depth + 1)
        return max(left_depth, right_depth)
    elif isinstance(node, UnaryOpNode):
        return compute_max_depth(node.child, current_depth + 1)
    elif isinstance(node, IfNode):
        cond_depth = compute_max_depth(node.condition, current_depth + 1)
        then_depth = compute_max_depth(node.then_branch, current_depth + 1)
        else_depth = compute_max_depth(node.else_branch, current_depth + 1)
        return max(cond_depth, then_depth, else_depth)
    else:
        return current_depth


def featurize_program(program: List[Dict[str, Any]]) -> torch.Tensor:
    """
    将程序转换为64维特征向量
    
    特征分组：
    - [0-12]: 节点类型统计（BinaryOpNode, UnaryOpNode, TerminalNode, IfNode等）
    - [13-25]: 操作符统计（+, -, *, >, <, abs, sin, cos等）
    - [26-35]: 变量使用统计（pos_err, vel_err, ang_err等）
    - [36-45]: 结构统计（规则数、深度、节点数、multiplier等）
    - [46-63]: 保留扩展（padding）
    
    Args:
        program: 规则列表，每个规则是dict {'condition': ..., 'action': ...}
    
    Returns:
        torch.Tensor: shape [64], dtype=float32
    """
    features = []
    
    # 统计计数器
    counts = {}
    
    # 遍历所有规则，统计节点
    for rule in program:
        if 'condition' in rule and rule['condition'] is not None:
            count_nodes_recursive(rule['condition'], counts)
        
        if 'action' in rule:
            action = rule['action']
            if isinstance(action, list):
                for act in action:
                    count_nodes_recursive(act, counts)
            else:
                count_nodes_recursive(action, counts)
    
    # === 特征组1: 节点类型统计 (13维) ===
    node_types = [
        'BinaryOpNode', 'UnaryOpNode', 'TerminalNode', 'IfNode',
        'const'  # 常量节点
    ]
    for nt in node_types:
        features.append(counts.get(nt, 0) / 10.0)  # 归一化
    
    # 额外类型特征
    features.append(counts.get('op_set', 0) / 5.0)  # set操作（用于action）
    
    # Padding到13维
    while len(features) < 13:
        features.append(0.0)
    
    # === 特征组2: 操作符统计 (13维) ===
    operators = [
        'op_+', 'op_-', 'op_*', 'op_/', 
        'op_>', 'op_<', 'op_==', 'op_!=',
        'op_max', 'op_min',
        'op_abs', 'op_sin', 'op_cos'
    ]
    for op in operators:
        features.append(counts.get(op, 0) / 5.0)
    
    # === 特征组3: 变量使用统计 (10维) ===
    variables = [
        'var_pos_err', 'var_vel_err', 'var_ang_err', 'var_ang_vel',
        'var_time', 'var_state', 'var_control',
        'var_x', 'var_y', 'var_z'
    ]
    for var in variables:
        features.append(counts.get(var, 0) / 5.0)
    
    # === 特征组4: 结构统计 (10维) ===
    # 规则数量
    num_rules = len(program)
    features.append(num_rules / 10.0)
    
    # 平均深度和最大深度
    depths = []
    total_nodes = 0
    for rule in program:
        if 'action' in rule:
            action = rule['action']
            if isinstance(action, list):
                for act in action:
                    d = compute_max_depth(act)
                    depths.append(d)
                    total_nodes += counts.get('BinaryOpNode', 0) + counts.get('UnaryOpNode', 0) + counts.get('TerminalNode', 0)
            else:
                d = compute_max_depth(action)
                depths.append(d)
    
    avg_depth = np.mean(depths) if depths else 0.0
    max_depth = max(depths) if depths else 0.0
    
    features.append(avg_depth / 20.0)
    features.append(max_depth / 20.0)
    
    # 总节点数
    features.append(total_nodes / 50.0)
    
    # Multiplier统计
    multipliers = []
    for rule in program:
        if 'multiplier' in rule:
            mult = rule['multiplier']
            if isinstance(mult, (list, tuple)):
                multipliers.extend(mult)
            else:
                multipliers.append(mult)
    
    if multipliers:
        features.append(np.mean(multipliers))
        features.append(np.std(multipliers))
        features.append(np.max(multipliers))
        features.append(np.min(multipliers))
    else:
        features.extend([1.0, 0.0, 1.0, 1.0])  # 默认值
    
    # 条件复杂度（If节点数量）
    features.append(counts.get('IfNode', 0) / 10.0)
    
    # === Padding到64维 ===
    while len(features) < 64:
        features.append(0.0)
    
    # 截断到64维
    features = features[:64]
    
    # 转换为tensor
    return torch.tensor(features, dtype=torch.float32)


def featurize_program_batch(programs: List[List[Dict[str, Any]]]) -> torch.Tensor:
    """
    批量特征化
    
    Args:
        programs: 程序列表
    
    Returns:
        torch.Tensor: shape [batch_size, 64]
    """
    features_list = [featurize_program(prog) for prog in programs]
    return torch.stack(features_list)


# 测试代码
if __name__ == '__main__':
    # 测试简单程序
    test_program = [
        {
            'name': 'rule1',
            'condition': BinaryOpNode('>', TerminalNode('pos_err'), TerminalNode(0.5)),
            'action': [
                BinaryOpNode('set', TerminalNode('control'), BinaryOpNode('*', TerminalNode('pos_err'), TerminalNode(1.0)))
            ],
            'multiplier': [1.0, 1.0, 1.0]
        }
    ]
    
    features = featurize_program(test_program)
    print(f"Feature shape: {features.shape}")
    print(f"Feature values (first 20): {features[:20]}")
    print(f"Non-zero features: {(features != 0).sum().item()}")

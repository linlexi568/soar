"""将DSL程序AST转换为图表示，用于GNN输入

参考论文：
- Learning to Represent Programs with Graphs (ICLR 2018)
- Learning Symbolic Physics with Graph Networks (ICLR 2020)
"""
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np

# 导入DSL节点类型
try:
    from ..core.dsl import ProgramNode, TerminalNode, ConstantNode, UnaryOpNode, BinaryOpNode, IfNode
except Exception:
    from core.dsl import ProgramNode, TerminalNode, ConstantNode, UnaryOpNode, BinaryOpNode, IfNode

# PyTorch Geometric (延迟导入，避免依赖问题)
try:
    from torch_geometric.data import Data, Batch
    PyG_AVAILABLE = True
except ImportError:
    PyG_AVAILABLE = False
    Data = None
    Batch = None


# ==================== 节点特征编码配置 ====================

# 节点类型 (4维 one-hot)
# 注意：保持总特征维度为 24，不改动下游 GNN 结构。
# ConstantNode 在数值特征里编码，而不是单独的类型维。
NODE_TYPES = ['Terminal', 'Unary', 'Binary', 'If']

# 操作符类型 (16维 one-hot + 1维 unknown)
OPERATORS = [
    '+', '-', '*', '/', '>', '<', '==', '!=',
    'max', 'min', 'abs', 'sin', 'cos', 'tan', 'log1p', 'sqrt'
]

# 常见变量名 (学习embedding，这里列出用于统计)
COMMON_VARIABLES = [
    'pos_err_x', 'pos_err_y', 'pos_err_z', 'pos_err',
    'vel_x', 'vel_y', 'vel_z', 'vel_err',
    'err_i_x', 'err_i_y', 'err_i_z',
    'err_d_x', 'err_d_y', 'err_d_z',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
    'err_p_roll', 'err_p_pitch', 'err_p_yaw',
    'u_fz', 'u_tx', 'u_ty', 'u_tz'
]

# 边类型 (7种)：保持原有 6 类，并新增 'param' 用于算子参数
EDGE_TYPES = ['left', 'right', 'child', 'condition', 'then', 'else', 'param']


def get_node_type_features(node: Any) -> List[float]:
    """节点类型 one-hot (4维)"""
    features = [0.0] * len(NODE_TYPES)
    node_type = type(node).__name__
    if node_type in NODE_TYPES:
        idx = NODE_TYPES.index(node_type)
        features[idx] = 1.0
    return features


def get_operator_features(node: Any) -> List[float]:
    """操作符 one-hot (17维: 16个已知 + 1个unknown)"""
    features = [0.0] * (len(OPERATORS) + 1)
    
    if isinstance(node, (UnaryOpNode, BinaryOpNode)):
        op = getattr(node, 'op', None)
        if op in OPERATORS:
            idx = OPERATORS.index(op)
            features[idx] = 1.0
        else:
            features[-1] = 1.0  # unknown operator
    
    return features


def get_variable_id(var_name: str) -> int:
    """变量名映射到ID (用于embedding)"""
    if var_name in COMMON_VARIABLES:
        return COMMON_VARIABLES.index(var_name)
    else:
        # 未知变量映射到一个特殊ID
        return len(COMMON_VARIABLES)


def get_numerical_features(node: Any) -> List[float]:
    """数值特征 (3维)

    维度保持与原版一致，以兼容已有 GNN：
    - [0]: 是否为常数节点 (TerminalNode/ConstantNode)
    - [1]: 常数值的 tanh 归一化
    - [2]: 变量 ID 归一化（TerminalNode with str）
    """
    features = [0.0] * 3

    # 常数节点（包括旧的 TerminalNode(float) 和新的 ConstantNode）
    if isinstance(node, TerminalNode) and isinstance(node.value, (int, float)):
        features[0] = 1.0
        features[1] = float(np.tanh(float(node.value)))
    elif isinstance(node, ConstantNode):
        features[0] = 1.0
        features[1] = float(np.tanh(float(node.value)))

    # 变量节点
    if isinstance(node, TerminalNode) and isinstance(node.value, str):
        var_id = get_variable_id(node.value)
        features[2] = var_id / (len(COMMON_VARIABLES) + 1)

    return features


def encode_node_features(node: Any) -> List[float]:
    """
    编码单个节点的特征向量
    
    特征组成 (24维，与原版保持一致):
    - [0-3]: 节点类型 one-hot (4: Terminal/Unary/Binary/If)
    - [4-20]: 操作符 one-hot (17)
    - [21-23]: 数值特征 (3维)
    
    Returns:
        28维特征向量
    """
    features = []
    
    # 节点类型 (4维)
    features.extend(get_node_type_features(node))
    
    # 操作符 (17维)
    features.extend(get_operator_features(node))
    
    # 数值特征 (3维)
    num_feats = get_numerical_features(node)
    features.extend(num_feats)

    assert len(features) == 24, f"Feature dimension mismatch: {len(features)}"
    return features


def ast_to_graph_recursive(
    node: Any,
    nodes: List[Any],
    node_features: List[List[float]],
    edges: List[Tuple[int, int, str]],
    parent_idx: Optional[int] = None,
    edge_type: Optional[str] = None
) -> int:
    """
    递归遍历AST，构建图结构
    
    Args:
        node: 当前AST节点
        nodes: 节点列表（累积）
        node_features: 节点特征列表（累积）
        edges: 边列表 (src, dst, type)
        parent_idx: 父节点索引
        edge_type: 当前边的类型
    
    Returns:
        当前节点在列表中的索引
    """
    if node is None:
        return -1
    
    # 添加当前节点
    current_idx = len(nodes)
    nodes.append(node)
    node_features.append(encode_node_features(node))
    
    # 添加从父节点到当前节点的边
    if parent_idx is not None and edge_type is not None:
        edges.append((parent_idx, current_idx, edge_type))
    
    # 递归处理子节点
    if isinstance(node, BinaryOpNode):
        if hasattr(node, 'left') and node.left is not None:
            ast_to_graph_recursive(node.left, nodes, node_features, edges, current_idx, 'left')
        if hasattr(node, 'right') and node.right is not None:
            ast_to_graph_recursive(node.right, nodes, node_features, edges, current_idx, 'right')
    
    elif isinstance(node, UnaryOpNode):
        # 处理子节点
        if hasattr(node, 'child') and node.child is not None:
            ast_to_graph_recursive(node.child, nodes, node_features, edges, current_idx, 'child')
        
        # 处理参数字典中的 ConstantNode (新增)
        if hasattr(node, 'params') and node.params:
            for param_name, param_node in node.params.items():
                if isinstance(param_node, ConstantNode):
                    # 为参数创建边，标记为特殊边类型 'param'
                    ast_to_graph_recursive(param_node, nodes, node_features, edges, current_idx, 'param')
    
    elif isinstance(node, IfNode):
        if hasattr(node, 'condition') and node.condition is not None:
            ast_to_graph_recursive(node.condition, nodes, node_features, edges, current_idx, 'condition')
        if hasattr(node, 'then_branch') and node.then_branch is not None:
            ast_to_graph_recursive(node.then_branch, nodes, node_features, edges, current_idx, 'then')
        if hasattr(node, 'else_branch') and node.else_branch is not None:
            ast_to_graph_recursive(node.else_branch, nodes, node_features, edges, current_idx, 'else')
    
    return current_idx


def program_to_graph(program: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[Tuple[int, int]], List[int]]:
    """
    将程序（多条规则）转换为单个图
    
    策略：为每条规则创建一个虚拟根节点，所有规则的根节点连接到一个全局根
    
    Args:
        program: 规则列表
    
    Returns:
        (node_features, edges, edge_types)
        - node_features: [num_nodes, 24]
        - edges: [(src, dst), ...] 
        - edge_types: [edge_type_id, ...]
    """
    nodes = []
    node_features = []
    edges_with_type = []
    
    # 创建全局根节点（虚拟节点，表示整个程序）
    global_root_idx = len(nodes)
    nodes.append('PROGRAM_ROOT')
    # 全局根的特征：全零 + 标记位
    root_features = [0.0] * 24
    root_features[0] = 1.0  # 标记为特殊节点
    node_features.append(root_features)
    
    # 处理每条规则
    for rule_idx, rule in enumerate(program):
        # 创建规则根节点
        rule_root_idx = len(nodes)
        nodes.append(f'RULE_{rule_idx}')
        rule_root_features = [0.0] * 24
        rule_root_features[1] = 1.0  # 标记为规则节点
        node_features.append(rule_root_features)
        
        # 连接到全局根
        edges_with_type.append((global_root_idx, rule_root_idx, 'child'))
        
        # 处理条件
        if 'condition' in rule and rule['condition'] is not None:
            ast_to_graph_recursive(
                rule['condition'], 
                nodes, 
                node_features, 
                edges_with_type,
                rule_root_idx,
                'condition'
            )
        
        # 处理动作
        if 'action' in rule:
            actions = rule['action']
            if not isinstance(actions, list):
                actions = [actions]
            
            for action in actions:
                if action is not None:
                    ast_to_graph_recursive(
                        action,
                        nodes,
                        node_features,
                        edges_with_type,
                        rule_root_idx,
                        'then'  # 动作视为"then分支"
                    )
    
    # 转换边格式: (src, dst, type_str) -> (src, dst), [type_id]
    edges = [(src, dst) for src, dst, _ in edges_with_type]
    edge_types = [EDGE_TYPES.index(etype) if etype in EDGE_TYPES else 0 
                  for _, _, etype in edges_with_type]
    
    return node_features, edges, edge_types


def ast_to_pyg_graph(program: List[Dict[str, Any]]) -> Any:
    """
    将程序转换为PyTorch Geometric的Data对象
    
    Args:
        program: 规则列表
    
    Returns:
        torch_geometric.data.Data 对象，包含：
        - x: [num_nodes, 24] 节点特征
        - edge_index: [2, num_edges] 边索引
        - edge_attr: [num_edges, 1] 边类型
    """
    if not PyG_AVAILABLE:
        raise ImportError(
            "PyTorch Geometric not installed. "
            "Install with: pip install torch-geometric torch-scatter torch-sparse"
        )
    
    # 转换为图
    node_features, edges, edge_types = program_to_graph(program)
    
    # 转为tensor
    x = torch.tensor(node_features, dtype=torch.float32)
    
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_types, dtype=torch.long).unsqueeze(1)
    else:
        # 空图（只有根节点）
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.long)
    
    # 创建Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


def batch_programs_to_graphs(programs: List[List[Dict[str, Any]]]) -> Any:
    """
    批量转换程序为图，并打包成batch
    
    Args:
        programs: 程序列表
    
    Returns:
        torch_geometric.data.Batch 对象
    """
    if not PyG_AVAILABLE:
        raise ImportError("PyTorch Geometric not installed.")
    
    graph_list = [ast_to_pyg_graph(prog) for prog in programs]
    batch = Batch.from_data_list(graph_list)
    
    return batch


# ==================== 测试代码 ====================

if __name__ == '__main__':
    import sys
    import pathlib
    
    # 添加路径以导入dsl
    _parent = pathlib.Path(__file__).resolve().parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    
    from core.dsl import TerminalNode, BinaryOpNode
    
    print("=" * 60)
    print("测试 AST → 图转换")
    print("=" * 60)
    
    # 测试程序：u_fz = 2.0 * pos_err_z - 0.5 * vel_z
    test_program = [
        {
            'name': 'rule1',
            'condition': BinaryOpNode('>', TerminalNode('pos_err_z'), TerminalNode(0.1)),
            'action': [
                BinaryOpNode(
                    'set',
                    TerminalNode('u_fz'),
                    BinaryOpNode(
                        '-',
                        BinaryOpNode('*', TerminalNode(2.0), TerminalNode('pos_err_z')),
                        BinaryOpNode('*', TerminalNode(0.5), TerminalNode('vel_z'))
                    )
                )
            ]
        }
    ]
    
    print("\n1. 测试节点特征编码...")
    test_node = BinaryOpNode('+', TerminalNode('pos_err_z'), TerminalNode(1.0))
    features = encode_node_features(test_node)
    print(f"  Binary节点特征维度: {len(features)}")
    print(f"  前10维: {features[:10]}")
    
    print("\n2. 测试程序→图转换...")
    node_features, edges, edge_types = program_to_graph(test_program)
    print(f"  节点数: {len(node_features)}")
    print(f"  边数: {len(edges)}")
    print(f"  特征维度: {len(node_features[0]) if node_features else 0}")
    
    if PyG_AVAILABLE:
        print("\n3. 测试PyG Data对象...")
        data = ast_to_pyg_graph(test_program)
        print(f"  节点特征shape: {data.x.shape}")
        print(f"  边索引shape: {data.edge_index.shape}")
        print(f"  边特征shape: {data.edge_attr.shape}")
        print(f"  ✅ PyG Data创建成功")
        
        print("\n4. 测试批处理...")
        batch_programs = [test_program, test_program]
        batch = batch_programs_to_graphs(batch_programs)
        print(f"  Batch节点数: {batch.x.shape[0]}")
        print(f"  Batch边数: {batch.edge_index.shape[1]}")
        print(f"  Batch size: {batch.num_graphs}")
        print(f"  ✅ 批处理成功")
    else:
        print("\n⚠️  PyTorch Geometric未安装，跳过PyG测试")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过")
    print("=" * 60)

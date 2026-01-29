"""基于图神经网络(GNN)的策略-价值网络

使用Graph Attention Network (GAT) 处理可变大小的程序AST
参考论文：
- Graph Attention Networks (ICLR 2018)
- Learning to Represent Programs with Graphs (ICLR 2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# PyTorch Geometric
try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
    try:
        # Try new API first (PyG >= 2.3.0)
        from torch_geometric.nn.aggr import AttentionalAggregation as GlobalAttention
    except ImportError:
        # Fall back to old API
        from torch_geometric.nn import GlobalAttention
    from torch_geometric.data import Data, Batch
    PyG_AVAILABLE = True
except ImportError:
    PyG_AVAILABLE = False
    GATv2Conv = None
    global_mean_pool = None
    global_add_pool = None
    GlobalAttention = None
    Data = None
    Batch = None


class ExpressionGNN(nn.Module):
    """
    表达式AST编码器：3层GAT网络
    
    输入：程序AST图
    输出：64维程序embedding
    """
    def __init__(
        self,
        node_feature_dim: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention_pooling: bool = True
    ):
        super().__init__()
        
        if not PyG_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric not installed. "
                "Install with: pip install torch-geometric torch-scatter torch-sparse"
            )
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_attention_pooling = use_attention_pooling
        
        # 输入投影：24 → 64
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # GAT层：使用GATv2Conv（更稳定）
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim // num_heads  # 每个head输出
            
            gat_layer = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_heads,
                dropout=dropout,
                concat=True,  # 拼接所有head的输出
                add_self_loops=True,
                edge_dim=None  # 暂不使用边特征
            )
            self.gat_layers.append(gat_layer)
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 全局池化：attention pooling或mean pooling
        if use_attention_pooling:
            # 学习每个节点的重要性权重
            gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.global_pool = GlobalAttention(gate_nn)
        else:
            self.global_pool = None  # 使用mean pooling
        
        # 输出投影：保持64维
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyG Data对象，包含x, edge_index, batch
        
        Returns:
            [batch_size, 64] 程序embedding
        """
        x = data.x  # [num_nodes, 24]
        edge_index = data.edge_index  # [2, num_edges]
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 输入投影
        x = self.input_projection(x)  # [num_nodes, 64]
        x = F.relu(x)
        
        # GAT层 + 残差连接 + LayerNorm
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_residual = x
            x = gat(x, edge_index)  # [num_nodes, 64]
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_residual  # 残差连接
            x = norm(x)  # LayerNorm
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        # 全局池化：图级表示
        if self.global_pool is not None:
            # Attention pooling
            graph_embedding = self.global_pool(x, batch)  # [batch_size, 64]
        else:
            # Mean pooling
            graph_embedding = global_mean_pool(x, batch)  # [batch_size, 64]
        
        # 输出投影
        graph_embedding = self.output_projection(graph_embedding)
        graph_embedding = F.relu(graph_embedding)
        
        return graph_embedding


class GNNPolicyValueNet(nn.Module):
    """
    GNN策略-价值网络
    
    结构：
    - 共享GNN编码器（3层GAT）
    - 策略头：64 → 32 → 14 (mutation logits)
    - 价值头：64 → 32 → 1 (performance estimate)
    """
    def __init__(
        self,
        node_feature_dim: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        policy_output_dim: int = 14,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if not PyG_AVAILABLE:
            raise ImportError("PyTorch Geometric not installed.")
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.policy_output_dim = policy_output_dim
        
        # 共享GNN编码器
        self.gnn_encoder = ExpressionGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_attention_pooling=True
        )
        
        # 策略头：预测mutation类型的概率
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, policy_output_dim)
        )
        
        # 价值头：预测程序性能
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: PyG Data或Batch对象
        
        Returns:
            (policy_logits, value)
            - policy_logits: [batch_size, 14] 未归一化的logits
            - value: [batch_size, 1] 价值估计
        """
        # GNN编码
        graph_embedding = self.gnn_encoder(data)  # [batch_size, 64]
        
        # 策略头
        policy_logits = self.policy_head(graph_embedding)  # [batch_size, 14]
        
        # 价值头
        value = self.value_head(graph_embedding)  # [batch_size, 1]
        
        return policy_logits, value
    
    def predict_policy(self, data: Data) -> torch.Tensor:
        """仅预测策略（推理时使用）"""
        with torch.no_grad():
            policy_logits, _ = self.forward(data)
            return F.softmax(policy_logits, dim=-1)
    
    def predict_value(self, data: Data) -> torch.Tensor:
        """仅预测价值（推理时使用）"""
        with torch.no_grad():
            _, value = self.forward(data)
            return value.squeeze(-1)
    
    def get_num_params(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gnn_policy_value_net(
    node_feature_dim: int = 24,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_heads: int = 4,
    policy_output_dim: int = 14,
    dropout: float = 0.1,
    device: str = 'cuda'
) -> GNNPolicyValueNet:
    """
    工厂函数：创建GNN策略-价值网络
    
    Args:
        node_feature_dim: 节点特征维度 (默认24)
        hidden_dim: 隐藏层维度 (默认64)
        num_layers: GAT层数 (默认3)
        num_heads: 注意力头数 (默认4)
        policy_output_dim: 策略输出维度 (mutation类型数，默认14)
        dropout: Dropout率
        device: 设备
    
    Returns:
        GNNPolicyValueNet实例
    """
    model = GNNPolicyValueNet(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        policy_output_dim=policy_output_dim,
        dropout=dropout
    )
    
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"GNN策略-价值网络创建成功:")
    print(f"  参数量: {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  GAT层数: {num_layers}")
    print(f"  注意力头数: {num_heads}")
    print(f"  设备: {device}")
    
    return model


# ==================== 测试代码 ====================

if __name__ == '__main__':
    import sys
    import pathlib
    
    # 添加路径
    _parent = pathlib.Path(__file__).resolve().parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    
    if not PyG_AVAILABLE:
        print("⚠️  PyTorch Geometric未安装，跳过测试")
        sys.exit(0)
    
    from gnn_features import ast_to_pyg_graph, batch_programs_to_graphs
    from dsl import TerminalNode, BinaryOpNode
    
    print("=" * 60)
    print("测试 GNN 策略-价值网络")
    print("=" * 60)
    
    # 创建测试程序
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
    
    print("\n1. 测试单个程序前向传播...")
    data = ast_to_pyg_graph(test_program)
    print(f"  图节点数: {data.x.shape[0]}")
    print(f"  图边数: {data.edge_index.shape[1]}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_gnn_policy_value_net(device=device)
    
    data = data.to(device)
    policy_logits, value = model(data)
    print(f"  策略logits shape: {policy_logits.shape}")
    print(f"  价值 shape: {value.shape}")
    print(f"  价值范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
    print(f"  ✅ 单程序前向传播成功")
    
    print("\n2. 测试批处理前向传播...")
    batch_programs = [test_program, test_program, test_program]
    batch = batch_programs_to_graphs(batch_programs).to(device)
    print(f"  Batch大小: {batch.num_graphs}")
    print(f"  Batch节点数: {batch.x.shape[0]}")
    
    policy_logits, value = model(batch)
    print(f"  策略logits shape: {policy_logits.shape}")
    print(f"  价值 shape: {value.shape}")
    print(f"  ✅ 批处理前向传播成功")
    
    print("\n3. 测试策略/价值单独预测...")
    policy_probs = model.predict_policy(data)
    value_pred = model.predict_value(data)
    print(f"  策略概率 shape: {policy_probs.shape}")
    print(f"  策略概率和: {policy_probs.sum().item():.4f}")
    print(f"  价值预测 shape: {value_pred.shape}")
    print(f"  ✅ 单独预测成功")
    
    print("\n4. 测试梯度反向传播...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    policy_logits, value = model(batch)
    
    # 模拟损失
    policy_target = torch.randint(0, 14, (3,), device=device)
    value_target = torch.randn(3, 1, device=device)
    
    policy_loss = F.cross_entropy(policy_logits, policy_target)
    value_loss = F.mse_loss(value, value_target)
    total_loss = policy_loss + value_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"  策略损失: {policy_loss.item():.4f}")
    print(f"  价值损失: {value_loss.item():.4f}")
    print(f"  总损失: {total_loss.item():.4f}")
    print(f"  ✅ 梯度反向传播成功")
    
    print("\n5. 参数量对比...")
    gnn_params = model.get_num_params()
    print(f"  GNN网络: {gnn_params:,} ({gnn_params / 1e6:.2f}M)")
    
    # 对比固定特征网络 (64→256→14+1)
    baseline_params = 64 * 256 + 256 + 256 * 14 + 14 + 256 * 1 + 1
    print(f"  固定特征网络(估算): {baseline_params:,} ({baseline_params / 1e6:.2f}M)")
    print(f"  参数减少: {(1 - gnn_params / baseline_params) * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过")
    print("=" * 60)

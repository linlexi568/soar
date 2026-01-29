"""gnn_policy_nn_v2.py

分层双网络 (Hierarchical Dual Network) 版本:
- 结构编码器 (StructureEncoder): 深层 GATv2Conv 处理程序图的结构与语义关系。
- 特征编码器 (FeatureEncoder): 将原始节点特征序列化后经 TransformerEncoder 捕获跨节点的顺序/组合模式。
- 融合层 (CrossFusion): 使用多头交叉注意力 (structure embedding 作为 query, feature tokens 作为 key/value) 获得融合表示。
- Policy / Value Heads: 更深的两层 MLP 输出策略 logits 与价值标量。

设计目标:
1. 参数量 ~2.5M (在 8GB GPU 可接受范围内,为后续零先验扩展做容量预留)。
2. 与现有 create_gnn_policy_value_net 接口兼容,方便 A/B 替换。
3. 支持可变大小图 (batch 内不同节点数) 以及 padding mask。
4. 尽量避免引入额外复杂依赖 (仅使用 PyTorch / PyTorch Geometric)。

使用方式:
from 01_soar.gnn_policy_nn_v2 import create_gnn_policy_value_net_v2
model = create_gnn_policy_value_net_v2(node_feature_dim=24, policy_output_dim=14)
policy_logits, value = model(graph_batch)

测试:
python 01_soar/gnn_policy_nn_v2.py  # 将运行自检 (随机图) 并打印参数统计与一次前向+反向时间。
"""
from __future__ import annotations
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import add_self_loops, remove_self_loops, coalesce
except ImportError:
    raise ImportError("需要安装 torch-geometric 才能使用该模块。")

# ----------------------------- 工具函数 ----------------------------- #

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# padding 序列的简易函数

def pad_sequences(seqs: list[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """将长度不等的 [L_i, D] 张量列表 pad 成 [B, L_max, D] 并返回 mask:[B, L_max] (True=有效)。"""
    if not seqs:
        return torch.empty(0), torch.empty(0)
    max_len = max(s.size(0) for s in seqs)
    dim = seqs[0].size(1)
    batch = len(seqs)
    out = seqs[0].new_full((batch, max_len, dim), pad_value)
    mask = torch.zeros(batch, max_len, dtype=torch.bool, device=seqs[0].device)
    for i, s in enumerate(seqs):
        l = s.size(0)
        out[i, :l] = s
        mask[i, :l] = True
    return out, mask

# ----------------------------- 结构编码器 ----------------------------- #

class StructureEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 5, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        # 🔧 恢复 GATv2Conv，但添加健壮的边索引处理
        out_per_head = hidden_dim // heads
        for i in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=out_per_head,
                heads=heads,
                dropout=dropout,
                edge_dim=None,
                add_self_loops=False,  # 手动添加自环以确保正确性
                share_weights=False,
            )
            self.layers.append(conv)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        # GATv2Conv 在 forward 内部通过 self._alpha 暂存注意力，不是线程安全的；
        # 异步训练+MCTS 并发访问时需要锁以避免 alpha=None 断言。
        self._conv_lock = threading.Lock()

    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 输入节点特征: x [N_total, in_dim]
        x = self.proj_in(data.x)  # [N, hidden]
        edge_index = data.edge_index
        num_nodes = x.size(0)
        
        # 🔧 关键修复：确保每个节点都有自环
        # GATv2Conv要求每个节点至少有一条边（否则alpha=None）
        # 策略：使用PyG的add_self_loops，它会正确处理批次索引
        from torch_geometric.utils import add_self_loops, remove_self_loops, coalesce
        
        # 先移除可能的自环，避免重复
        edge_index, _ = remove_self_loops(edge_index)
        
        # 使用PyG的add_self_loops，它会自动为0到num_nodes-1的所有节点添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # 确保边索引去重并排序（coalesce会合并重复边）
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        
        with self._conv_lock:
            for conv, ln in zip(self.layers, self.norms):
                # GATv2Conv 现在保证每个节点都有边，不会导致 alpha=None
                h = conv(x, edge_index)
                # 残差 + 层归一化
                x = ln(x + self.dropout(F.elu(h)))
        # 图级池化
        graph_emb = global_mean_pool(x, data.batch)  # [B, hidden]
        return graph_emb, x, data.batch

# ----------------------------- 特征编码器 ----------------------------- #

class FeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 3, nhead: int = 8, dropout: float = 0.1, ff_multiplier: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        # 可学习的 CLS token 初始向量
        self.cls_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, node_features: torch.Tensor, batch_index: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        # node_features: [N_total, hidden_dim] (来自结构编码器的节点嵌入, 或原始 x 再投影)
        # batch_index: [N_total]
        # 将节点按图分组
        device = node_features.device
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        sequences = []
        for g in range(num_graphs):
            mask = batch_index == g
            seq = node_features[mask]  # [L_g, hidden]
            # 插入 CLS token
            if seq.numel() == 0:
                seq = self.cls_token.unsqueeze(0).expand(1, -1)
            else:
                cls = self.cls_token.unsqueeze(0).expand(1, -1)
                seq = torch.cat([cls, seq], dim=0)  # [L_g+1, hidden]
            sequences.append(seq)
        if not sequences:
            return torch.empty(0, device=device), []
        # pad -> [B, L_max, hidden]
        padded, mask = pad_sequences(sequences, pad_value=0.0)
        # 投影 (若上游不是 hidden_dim 可在此投影; 这里输入已经是 hidden_dim, 但为了灵活性仍调用 self.proj 若形状不同)
        if padded.size(-1) != self.proj.out_features:
            padded = self.proj(padded)
        padded = self.dropout(padded)
        # Transformer 编码 (使用 src_key_padding_mask: False=保留, True=忽略)
        key_padding_mask = ~mask  # [B, L_max]
        encoded = self.encoder(padded, src_key_padding_mask=key_padding_mask)  # [B, L_max, hidden]
        # 取 CLS 位置作为图级特征
        cls_emb = encoded[:, 0]  # [B, hidden]
        return cls_emb, sequences

# ----------------------------- 融合与最终网络 ----------------------------- #

class GNNPolicyValueNetV2(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        policy_output_dim: int,
        structure_hidden: int = 256,
        structure_layers: int = 5,
        structure_heads: int = 8,
        feature_layers: int = 3,
        feature_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 结构编码器 + 特征编码器
        self.structure_encoder = StructureEncoder(
            in_dim=node_feature_dim,
            hidden_dim=structure_hidden,
            num_layers=structure_layers,
            heads=structure_heads,
            dropout=dropout,
        )
        self.feature_encoder = FeatureEncoder(
            in_dim=structure_hidden,
            hidden_dim=structure_hidden,
            num_layers=feature_layers,
            nhead=feature_heads,
            dropout=dropout,
        )
        # 交叉注意力 (查询=结构, 键值=特征序列) 这里简化为再一次融合层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=structure_hidden,
            num_heads=structure_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(structure_hidden)
        # 融合线性
        self.fuse = nn.Sequential(
            nn.Linear(structure_hidden * 2, structure_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(structure_hidden),
        )
        # Policy head (deeper)
        self.policy_head = nn.Sequential(
            nn.Linear(structure_hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, policy_output_dim),
        )
        # Value head: 多任务输出（总价值标量 + 8个组件向量）
        # 共享主干
        self.value_backbone = nn.Sequential(
            nn.Linear(structure_hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 总价值头（标量，用于MCTS）
        self.value_scalar_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        # 组件价值头（8维向量：position/settling/control/smoothness/gain/saturation/peak/high_freq）
        self.value_components_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 8),  # 8个组件
            nn.Tanh(),  # 各组件归一化到[-1,1]
        )

    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播，返回多任务输出
        
        Returns:
            policy_logits: [B, action_dim] 策略logits
            value_scalar: [B] 总价值标量（用于MCTS）
            value_components: [B, 8] 组件价值向量（用于训练梯度密集化）
        """
        # 1. 结构编码 (得到图嵌入 + 节点嵌入)
        graph_emb, node_emb, batch_index = self.structure_encoder(data)  # [B, H], [N, H]
        # 2. 特征编码 (使用节点嵌入作为序列输入, 产出 CLS token 图特征)
        feature_graph_emb, sequences = self.feature_encoder(node_emb, batch_index)  # [B, H]
        # 3. 交叉注意力: query=结构图嵌入, key/value=特征 CLS 扩展后 or 整个序列 (简化: 使用特征 CLS Emb 单 token)
        # 为了让注意力有意义, 我们构建 key/value = feature_graph_emb 作为单长度序列
        B, H = graph_emb.size()
        q = graph_emb.unsqueeze(1)  # [B, 1, H]
        kv = feature_graph_emb.unsqueeze(1)  # [B, 1, H]
        attn_out, _ = self.cross_attn(q, kv, kv)  # [B, 1, H]
        attn_out = attn_out.squeeze(1)
        attn_out = self.cross_norm(attn_out + graph_emb)  # 残差 + LN
        # 4. 融合
        fused = self.fuse(torch.cat([attn_out, feature_graph_emb], dim=-1))  # [B, H]
        # 缓存融合嵌入，供 get_embedding() 使用（用于ranking网络）
        self._cached_embedding = fused
        # 5. heads 输出
        policy_logits = self.policy_head(fused)
        # 多任务价值输出
        value_features = self.value_backbone(fused)  # [B, 256]
        value_scalar = self.value_scalar_head(value_features).squeeze(-1)  # [B]
        value_components = self.value_components_head(value_features)  # [B, 8]
        return policy_logits, value_scalar, value_components

    def get_embedding(self, data: Batch) -> torch.Tensor:
        """
        提取程序的嵌入表示（fusion后的128维向量），供ranking网络使用
        
        Args:
            data: PyG Batch对象
            
        Returns:
            embedding: [B, hidden] 融合后的程序嵌入
        """
        # 1. 结构编码
        graph_emb, node_emb, batch_index = self.structure_encoder(data)  # [B, H], [N, H]
        # 2. 特征编码
        feature_graph_emb, sequences = self.feature_encoder(node_emb, batch_index)  # [B, H]
        # 3. 交叉注意力
        B, H = graph_emb.size()
        q = graph_emb.unsqueeze(1)  # [B, 1, H]
        kv = feature_graph_emb.unsqueeze(1)  # [B, 1, H]
        attn_out, _ = self.cross_attn(q, kv, kv)  # [B, 1, H]
        attn_out = attn_out.squeeze(1)
        attn_out = self.cross_norm(attn_out + graph_emb)  # 残差 + LN
        # 4. 融合（这是我们要提取的嵌入）
        fused = self.fuse(torch.cat([attn_out, feature_graph_emb], dim=-1))  # [B, H]
        return fused

# ----------------------------- 工厂函数 ----------------------------- #

def create_gnn_policy_value_net_v2(
    node_feature_dim: int,
    policy_output_dim: int,
    structure_hidden: int = 256,
    structure_layers: int = 5,
    structure_heads: int = 8,
    feature_layers: int = 3,
    feature_heads: int = 8,
    dropout: float = 0.1,
) -> GNNPolicyValueNetV2:
    model = GNNPolicyValueNetV2(
        node_feature_dim=node_feature_dim,
        policy_output_dim=policy_output_dim,
        structure_hidden=structure_hidden,
        structure_layers=structure_layers,
        structure_heads=structure_heads,
        feature_layers=feature_layers,
        feature_heads=feature_heads,
        dropout=dropout,
    )
    return model

# ----------------------------- 自检与快速测试 ----------------------------- #

def _synthetic_graph_batch(batch_size: int = 8, avg_nodes: int = 18, node_feature_dim: int = 24, edge_prob: float = 0.15) -> Batch:
    import random
    datas = []
    for _ in range(batch_size):
        n = max(4, int(random.gauss(avg_nodes, 3)))
        x = torch.randn(n, node_feature_dim)
        # 随机无向图 -> 转为双向边
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and random.random() < edge_prob:
                    edges.append((i, j))
        if not edges:
            # 至少连一条边
            if n > 1:
                edges.append((0, 1))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        datas.append(data)
    return Batch.from_data_list(datas)


def _run_quick_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _synthetic_graph_batch(batch_size=12).to(device)
    model = create_gnn_policy_value_net_v2(node_feature_dim=24, policy_output_dim=14).to(device)
    print("模型参数总数:", count_parameters(model))
    t0 = time.time()
    policy_logits, value_scalar, value_components = model(batch)
    t1 = time.time()
    print("前向输出 shapes:", policy_logits.shape, value_scalar.shape, value_components.shape)
    loss = policy_logits.mean() + value_scalar.mean() + value_components.mean()
    loss.backward()
    t2 = time.time()
    total_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.abs().mean().item()
    print(f"前向耗时: {t1 - t0:.4f}s, 反向耗时: {t2 - t1:.4f}s, 平均梯度: {total_grad:.4f}")
    # 内存占用估计 (仅 GPU)
    if device.type == "cuda":
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated(device) / (1024**2)
        print(f"GPU 当前显存占用 ~{mem:.2f} MB")

if __name__ == "__main__":
    _run_quick_test()

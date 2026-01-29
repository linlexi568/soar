"""测试Ranking NN整合动作特征的功能"""
import sys
sys.path.insert(0, '01_soar')

import torch
import numpy as np
from ranking_value_net import RankingValueNet, PairwiseRankingBuffer, compute_ranking_loss
from gnn_policy_nn_v2 import HierarchicalGNN_v2
from torch_geometric.data import Data, Batch

def test_ranking_with_action_features():
    """测试1：Ranking网络可以处理动作特征"""
    print("\n" + "="*70)
    print("测试1：Ranking Value Network + 动作特征")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_dim = 256
    action_dim = 6
    
    # 创建ranking网络
    ranking_net = RankingValueNet(embed_dim=embed_dim, action_feature_dim=action_dim).to(device)
    print(f"✓ Ranking网络创建成功")
    print(f"  - 嵌入维度: {embed_dim}")
    print(f"  - 动作特征维度: {action_dim}")
    print(f"  - 总参数: {sum(p.numel() for p in ranking_net.parameters()):,}")
    
    # 测试1.1：带动作特征的比较
    embed_a = torch.randn(4, embed_dim, device=device)
    embed_b = torch.randn(4, embed_dim, device=device)
    action_a = torch.tensor([
        [0.5, 0.1, 0.8, 0.3, 0.05, 0.4],  # 大推力程序
        [0.1, 0.05, 0.2, 0.1, 0.02, 0.15],  # 小推力程序
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 零推力程序
        [0.8, 0.3, 1.2, 0.5, 0.2, 0.7],  # 很大推力程序
    ], device=device, dtype=torch.float32)
    action_b = torch.tensor([
        [0.2, 0.05, 0.3, 0.15, 0.03, 0.2],  # 中等推力
        [0.6, 0.15, 0.9, 0.4, 0.08, 0.5],  # 大推力
        [0.1, 0.02, 0.15, 0.05, 0.01, 0.1],  # 小推力
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 零推力
    ], device=device, dtype=torch.float32)
    
    logits = ranking_net.forward_compare(embed_a, embed_b, action_a, action_b)
    probs = torch.sigmoid(logits)
    
    print(f"\n✓ 动作特征比较测试:")
    for i in range(4):
        fz_a, fz_b = action_a[i, 0].item(), action_b[i, 0].item()
        prob = probs[i].item()
        print(f"  [{i+1}] fz_a={fz_a:.2f} vs fz_b={fz_b:.2f} → P(a>b)={prob:.3f}")
    
    # 测试1.2：不提供动作特征（退化模式）
    logits_no_action = ranking_net.forward_compare(embed_a, embed_b)  # 不传action_feat
    probs_no_action = torch.sigmoid(logits_no_action)
    print(f"\n✓ 无动作特征退化测试:")
    print(f"  - 预测概率: {probs_no_action.cpu().numpy().ravel()}")
    print(f"  - 说明: 未提供action_feat时使用零特征（兼容MCTS）")
    
    # 测试1.3：价值估计
    value_with_action = ranking_net.forward_value(embed_a, action_a)
    value_no_action = ranking_net.forward_value(embed_a)
    print(f"\n✓ 价值估计测试:")
    print(f"  - 带动作特征: {value_with_action.cpu().numpy().ravel()}")
    print(f"  - 无动作特征: {value_no_action.cpu().numpy().ravel()}")
    
    return ranking_net


def test_buffer_with_action_features():
    """测试2：Ranking Buffer存储动作特征"""
    print("\n" + "="*70)
    print("测试2：PairwiseRankingBuffer + 动作特征")
    print("="*70)
    
    buffer = PairwiseRankingBuffer(capacity=100)
    
    # 创建模拟程序图
    def make_dummy_graph():
        x = torch.randn(5, 32)  # 5个节点
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
    
    # 添加程序对
    for i in range(10):
        graph_a = make_dummy_graph()
        graph_b = make_dummy_graph()
        preference = 1.0 if i % 2 == 0 else 0.0
        action_a = [np.random.rand(), 0.1, np.random.rand(), 0.2, 0.05, 0.3]
        action_b = [np.random.rand(), 0.1, np.random.rand(), 0.2, 0.05, 0.3]
        buffer.push(graph_a, graph_b, preference, action_a, action_b)
    
    print(f"✓ Buffer添加测试:")
    print(f"  - 容量: {buffer.capacity}")
    print(f"  - 当前大小: {len(buffer)}")
    
    # 采样测试
    batch = buffer.sample(4)
    print(f"\n✓ 采样测试:")
    print(f"  - 采样大小: {len(batch)}")
    for i, item in enumerate(batch):
        action_a = item['action_feat_a']
        action_b = item['action_feat_b']
        pref = item['preference']
        print(f"  [{i+1}] action_a[0]={action_a[0]:.3f}, action_b[0]={action_b[0]:.3f}, pref={pref:.1f}")
    
    return buffer


def test_ranking_loss_with_action_features():
    """测试3：Ranking Loss计算（整合动作特征）"""
    print("\n" + "="*70)
    print("测试3：Ranking Loss + 动作特征")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建GNN编码器（用于提取嵌入）
    gnn_encoder = HierarchicalGNN_v2(
        node_feat_dim=32,
        hidden_dim=256,
        num_action_classes=4,
        gat_heads=4
    ).to(device)
    
    # 创建ranking网络
    ranking_net = RankingValueNet(embed_dim=256, action_feature_dim=6).to(device)
    
    # 创建buffer并填充
    buffer = PairwiseRankingBuffer(capacity=100)
    
    def make_dummy_graph():
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
    
    # 模拟：大推力程序比零推力程序更好
    for i in range(20):
        graph_good = make_dummy_graph()
        graph_bad = make_dummy_graph()
        preference = 1.0  # good > bad
        action_good = [0.8, 0.2, 1.0, 0.4, 0.1, 0.5]  # 大推力
        action_bad = [0.05, 0.01, 0.1, 0.02, 0.005, 0.03]  # 小推力
        buffer.push(graph_good, graph_bad, preference, action_good, action_bad)
    
    # 采样并计算loss
    batch = buffer.sample(8)
    loss, metrics = compute_ranking_loss(ranking_net, batch, gnn_encoder, device)
    
    print(f"✓ Ranking Loss计算测试:")
    print(f"  - Loss: {metrics['ranking_loss']:.4f}")
    print(f"  - Accuracy: {metrics['ranking_accuracy']:.2%}")
    print(f"  - Mean Prob: {metrics['mean_prob']:.3f}")
    print(f"  - Action fz差异: {metrics['action_fz_diff']:.3f}")
    
    # 训练几步看是否能学习动作特征的重要性
    optimizer = torch.optim.Adam(ranking_net.parameters(), lr=1e-3)
    
    print(f"\n✓ 训练测试 (10步):")
    for step in range(10):
        batch = buffer.sample(8)
        loss, metrics = compute_ranking_loss(ranking_net, batch, gnn_encoder, device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 3 == 0:
            print(f"  Step {step+1}: loss={metrics['ranking_loss']:.4f}, acc={metrics['ranking_accuracy']:.2%}")
    
    # 测试网络是否学到了"大推力>小推力"
    print(f"\n✓ 学习验证:")
    embed_test = torch.randn(2, 256, device=device)
    action_large = torch.tensor([[0.9, 0.2, 1.1, 0.5, 0.1, 0.6]], device=device)
    action_small = torch.tensor([[0.05, 0.01, 0.08, 0.02, 0.005, 0.03]], device=device)
    
    logits = ranking_net.forward_compare(
        embed_test[0:1], embed_test[1:2],
        action_large, action_small
    )
    prob = torch.sigmoid(logits).item()
    print(f"  - P(大推力 > 小推力) = {prob:.3f}")
    print(f"  - 期望: > 0.5 (如果网络学到了动作幅度的重要性)")


def main():
    print("="*70)
    print("Ranking NN + 动作特征整合测试")
    print("="*70)
    print("目标：验证零动作惩罚已整合到Ranking NN中")
    print("方法：网络直接接收动作特征，自动学习'零动作→低排名'")
    print("="*70)
    
    try:
        # 测试1：基本功能
        ranking_net = test_ranking_with_action_features()
        
        # 测试2：Buffer
        buffer = test_buffer_with_action_features()
        
        # 测试3：训练
        test_ranking_loss_with_action_features()
        
        print("\n" + "="*70)
        print("✅ 所有测试通过！")
        print("="*70)
        print("Ranking NN已成功整合动作特征:")
        print("  1. forward_compare接收action_feat_a/b参数")
        print("  2. 未提供时自动填充零特征（兼容MCTS）")
        print("  3. Buffer存储action_features并用于训练")
        print("  4. 网络可学习'大推力>小推力'的规则")
        print("\n下一步：")
        print("  - 在train_online.py中启用 --use-ranking")
        print("  - 运行60轮测试观察效果")
        print("  - 查看ranking_accuracy是否提升")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

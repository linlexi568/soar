"""测试Ranking Value Network集成

验证：
1. GNN的get_embedding()方法能正常提取嵌入
2. RankingValueNet能正常比较两个程序
3. 训练循环能正常运行
"""
import torch
import sys
from pathlib import Path

# 添加路径
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / '01_soar'))

from gnn_features import ast_to_pyg_graph
from gnn_policy_nn_v2 import create_gnn_policy_value_net_v2
from ranking_value_net import (
    RankingValueNet, PairwiseRankingBuffer,
    compute_ranking_loss, setup_ranking_training
)
from torch_geometric.data import Batch

def test_gnn_embedding():
    """测试GNN嵌入提取"""
    print("\n=== 测试1: GNN嵌入提取 ===")
    
    # 创建GNN模型
    gnn = create_gnn_policy_value_net_v2(
        node_feature_dim=24,
        policy_output_dim=8,
        structure_hidden=256,
        structure_layers=5,
        structure_heads=8,
        feature_layers=3,
        feature_heads=8,
        dropout=0.1
    )
    
    # 创建简单程序
    program = [
        {'if': {'condition': ('pos_err_z', '>', 0.0), 'then': [{'set': ('u_z', 1.0)}]}}
    ]
    
    # 转换为图
    graph = ast_to_pyg_graph(program)
    batch = Batch.from_data_list([graph])
    
    # 测试forward
    policy_logits, value_scalar, value_components = gnn(batch)
    print(f"✓ Forward成功: policy={policy_logits.shape}, value={value_scalar.shape}, components={value_components.shape}")
    
    # 测试get_embedding
    embedding = gnn.get_embedding(batch)
    print(f"✓ get_embedding成功: embedding={embedding.shape}")
    
    assert embedding.shape == (1, 256), f"嵌入维度错误: {embedding.shape}"
    print("✓ 测试1通过！\n")
    return gnn


def test_ranking_network(gnn):
    """测试Ranking网络"""
    print("=== 测试2: Ranking网络 ===")
    
    # 创建Ranking网络
    ranking_net = RankingValueNet(embed_dim=256)
    
    # 创建两个程序
    prog_a = [
        {'if': {'condition': ('pos_err_z', '>', 0.0), 'then': [{'set': ('u_z', 1.0)}]}}
    ]
    prog_b = [
        {'if': {'condition': ('pos_err_z', '>', 0.0), 'then': [{'set': ('u_z', 2.0)}]}}
    ]
    
    # 获取嵌入
    graph_a = ast_to_pyg_graph(prog_a)
    graph_b = ast_to_pyg_graph(prog_b)
    batch = Batch.from_data_list([graph_a, graph_b])
    
    with torch.no_grad():
        embeddings = gnn.get_embedding(batch)
    
    embed_a = embeddings[0:1]
    embed_b = embeddings[1:2]
    
    # 测试比较
    with torch.no_grad():
        compare_score = ranking_net.forward_compare(embed_a, embed_b)
        value_a = ranking_net.forward_value(embed_a)
        value_b = ranking_net.forward_value(embed_b)
    
    print(f"✓ forward_compare成功: score={compare_score.item():.4f}")
    print(f"✓ forward_value成功: value_a={value_a.item():.4f}, value_b={value_b.item():.4f}")
    print("✓ 测试2通过！\n")
    return ranking_net


def test_ranking_training(gnn, ranking_net):
    """测试Ranking训练"""
    print("=== 测试3: Ranking训练 ===")
    
    # 创建buffer和optimizer
    ranking_buffer = PairwiseRankingBuffer(capacity=1000)
    ranking_optimizer = torch.optim.Adam(ranking_net.parameters(), lr=1e-3)
    
    # 生成训练数据
    program_buffer = []
    for i in range(10):
        prog = [
            {'if': {'condition': ('pos_err_z', '>', 0.0), 'then': [{'set': ('u_z', float(i))}]}}
        ]
        graph = ast_to_pyg_graph(prog)
        program_buffer.append({
            'graph': graph,
            'reward': float(i) - 5.0,  # -5到4的奖励
            'program': prog
        })
    
    # 生成程序对
    from ranking_value_net import generate_program_pairs
    pairs = generate_program_pairs(program_buffer, reward_threshold=0.5)
    print(f"✓ 生成 {len(pairs)} 个程序对")
    
    # pairs已经包含graph了，我们需要转换为program格式用于buffer
    # 但实际训练中我们直接用graph，所以这里简化测试，手动创建程序对
    for i in range(10):
        for j in range(i+1, 10):
            prog_a = [{'if': {'condition': ('pos_err_z', '>', 0.0), 'then': [{'set': ('u_z', float(i))}]}}]
            prog_b = [{'if': {'condition': ('pos_err_z', '>', 0.0), 'then': [{'set': ('u_z', float(j))}]}}]
            graph_a = ast_to_pyg_graph(prog_a)
            graph_b = ast_to_pyg_graph(prog_b)
            # j > i，所以prog_b更好
            ranking_buffer.push(graph_a, graph_b, 1.0)
    
    print(f"✓ Ranking buffer大小: {len(ranking_buffer)}")
    
    # 训练一步
    batch_size = 8
    batch = ranking_buffer.sample(batch_size)
    
    # 使用compute_ranking_loss（自动处理嵌入提取）
    device = torch.device('cpu')
    loss, metrics = compute_ranking_loss(
        ranking_net, batch, gnn, device
    )
    
    accuracy = metrics['ranking_accuracy']
    
    print(f"✓ Ranking loss计算成功: loss={loss.item():.4f}, accuracy={accuracy:.2%}")
    
    # 反向传播
    ranking_optimizer.zero_grad()
    loss.backward()
    ranking_optimizer.step()
    
    print("✓ 反向传播成功")
    print("✓ 测试3通过！\n")


def main():
    print("=" * 60)
    print("Ranking Value Network 集成测试")
    print("=" * 60)
    
    try:
        # 测试1: GNN嵌入提取
        gnn = test_gnn_embedding()
        
        # 测试2: Ranking网络
        ranking_net = test_ranking_network(gnn)
        
        # 测试3: Ranking训练
        test_ranking_training(gnn, ranking_net)
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n下一步：")
        print("1. 运行短期训练测试（20轮）验证完整流程")
        print("2. 观察ranking_loss是否下降、accuracy是否提升")
        print("3. 验证policy_loss是否不再为0")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

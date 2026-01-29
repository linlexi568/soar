"""快速测试在线训练系统"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

print("测试1: 导入模块...")
try:
    from mcts_training.program_features import featurize_program
    from mcts_training.policy.policy_nn import PolicyValueNNLarge
    print("✅ 模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

print("\n测试2: 特征化程序...")
try:
    test_program = [
        {'name': 'rule1', 'condition': None, 'action': [], 'multiplier': [1, 1, 1]}
    ]
    features = featurize_program(test_program)
    print(f"✅ 特征维度: {features.shape} (期望: [64])")
    assert features.shape[0] == 64, "特征维度错误"
except Exception as e:
    print(f"❌ 特征化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n测试3: 初始化NN...")
try:
    import torch
    model = PolicyValueNNLarge(in_dim=64, hidden=256)
    print(f"✅ NN参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    features_batch = features.unsqueeze(0)
    policy_logits, value = model(features_batch)
    print(f"✅ 策略输出维度: {policy_logits.shape}")
    print(f"✅ 价值输出维度: {value.shape}")
except Exception as e:
    print(f"❌ NN初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("🎉 所有测试通过！系统就绪。")
print("="*50)
print("\n运行完整训练：")
print("python 01_soar\\train_online.py --total-iters 100 --mcts-simulations 200")

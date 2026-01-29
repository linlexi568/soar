#!/usr/bin/env python3
"""
SAC 测试脚本 - 使用 SCG 精确奖励（与训练脚本/Soar/传统控制器对齐）
所有参数写在脚本顶部，直接修改即可

奖励函数: r_t = -(x_err^T Q x_err + u^T R u)
  Q = diag([1, 0.01, 1, 0.01, 1, 0.01, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
  R = 0.0001
"""

import sys
from pathlib import Path

# ============================================================================
# 路径设置 (Isaac Gym 必须在 torch 之前导入)
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '01_soar'))

# ============================================================================
#                    ★★★ 测试参数配置 (修改这里) ★★★
# ============================================================================

# --------------------- 任务配置 ---------------------
TASK = "square"          # 选择: circle, square, helix, figure8 (必须与训练时一致)
DURATION = 5.0            # 每个episode时长(秒)

# --------------------- 测试配置 ---------------------
NUM_ENVS = 1024           # 并行环境数 (与 Soar/CPID 评估对齐)
EPISODES = 10             # 测试episode数
DEVICE = "cuda:0"         # 设备

# --------------------- 模型路径 ---------------------
# 修改为你的模型路径
MODEL_PATH = ROOT / "results" / "sac" / TASK / "sac_quadrotor_latest.zip"

# --------------------- 输出配置 ---------------------
OUTPUT_PATH = ROOT / "03_SAC" / "results" / f"eval_{TASK}.json"

# ============================================================================
#                         测试代码 (不需要修改)
# ============================================================================

def main():
    import json
    from datetime import datetime
    import numpy as np
    
    from scg_vec_env import IsaacSCGVecEnv
    from stable_baselines3 import SAC
    
    print("=" * 70)
    print(f"SAC 测试 - 任务: {TASK}")
    print("=" * 70)
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  并行环境: {NUM_ENVS}")
    print(f"  测试episodes: {EPISODES}")
    print(f"  时长: {DURATION}s")
    print()
    
    # 检查模型文件
    if not MODEL_PATH.exists():
        print(f"❌ 错误: 模型文件不存在: {MODEL_PATH}")
        print("   请先运行 train_sac.py 训练模型，或修改 MODEL_PATH 指向正确的模型文件")
        return
    
    # 创建输出目录
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建环境 (使用 SCG 精确奖励)
    env = IsaacSCGVecEnv(
        num_envs=NUM_ENVS,
        device=DEVICE,
        task=TASK,
        duration=DURATION,
    )
    
    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = SAC.load(str(MODEL_PATH), device=DEVICE)
    
    # 测试
    print("\n开始测试...")
    episode_rewards = []
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += float(reward.sum())  # reward.sum() 是所有环境的奖励和
        
        # 计算单个环境的平均奖励
        ep_reward = total_reward / NUM_ENVS
        episode_rewards.append(ep_reward)
        print(f"  Episode {ep+1:02d}/{EPISODES}: reward_per_env = {ep_reward:.2f}")
    
    env.close()
    
    # 统计
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    
    print()
    print("=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"  平均奖励 (per env): {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Episodes: {EPISODES}")
    print(f"  环境数: {NUM_ENVS}")
    print()
    
    # 保存结果
    result = {
        "algorithm": "SAC",
        "task": TASK,
        "duration_sec": DURATION,
        "num_envs": NUM_ENVS,
        "episodes": EPISODES,
        "model_path": str(MODEL_PATH),
        "mean_reward_per_env": mean_reward,
        "std_reward_per_env": std_reward,
        "episode_rewards_per_env": [float(r) for r in episode_rewards],
        "reward_type": "scg_exact",
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

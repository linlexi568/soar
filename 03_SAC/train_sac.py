#!/usr/bin/env python3
"""
SAC 训练脚本 - 使用 SCG 精确奖励
所有超参数写在脚本顶部，方便修改

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
sys.path.insert(0, str(ROOT / '02_PPO'))

# ============================================================================
#                    ★★★ 超参数配置 (修改这里) ★★★
# ============================================================================

# --------------------- 任务配置 ---------------------
TASK = "circle"           # 选择: circle, square, helix, figure8
SEED = 42                 # 随机种子
DURATION = 5.0            # 每个episode时长(秒)

# --------------------- 训练配置 ---------------------
TOTAL_TIMESTEPS = 500_000  # 总训练步数
NUM_ENVS = 256            # 并行环境数
DEVICE = "cuda:0"         # 设备

# --------------------- SAC 超参数 ---------------------
LEARNING_RATE = 3e-4      # 学习率
BUFFER_SIZE = 100_000     # 回放缓冲区大小
BATCH_SIZE = 256          # 批次大小
GAMMA = 0.99              # 折扣因子
TAU = 0.005               # 软更新系数
ENT_COEF = "auto"         # 熵系数 ("auto" 或 float)
LEARNING_STARTS = 1000    # 开始学习前的步数
TRAIN_FREQ = 1            # 每N步训练一次
GRADIENT_STEPS = 1        # 每次训练的梯度步数

# --------------------- 网络结构 ---------------------
NET_ARCH = [256, 256]     # MLP隐藏层

# --------------------- 输出配置 ---------------------
SAVE_DIR = ROOT / "03_SAC" / "results" / "checkpoints"
LOG_DIR = ROOT / "03_SAC" / "results" / "logs"

# ============================================================================
#                         训练代码 (不需要修改)
# ============================================================================

def main():
    # 导入 (Isaac Gym 优先)
    from scg_vec_env import IsaacSCGVecEnv
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    import json
    from datetime import datetime
    
    print("=" * 70)
    print(f"SAC 训练 - 任务: {TASK}")
    print("=" * 70)
    print(f"  总步数: {TOTAL_TIMESTEPS:,}")
    print(f"  并行环境: {NUM_ENVS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  网络: {NET_ARCH}")
    print()
    
    # 创建目录
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建环境
    train_env = IsaacSCGVecEnv(
        num_envs=NUM_ENVS,
        device=DEVICE,
        task=TASK,
        duration=DURATION,
    )
    
    eval_env = IsaacSCGVecEnv(
        num_envs=64,
        device=DEVICE,
        task=TASK,
        duration=DURATION,
    )
    
    # 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(SAVE_DIR / TASK),
        name_prefix="sac",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(SAVE_DIR / TASK),
        log_path=str(LOG_DIR / TASK),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    # 创建模型
    policy_kwargs = dict(net_arch=dict(pi=NET_ARCH, qf=NET_ARCH))
    
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        ent_coef=ENT_COEF,
        learning_starts=LEARNING_STARTS,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
        device=DEVICE,
        tensorboard_log=str(LOG_DIR),
    )
    
    # 训练
    print("\n开始训练...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    
    # 保存最终模型
    final_path = SAVE_DIR / TASK / "sac_final.zip"
    model.save(final_path)
    print(f"\n模型已保存到: {final_path}")
    
    # 测试
    print(f"\n测试 SAC on {TASK}...")
    test_env = IsaacSCGVecEnv(
        num_envs=1024,
        device=DEVICE,
        task=TASK,
        duration=DURATION,
    )
    
    episode_rewards = []
    for ep in range(10):
        obs = test_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = test_env.step(action)
            total_reward += reward.sum()
            if dones.any():
                done = True
        # 平均每个环境的奖励
        ep_reward = total_reward / 1024
        episode_rewards.append(ep_reward)
        print(f"  Episode {ep+1}/10: {ep_reward:.2f}")
    
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    std_reward = (sum((r - mean_reward)**2 for r in episode_rewards) / len(episode_rewards)) ** 0.5
    print(f"\n✓ SAC on {TASK}: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 保存结果
    result = {
        "task": TASK,
        "algorithm": "SAC",
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "episodes": [float(r) for r in episode_rewards],
        "hyperparameters": {
            "total_timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS,
            "learning_rate": LEARNING_RATE,
            "buffer_size": BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "tau": TAU,
            "net_arch": NET_ARCH,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    result_path = SAVE_DIR / TASK / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"结果已保存到: {result_path}")
    
    train_env.close()
    eval_env.close()
    test_env.close()


if __name__ == "__main__":
    main()

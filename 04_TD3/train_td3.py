#!/usr/bin/env python3
"""
TD3 训练脚本 - 使用 SCG 精确奖励
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

# --------------------- TD3 超参数 ---------------------
LEARNING_RATE = 3e-4      # 学习率
BUFFER_SIZE = 100_000     # 回放缓冲区大小
BATCH_SIZE = 256          # 批次大小
GAMMA = 0.99              # 折扣因子
TAU = 0.005               # 软更新系数
POLICY_DELAY = 2          # 策略更新延迟
TARGET_POLICY_NOISE = 0.2 # 目标策略噪声
TARGET_NOISE_CLIP = 0.5   # 噪声裁剪
LEARNING_STARTS = 1000    # 开始学习前的步数
TRAIN_FREQ = 1            # 每N步训练一次
GRADIENT_STEPS = 1        # 每次训练的梯度步数

# --------------------- 探索噪声 ---------------------
ACTION_NOISE_STD = 0.1    # 动作噪声标准差

# --------------------- 网络结构 ---------------------
NET_ARCH = [256, 256]     # MLP隐藏层

# --------------------- 输出配置 ---------------------
SAVE_DIR = ROOT / "04_TD3" / "results" / "checkpoints"
LOG_DIR = ROOT / "04_TD3" / "results" / "logs"

# ============================================================================
#                         训练代码 (不需要修改)
# ============================================================================

def main():
    # 导入 (Isaac Gym 优先)
    from scg_vec_env import IsaacSCGVecEnv
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    import numpy as np
    import json
    from datetime import datetime
    
    print("=" * 70)
    print(f"TD3 训练 - 任务: {TASK}")
    print("=" * 70)
    print(f"  总步数: {TOTAL_TIMESTEPS:,}")
    print(f"  并行环境: {NUM_ENVS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  网络: {NET_ARCH}")
    print(f"  策略延迟: {POLICY_DELAY}")
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
    
    # 动作噪声
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=ACTION_NOISE_STD * np.ones(n_actions)
    )
    
    # 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(SAVE_DIR / TASK),
        name_prefix="td3",
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
    
    model = TD3(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        policy_delay=POLICY_DELAY,
        target_policy_noise=TARGET_POLICY_NOISE,
        target_noise_clip=TARGET_NOISE_CLIP,
        learning_starts=LEARNING_STARTS,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        action_noise=action_noise,
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
    final_path = SAVE_DIR / TASK / "td3_final.zip"
    model.save(final_path)
    print(f"\n模型已保存到: {final_path}")
    
    # 测试
    print(f"\n测试 TD3 on {TASK}...")
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
    print(f"\n✓ TD3 on {TASK}: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 保存结果
    result = {
        "task": TASK,
        "algorithm": "TD3",
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
            "policy_delay": POLICY_DELAY,
            "action_noise_std": ACTION_NOISE_STD,
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

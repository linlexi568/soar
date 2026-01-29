"""SB3 (Stable-Baselines3) 集成模块。

提供与 SB3 兼容的环境、训练脚本和评估工具。

模块结构:
    - quadrotor_env.py: SB3 标准 Gymnasium 接口的四旋翼环境
    - train_sb3.py: 使用 SB3 训练 PPO/SAC/TD3
    - soar_policy.py: 将 Soar DSL 程序转换为 SB3 兼容策略
    - compare_eval.py: 对比评估 SB3 模型和 Soar 程序
    - isaac_gym_wrapper.py: Isaac Gym 的 SB3 向量化环境封装

使用流程:
    1. 训练 Soar: ./run.sh (使用 sb3_standard reward profile)
    2. 训练 SB3 baseline: python scripts/sb3/train_sb3.py --algo ppo
    3. 对比评估: python scripts/sb3/compare_eval.py --sb3-model ... --soar-program ...

快速开始:
    ```python
    # 1. 直接使用 SB3 训练
    from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv
    from stable_baselines3 import PPO
    
    env = QuadrotorTrackingEnv(trajectory='figure8')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
    
    # 2. 在 SB3 环境中测试 Soar 程序
    from scripts.sb3.soar_policy import SoarPolicy, evaluate_soar_in_sb3
    
    results = evaluate_soar_in_sb3(
        'results/figure8-sb3_standard.json',
        trajectory='figure8',
        n_episodes=10,
    )
    
    # 3. 对比评估
    from scripts.sb3.compare_eval import compare_methods
    
    compare_methods(
        sb3_model_path='results/sb3/ppo_figure8_final.zip',
        soar_program_path='results/figure8-sb3_standard.json',
        trajectory='figure8',
    )
    ```

Reward 设计:
    SB3 标准环境使用简洁的 reward:
        reward = alive_bonus - pos_cost - ctrl_cost
    
    对应的 Soar reward profile: sb3_standard
    
    使用方法:
        ./run.sh  # 修改 REWARD_PROFILE="sb3_standard"
        或
        python train_online.py --reward-profile sb3_standard
"""
from __future__ import annotations

# 导出主要类和函数
from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv, register_quadrotor_envs
from scripts.sb3.soar_policy import (
    SoarPolicy,
    load_soar_program,
    evaluate_soar_in_sb3,
)

__all__ = [
    'QuadrotorTrackingEnv',
    'register_quadrotor_envs',
    'SoarPolicy',
    'load_soar_program',
    'evaluate_soar_in_sb3',
]

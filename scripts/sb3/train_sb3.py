"""SB3 训练和评估脚本。

使用 Stable-Baselines3 训练 PPO/SAC，并与 Soar 生成的程序对比。

使用示例:
    # 训练 PPO
    python scripts/sb3/train_sb3.py --algo ppo --trajectory figure8 --timesteps 100000
    
    # 训练 SAC
    python scripts/sb3/train_sb3.py --algo sac --trajectory hover --timesteps 50000
    
    # 评估已保存的模型
    python scripts/sb3/train_sb3.py --eval --model results/sb3/ppo_figure8.zip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 添加项目路径
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "01_soar"))

# 导入环境
from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv


def train_sb3(
    algo: str = 'ppo',
    trajectory: str = 'figure8',
    total_timesteps: int = 100_000,
    save_dir: str = 'results/sb3',
    reward_weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
    verbose: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """使用 SB3 训练 RL 智能体。
    
    Args:
        algo: 算法 ('ppo', 'sac', 'td3', 'a2c')
        trajectory: 轨迹类型
        total_timesteps: 训练步数
        save_dir: 保存目录
        reward_weights: Reward 权重覆盖
        seed: 随机种子
        verbose: 日志级别
    
    Returns:
        训练结果统计
    """
    try:
        from stable_baselines3 import PPO, SAC, TD3, A2C
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("❌ 请先安装 stable-baselines3:")
        print("   pip install stable-baselines3[extra]")
        return {}
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo}_{trajectory}_{timestamp}"
    
    print(f"=" * 60)
    print(f"SB3 Training: {algo.upper()} on {trajectory}")
    print(f"=" * 60)
    
    # 创建环境
    env_kwargs = {
        'trajectory': trajectory,
        'duration': 5.0,
        'control_freq': 50,
    }
    if reward_weights:
        env_kwargs['reward_weights'] = reward_weights
    
    def make_env():
        env = QuadrotorTrackingEnv(**env_kwargs)
        return Monitor(env)
    
    # 向量化环境
    n_envs = 4
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    
    # 评估环境
    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 100)
    
    # 选择算法
    algo_map = {
        'ppo': PPO,
        'sac': SAC,
        'td3': TD3,
        'a2c': A2C,
    }
    
    if algo.lower() not in algo_map:
        print(f"❌ 不支持的算法: {algo}")
        print(f"   支持: {list(algo_map.keys())}")
        return {}
    
    AlgoClass = algo_map[algo.lower()]
    
    # 算法超参数
    if algo.lower() == 'ppo':
        model = AlgoClass(
            'MlpPolicy',
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=verbose,
            seed=seed,
            tensorboard_log=str(save_path / "tb_logs"),
        )
    elif algo.lower() == 'sac':
        model = AlgoClass(
            'MlpPolicy',
            vec_env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=verbose,
            seed=seed,
            tensorboard_log=str(save_path / "tb_logs"),
        )
    elif algo.lower() == 'td3':
        model = AlgoClass(
            'MlpPolicy',
            vec_env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=verbose,
            seed=seed,
            tensorboard_log=str(save_path / "tb_logs"),
        )
    else:  # a2c
        model = AlgoClass(
            'MlpPolicy',
            vec_env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=verbose,
            seed=seed,
            tensorboard_log=str(save_path / "tb_logs"),
        )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(save_path / "eval_logs"),
        eval_freq=max(total_timesteps // 20, 1000),
        n_eval_episodes=5,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 5000),
        save_path=str(save_path / "checkpoints"),
        name_prefix=run_name,
    )
    
    # 训练
    print(f"\n开始训练: {total_timesteps:,} 步")
    print(f"并行环境: {n_envs}")
    print(f"保存路径: {save_path}")
    print()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    # 保存最终模型
    final_model_path = save_path / f"{run_name}_final.zip"
    model.save(str(final_model_path))
    print(f"\n✅ 模型已保存: {final_model_path}")
    
    # 评估最终性能
    print("\n评估最终性能...")
    eval_results = evaluate_model(model, eval_env, n_eval_episodes=10)
    
    # 保存结果
    results = {
        'algo': algo,
        'trajectory': trajectory,
        'total_timesteps': total_timesteps,
        'seed': seed,
        'reward_weights': reward_weights,
        'eval_results': eval_results,
        'model_path': str(final_model_path),
        'timestamp': timestamp,
    }
    
    results_path = save_path / f"{run_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ 结果已保存: {results_path}")
    
    # 清理
    vec_env.close()
    eval_env.close()
    
    return results


def evaluate_model(
    model,
    env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """评估模型性能。
    
    Returns:
        评估结果统计
    """
    episode_rewards = []
    episode_lengths = []
    episode_pos_errors = []
    
    for ep in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        pos_errors = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            if 'pos_error' in info[0]:
                pos_errors.append(info[0]['pos_error'])
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if pos_errors:
            episode_pos_errors.append(np.mean(pos_errors))
    
    results = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_pos_error': float(np.mean(episode_pos_errors)) if episode_pos_errors else None,
        'n_episodes': n_eval_episodes,
    }
    
    print(f"\n📊 评估结果 ({n_eval_episodes} episodes):")
    print(f"   Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   Mean Length: {results['mean_length']:.1f}")
    if results['mean_pos_error'] is not None:
        print(f"   Mean Pos Error: {results['mean_pos_error']:.4f} m")
    
    return results


def load_and_evaluate(
    model_path: str,
    trajectory: str = 'figure8',
    n_eval_episodes: int = 10,
) -> Dict[str, Any]:
    """加载并评估已保存的模型。"""
    try:
        from stable_baselines3 import PPO, SAC, TD3, A2C
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("❌ 请先安装 stable-baselines3")
        return {}
    
    # 根据文件名猜测算法
    model_name = Path(model_path).stem.lower()
    if 'ppo' in model_name:
        AlgoClass = PPO
    elif 'sac' in model_name:
        AlgoClass = SAC
    elif 'td3' in model_name:
        AlgoClass = TD3
    elif 'a2c' in model_name:
        AlgoClass = A2C
    else:
        # 默认尝试 PPO
        AlgoClass = PPO
    
    print(f"加载模型: {model_path}")
    model = AlgoClass.load(model_path)
    
    # 创建评估环境
    env = DummyVecEnv([lambda: Monitor(QuadrotorTrackingEnv(trajectory=trajectory))])
    
    results = evaluate_model(model, env, n_eval_episodes)
    
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description='SB3 Quadrotor Training')
    
    # 模式
    parser.add_argument('--eval', action='store_true', help='评估模式')
    parser.add_argument('--model', type=str, help='要评估的模型路径')
    
    # 训练参数
    parser.add_argument('--algo', type=str, default='ppo',
                       choices=['ppo', 'sac', 'td3', 'a2c'],
                       help='RL 算法')
    parser.add_argument('--trajectory', type=str, default='figure8',
                       choices=['hover', 'figure8', 'circle'],
                       help='轨迹类型')
    parser.add_argument('--timesteps', type=int, default=100_000,
                       help='训练步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save-dir', type=str, default='results/sb3',
                       help='保存目录')
    
    # Reward 权重
    parser.add_argument('--pos-weight', type=float, default=1.0,
                       help='位置误差权重')
    parser.add_argument('--ctrl-weight', type=float, default=0.001,
                       help='控制代价权重')
    
    args = parser.parse_args()
    
    if args.eval:
        if not args.model:
            print("❌ 评估模式需要指定 --model 参数")
            return
        load_and_evaluate(args.model, args.trajectory)
    else:
        reward_weights = {
            'pos_cost_weight': args.pos_weight,
            'ctrl_cost_weight': args.ctrl_weight,
        }
        
        train_sb3(
            algo=args.algo,
            trajectory=args.trajectory,
            total_timesteps=args.timesteps,
            save_dir=args.save_dir,
            reward_weights=reward_weights,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()

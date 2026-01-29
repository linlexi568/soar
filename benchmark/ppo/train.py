#!/usr/bin/env python3
"""PPO Training Script for Benchmark"""
import os
import sys
import argparse
from pathlib import Path

# 配置 Isaac Gym 路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaacgym" / "python"))

_ISAAC_BINDINGS = ROOT / "isaacgym" / "python" / "isaacgym" / "_bindings" / "linux-x86_64"
if _ISAAC_BINDINGS.exists():
    os.environ.setdefault("LD_LIBRARY_PATH", str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))

try:
    from isaacgym import gymapi
except Exception:
    pass

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# 导入配置和环境
BENCHMARK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from ppo.config import TRAJECTORIES, PPO_CONFIG, TRAIN_CONFIG, REWARD_CONFIG
from envs.isaac_gym_wrapper import IsaacGymSB3VecEnv


class BestModelCallback(BaseCallback):
    """保存最佳模型"""
    def __init__(self, save_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.best_reward = -np.inf
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue
            reward = info.get("raw_episode_r", info["episode"].get("r", -np.inf))
            if reward > self.best_reward:
                self.best_reward = reward
                self.model.save(str(self.save_dir / "best_model.zip"))
                if hasattr(self.training_env, "save"):
                    self.training_env.save(str(self.save_dir / "vec_normalize.pkl"))
                if self.verbose:
                    print(f"[Best] New best reward: {reward:.2f}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=list(TRAJECTORIES.keys()),
                        help='Trajectory type')
    parser.add_argument('--num-envs', type=int, default=TRAIN_CONFIG['num_envs'],
                        help='Number of parallel environments')
    parser.add_argument('--max-steps', type=int, default=TRAIN_CONFIG['max_steps'],
                        help='Maximum training steps')
    parser.add_argument('--seed', type=int, default=TRAIN_CONFIG['seed'],
                        help='Random seed')
    args = parser.parse_args()

    # 创建结果目录
    results_dir = BENCHMARK_DIR / "results" / "ppo" / args.task
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"🚀 Training PPO on {args.task} task")
    print(f"   Num envs: {args.num_envs}")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Results: {results_dir}")

    # 构建轨迹参数
    traj_config = TRAJECTORIES[args.task].copy()
    traj_config['trajectory_type'] = args.task

    # 创建环境
    env = IsaacGymSB3VecEnv(
        num_envs=args.num_envs,
        trajectory_type=args.task,
        trajectory_params=traj_config,
        reward_type='scg_exact',
        shaping_cfg=REWARD_CONFIG,
        device='cuda:0',
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    # 学习率调度
    def lr_schedule(progress_remaining: float) -> float:
        return PPO_CONFIG['learning_rate_end'] + \
               (PPO_CONFIG['learning_rate_start'] - PPO_CONFIG['learning_rate_end']) * progress_remaining

    # 创建模型
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=lr_schedule,
        n_steps=PPO_CONFIG['n_steps'],
        batch_size=PPO_CONFIG['batch_size'],
        n_epochs=PPO_CONFIG['n_epochs'],
        gamma=PPO_CONFIG['gamma'],
        gae_lambda=PPO_CONFIG['gae_lambda'],
        clip_range=PPO_CONFIG['clip_range'],
        ent_coef=PPO_CONFIG['ent_coef'],
        policy_kwargs={
            'net_arch': PPO_CONFIG['net_arch'],
            'activation_fn': PPO_CONFIG['activation_fn'],
        },
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
    )

    # 训练
    callback = BestModelCallback(save_dir=results_dir, verbose=1)
    try:
        model.learn(total_timesteps=args.max_steps, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    finally:
        # 保存最终模型
        model.save(str(results_dir / "final_model.zip"))
        env.save(str(results_dir / "vec_normalize_final.pkl"))
        print(f"✅ Model saved to {results_dir}")
        env.close()


if __name__ == '__main__':
    main()

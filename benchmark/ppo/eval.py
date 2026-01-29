#!/usr/bin/env python3
"""PPO Evaluation Script for Benchmark"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 配置 Isaac Gym 路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_soar"))

# 尝试多个可能的 Isaac Gym 位置（优先使用已知工作的版本）
isaac_gym_candidates = [
    Path("/home/linlexi/桌面/soar/isaacgym/python"),  # 已知工作版本
    ROOT / "isaacgym" / "python",
    ROOT / "IsaacGym_Preview_4_Package" / "isaacgym" / "python",
]
for isaac_path in isaac_gym_candidates:
    if isaac_path.exists():
        sys.path.insert(0, str(isaac_path))
        _ISAAC_BINDINGS = isaac_path / "isaacgym" / "_bindings" / "linux-x86_64"
        if _ISAAC_BINDINGS.exists():
            os.environ.setdefault("LD_LIBRARY_PATH", str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
        break

try:
    from isaacgym import gymapi
except Exception:
    pass

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

BENCHMARK_DIR = Path(__file__).parent.parent
# CRITICAL: 必须在导入 isaac_gym_wrapper 之前添加路径
sys.path.insert(0, str(ROOT / "01_soar"))
sys.path.insert(0, str(BENCHMARK_DIR))

from ppo.config import TRAJECTORIES

# Import from scripts (original working version)
try:
    from scripts.sb3.isaac_gym_wrapper import IsaacGymSB3VecEnv
except ImportError:
    # Fallback to benchmark version
    from envs.isaac_gym_wrapper import IsaacGymSB3VecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=list(TRAJECTORIES.keys()))
    parser.add_argument('--model', type=str, default=None, help='Path to model (default: auto-detect)')
    parser.add_argument('--episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--num-envs', type=int, default=256, help='Number of parallel environments')
    parser.add_argument('--use-best', action='store_true', help='Use best model instead of final')
    args = parser.parse_args()

    # 定位模型路径
    results_dir = BENCHMARK_DIR / "results" / "ppo" / args.task
    if args.model:
        model_path = Path(args.model)
        vec_path = model_path.parent / "vec_normalize.pkl"
    else:
        if args.use_best:
            # 尝试多个可能的命名
            best_candidates = [
                results_dir / "best" / "ppo_best_scg.zip",
                results_dir / "best_model.zip",
            ]
            model_path = None
            for candidate in best_candidates:
                if candidate.exists():
                    model_path = candidate
                    break
            if model_path is None:
                model_path = best_candidates[0]
            
            vec_candidates = [
                results_dir / "best" / "vec_normalize_best.pkl",
                results_dir / "vec_normalize.pkl",
            ]
            vec_path = None
            for candidate in vec_candidates:
                if candidate.exists():
                    vec_path = candidate
                    break
            if vec_path is None:
                vec_path = vec_candidates[0]
        else:
            model_path = results_dir / "ppo_final.zip"
            vec_path = results_dir / "vec_normalize.pkl"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"📊 Evaluating PPO on {args.task}")
    print(f"   Model: {model_path}")
    print(f"   Episodes: {args.episodes}")

    # 构建轨迹参数
    traj_config = TRAJECTORIES[args.task].copy()
    traj_config['trajectory_type'] = args.task

    # 创建环境
    env = IsaacGymSB3VecEnv(
        num_envs=args.num_envs,
        trajectory_type=args.task,
        trajectory_params=traj_config,
        reward_type='scg_exact',
        shaping_cfg={'use_pure_scg': True},
        device='cuda:0',
    )
    
    # 加载归一化统计
    if vec_path.exists():
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False

    # 加载模型
    model = PPO.load(str(model_path), env=env, device='cuda:0')

    # 评估
    episode_rewards = []
    episode_lengths = []
    traj_x, traj_y, traj_z = [], [], []
    ref_x, ref_y, ref_z = [], [], []
    
    sb3_env = env.venv if hasattr(env, 'venv') else env
    obs = env.reset()
    isaac_env = sb3_env._isaac_env
    env_0_done = False
    
    # 记录初始位置
    pos_init = isaac_env.pos[0].cpu().numpy()
    traj_x.append(pos_init[0])
    traj_y.append(pos_init[1])
    traj_z.append(pos_init[2])
    
    target_init = sb3_env._target_pos[0].copy()
    ref_x.append(target_init[0])
    ref_y.append(target_init[1])
    ref_z.append(target_init[2])

    while len(episode_rewards) < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        # 收集轨迹（仅环境0）
        if not env_0_done:
            pos = isaac_env.pos[0].cpu().numpy()
            traj_x.append(pos[0])
            traj_y.append(pos[1])
            traj_z.append(pos[2])
            
            target = sb3_env._target_pos[0].copy()
            ref_x.append(target[0])
            ref_y.append(target[1])
            ref_z.append(target[2])
            
            if dones[0]:
                env_0_done = True

        # 收集episode统计
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
                if len(episode_rewards) >= args.episodes:
                    break

    env.close()

    # 统计结果
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    
    print("\n" + "=" * 60)
    print(f"PPO Evaluation Results ({args.task})")
    print("=" * 60)
    print(f"Episodes:      {len(episode_rewards)}")
    print(f"Mean Reward:   {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"Mean Length:   {mean_length:.2f}")
    print("=" * 60)

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.plot(ref_x, ref_y, 'r--', label='Reference', linewidth=2)
    plt.plot(traj_x, traj_y, 'b-', label='PPO', linewidth=1.5, alpha=0.8)
    plt.title(f'PPO Trajectory Tracking ({args.task})')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plot_path = results_dir / f"eval_{args.task}.png"
    plt.savefig(str(plot_path), dpi=150)
    print(f"\n📈 Plot saved to {plot_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train TD3 with SCG-aligned reward on Soar's Isaac Gym backend."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scg_vec_env import IsaacSCGVecEnv  # type: ignore

# =============================================================================
# Configurable constants (edit here only)
# =============================================================================
TASK = 'square'
DURATION_SEC = 5.0
DEVICE = 'cuda:0'
TRAIN_ENVS = 512
EVAL_ENVS = 1
TOTAL_TIMESTEPS = 1_000_000
SEED = 7

# TD3 hyper-parameters
LEARNING_RATE = 1e-3
BUFFER_SIZE = 300_000
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
LEARNING_STARTS = 20_000
POLICY_DELAY = 2
NET_ARCH = [400, 300]
NOISE_STD = 0.1

# Logging/checkpoint
SAVE_DIR = Path('results/aligned_rl/td3')
LOG_DIR = Path('results/aligned_rl/td3_logs')
CHECKPOINT_EVERY = 50_000
EVAL_EVERY = 25_000
EVAL_EPISODES = 5
TEST_EPISODES = 10


def _rollout(model, env: IsaacSCGVecEnv, episodes: int) -> List[float]:
    returns: List[float] = []
    for ep in range(episodes):
        obs = env.reset()
        done = np.array([False])
        reward_sum = 0.0
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            reward_sum += float(reward[0])
        returns.append(reward_sum)
        print(f"[Eval] Episode {ep+1}/{episodes}: reward={reward_sum:.2f}")
    return returns


def main() -> None:
    from stable_baselines3 import TD3
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.noise import NormalActionNoise

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TD3 Training (SCG aligned)")
    print(f" Task={TASK} | Duration={DURATION_SEC}s | Train envs={TRAIN_ENVS} | Seed={SEED}")
    print(f" Total timesteps={TOTAL_TIMESTEPS:,} | Device={DEVICE}")
    print("=" * 80)

    train_env = IsaacSCGVecEnv(num_envs=TRAIN_ENVS, task=TASK, duration=DURATION_SEC, device=DEVICE)
    eval_env = IsaacSCGVecEnv(num_envs=EVAL_ENVS, task=TASK, duration=DURATION_SEC, device=DEVICE)
    test_env = None

    policy_kwargs = dict(net_arch=dict(pi=NET_ARCH, qf=NET_ARCH))
    action_noise = NormalActionNoise(
        mean=np.zeros(train_env.action_space.shape[0], dtype=np.float32),
        sigma=np.full(train_env.action_space.shape[0], NOISE_STD, dtype=np.float32),
    )

    model = TD3(
        'MlpPolicy',
        train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        learning_starts=LEARNING_STARTS,
        policy_delay=POLICY_DELAY,
        action_noise=action_noise,
        seed=SEED,
        tensorboard_log=str(LOG_DIR),
        verbose=1,
        device=DEVICE,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY // max(1, TRAIN_ENVS // 64),
        save_path=str(SAVE_DIR / TASK / 'checkpoints'),
        name_prefix='td3_scg',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(SAVE_DIR / TASK / 'best'),
        log_path=str(SAVE_DIR / TASK / 'eval_logs'),
        eval_freq=EVAL_EVERY // max(1, TRAIN_ENVS // 64),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )

    try:
        print("\n🚀 Start training...")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_cb, eval_cb], progress_bar=True)

        final_path = SAVE_DIR / TASK / 'td3_final.zip'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(final_path)
        print(f"\n💾 Final policy saved to: {final_path}")

        print("\n🔎 Final evaluation (deterministic)...")
        test_env = IsaacSCGVecEnv(num_envs=1, task=TASK, duration=DURATION_SEC, device=DEVICE)
        returns = _rollout(model, test_env, TEST_EPISODES)
        mean_reward = float(np.mean(returns))
        std_reward = float(np.std(returns))
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f} over {TEST_EPISODES} episodes")

        summary = {
            'algorithm': 'TD3',
            'task': TASK,
            'duration_sec': DURATION_SEC,
            'train_envs': TRAIN_ENVS,
            'total_timesteps': TOTAL_TIMESTEPS,
            'seed': SEED,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'episode_returns': returns,
            'timestamp': datetime.utcnow().isoformat(),
            'hyperparameters': {
                'learning_rate': LEARNING_RATE,
                'buffer_size': BUFFER_SIZE,
                'batch_size': BATCH_SIZE,
                'gamma': GAMMA,
                'tau': TAU,
                'policy_delay': POLICY_DELAY,
                'net_arch': NET_ARCH,
            },
        }
        summary_path = SAVE_DIR / TASK / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to: {summary_path}")
    finally:
        train_env.close()
        eval_env.close()
        if test_env is not None:
            test_env.close()


if __name__ == '__main__':
    main()

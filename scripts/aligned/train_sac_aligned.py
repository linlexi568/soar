#!/usr/bin/env python3
"""Train SAC on Soar's Isaac Gym env with SCG-aligned reward.

All hyper-parameters live inside this file per lab policy. Modify the
constants below instead of passing CLI flags.
"""
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
# ⚙️  Config (edit below)
# =============================================================================
TASK = 'square'              # {'hover','circle','square','figure8','helix'}
DURATION_SEC = 5.0           # Episode horizon in seconds
DEVICE = 'cuda:0'
TRAIN_ENVS = 512             # Parallel envs for data collection
EVAL_ENVS = 1                # Use single-env eval for stable metrics
TOTAL_TIMESTEPS = 800_000
SEED = 42

# SAC hyper-parameters
LEARNING_RATE = 3e-4
BUFFER_SIZE = 200_000
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 0.005
ENT_COEF = 'auto'
LEARNING_STARTS = 10_000
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
NET_ARCH = [256, 256, 256]

# Logging / checkpointing
SAVE_DIR = Path('results/aligned_rl/sac')
LOG_DIR = Path('results/aligned_rl/sac_logs')
CHECKPOINT_EVERY = 50_000
EVAL_EVERY = 25_000
EVAL_EPISODES = 5
TEST_EPISODES = 10

# =============================================================================
# 🧠  Training entry
# =============================================================================

def _run_rollouts(model, env: IsaacSCGVecEnv, episodes: int) -> List[float]:
    """Deterministic evaluation helper returning per-episode returns."""
    returns: List[float] = []
    for ep in range(episodes):
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += float(reward[0])
        returns.append(total_reward)
        print(f"[Eval] Episode {ep+1}/{episodes}: reward={total_reward:.2f}")
    return returns


def main() -> None:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SAC Training (SCG aligned)")
    print(f" Task={TASK} | Duration={DURATION_SEC}s | Train envs={TRAIN_ENVS} | Seed={SEED}")
    print(f" Total timesteps={TOTAL_TIMESTEPS:,} | Device={DEVICE}")
    print("=" * 80)

    train_env = IsaacSCGVecEnv(num_envs=TRAIN_ENVS, task=TASK, duration=DURATION_SEC, device=DEVICE)
    eval_env = IsaacSCGVecEnv(num_envs=EVAL_ENVS, task=TASK, duration=DURATION_SEC, device=DEVICE)
    test_env = None

    policy_kwargs = dict(net_arch=dict(pi=NET_ARCH, qf=NET_ARCH))

    model = SAC(
        'MlpPolicy',
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
        seed=SEED,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device=DEVICE,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY // max(1, TRAIN_ENVS // 64),
        save_path=str(SAVE_DIR / TASK / 'checkpoints'),
        name_prefix='sac_scg',
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

        final_path = SAVE_DIR / TASK / 'sac_final.zip'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(final_path)
        print(f"\n💾 Final policy saved to: {final_path}")

        print("\n🔎 Final evaluation (deterministic)...")
        test_env = IsaacSCGVecEnv(num_envs=1, task=TASK, duration=DURATION_SEC, device=DEVICE)
        returns = _run_rollouts(model, test_env, TEST_EPISODES)
        mean_reward = float(np.mean(returns))
        std_reward = float(np.std(returns))
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f} over {TEST_EPISODES} episodes")

        summary = {
        'algorithm': 'SAC',
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
            'ent_coef': ENT_COEF,
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

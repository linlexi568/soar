from __future__ import annotations

"""Evaluation helpers for baseline policies executed in Isaac Gym."""

import math
from typing import Dict

import numpy as np


def evaluate_model(model, env, *, duration: float, n_eval: int, deterministic: bool = True) -> Dict[str, object]:
    """Roll out a Stable-Baselines3 model in a VecEnv and aggregate metrics."""
    num_envs = env.num_envs
    max_steps = max(1, int(round(duration / env.dt)))
    n_batches = max(1, math.ceil(n_eval / num_envs))

    all_rewards = []
    all_lengths = []
    batch_stats = []

    for batch_idx in range(n_batches):
        obs = env.reset()
        rewards_acc = np.zeros(num_envs, dtype=np.float64)
        lengths_acc = np.zeros(num_envs, dtype=np.float64)
        dones = np.zeros(num_envs, dtype=bool)

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, step_dones, _ = env.step(action)
            active = ~dones
            rewards_acc[active] += rewards[active]
            lengths_acc[active] += 1
            dones |= step_dones
            if dones.all():
                break

        all_rewards.extend(rewards_acc.tolist())
        all_lengths.extend(lengths_acc.tolist())
        batch_stats.append(
            {
                "batch_index": batch_idx + 1,
                "mean_reward": float(rewards_acc.mean()),
                "std_reward": float(rewards_acc.std()),
                "mean_length": float(lengths_acc.mean()),
            }
        )

    rewards_arr = np.asarray(all_rewards, dtype=np.float64)[:n_eval]
    lengths_arr = np.asarray(all_lengths, dtype=np.float64)[:n_eval]

    summary = {
        "episodes": int(len(rewards_arr)),
        "mean_reward": float(rewards_arr.mean()) if rewards_arr.size else 0.0,
        "std_reward": float(rewards_arr.std()) if rewards_arr.size else 0.0,
        "mean_length": float(lengths_arr.mean()) if lengths_arr.size else 0.0,
        "reward_per_episode": rewards_arr,
        "length_per_episode": lengths_arr,
        "batch_stats": batch_stats,
    }
    return summary

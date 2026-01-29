#!/usr/bin/env python3
"""Evaluate safe-control-gym PPO/SAC baselines in the Isaac Gym environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3 import PPO, SAC

if __package__ is None or __package__ == "":
    import sys
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT = CURRENT_DIR.parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))

from eval_utils import evaluate_model
from isaac_vec_env import INITIAL_XYZ_DEFAULT, IsaacGymVecEnv, default_trajectory_params


MODEL_LOADERS = {
    "ppo": PPO,
    "sac": SAC,
}


def _build_env(cfg: argparse.Namespace) -> IsaacGymVecEnv:
    traj_catalog = default_trajectory_params()
    traj_params = dict(traj_catalog.get(cfg.task, {}))
    initial_xyz = traj_params.pop("center", INITIAL_XYZ_DEFAULT)
    env = IsaacGymVecEnv(
        num_envs=cfg.isaac_num_envs,
        device=cfg.device,
        task=cfg.task,
        duration=cfg.duration,
        reward_profile=cfg.reward_profile,
        traj_params=traj_params,
        initial_xyz=initial_xyz,
    )
    return env


def _load_model(policy_type: str, model_path: Path, env: IsaacGymVecEnv, device: str):
    policy_type = policy_type.lower()
    if policy_type not in MODEL_LOADERS:
        raise ValueError(f"Unsupported policy type '{policy_type}'. Expected one of {list(MODEL_LOADERS)}")
    loader = MODEL_LOADERS[policy_type]
    try_path = model_path
    if try_path.is_dir():
        try_path = try_path / "final_model.zip"
    elif try_path.suffix == "":
        candidate = Path(f"{try_path}.zip")
        try_path = candidate if candidate.exists() else try_path
    if not try_path.exists():
        raise FileNotFoundError(f"Model file '{try_path}' not found")
    model = loader.load(str(try_path), env=env, device=device)
    return model, try_path


def _save_results(results: Dict[str, object], output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean_reward": results["mean_reward"],
        "std_reward": results["std_reward"],
        "mean_length": results["mean_length"],
        "episodes": results["episodes"],
        "reward_per_episode": results["reward_per_episode"].tolist(),
        "length_per_episode": results["length_per_episode"].tolist(),
        "batch_stats": results["batch_stats"],
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Eval] Results saved to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate safe-control baselines inside Isaac Gym")
    parser.add_argument("--policy-type", choices=MODEL_LOADERS.keys(), default="ppo")
    parser.add_argument("--model-path", required=True, help="Path to the trained model (zip file or directory)")
    parser.add_argument("--task", default="figure8", choices=["hover", "figure8", "circle", "square", "helix"], help="Trajectory task")
    parser.add_argument("--duration", type=float, default=10.0, help="Episode duration in seconds")
    parser.add_argument("--isaac-num-envs", type=int, default=512, help="Parallel Isaac env count")
    parser.add_argument(
        "--reward-profile",
        default="safe_control_tracking",
        help="Reward profile (default matches safe-control-gym quadrotor_3D_track")
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy evaluation")
    parser.add_argument("--output", type=str, default="", help="Optional JSON file to store metrics")
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    env = _build_env(cfg)
    try:
        model, resolved_path = _load_model(cfg.policy_type, Path(cfg.model_path), env, cfg.device)
        print(f"[Eval] Loaded {cfg.policy_type.upper()} model from {resolved_path}")

        results = evaluate_model(
            model,
            env,
            duration=cfg.duration,
            n_eval=cfg.episodes,
            deterministic=cfg.deterministic,
        )

        for batch_stat in results.get("batch_stats", []):
            print(
                f"  Batch {batch_stat['batch_index']}: "
                f"mean_reward={batch_stat['mean_reward']:.3f}, "
                f"std={batch_stat['std_reward']:.3f}, "
                f"mean_length={batch_stat['mean_length']:.1f}"
            )

        print(
            f"\n[Summary] {cfg.policy_type.upper()} on {cfg.task}: "
            f"mean_reward={results['mean_reward']:.3f} ± {results['std_reward']:.3f}, "
            f"episodes={results['episodes']}"
        )

        output_path = Path(cfg.output) if cfg.output else None
        _save_results(results, output_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()

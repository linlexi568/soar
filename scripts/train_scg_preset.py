#!/usr/bin/env python3
"""SCG-only training launcher with inline hyperparameters."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "01_soar"
for _path in (REPO_ROOT, PKG_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_online  # noqa: E402
OnlineTrainer = train_online.OnlineTrainer
parse_args = train_online.parse_args

# Edit these dictionaries directly instead of passing CLI flags.
ENV_OVERRIDES = {
    "RANKING_GNN_CHUNK": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64",
    "BATCH_EVAL_QUIET": "1",
    "BATCH_EVAL_LOG_LEVEL": "WARN",
    "OMP_NUM_THREADS": "64",
    "MKL_NUM_THREADS": "64",
    "NUMEXPR_NUM_THREADS": "64",
    "PYTORCH_NUM_THREADS": "64",
    "CUDA_LAUNCH_BLOCKING": "0",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_THREADING_LAYER": "GNU",
}

TRAINING_OVERRIDES = {
    # Core schedule
    "total_iters": 800,
    "mcts_simulations": 200,
    "update_freq": 999_999,
    "train_steps_per_update": 10,
    "isaac_num_envs": 8196,
    "mcts_leaf_batch_size": 1024,
    "async_training": False,
    "async_update_interval": 0.15,
    "async_max_steps_per_iter": 1,
    # Reward + evaluation knobs
    "reward_profile": "safe_control_tracking",
    "reward_reduction": "mean",
    "traj": "figure8",
    "duration": 5,
    "eval_replicas_per_program": 1,
    "min_steps_frac": 0.0,
    "zero_action_penalty": 5.0,
    "zero_action_penalty_decay": 0.98,
    "zero_action_penalty_min": 1.0,
    "action_scale_multiplier": 1.0,
    "enable_output_mad": True,
    "mad_min_fz": 0.0,
    "mad_max_fz": 7.5,
    "mad_max_xy": 0.12,
    "mad_max_yaw": 0.04,
    "mad_max_delta_fz": 1.5,
    "mad_max_delta_xy": 0.03,
    "mad_max_delta_yaw": 0.02,
    # Search heuristics
    "root_dirichlet_eps_init": 0.25,
    "root_dirichlet_eps_final": 0.10,
    "root_dirichlet_alpha_init": 0.30,
    "root_dirichlet_alpha_final": 0.20,
    "heuristic_decay_window": 350,
    "use_fast_path": True,
    "disable_gpu_expression": False,
    # Model + optimizer
    "batch_size": 128,
    "learning_rate": 1e-3,
    "replay_capacity": 50_000,
    "gnn_structure_hidden": 192,
    "gnn_structure_layers": 4,
    "gnn_structure_heads": 6,
    "gnn_feature_layers": 4,
    "gnn_feature_heads": 8,
    "gnn_dropout": 0.2,
    # Ranking/value toggles
    "use_ranking": False,
    "enable_ranking_mcts_bias": False,
    "enable_value_head": False,
    "enable_ranking_reweight": False,
    "ranking_bias_beta": 0.3,
    "ranking_reweight_beta": 0.2,
    "ranking_blend_init": 0.3,
    "ranking_blend_max": 0.8,
    "ranking_blend_warmup": 100,
    # Prior / Bayes
    "prior_profile": "none",
    "structure_prior_weight": 0.0,
    "stability_prior_weight": 0.0,
    "enable_bayesian_tuning": True,
    "bo_batch_size": 12,
    "bo_iterations": 2,
    "prior_level": 2,
    # Misc
    "use_meta_rl": False,
    "save_path": str(REPO_ROOT / "results" / "figure8-safe_control_tracking.json"),
    "checkpoint_freq": 50,
    "warm_start": None,
    "elite_archive_size": 50,
    "curriculum_mode": "none",
    "program_history_path": str(REPO_ROOT / "01_soar" / "results" / "program_history.jsonl"),
    "debug_programs": False,
    "debug_programs_limit": 20,
    "debug_rewards": False,
    "ast_pipeline": False,
    "policy_temperature": 1.0,
    "exploration_weight": 2.5,
    "puct_c": 1.5,
    "max_depth": 12,
}


def apply_environment() -> None:
    for key, value in ENV_OVERRIDES.items():
        os.environ.setdefault(key, value)
    isaac_bindings = REPO_ROOT / "isaacgym" / "python" / "isaacgym" / "_bindings" / "linux-x86_64"
    if isaac_bindings.is_dir():
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{isaac_bindings}:{current}" if current else str(isaac_bindings)


def build_args():
    args = parse_args([])
    for key, value in TRAINING_OVERRIDES.items():
        setattr(args, key, value)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return args


def main():
    apply_environment()
    args = build_args()
    print("[train_scg_preset] Launching with inline parameters:")
    for key in sorted(TRAINING_OVERRIDES):
        print(f"  - {key} = {TRAINING_OVERRIDES[key]}")
    trainer = OnlineTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal evaluation runner: safe-control-gym × Soar adapter

Usage (after installing safe-control-gym):
    python scripts/eval_safecontrol_soar.py \
        --env-id quadrotor_tracking \
        --episodes 3 \
        --controller pid \
        --action-space thrust_torque

Optional (using Soar DSL program):
    python scripts/eval_safecontrol_soar.py \
        --env-id quadrotor_tracking \
        --episodes 3 \
        --controller soar \
        --program-json results/figure8-balanced.json \
        --device cuda:0

Metrics recorded:
- RMSE of position (m)
- Max position error (m)
- Control effort (mean squared action)
- Violation ratio (if env exposes constraints; otherwise based on action clipping)
- Success ratio (heuristic threshold on tracking RMSE)
"""
from __future__ import annotations
import argparse
import sys
import time
import math
from typing import Any, Dict, Tuple

import numpy as np

from adapters.safecontrol_soar_adapter import SoarPolicyAdapter

CONTROLLER_CHOICES = ('pid', 'lqr', 'soar')

try:
    # safe-control-gym uses gymnasium in newer versions; try both
    import gymnasium as gymnasium
except Exception:
    gymnasium = None

try:
    import gym
except Exception:
    gym = None


def _make_env(env_id: str, seed: int = 0):
    # Try safe-control-gym factory first so we always exercise the native SCG environment.
    try:
        from safe_control_gym.envs import make as scg_make  # type: ignore
        env = scg_make(env_id=env_id, seed=seed)
        return env
    except Exception:
        pass

    # Fall back to gymnasium first, then classic gym. Users may need to register SCG envs here.
    if gymnasium is not None:
        try:
            env = gymnasium.make(env_id)
            try:
                env.reset(seed=seed)
            except Exception:
                pass
            return env
        except Exception:
            pass
    if gym is not None:
        try:
            env = gym.make(env_id)
            try:
                env.reset(seed=seed)
            except Exception:
                pass
            return env
        except Exception:
            pass
    raise RuntimeError(
        f"Failed to create env '{env_id}'. Please ensure safe-control-gym is installed and env_id is valid."
    )


def _metrics_init():
    return {
        'pos_err_sq_sum': 0.0,
        'pos_err_max': 0.0,
        'action_sq_sum': 0.0,
        'steps': 0,
        'violations': 0,
        'episodes': 0,
        'success': 0,
    }


def _extract_pos_err(info: Dict[str, Any]) -> Tuple[float, float, float]:
    # Try common keys; fallback to (0,0,0)
    for k in ('pos_err', 'position_error', 'track_error'):
        if isinstance(info, dict) and k in info:
            v = np.asarray(info[k], dtype=np.float64).reshape(-1)
            if v.size >= 3:
                return float(v[0]), float(v[1]), float(v[2])
    return 0.0, 0.0, 0.0


def _maybe_violation(info: Dict[str, Any], action: np.ndarray) -> bool:
    # If env exposes constraint flags, use them; else consider large actions as potential violations.
    if isinstance(info, dict):
        for k in ('violation', 'violations', 'constraint_violations'):
            if k in info:
                v = info[k]
                if isinstance(v, (int, float)):
                    return bool(v)
                if isinstance(v, (list, tuple, np.ndarray)):
                    return any(bool(x) for x in v)
    return bool(np.any(np.abs(action) > 0.99))


def evaluate(env_id: str, episodes: int, controller: str, action_space: str, program_json: str, device: str, seed: int):
    env = _make_env(env_id=env_id, seed=seed)
    adapter = SoarPolicyAdapter(
        mode=(controller if controller in CONTROLLER_CHOICES else 'pid'),
        action_space=action_space,
        program_json=program_json if controller == 'soar' else None,
        device=device,
        normalize_motors=(action_space == 'motors'),
    )
    M = _metrics_init()
    success_threshold_rmse = 0.15  # m, adjust per benchmark difficulty

    for ep in range(episodes):
        if hasattr(env, 'reset'):  # gymnasium returns (obs, info)
            out = env.reset(seed=seed + ep)
            if isinstance(out, tuple):
                obs, info = out[0], out[1]
            else:
                obs, info = out, {}
        else:
            obs, info = env.reset(), {}
        pos_err_sq, pos_err_cnt = 0.0, 0
        pos_err_max = 0.0
        action_sq, step_count = 0.0, 0
        violated = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = adapter.act(obs)
            # step
            try:
                step_out = env.step(action)
            except Exception as e:
                raise RuntimeError(f"Env.step failed with action shape {action.shape}: {e}")
            # Unpack gym/gymnasium outputs
            if len(step_out) == 5:
                obs, reward, done, truncated, info = step_out
            elif len(step_out) == 4:
                obs, reward, done, info = step_out
                truncated = False
            else:
                raise RuntimeError('Unexpected env.step return tuple')
            # metrics
            pe = _extract_pos_err(info)
            pe_norm = math.sqrt(pe[0]**2 + pe[1]**2 + pe[2]**2)
            pos_err_sq += pe_norm * pe_norm
            pos_err_cnt += 1
            pos_err_max = max(pos_err_max, pe_norm)
            action_sq += float(np.sum(action.astype(np.float64)**2))
            violated += 1 if _maybe_violation(info, action) else 0
            step_count += 1
        # ep done
        rmse = math.sqrt(pos_err_sq / max(1, pos_err_cnt))
        M['pos_err_sq_sum'] += pos_err_sq
        M['pos_err_max'] = max(M['pos_err_max'], pos_err_max)
        M['action_sq_sum'] += action_sq
        M['steps'] += step_count
        M['violations'] += violated
        M['episodes'] += 1
        M['success'] += 1 if rmse <= success_threshold_rmse else 0
        print(f"[Episode {ep+1}/{episodes}] RMSE={rmse:.3f}m, e_max={pos_err_max:.3f}m, action_energy={action_sq:.2f}, viol={violated}")

    # aggregate
    steps = max(1, M['steps'])
    eps = max(1, M['episodes'])
    rmse_mean = math.sqrt(M['pos_err_sq_sum'] / max(1, M['steps']))
    print("\n=== Summary ===")
    print(f"Episodes: {eps}")
    print(f"RMSE_pos (mean over steps): {rmse_mean:.3f} m")
    print(f"e_max (over episodes): {M['pos_err_max']:.3f} m")
    print(f"Control effort (mean per step): {M['action_sq_sum']/steps:.3f}")
    print(f"Violations/step: {M['violations']/steps:.4f}")
    print(f"Success ratio: {M['success']/eps:.2%}")
    try:
        env.close()
    except Exception:
        pass
    try:
        adapter.close()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-id', type=str, default='quadrotor_tracking', help='safe-control-gym env id')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--controller', type=str, default='pid', choices=list(CONTROLLER_CHOICES))
    ap.add_argument('--action-space', type=str, default='thrust_torque', choices=['thrust_torque', 'motors'])
    ap.add_argument('--program-json', type=str, default='')
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    # Friendly checks
    if args.controller == 'soar' and not args.program_json:
        print('[Warn] --controller soar specified but --program-json is empty; falling back to PID baseline.')
        args.controller = 'pid'
    try:
        evaluate(
            env_id=args.env_id,
            episodes=args.episodes,
            controller=args.controller,
            action_space=args.action_space,
            program_json=args.program_json,
            device=args.device,
            seed=args.seed,
        )
    except Exception as e:
        print('[Error] Evaluation failed:', e)
        print('Hint: ensure safe-control-gym is installed, and env-id/action-space align with your install.')
        sys.exit(1)


if __name__ == '__main__':
    main()

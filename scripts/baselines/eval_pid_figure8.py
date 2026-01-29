#!/usr/bin/env python3
"""Evaluate PID inside safe-control-gym (testing path).

所有配置写在常量里，禁止通过终端传参；如需调整测试配置请直接修改常量。
训练仍在 Isaac Gym 进行，本脚本专门用于 SCG 测试，方便对齐外部基线。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import gymnasium as _gym
    GYMN_EXT = "gymnasium"
except Exception:
    _gym = None
    GYMN_EXT = ""

try:
    import gym as _legacy_gym
except Exception:
    _legacy_gym = None

if __package__ is None or __package__ == "":
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    ROOT = CURRENT_DIR.parent.parent
    for p in (CURRENT_DIR, ROOT, ROOT / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

from adapters.safecontrol_soar_adapter import SoarPolicyAdapter  # type: ignore


# ---------------------------------------------------------------------------
# 固定测试配置：SCG 提供标准测试，Isaac 专注训练
# ---------------------------------------------------------------------------

ENV_ID = "quadrotor_tracking"     # safe-control-gym env
EPISODES = 5                      # 测试 episode 数
SEED = 0                          # 复现实验用
ACTION_SPACE = "thrust_torque"    # 也可改成 "motors"
CONTROLLER_MODE = "pd"            # 'pd' 或 'soar'
PROGRAM_JSON = ""                 # 如需测试某个 Soar 程序，填结果 JSON 路径
DEVICE = "cpu"                    # 仅对 soar 模式生效
SUCCESS_RMSE = 0.15               # m，成功阈值


def _make_env(seed: int):
    if _gym is not None:
        try:
            env = _gym.make(ENV_ID)
            env.reset(seed=seed)
            print(f"[SCG] using {GYMN_EXT} env {ENV_ID}")
            return env
        except Exception as exc:
            print(f"[SCG] gymnasium failed: {exc}")
    if _legacy_gym is not None:
        env = _legacy_gym.make(ENV_ID)
        try:
            env.reset(seed=seed)
        except Exception:
            pass
        print(f"[SCG] using gym env {ENV_ID}")
        return env
    raise RuntimeError(
        f"safe-control-gym env '{ENV_ID}' unavailable. 请先安装 safe-control-gym，并确保 gym/gymnasium 可用。"
    )


def _metric_bucket():
    return {
        "pos_err_sq": 0.0,
        "pos_err_max": 0.0,
        "action_energy": 0.0,
        "steps": 0,
        "episodes": 0,
        "violations": 0,
        "success": 0,
    }


def _pos_err(info: Dict[str, Any]) -> float:
    for key in ("pos_err", "position_error", "track_error"):
        if isinstance(info, dict) and key in info:
            arr = np.asarray(info[key], dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                return float(np.linalg.norm(arr[:3]))
    return 0.0


def _violated(info: Dict[str, Any], action: np.ndarray) -> bool:
    if isinstance(info, dict):
        for key in ("violation", "violations", "constraint_violations"):
            if key in info:
                val = info[key]
                if isinstance(val, (int, float)):
                    return bool(val)
                if isinstance(val, (list, tuple, np.ndarray)):
                    return any(bool(x) for x in val)
    return bool(np.any(np.abs(action) > 0.99))


def run_tests():
    env = _make_env(SEED)
    adapter = SoarPolicyAdapter(
        mode=CONTROLLER_MODE,
        action_space=ACTION_SPACE,
        program_json=PROGRAM_JSON or None,
        device=DEVICE,
        normalize_motors=(ACTION_SPACE == "motors"),
    )
    stats = _metric_bucket()

    try:
        for ep in range(EPISODES):
            out = env.reset(seed=SEED + ep)
            if isinstance(out, tuple):
                obs, info = out[0], out[1]
            else:
                obs, info = out, {}
            done = False
            truncated = False
            ep_err_sq = 0.0
            ep_steps = 0
            ep_max_err = 0.0
            ep_energy = 0.0
            ep_viol = 0
            while not (done or truncated):
                action = adapter.act(obs)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, done, truncated, info = step_out
                elif len(step_out) == 4:
                    obs, reward, done, info = step_out
                    truncated = False
                else:
                    raise RuntimeError("Unexpected env.step output")
                err = _pos_err(info)
                ep_err_sq += err * err
                ep_max_err = max(ep_max_err, err)
                ep_energy += float(np.sum(action.astype(np.float64) ** 2))
                ep_viol += 1 if _violated(info, action) else 0
                ep_steps += 1
            stats["pos_err_sq"] += ep_err_sq
            stats["pos_err_max"] = max(stats["pos_err_max"], ep_max_err)
            stats["action_energy"] += ep_energy
            stats["violations"] += ep_viol
            stats["steps"] += ep_steps
            rmse_ep = math.sqrt(ep_err_sq / max(ep_steps, 1))
            stats["episodes"] += 1
            stats["success"] += 1 if rmse_ep <= SUCCESS_RMSE else 0
            print(
                f"[SCG Test] Episode {ep + 1}/{EPISODES} | RMSE={rmse_ep:.3f} m | "
                f"e_max={ep_max_err:.3f} m"
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
        adapter.close()

    mean_rmse = math.sqrt(stats["pos_err_sq"] / max(stats["steps"], 1))
    mean_energy = stats["action_energy"] / max(stats["steps"], 1)
    viol_rate = stats["violations"] / max(stats["steps"], 1)
    print("\n===== safe-control-gym Testing Summary =====")
    print(f"Env: {ENV_ID} | Controller: {CONTROLLER_MODE} | Episodes: {stats['episodes']}")
    print(f"RMSE (mean over steps): {mean_rmse:.3f} m")
    print(f"Max position error: {stats['pos_err_max']:.3f} m")
    print(f"Control effort per step: {mean_energy:.3f}")
    print(f"Violation rate: {viol_rate:.4f}")
    print(f"Success ratio: {stats['success'] / max(stats['episodes'], 1):.2%}")


def main():
    run_tests()


if __name__ == "__main__":
    main()

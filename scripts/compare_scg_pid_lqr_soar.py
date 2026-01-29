#!/usr/bin/env python3
"""在 Safe-Control-Gym 环境中对比 PID / LQR / Soar 的真实奖励。

- 环境：safe-control-gym quadrotor (TRAJ_TRACKING + figure8)
- 奖励：严格使用 SCG 论文的二次代价 r = -(state_cost + action_cost)
- 控制器：
    1) PID：使用在简化模型中调到的最优参数
    2) LQR：使用简化模型调参得到的近似 LQR 参数
    3) Soar：通过 SCG 适配器，把 DSL 程序当控制律执行

注意：该脚本只做 SCG 端的比较，不会触碰 Isaac / 训练进程。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 尝试导入 safe-control-gym
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
for candidate in (CURRENT_DIR, ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from safe_control_gym.envs.benchmark_env import Task
    from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
    from safe_control_gym.utils.registration import make as scg_make
    SCG_AVAILABLE = True
except Exception:
    SCG_AVAILABLE = False

# ---------------------------------------------------------------------------
# 导入 Soar 程序适配器（走已有的 SCG 适配路径）
# ---------------------------------------------------------------------------
try:
    from adapters.safecontrol_soar_adapter import SoarPolicyAdapter  # type: ignore
except Exception:
    SoarPolicyAdapter = None  # type: ignore


# ---------------------------------------------------------------------------
# SCG 论文的 Q, R 权重矩阵
# ---------------------------------------------------------------------------
SCG_Q_DIAG = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
SCG_R_DIAG = np.array([0.0001, 0.0001, 0.0001, 0.0001], dtype=np.float64)


def compute_scg_cost(state: np.ndarray, target_state: np.ndarray, action: np.ndarray) -> Tuple[float, float, float]:
    """计算 SCG 论文的二次代价。"""
    state = np.atleast_2d(state)
    target_state = np.atleast_2d(target_state)
    action = np.atleast_2d(action)
    err = state - target_state
    state_cost = np.sum((err ** 2) * SCG_Q_DIAG, axis=-1)
    action_cost = np.sum((action ** 2) * SCG_R_DIAG, axis=-1)
    total = state_cost + action_cost
    return float(state_cost.sum()), float(action_cost.sum()), float(total.sum())


def extract_state_from_obs(obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    """从 SCG 观测中构造 12 维状态向量。

    尽量从 info['state'] 里直接取；否则按 [pos, vel, rpy, omega] 结构猜测。
    """
    if isinstance(info, dict) and "state" in info:
        s = np.asarray(info["state"], dtype=np.float64).reshape(-1)
        if s.size >= 12:
            return s[:12]
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    state = np.zeros(12, dtype=np.float64)
    if o.size >= 12:
        state[:12] = o[:12]
    else:
        if o.size > 0:
            state[0] = o[0]
        if o.size > 1:
            state[2] = o[1]
        if o.size > 2:
            state[4] = o[2]
    return state


def get_target_state(info: Dict[str, Any]) -> np.ndarray:
    """从 info 中提取参考状态（只关心 xyz 位置）。"""
    target = np.zeros(12, dtype=np.float64)
    if not isinstance(info, dict):
        return target
    for key in ("reference", "target", "goal", "target_pos"):
        if key in info:
            v = np.asarray(info[key], dtype=np.float64).reshape(-1)
            if v.size >= 3:
                target[0] = v[0]
                target[2] = v[1]
                target[4] = v[2]
                return target
    target[4] = 1.0
    return target


class PIDControllerSCG:
    """在 SCG 状态空间上运行的 PID（使用已调好的参数）。"""

    def __init__(self):
        # 与 tune_pid_lqr.py 中 PID 最优参数保持一致
        self.kp_xy = 38.0
        self.kp_z = 70.0
        self.kd_xy = 8.0
        self.kd_z = 12.0
        self.kp_att = 60.0
        self.kd_att = 0.1
        self.att_scale = 0.6
        self.mass = 0.027
        self.g = 9.81
        self.hover_thrust = self.mass * self.g

    def compute(self, state: np.ndarray, target_state: np.ndarray) -> np.ndarray:
        pos = np.array([state[0], state[2], state[4]])
        vel = np.array([state[1], state[3], state[5]])
        rpy = np.array([state[6], state[7], state[8]])
        omega = np.array([state[9], state[10], state[11]])
        target_pos = np.array([target_state[0], target_state[2], target_state[4]])
        target_vel = np.array([target_state[1], target_state[3], target_state[5]])

        pos_err = target_pos - pos
        vel_err = target_vel - vel

        acc_des = np.array([
            self.kp_xy * pos_err[0] + self.kd_xy * vel_err[0],
            self.kp_xy * pos_err[1] + self.kd_xy * vel_err[1],
            self.kp_z * pos_err[2] + self.kd_z * vel_err[2] + self.g,
        ])

        fz = self.mass * acc_des[2]
        fz = np.clip(fz, 0.0, 2.0 * self.hover_thrust)

        roll_des = -acc_des[1] / self.g * self.att_scale
        pitch_des = acc_des[0] / self.g * self.att_scale
        rpy_des = np.array([roll_des, pitch_des, 0.0])

        rpy_err = rpy_des - rpy
        torque = self.kp_att * rpy_err - self.kd_att * omega
        torque = np.clip(torque, -0.1, 0.1)

        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float64)


class LQRControllerSCG:
    """在 SCG 状态空间上运行的 LQR 近似控制器。"""

    def __init__(self):
        # 与 tune_pid_lqr.py 中 LQR 最优参数保持一致
        self.k_pos = 75.0
        self.k_vel = 12.0
        self.k_att = 300.0
        self.k_omega = 0.7
        self.att_scale = 0.85
        self.mass = 0.027
        self.g = 9.81
        self.hover_thrust = self.mass * self.g

    def compute(self, state: np.ndarray, target_state: np.ndarray) -> np.ndarray:
        pos = np.array([state[0], state[2], state[4]])
        vel = np.array([state[1], state[3], state[5]])
        rpy = np.array([state[6], state[7], state[8]])
        omega = np.array([state[9], state[10], state[11]])
        target_pos = np.array([target_state[0], target_state[2], target_state[4]])
        target_vel = np.array([target_state[1], target_state[3], target_state[5]])

        pos_err = target_pos - pos
        vel_err = target_vel - vel

        acc_des = self.k_pos * pos_err + self.k_vel * vel_err
        acc_des[2] += self.g

        fz = self.mass * acc_des[2]
        fz = np.clip(fz, 0.0, 2.0 * self.hover_thrust)

        roll_des = -acc_des[1] / self.g * self.att_scale
        pitch_des = acc_des[0] / self.g * self.att_scale
        rpy_des = np.array([roll_des, pitch_des, 0.0])

        rpy_err = rpy_des - rpy
        torque = self.k_att * rpy_err - self.k_omega * omega
        torque = np.clip(torque, -0.1, 0.1)

        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float64)


def make_scg_env(duration: float, seed: int = 0):
    """创建 figure8 轨迹的 SCG quadrotor 环境。"""
    if not SCG_AVAILABLE:
        raise RuntimeError("safe-control-gym 未安装，无法创建 SCG 环境")
    make_kwargs = dict(
        task=Task.TRAJ_TRACKING,
        task_info={
            "trajectory_type": "figure8",
            "stabilization_goal": [0.0, 1.0],
            "num_cycles": 1,
            "trajectory_plane": "zx",
            "trajectory_position_offset": [0.5, 0.0],
            "trajectory_scale": -0.5,
        },
        ctrl_freq=48,
        pyb_freq=240,
        episode_len_sec=duration,
        quad_type=QuadType.TWO_D,
        gui=False,
    )

    try:
        env = scg_make("quadrotor", **make_kwargs)
    except TypeError:
        # 部分旧版 SCG 需要 task_info 以外的 kw 或不支持某些 kw；此时回退去掉 pyb_freq 之类
        fallback_kwargs = make_kwargs.copy()
        fallback_kwargs.pop("pyb_freq", None)
        env = scg_make("quadrotor", **fallback_kwargs)
    env.seed(seed)
    return env


def reset_env_with_seed(env, seed: int):
    """Gym reset helper兼容老接口。"""
    try:
        out = env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        out = env.reset()
    if isinstance(out, tuple):
        obs = out[0]
        info = out[1] if len(out) > 1 else {}
    else:
        obs, info = out, {}
    return obs, info


def format_action_for_env(raw_action: np.ndarray, env) -> np.ndarray:
    """将 (fz, τ) 动作转成 SCG 环境需要的力输入。"""
    raw = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    if not hasattr(env, "action_space") or env.action_space.shape is None:
        return raw
    action_dim = int(np.prod(env.action_space.shape))
    if action_dim == raw.size:
        return raw
    if action_dim == 2 and raw.size >= 3:
        fz = raw[0]
        tau_pitch = raw[2] if raw.size >= 3 else 0.0
        arm = float(getattr(env, "L", 0.04))
        arm = max(arm, 1e-3)
        half = 0.5 * fz
        delta = tau_pitch / (2.0 * arm)
        t1 = half - delta
        t2 = half + delta
        return np.array([t1, t2], dtype=np.float64)
    if action_dim == 1:
        return np.array([raw[0]], dtype=np.float64)
    padded = np.zeros(action_dim, dtype=np.float64)
    padded[: min(raw.size, action_dim)] = raw[: min(raw.size, action_dim)]
    return padded


def run_controller_in_scg(name: str, controller, episodes: int, duration: float, seed: int = 0) -> Dict[str, float]:
    """在 SCG env 中运行给定控制器，返回真实奖励统计。"""
    env = make_scg_env(duration, seed=seed)
    ctrl_freq = 48.0
    max_steps = int(duration * ctrl_freq)

    rewards = []
    state_costs = []
    action_costs = []

    for ep in range(episodes):
        obs, info = reset_env_with_seed(env, seed + ep)

        done = False
        truncated = False
        total_state_cost = 0.0
        total_action_cost = 0.0

        for _ in range(max_steps):
            if done or truncated:
                break
            state = extract_state_from_obs(obs, info)
            target_state = get_target_state(info)
            raw_action = controller.compute(state, target_state)
            sc, ac, _ = compute_scg_cost(state, target_state, raw_action)
            total_state_cost += sc
            total_action_cost += ac

            env_action = format_action_for_env(raw_action, env)
            step_out = env.step(env_action)
            if len(step_out) == 5:
                obs, reward, done, truncated, info = step_out
            elif len(step_out) == 4:
                obs, reward, done, info = step_out
                truncated = False
            else:
                raise RuntimeError("Unexpected env.step output from SCG")

        true_reward = -(total_state_cost + total_action_cost)
        rewards.append(true_reward)
        state_costs.append(total_state_cost)
        action_costs.append(total_action_cost)
        print(f"[{name}] Episode {ep+1}/{episodes}: true_reward={true_reward:.4f}, state={total_state_cost:.4f}, action={total_action_cost:.6f}")

    env.close()

    rewards_arr = np.array(rewards, dtype=np.float64)
    state_arr = np.array(state_costs, dtype=np.float64)
    action_arr = np.array(action_costs, dtype=np.float64)
    return {
        "mean_true_reward": float(rewards_arr.mean()),
        "std_true_reward": float(rewards_arr.std()),
        "mean_state_cost": float(state_arr.mean()),
        "mean_action_cost": float(action_arr.mean()),
    }


def run_soar_in_scg(program_json: Path, episodes: int, duration: float, seed: int = 0) -> Dict[str, float]:
    """使用 SCG 适配器运行已有的 Soar 程序（若适配器可用）。

    注意：这里直接使用 SCG 的原生 reward（环境返回），同时用 compute_scg_cost 复算一份，
    方便与 PID/LQR 完全对齐。
    """
    if SoarPolicyAdapter is None:
        raise RuntimeError("SCG 适配器未就绪，无法在 SCG 内运行 Soar 程序")

    env = make_scg_env(duration, seed=seed)
    adapter = SoarPolicyAdapter(
        mode="soar",
        action_space="thrust_torque",
        program_json=str(program_json),
        device="cpu",
        normalize_motors=False,
    )

    ctrl_freq = 48.0
    max_steps = int(duration * ctrl_freq)

    rewards_env = []
    rewards_true = []
    state_costs = []
    action_costs = []

    for ep in range(episodes):
        obs, info = reset_env_with_seed(env, seed + ep)

        done = False
        truncated = False
        total_state_cost = 0.0
        total_action_cost = 0.0
        total_env_reward = 0.0

        for _ in range(max_steps):
            if done or truncated:
                break
            raw_action = adapter.act(obs)
            state = extract_state_from_obs(obs, info)
            target_state = get_target_state(info)
            sc, ac, _ = compute_scg_cost(state, target_state, raw_action)
            total_state_cost += sc
            total_action_cost += ac

            env_action = format_action_for_env(raw_action, env)
            step_out = env.step(env_action)
            if len(step_out) == 5:
                obs, reward, done, truncated, info = step_out
            elif len(step_out) == 4:
                obs, reward, done, info = step_out
                truncated = False
            else:
                raise RuntimeError("Unexpected env.step output from SCG")
            total_env_reward += float(reward)

        true_reward = -(total_state_cost + total_action_cost)
        rewards_true.append(true_reward)
        rewards_env.append(total_env_reward)
        state_costs.append(total_state_cost)
        action_costs.append(total_action_cost)
        print(f"[Soar] Episode {ep+1}/{episodes}: true_reward={true_reward:.4f}, env_reward={total_env_reward:.4f}, state={total_state_cost:.4f}, action={total_action_cost:.6f}")

    env.close()
    adapter.close()

    def _summ(x):
        arr = np.array(x, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    mean_true, std_true = _summ(rewards_true)
    mean_env, std_env = _summ(rewards_env)
    mean_state, _ = _summ(state_costs)
    mean_action, _ = _summ(action_costs)

    return {
        "mean_true_reward": mean_true,
        "std_true_reward": std_true,
        "mean_state_cost": mean_state,
        "mean_action_cost": mean_action,
        "mean_env_reward": mean_env,
        "std_env_reward": std_env,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在 SCG 中对比 PID / LQR / Soar 的真实奖励")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--soar-json", type=str, default="results/scg_aligned/figure8_safe_control_tracking_best.json")
    parser.add_argument("--output", type=str, default="results/scg_compare_pid_lqr_soar.json")
    args = parser.parse_args()

    if not SCG_AVAILABLE:
        raise SystemExit("safe-control-gym 未安装，无法运行对比")

    print("================ SCG 对比：PID / LQR / Soar ================")
    print(f"episodes={args.episodes}, duration={args.duration}s, seed={args.seed}")

    pid_ctrl = PIDControllerSCG()
    lqr_ctrl = LQRControllerSCG()

    pid_stats = run_controller_in_scg("PID", pid_ctrl, args.episodes, args.duration, seed=args.seed)
    lqr_stats = run_controller_in_scg("LQR", lqr_ctrl, args.episodes, args.duration, seed=args.seed)

    soar_path = Path(args.soar_json)
    if soar_path.is_file() and SoarPolicyAdapter is not None:
        soar_stats = run_soar_in_scg(soar_path, args.episodes, args.duration, seed=args.seed)
    else:
        print("[Warning] 找不到 Soar 程序或 SCG 适配器不可用，只对比 PID/LQR。")
        soar_stats = None

    print("\n================ 汇总 (SCG + SCG cost) ================")
    def _fmt_row(name: str, s: Dict[str, float] | None) -> str:
        if s is None:
            return f"{name:<10} {'-':<12} {'-':<12} {'-':<12}"
        return (
            f"{name:<10} "
            f"{s['mean_true_reward']:<12.4f} "
            f"{s['mean_state_cost']:<12.4f} "
            f"{s['mean_action_cost']:<12.6f}"
        )

    print(f"{'Controller':<10} {'true_reward':<12} {'state_cost':<12} {'action_cost':<12}")
    print("-" * 55)
    print(_fmt_row("PID", pid_stats))
    print(_fmt_row("LQR", lqr_stats))
    print(_fmt_row("Soar", soar_stats))

    out = {
        "episodes": args.episodes,
        "duration": args.duration,
        "seed": args.seed,
        "pid": pid_stats,
        "lqr": lqr_stats,
        "soar": soar_stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()

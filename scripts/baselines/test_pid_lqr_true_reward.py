#!/usr/bin/env python3
"""测试 PID 和 LQR 控制器在 Safe-Control-Gym 环境中的真实奖励。

真实奖励 = -(state_cost + action_cost)，即 SCG 论文中的二次代价。
只使用 SCG 原生环境，不经过 Isaac Gym。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity
except ImportError as exc:
    raise ImportError(f"无法导入 trajectory_presets: {exc}")

# ---------------------------------------------------------------------------
# 尝试导入 safe-control-gym
# ---------------------------------------------------------------------------
try:
    from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
    from safe_control_gym.utils.registration import make as scg_make
    from safe_control_gym.utils.configuration import ConfigFactory
    SCG_AVAILABLE = True
except ImportError:
    SCG_AVAILABLE = False
    print("[Warning] safe-control-gym 未安装，将使用简化模拟")

# ---------------------------------------------------------------------------
# SCG 论文的 Q, R 权重矩阵（与 reward_scg_exact.py 保持一致）
# ---------------------------------------------------------------------------
# 状态: [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r] (12维)
SCG_Q_DIAG = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
SCG_R_DIAG = np.array([0.0001, 0.0001, 0.0001, 0.0001], dtype=np.float64)  # 4 个控制输入


def compute_scg_cost(
    state: np.ndarray,
    target_state: np.ndarray,
    action: np.ndarray,
) -> Tuple[float, float, float]:
    """计算 SCG 论文的二次代价。
    
    Args:
        state: 当前状态 [12] 或 [N, 12]
        target_state: 目标状态 [12] 或 [N, 12]
        action: 控制输入 [4] 或 [N, 4]
    
    Returns:
        (state_cost, action_cost, total_cost)
    """
    state = np.atleast_2d(state)
    target_state = np.atleast_2d(target_state)
    action = np.atleast_2d(action)
    
    # 状态误差
    state_err = state - target_state  # [N, 12]
    # 状态代价: err^T Q err
    state_cost = np.sum((state_err ** 2) * SCG_Q_DIAG, axis=-1)  # [N]
    # 控制代价: u^T R u
    action_cost = np.sum((action ** 2) * SCG_R_DIAG, axis=-1)  # [N]
    
    total_cost = state_cost + action_cost
    
    return float(state_cost.sum()), float(action_cost.sum()), float(total_cost.sum())


def create_scg_env(task: str = "tracking", duration: float = 10.0, seed: int = 0):
    """创建 Safe-Control-Gym 环境。
    
    注意：SCG 只支持 ONE_D 和 TWO_D 四旋翼，不支持完整 3D。
    这里使用 TWO_D 作为近似测试。
    """
    if not SCG_AVAILABLE:
        raise RuntimeError("safe-control-gym 未安装")
    
    # 根据任务类型设置轨迹
    if task == "hover":
        task_config = {
            "task": "STABILIZATION",
            "task_info": {
                "stabilization_goal": [0.0, 1.0],  # 2D: [x, z]
            }
        }
    else:  # tracking / figure8
        task_config = {
            "task": "TRAJ_TRACKING",
            "task_info": {
                "trajectory_type": "figure8" if task in ("figure8", "tracking") else task,
            }
        }
    
    # 创建环境 - 使用 TWO_D（SCG 不支持 THREE_D）
    env = scg_make(
        "quadrotor",
        **{
            "ctrl_freq": 48,
            "pyb_freq": 240,
            "episode_len_sec": duration,
            "quad_type": QuadType.TWO_D,
            "gui": False,
            "done_on_out_of_bound": False,
            **task_config,
        }
    )
    env.seed(seed)
    return env


class SimplePIDController:
    """简单 PD 控制器（位置 + 姿态）。"""
    
    def __init__(
        self,
        kp_pos: np.ndarray = np.array([5.0, 5.0, 8.0]),
        kd_pos: np.ndarray = np.array([3.0, 3.0, 5.0]),
        kp_att: np.ndarray = np.array([10.0, 10.0, 5.0]),
        kd_att: np.ndarray = np.array([1.0, 1.0, 0.5]),
        mass: float = 0.027,
        g: float = 9.81,
    ):
        self.kp_pos = np.array(kp_pos, dtype=np.float64)
        self.kd_pos = np.array(kd_pos, dtype=np.float64)
        self.kp_att = np.array(kp_att, dtype=np.float64)
        self.kd_att = np.array(kd_att, dtype=np.float64)
        self.mass = mass
        self.g = g
        self.hover_thrust = mass * g
    
    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        rpy: np.ndarray,
        omega: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray = None,
    ) -> np.ndarray:
        """计算推力和力矩 [fz, tx, ty, tz]。"""
        if target_vel is None:
            target_vel = np.zeros(3)
        
        # 位置误差
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        
        # 期望加速度
        acc_des = self.kp_pos * pos_err + self.kd_pos * vel_err
        acc_des[2] += self.g  # 补偿重力
        
        # 推力
        fz = self.mass * acc_des[2]
        fz = np.clip(fz, 0.0, 2.0 * self.hover_thrust)
        
        # 期望姿态（简化：直接用位置误差映射）
        roll_des = -acc_des[1] / self.g * 0.1
        pitch_des = acc_des[0] / self.g * 0.1
        yaw_des = 0.0
        rpy_des = np.array([roll_des, pitch_des, yaw_des])
        
        # 姿态误差
        rpy_err = rpy_des - rpy
        
        # 力矩
        torque = self.kp_att * rpy_err - self.kd_att * omega
        torque = np.clip(torque, -0.1, 0.1)
        
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float64)


class SimpleLQRController:
    """简化 LQR 控制器（线性化后的 PD 形式）。
    
    实际 LQR 需要解 Riccati 方程，这里用预设增益近似。
    """
    
    def __init__(
        self,
        k_pos: np.ndarray = np.array([3.16, 3.16, 3.16]),  # sqrt(10)
        k_vel: np.ndarray = np.array([3.16, 3.16, 3.16]),
        k_att: np.ndarray = np.array([10.0, 10.0, 10.0]),
        k_omega: np.ndarray = np.array([3.16, 3.16, 3.16]),
        mass: float = 0.027,
        g: float = 9.81,
    ):
        self.k_pos = np.array(k_pos, dtype=np.float64)
        self.k_vel = np.array(k_vel, dtype=np.float64)
        self.k_att = np.array(k_att, dtype=np.float64)
        self.k_omega = np.array(k_omega, dtype=np.float64)
        self.mass = mass
        self.g = g
        self.hover_thrust = mass * g
    
    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        rpy: np.ndarray,
        omega: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray = None,
    ) -> np.ndarray:
        """计算控制输入。"""
        if target_vel is None:
            target_vel = np.zeros(3)
        
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        
        # LQR 形式的位置控制
        acc_des = self.k_pos * pos_err + self.k_vel * vel_err
        acc_des[2] += self.g
        
        fz = self.mass * acc_des[2]
        fz = np.clip(fz, 0.0, 2.0 * self.hover_thrust)
        
        # 期望姿态
        roll_des = -acc_des[1] / self.g * 0.1
        pitch_des = acc_des[0] / self.g * 0.1
        rpy_des = np.array([roll_des, pitch_des, 0.0])
        
        rpy_err = rpy_des - rpy
        torque = self.k_att * rpy_err - self.k_omega * omega
        torque = np.clip(torque, -0.1, 0.1)
        
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float64)


def extract_state_from_obs(obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    """从观测中提取 12 维状态向量。
    
    SCG 状态: [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
    """
    # 尝试从 info 中获取完整状态
    if "state" in info:
        state = np.array(info["state"], dtype=np.float64)
        if state.shape[-1] >= 12:
            return state[:12]
    
    # 否则从 obs 中解析
    # 假设 obs 结构: [pos(3), vel(3), rpy(3), omega(3), ...]
    obs = np.asarray(obs, dtype=np.float64).flatten()
    if len(obs) >= 12:
        pos = obs[0:3]
        vel = obs[3:6]
        rpy = obs[6:9]
        omega = obs[9:12]
        # 转换为 SCG 格式
        state = np.array([
            pos[0], vel[0],  # x, x_dot
            pos[1], vel[1],  # y, y_dot
            pos[2], vel[2],  # z, z_dot
            rpy[0], rpy[1], rpy[2],  # phi, theta, psi
            omega[0], omega[1], omega[2],  # p, q, r
        ], dtype=np.float64)
        return state
    
    # 最小化：只有位置
    state = np.zeros(12, dtype=np.float64)
    state[0] = obs[0] if len(obs) > 0 else 0.0  # x
    state[2] = obs[1] if len(obs) > 1 else 0.0  # y
    state[4] = obs[2] if len(obs) > 2 else 0.0  # z
    return state


def get_target_state(info: Dict[str, Any], default_pos: np.ndarray = None) -> np.ndarray:
    """获取目标状态。"""
    target = np.zeros(12, dtype=np.float64)
    
    # 尝试从 info 获取
    for key in ("target", "target_pos", "reference", "goal"):
        if key in info:
            val = np.asarray(info[key], dtype=np.float64).flatten()
            if len(val) >= 3:
                target[0] = val[0]  # x
                target[2] = val[1]  # y
                target[4] = val[2]  # z
                return target
    
    # 默认位置
    if default_pos is not None:
        target[0] = default_pos[0]
        target[2] = default_pos[1]
        target[4] = default_pos[2]
    else:
        target[4] = 1.0  # 默认悬停高度 1m
    
    return target


def run_episode(
    env,
    controller,
    duration: float,
    ctrl_freq: float = 48.0,
) -> Dict[str, Any]:
    """运行一个 episode，收集真实奖励。"""
    max_steps = int(duration * ctrl_freq)
    
    out = env.reset()
    if isinstance(out, tuple):
        obs, info = out[0], out[1] if len(out) > 1 else {}
    else:
        obs, info = out, {}
    
    total_state_cost = 0.0
    total_action_cost = 0.0
    total_reward = 0.0
    steps = 0
    pos_errors = []
    
    done = False
    truncated = False
    
    for step in range(max_steps):
        if done or truncated:
            break
        
        # 提取状态
        state = extract_state_from_obs(obs, info)
        target_state = get_target_state(info, default_pos=np.array([0.0, 0.0, 1.0]))
        
        # 提取位置和速度用于控制器
        pos = np.array([state[0], state[2], state[4]])
        vel = np.array([state[1], state[3], state[5]])
        rpy = np.array([state[6], state[7], state[8]])
        omega = np.array([state[9], state[10], state[11]])
        target_pos = np.array([target_state[0], target_state[2], target_state[4]])
        
        # 计算控制输入
        action = controller.compute(pos, vel, rpy, omega, target_pos)
        
        # 计算 SCG 代价
        state_cost, action_cost, _ = compute_scg_cost(state, target_state, action)
        total_state_cost += state_cost
        total_action_cost += action_cost
        
        # 位置误差
        pos_err = np.linalg.norm(pos - target_pos)
        pos_errors.append(pos_err)
        
        # 环境步进
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, done, truncated, info = step_out
        elif len(step_out) == 4:
            obs, reward, done, info = step_out
            truncated = False
        else:
            raise RuntimeError(f"Unexpected env.step output: {len(step_out)} elements")
        
        total_reward += reward
        steps += 1
    
    # 真实奖励 = -(state_cost + action_cost)
    true_reward = -(total_state_cost + total_action_cost)
    
    return {
        "steps": steps,
        "state_cost": total_state_cost,
        "action_cost": total_action_cost,
        "true_reward": true_reward,
        "env_reward": total_reward,
        "mean_pos_err": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "max_pos_err": float(np.max(pos_errors)) if pos_errors else 0.0,
        "rmse": float(np.sqrt(np.mean(np.array(pos_errors) ** 2))) if pos_errors else 0.0,
    }


def test_controller(
    controller_type: str,
    task: str = "figure8",
    duration: float = 5.0,
    episodes: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """测试指定控制器。"""
    print(f"\n{'='*60}")
    print(f"测试 {controller_type.upper()} 控制器 | 任务: {task} | 时长: {duration}s")
    print(f"{'='*60}")
    
    # 创建控制器
    if controller_type.lower() == "pid":
        controller = SimplePIDController()
    elif controller_type.lower() == "lqr":
        controller = SimpleLQRController()
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    results = []
    
    # 如果 SCG 可用，使用真实环境
    if SCG_AVAILABLE:
        try:
            env = create_scg_env(task=task, duration=duration, seed=seed)
            for ep in range(episodes):
                env.seed(seed + ep)
                ep_result = run_episode(env, controller, duration)
                results.append(ep_result)
                print(
                    f"  Episode {ep+1}/{episodes}: "
                    f"true_reward={ep_result['true_reward']:.4f} "
                    f"(state={ep_result['state_cost']:.4f}, action={ep_result['action_cost']:.6f}) | "
                    f"RMSE={ep_result['rmse']:.4f}m"
                )
            env.close()
        except Exception as e:
            print(f"[Warning] SCG 环境运行失败: {e}")
            print("  使用模拟测试...")
            results = _run_simulated_test(controller, task, duration, episodes, seed)
    else:
        print("  [Warning] SCG 不可用，使用模拟测试")
        results = _run_simulated_test(controller, task, duration, episodes, seed)
    
    # 汇总统计
    if results:
        mean_true_reward = np.mean([r["true_reward"] for r in results])
        std_true_reward = np.std([r["true_reward"] for r in results])
        mean_state_cost = np.mean([r["state_cost"] for r in results])
        mean_action_cost = np.mean([r["action_cost"] for r in results])
        mean_rmse = np.mean([r["rmse"] for r in results])
        
        summary = {
            "controller": controller_type,
            "task": task,
            "duration": duration,
            "episodes": len(results),
            "mean_true_reward": float(mean_true_reward),
            "std_true_reward": float(std_true_reward),
            "mean_state_cost": float(mean_state_cost),
            "mean_action_cost": float(mean_action_cost),
            "mean_rmse": float(mean_rmse),
            "episode_results": results,
        }
        
        print(f"\n--- {controller_type.upper()} 汇总 ---")
        print(f"  真实奖励: {mean_true_reward:.4f} ± {std_true_reward:.4f}")
        print(f"  状态代价: {mean_state_cost:.4f}")
        print(f"  控制代价: {mean_action_cost:.6f}")
        print(f"  平均RMSE: {mean_rmse:.4f}m")
        
        return summary
    
    return {}


def _run_simulated_test(
    controller,
    task: str,
    duration: float,
    episodes: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """简化模拟测试（无物理引擎）。"""
    np.random.seed(seed)
    results = []
    ctrl_freq = 48.0
    dt = 1.0 / ctrl_freq
    max_steps = int(duration * ctrl_freq)
    
    traj_cfg = get_scg_trajectory_config(task)
    center = np.array(traj_cfg.center, dtype=np.float64)

    for ep in range(episodes):
        # 初始状态
        pos = np.array([0.0, 0.0, 0.0])
        vel = np.zeros(3)
        rpy = np.zeros(3)
        omega = np.zeros(3)
        
        total_state_cost = 0.0
        total_action_cost = 0.0
        pos_errors = []
        
        for step in range(max_steps):
            t = step * dt
            target_pos, _ = scg_position_velocity(task, t, params=traj_cfg.params, center=center)
            
            # 控制
            action = controller.compute(pos, vel, rpy, omega, target_pos)
            
            # 构建状态向量
            state = np.array([
                pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
                rpy[0], rpy[1], rpy[2], omega[0], omega[1], omega[2]
            ])
            target_state = np.array([
                target_pos[0], 0.0, target_pos[1], 0.0, target_pos[2], 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
            
            # 计算代价
            sc, ac, _ = compute_scg_cost(state, target_state, action)
            total_state_cost += sc
            total_action_cost += ac
            pos_errors.append(np.linalg.norm(pos - target_pos))
            
            # 简化动力学更新
            mass = 0.027
            g = 9.81
            acc = np.array([0.0, 0.0, action[0] / mass - g])
            vel += acc * dt
            pos += vel * dt
            # 姿态简化
            rpy += omega * dt
            omega += action[1:4] * 10.0 * dt  # 简化力矩到角加速度
        
        true_reward = -(total_state_cost + total_action_cost)
        results.append({
            "steps": max_steps,
            "state_cost": total_state_cost,
            "action_cost": total_action_cost,
            "true_reward": true_reward,
            "env_reward": 0.0,
            "mean_pos_err": float(np.mean(pos_errors)),
            "max_pos_err": float(np.max(pos_errors)),
            "rmse": float(np.sqrt(np.mean(np.array(pos_errors) ** 2))),
        })
        print(
            f"  Episode {ep+1}/{episodes} (sim): "
            f"true_reward={true_reward:.4f} "
            f"(state={total_state_cost:.4f}, action={total_action_cost:.6f}) | "
            f"RMSE={results[-1]['rmse']:.4f}m"
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="测试 PID/LQR 真实奖励")
    parser.add_argument("--task", default="figure8", choices=["hover", "figure8", "circle", "square", "helix"])
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="", help="保存结果 JSON")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PID / LQR 真实奖励测试")
    print(f"任务: {args.task} | 时长: {args.duration}s | Episodes: {args.episodes}")
    print("=" * 60)
    
    # 测试 PID
    pid_results = test_controller(
        "pid",
        task=args.task,
        duration=args.duration,
        episodes=args.episodes,
        seed=args.seed,
    )
    
    # 测试 LQR
    lqr_results = test_controller(
        "lqr",
        task=args.task,
        duration=args.duration,
        episodes=args.episodes,
        seed=args.seed,
    )
    
    # 对比总结
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"{'控制器':<10} {'真实奖励':<20} {'状态代价':<15} {'控制代价':<15} {'RMSE(m)':<10}")
    print("-" * 70)
    if pid_results:
        print(f"{'PID':<10} {pid_results['mean_true_reward']:<20.4f} {pid_results['mean_state_cost']:<15.4f} {pid_results['mean_action_cost']:<15.6f} {pid_results['mean_rmse']:<10.4f}")
    if lqr_results:
        print(f"{'LQR':<10} {lqr_results['mean_true_reward']:<20.4f} {lqr_results['mean_state_cost']:<15.4f} {lqr_results['mean_action_cost']:<15.6f} {lqr_results['mean_rmse']:<10.4f}")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_results = {
            "task": args.task,
            "duration": args.duration,
            "episodes": args.episodes,
            "pid": pid_results,
            "lqr": lqr_results,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

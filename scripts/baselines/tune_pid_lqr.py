#!/usr/bin/env python3
"""调优 PID 和 LQR 控制器参数，最大化真实奖励（最小化 SCG 代价）。

使用网格搜索 + 局部优化找到最佳参数组合。
"""

from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import time

import numpy as np

try:
    from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity
except ImportError as exc:
    raise ImportError(f"无法导入 trajectory_presets: {exc}")

# ---------------------------------------------------------------------------
# SCG 论文的 Q, R 权重矩阵
# ---------------------------------------------------------------------------
# 状态: [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r] (12维)
# 只有位置 x, y, z 有权重
SCG_Q_DIAG = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
SCG_R_DIAG = np.array([0.0001, 0.0001, 0.0001, 0.0001], dtype=np.float64)


def compute_scg_cost(
    state: np.ndarray,
    target_state: np.ndarray,
    action: np.ndarray,
) -> Tuple[float, float, float]:
    """计算 SCG 论文的二次代价。"""
    state = np.atleast_2d(state)
    target_state = np.atleast_2d(target_state)
    action = np.atleast_2d(action)
    
    state_err = state - target_state
    state_cost = np.sum((state_err ** 2) * SCG_Q_DIAG, axis=-1)
    action_cost = np.sum((action ** 2) * SCG_R_DIAG, axis=-1)
    total_cost = state_cost + action_cost
    
    return float(state_cost.sum()), float(action_cost.sum()), float(total_cost.sum())



@dataclass
class PIDParams:
    """PID 控制器参数"""
    kp_xy: float = 5.0
    kp_z: float = 8.0
    kd_xy: float = 3.0
    kd_z: float = 5.0
    kp_att: float = 10.0
    kd_att: float = 1.0
    att_scale: float = 0.1  # 位置误差到姿态的映射比例
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "kp_xy": self.kp_xy,
            "kp_z": self.kp_z,
            "kd_xy": self.kd_xy,
            "kd_z": self.kd_z,
            "kp_att": self.kp_att,
            "kd_att": self.kd_att,
            "att_scale": self.att_scale,
        }


@dataclass  
class LQRParams:
    """LQR 控制器参数"""
    k_pos: float = 3.16
    k_vel: float = 3.16
    k_att: float = 10.0
    k_omega: float = 3.16
    att_scale: float = 0.1
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "k_pos": self.k_pos,
            "k_vel": self.k_vel,
            "k_att": self.k_att,
            "k_omega": self.k_omega,
            "att_scale": self.att_scale,
        }


class TunablePIDController:
    """可调参数的 PID 控制器"""
    
    def __init__(self, params: PIDParams, mass: float = 0.027, g: float = 9.81):
        self.params = params
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
        if target_vel is None:
            target_vel = np.zeros(3)
        
        p = self.params
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        
        # 位置控制
        acc_des = np.array([
            p.kp_xy * pos_err[0] + p.kd_xy * vel_err[0],
            p.kp_xy * pos_err[1] + p.kd_xy * vel_err[1],
            p.kp_z * pos_err[2] + p.kd_z * vel_err[2] + self.g,
        ])
        
        # 推力
        fz = self.mass * acc_des[2]
        fz = np.clip(fz, 0.0, 2.0 * self.hover_thrust)
        
        # 期望姿态
        roll_des = -acc_des[1] / self.g * p.att_scale
        pitch_des = acc_des[0] / self.g * p.att_scale
        rpy_des = np.array([roll_des, pitch_des, 0.0])
        
        # 姿态控制
        rpy_err = rpy_des - rpy
        torque = p.kp_att * rpy_err - p.kd_att * omega
        torque = np.clip(torque, -0.1, 0.1)
        
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float64)


class TunableLQRController:
    """可调参数的 LQR 控制器"""
    
    def __init__(self, params: LQRParams, mass: float = 0.027, g: float = 9.81):
        self.params = params
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
        if target_vel is None:
            target_vel = np.zeros(3)
        
        p = self.params
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        
        # LQR 形式
        acc_des = p.k_pos * pos_err + p.k_vel * vel_err
        acc_des[2] += self.g
        
        fz = self.mass * acc_des[2]
        fz = np.clip(fz, 0.0, 2.0 * self.hover_thrust)
        
        roll_des = -acc_des[1] / self.g * p.att_scale
        pitch_des = acc_des[0] / self.g * p.att_scale
        rpy_des = np.array([roll_des, pitch_des, 0.0])
        
        rpy_err = rpy_des - rpy
        torque = p.k_att * rpy_err - p.k_omega * omega
        torque = np.clip(torque, -0.1, 0.1)
        
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float64)


def simulate_episode(
    controller,
    task: str = "figure8",
    duration: float = 5.0,
    ctrl_freq: float = 48.0,
) -> Dict[str, float]:
    """模拟一个 episode，返回代价指标。"""
    dt = 1.0 / ctrl_freq
    max_steps = int(duration * ctrl_freq)
    mass = 0.027
    g = 9.81
    
    # 初始状态
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.zeros(3)
    rpy = np.zeros(3)
    omega = np.zeros(3)
    
    total_state_cost = 0.0
    total_action_cost = 0.0
    pos_errors = []
    
    traj_cfg = get_scg_trajectory_config(task)
    center = np.array(traj_cfg.center, dtype=np.float64)

    for step in range(max_steps):
        t = step * dt
        target_pos, target_vel = scg_position_velocity(task, t, params=traj_cfg.params, center=center)
        
        # 控制输入
        action = controller.compute(pos, vel, rpy, omega, target_pos, target_vel)

        pos_err_vec = target_pos - pos
        vel_err_vec = target_vel - vel
        rpy_err_vec = -rpy  # target姿态恒为0
        omega_err_vec = -omega

        pos_errors.append(float(np.linalg.norm(pos_err_vec)))
        
        # 构建状态向量
        state = np.array([
            pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
            rpy[0], rpy[1], rpy[2], omega[0], omega[1], omega[2]
        ])
        sin_roll = np.sin(rpy[0])
        sin_pitch = np.sin(rpy[1])
        cos_roll = np.cos(rpy[0])
        cos_pitch = np.cos(rpy[1])
        
        # SCG 风格状态/动作代价（积分形式）
        state_cost = (
            float(pos_err_vec @ pos_err_vec)
            + 0.1 * float(vel_err_vec @ vel_err_vec)
            + 0.5 * float(rpy_err_vec @ rpy_err_vec)
            + 0.05 * float(omega_err_vec @ omega_err_vec)
        )
        action_cost = 1e-3 * float(action @ action)
        total_state_cost += state_cost * dt
        total_action_cost += action_cost * dt

        # 推力在世界坐标系的分解
        thrust_world = action[0] / mass * np.array([
            sin_pitch * cos_roll,
            -sin_roll,
            cos_pitch * cos_roll
        ])
        
        acc = thrust_world - np.array([0.0, 0.0, g])
        vel += acc * dt
        pos += vel * dt
        
        # 姿态动力学（简化）
        # 力矩产生角加速度（假设单位惯性矩阵）
        inertia = np.array([1e-5, 1e-5, 2e-5])  # 典型四旋翼惯量
        alpha = action[1:4] / inertia
        omega += alpha * dt
        omega = np.clip(omega, -10.0, 10.0)  # 限制角速度
        rpy += omega * dt
        rpy = np.clip(rpy, -0.5, 0.5)  # 限制姿态角
    
    true_reward = -(total_state_cost + total_action_cost)
    
    return {
        "state_cost": total_state_cost,
        "action_cost": total_action_cost,
        "true_reward": true_reward,
        "rmse": float(np.sqrt(np.mean(np.array(pos_errors) ** 2))),
        "mean_pos_err": float(np.mean(pos_errors)),
        "max_pos_err": float(np.max(pos_errors)),
    }


def evaluate_pid_params(params: PIDParams, task: str, duration: float, n_episodes: int = 3) -> float:
    """评估 PID 参数，返回平均真实奖励。"""
    controller = TunablePIDController(params)
    rewards = []
    for _ in range(n_episodes):
        result = simulate_episode(controller, task, duration)
        rewards.append(result["true_reward"])
    return np.mean(rewards)


def evaluate_lqr_params(params: LQRParams, task: str, duration: float, n_episodes: int = 3) -> float:
    """评估 LQR 参数，返回平均真实奖励。"""
    controller = TunableLQRController(params)
    rewards = []
    for _ in range(n_episodes):
        result = simulate_episode(controller, task, duration)
        rewards.append(result["true_reward"])
    return np.mean(rewards)


def grid_search_pid(task: str, duration: float) -> Tuple[PIDParams, float]:
    """网格搜索 PID 最佳参数（精简版，降低算力消耗）。"""
    print("\n[PID] 开始网格搜索（精简）...")
    
    # 搜索范围（精简：每参数 3-4 个值）
    kp_xy_range = [4.0, 8.0, 15.0]
    kp_z_range = [8.0, 14.0, 20.0]
    kd_xy_range = [2.0, 4.0, 6.0]
    kd_z_range = [4.0, 6.0]
    kp_att_range = [10.0, 20.0, 30.0]
    kd_att_range = [1.0, 2.0]
    att_scale_range = [0.1, 0.2]
    
    best_params = PIDParams()
    best_reward = float("-inf")
    total = len(kp_xy_range) * len(kp_z_range) * len(kd_xy_range) * len(kd_z_range) * len(kp_att_range) * len(kd_att_range) * len(att_scale_range)
    
    # 粗搜索：先固定一些参数
    print("  阶段1: 粗搜索位置增益...")
    for kp_xy in kp_xy_range:
        for kp_z in kp_z_range:
            for kd_xy in [2.0, 4.0]:
                for kd_z in [4.0, 6.0]:
                    params = PIDParams(kp_xy=kp_xy, kp_z=kp_z, kd_xy=kd_xy, kd_z=kd_z)
                    reward = evaluate_pid_params(params, task, duration, n_episodes=1)
                    if reward > best_reward:
                        best_reward = reward
                        best_params = params
    
    print(f"    粗搜索最佳: kp_xy={best_params.kp_xy}, kp_z={best_params.kp_z}, reward={best_reward:.4f}")
    
    # 细搜索：在最佳位置增益附近搜索
    print("  阶段2: 细搜索阻尼和姿态增益...")
    base_kp_xy = best_params.kp_xy
    base_kp_z = best_params.kp_z
    
    for kd_xy in kd_xy_range:
        for kd_z in kd_z_range:
            for kp_att in kp_att_range:
                for kd_att in kd_att_range:
                    for att_scale in att_scale_range:
                        params = PIDParams(
                            kp_xy=base_kp_xy, kp_z=base_kp_z,
                            kd_xy=kd_xy, kd_z=kd_z,
                            kp_att=kp_att, kd_att=kd_att,
                            att_scale=att_scale
                        )
                        reward = evaluate_pid_params(params, task, duration, n_episodes=1)
                        if reward > best_reward:
                            best_reward = reward
                            best_params = params
    
    # 最终评估
    final_reward = evaluate_pid_params(best_params, task, duration, n_episodes=5)
    print(f"  最终 PID 参数: {best_params.to_dict()}")
    print(f"  最终真实奖励: {final_reward:.4f}")
    
    return best_params, final_reward


def grid_search_lqr(task: str, duration: float) -> Tuple[LQRParams, float]:
    """网格搜索 LQR 最佳参数（精简版，降低算力消耗）。"""
    print("\n[LQR] 开始网格搜索（精简）...")
    
    # 搜索范围（精简：每参数 3-4 个值）
    k_pos_range = [2.0, 4.0, 6.0, 8.0]
    k_vel_range = [2.0, 4.0, 6.0]
    k_att_range = [10.0, 20.0, 30.0]
    k_omega_range = [2.0, 4.0]
    att_scale_range = [0.1, 0.2]
    
    best_params = LQRParams()
    best_reward = float("-inf")
    
    # 粗搜索
    print("  阶段1: 粗搜索位置/速度增益...")
    for k_pos in k_pos_range:
        for k_vel in k_vel_range:
            params = LQRParams(k_pos=k_pos, k_vel=k_vel)
            reward = evaluate_lqr_params(params, task, duration, n_episodes=1)
            if reward > best_reward:
                best_reward = reward
                best_params = params
    
    print(f"    粗搜索最佳: k_pos={best_params.k_pos}, k_vel={best_params.k_vel}, reward={best_reward:.4f}")
    
    # 细搜索
    print("  阶段2: 细搜索姿态增益...")
    base_k_pos = best_params.k_pos
    base_k_vel = best_params.k_vel
    
    for k_att in k_att_range:
        for k_omega in k_omega_range:
            for att_scale in att_scale_range:
                params = LQRParams(
                    k_pos=base_k_pos, k_vel=base_k_vel,
                    k_att=k_att, k_omega=k_omega,
                    att_scale=att_scale
                )
                reward = evaluate_lqr_params(params, task, duration, n_episodes=1)
                if reward > best_reward:
                    best_reward = reward
                    best_params = params
    
    # 最终评估
    final_reward = evaluate_lqr_params(best_params, task, duration, n_episodes=5)
    print(f"  最终 LQR 参数: {best_params.to_dict()}")
    print(f"  最终真实奖励: {final_reward:.4f}")
    
    return best_params, final_reward


def local_refine_pid(params: PIDParams, task: str, duration: float, iterations: int = 8) -> Tuple[PIDParams, float]:
    """局部优化 PID 参数（精简版，减少迭代次数）。"""
    print("\n[PID] 局部优化（精简）...")
    
    current = params
    current_reward = evaluate_pid_params(current, task, duration, n_episodes=3)
    
    step_sizes = {
        "kp_xy": 1.0, "kp_z": 1.0, "kd_xy": 0.5, "kd_z": 0.5,
        "kp_att": 2.0, "kd_att": 0.2, "att_scale": 0.02
    }
    
    for i in range(iterations):
        improved = False
        for param_name in step_sizes:
            step = step_sizes[param_name]
            
            # 尝试增加
            new_params = PIDParams(**current.to_dict())
            new_val = getattr(new_params, param_name) + step
            if new_val > 0:
                setattr(new_params, param_name, new_val)
                new_reward = evaluate_pid_params(new_params, task, duration, n_episodes=2)
                if new_reward > current_reward:
                    current = new_params
                    current_reward = new_reward
                    improved = True
                    continue
            
            # 尝试减少
            new_params = PIDParams(**current.to_dict())
            new_val = getattr(new_params, param_name) - step
            if new_val > 0:
                setattr(new_params, param_name, new_val)
                new_reward = evaluate_pid_params(new_params, task, duration, n_episodes=2)
                if new_reward > current_reward:
                    current = new_params
                    current_reward = new_reward
                    improved = True
        
        if not improved:
            # 减小步长
            for k in step_sizes:
                step_sizes[k] *= 0.5
        
        if i % 5 == 0:
            print(f"    迭代 {i}: reward={current_reward:.4f}")
    
    return current, current_reward


def local_refine_lqr(params: LQRParams, task: str, duration: float, iterations: int = 8) -> Tuple[LQRParams, float]:
    """局部优化 LQR 参数（精简版，减少迭代次数）。"""
    print("\n[LQR] 局部优化（精简）...")
    
    current = params
    current_reward = evaluate_lqr_params(current, task, duration, n_episodes=3)
    
    step_sizes = {
        "k_pos": 0.5, "k_vel": 0.5, "k_att": 2.0, "k_omega": 0.5, "att_scale": 0.02
    }
    
    for i in range(iterations):
        improved = False
        for param_name in step_sizes:
            step = step_sizes[param_name]
            
            new_params = LQRParams(**current.to_dict())
            new_val = getattr(new_params, param_name) + step
            if new_val > 0:
                setattr(new_params, param_name, new_val)
                new_reward = evaluate_lqr_params(new_params, task, duration, n_episodes=2)
                if new_reward > current_reward:
                    current = new_params
                    current_reward = new_reward
                    improved = True
                    continue
            
            new_params = LQRParams(**current.to_dict())
            new_val = getattr(new_params, param_name) - step
            if new_val > 0:
                setattr(new_params, param_name, new_val)
                new_reward = evaluate_lqr_params(new_params, task, duration, n_episodes=2)
                if new_reward > current_reward:
                    current = new_params
                    current_reward = new_reward
                    improved = True
        
        if not improved:
            for k in step_sizes:
                step_sizes[k] *= 0.5
        
        if i % 5 == 0:
            print(f"    迭代 {i}: reward={current_reward:.4f}")
    
    return current, current_reward


def detailed_evaluation(controller, controller_name: str, task: str, duration: float) -> Dict[str, Any]:
    """详细评估控制器性能。"""
    result = simulate_episode(controller, task, duration)
    print(f"\n{'='*60}")
    print(f"{controller_name} 详细评估 | 任务: {task}")
    print(f"{'='*60}")
    print(f"  真实奖励 (SCG): {result['true_reward']:.4f}")
    print(f"  状态代价: {result['state_cost']:.4f}")
    print(f"  控制代价: {result['action_cost']:.6f}")
    print(f"  位置 RMSE: {result['rmse']:.4f} m")
    print(f"  最大位置误差: {result['max_pos_err']:.4f} m")
    return result


def main():
    parser = argparse.ArgumentParser(description="调优 PID/LQR 控制器")
    parser.add_argument("--task", default="figure8", choices=["hover", "figure8", "circle", "square", "helix"])
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--pid-refine-iters", type=int, default=10, help="PID 局部精调迭代次数")
    parser.add_argument("--lqr-refine-iters", type=int, default=10, help="LQR 局部精调迭代次数")
    parser.add_argument("--output", type=str, default="results/tuned_baselines.json")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PID / LQR 参数调优")
    print(f"任务: {args.task} | 时长: {args.duration}s")
    print(f"PID 精调: {args.pid_refine_iters} 次 | LQR 精调: {args.lqr_refine_iters} 次")
    print("=" * 60)
    
    start_time = time.time()
    
    # 调优 PID
    pid_params, pid_grid_reward = grid_search_pid(args.task, args.duration)
    pid_params, pid_final_reward = local_refine_pid(pid_params, args.task, args.duration, iterations=args.pid_refine_iters)
    
    # 调优 LQR
    lqr_params, lqr_grid_reward = grid_search_lqr(args.task, args.duration)
    lqr_params, lqr_final_reward = local_refine_lqr(lqr_params, args.task, args.duration, iterations=args.lqr_refine_iters)
    
    elapsed = time.time() - start_time
    print(f"\n调优耗时: {elapsed:.1f}s")
    
    # 详细评估
    pid_controller = TunablePIDController(pid_params)
    lqr_controller = TunableLQRController(lqr_params)
    
    pid_result = detailed_evaluation(pid_controller, "PID", args.task, args.duration)
    lqr_result = detailed_evaluation(lqr_controller, "LQR", args.task, args.duration)
    
    # 对比
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"{'控制器':<10} {'真实奖励':<15} {'状态代价':<12} {'控制代价':<12} {'RMSE(m)':<10}")
    print("-" * 60)
    print(f"{'PID':<10} {pid_result['true_reward']:<15.4f} {pid_result['state_cost']:<12.4f} {pid_result['action_cost']:<12.6f} {pid_result['rmse']:<10.4f}")
    print(f"{'LQR':<10} {lqr_result['true_reward']:<15.4f} {lqr_result['state_cost']:<12.4f} {lqr_result['action_cost']:<12.6f} {lqr_result['rmse']:<10.4f}")
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "task": args.task,
        "duration": args.duration,
        "pid": {
            "params": pid_params.to_dict(),
            "true_reward": pid_result["true_reward"],
            "state_cost": pid_result["state_cost"],
            "action_cost": pid_result["action_cost"],
            "rmse": pid_result["rmse"],
        },
        "lqr": {
            "params": lqr_params.to_dict(),
            "true_reward": lqr_result["true_reward"],
            "state_cost": lqr_result["state_cost"],
            "action_cost": lqr_result["action_cost"],
            "rmse": lqr_result["rmse"],
        },
    }
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    
    # 输出最优参数代码
    print("\n" + "=" * 60)
    print("最优参数 (可直接复制使用)")
    print("=" * 60)
    print(f"""
# PID 最优参数
PID_PARAMS = {{
    "kp_xy": {pid_params.kp_xy},
    "kp_z": {pid_params.kp_z},
    "kd_xy": {pid_params.kd_xy},
    "kd_z": {pid_params.kd_z},
    "kp_att": {pid_params.kp_att},
    "kd_att": {pid_params.kd_att},
    "att_scale": {pid_params.att_scale},
}}

# LQR 最优参数
LQR_PARAMS = {{
    "k_pos": {lqr_params.k_pos},
    "k_vel": {lqr_params.k_vel},
    "k_att": {lqr_params.k_att},
    "k_omega": {lqr_params.k_omega},
    "att_scale": {lqr_params.att_scale},
}}
""")


if __name__ == "__main__":
    main()

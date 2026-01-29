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
    ki_xy: float = 0.0  # 积分项
    ki_z: float = 0.0
    kp_att: float = 10.0
    kd_att: float = 1.0
    att_scale: float = 0.1  # 位置误差到姿态的映射比例
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "kp_xy": self.kp_xy,
            "kp_z": self.kp_z,
            "kd_xy": self.kd_xy,
            "kd_z": self.kd_z,
            "ki_xy": self.ki_xy,
            "ki_z": self.ki_z,
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
    """可调参数的 PID 控制器（带积分项和更精确的推力模型）"""
    
    def __init__(self, params: PIDParams, mass: float = 0.027, g: float = 9.81):
        self.params = params
        self.mass = mass
        self.g = g
        self.hover_thrust = mass * g
        self.integral_xy = np.zeros(2)
        self.integral_z = 0.0
        self.max_integral_xy = 1.0  # 积分抗饱和
        self.max_integral_z = 1.5
    
    def reset_integral(self):
        """重置积分项"""
        self.integral_xy = np.zeros(2)
        self.integral_z = 0.0
    
    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        rpy: np.ndarray,
        omega: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray = None,
        dt: float = 0.02,
    ) -> np.ndarray:
        if target_vel is None:
            target_vel = np.zeros(3)
        
        p = self.params
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        
        # 积分项累积
        self.integral_xy += pos_err[:2] * dt
        self.integral_xy = np.clip(self.integral_xy, -self.max_integral_xy, self.max_integral_xy)
        self.integral_z += pos_err[2] * dt
        self.integral_z = np.clip(self.integral_z, -self.max_integral_z, self.max_integral_z)
        
        # 位置控制（PID）
        acc_des = np.array([
            p.kp_xy * pos_err[0] + p.kd_xy * vel_err[0] + p.ki_xy * self.integral_xy[0],
            p.kp_xy * pos_err[1] + p.kd_xy * vel_err[1] + p.ki_xy * self.integral_xy[1],
            p.kp_z * pos_err[2] + p.kd_z * vel_err[2] + p.ki_z * self.integral_z + self.g,
        ])
        
        # 推力（更大的限制范围，典型四旋翼可以达到4倍悬停推力）
        fz = self.mass * acc_des[2]
        fz = np.clip(fz, 0.01 * self.hover_thrust, 4.0 * self.hover_thrust)
        
        # 期望姿态（更激进的映射）
        roll_des = -acc_des[1] / self.g * p.att_scale
        pitch_des = acc_des[0] / self.g * p.att_scale
        roll_des = np.clip(roll_des, -0.6, 0.6)  # ±34度
        pitch_des = np.clip(pitch_des, -0.6, 0.6)
        rpy_des = np.array([roll_des, pitch_des, 0.0])
        
        # 姿态控制（PD）
        rpy_err = rpy_des - rpy
        torque = p.kp_att * rpy_err - p.kd_att * omega
        # 更大的力矩限制
        torque = np.clip(torque, -0.5, 0.5)
        
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float64)


class TunableLQRController:
    """可调参数的 LQR 控制器（更精确的推力和力矩模型）"""
    
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
        dt: float = 0.02,
    ) -> np.ndarray:
        if target_vel is None:
            target_vel = np.zeros(3)
        
        p = self.params
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        
        # LQR 形式（位置和速度独立增益）
        acc_des = p.k_pos * pos_err + p.k_vel * vel_err
        acc_des[2] += self.g
        
        # 推力（更大的范围）
        fz = self.mass * acc_des[2]
        fz = np.clip(fz, 0.01 * self.hover_thrust, 4.0 * self.hover_thrust)
        
        # 期望姿态
        roll_des = -acc_des[1] / self.g * p.att_scale
        pitch_des = acc_des[0] / self.g * p.att_scale
        roll_des = np.clip(roll_des, -0.6, 0.6)
        pitch_des = np.clip(pitch_des, -0.6, 0.6)
        rpy_des = np.array([roll_des, pitch_des, 0.0])
        
        # 姿态控制（更大的增益和力矩限制）
        rpy_err = rpy_des - rpy
        torque = p.k_att * rpy_err - p.k_omega * omega
        torque = np.clip(torque, -0.5, 0.5)
        
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float64)


def simulate_episode(
    controller,
    task: str = "figure8",
    duration: float = 5.0,
    ctrl_freq: float = 48.0,
) -> Dict[str, float]:
    """模拟一个 episode，使用更精确的四旋翼动力学模型。"""
    dt = 1.0 / ctrl_freq
    max_steps = int(duration * ctrl_freq)
    mass = 0.027  # Crazyflie 2.1 质量
    g = 9.81
    
    # Crazyflie 2.1 惯性矩阵（真实值）
    Ixx = 1.66e-5
    Iyy = 1.66e-5
    Izz = 2.92e-5
    inertia = np.array([Ixx, Iyy, Izz])
    
    # 初始状态
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.zeros(3)
    rpy = np.zeros(3)
    omega = np.zeros(3)
    
    # 重置积分项（如果是PID）
    if hasattr(controller, 'reset_integral'):
        controller.reset_integral()
    
    total_state_cost = 0.0
    total_action_cost = 0.0
    pos_errors = []
    vel_errors = []
    att_errors = []
    
    traj_cfg = get_scg_trajectory_config(task)
    center = np.array(traj_cfg.center, dtype=np.float64)

    for step in range(max_steps):
        t = step * dt
        target_pos, target_vel = scg_position_velocity(task, t, params=traj_cfg.params, center=center)
        
        # 控制输入
        action = controller.compute(pos, vel, rpy, omega, target_pos, target_vel, dt=dt)

        pos_err_vec = target_pos - pos
        vel_err_vec = target_vel - vel
        rpy_err_vec = -rpy  # target姿态恒为0
        omega_err_vec = -omega

        pos_errors.append(float(np.linalg.norm(pos_err_vec)))
        vel_errors.append(float(np.linalg.norm(vel_err_vec)))
        att_errors.append(float(np.linalg.norm(rpy_err_vec)))
        
        # SCG 风格状态/动作代价（更精确的权重）
        state_cost = (
            float(pos_err_vec @ pos_err_vec)  # 位置误差（权重1.0）
            + 0.1 * float(vel_err_vec @ vel_err_vec)  # 速度误差
            + 0.8 * float(rpy_err_vec @ rpy_err_vec)  # 姿态误差
            + 0.05 * float(omega_err_vec @ omega_err_vec)  # 角速度误差
        )
        action_cost = 1e-4 * float(action @ action)  # 控制代价
        total_state_cost += state_cost * dt
        total_action_cost += action_cost * dt

        # === 四旋翼动力学（更精确） ===
        roll, pitch, yaw = rpy
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # 旋转矩阵（Body -> World）
        R = np.array([
            [cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
            [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
            [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]
        ])
        
        # 推力在世界坐标系
        thrust_body = np.array([0.0, 0.0, action[0] / mass])
        thrust_world = R @ thrust_body
        
        # 加速度
        acc = thrust_world - np.array([0.0, 0.0, g])
        
        # 空气阻力（简化模型）
        drag_coeff = 0.02
        acc -= drag_coeff * vel
        
        # 更新速度和位置
        vel += acc * dt
        pos += vel * dt
        
        # 角加速度（欧拉动力学方程）
        torques = action[1:4]
        
        # 陀螺效应（简化）
        gyro_effect = np.array([
            omega[1] * omega[2] * (Iyy - Izz),
            omega[0] * omega[2] * (Izz - Ixx),
            omega[0] * omega[1] * (Ixx - Iyy)
        ]) / inertia
        
        alpha = torques / inertia - gyro_effect
        
        # 更新角速度
        omega += alpha * dt
        omega = np.clip(omega, -15.0, 15.0)  # 更大的角速度限制
        
        # 更新姿态角（使用欧拉角微分方程）
        # 简化版：小角度近似
        if abs(cos_pitch) > 0.01:
            rpy_dot = np.array([
                omega[0] + omega[1] * sin_roll * sin_pitch / cos_pitch + omega[2] * cos_roll * sin_pitch / cos_pitch,
                omega[1] * cos_roll - omega[2] * sin_roll,
                omega[1] * sin_roll / cos_pitch + omega[2] * cos_roll / cos_pitch
            ])
        else:
            rpy_dot = omega  # 退化到简单模型
        
        rpy += rpy_dot * dt
        
        # 姿态角限制（防止奇点）
        rpy[0] = np.clip(rpy[0], -0.8, 0.8)  # roll
        rpy[1] = np.clip(rpy[1], -0.8, 0.8)  # pitch
        # yaw可以不限制
        
        # 地面碰撞检测
        if pos[2] < -0.1:
            # 碰撞惩罚
            total_state_cost += 1000.0
            break
    
    true_reward = -(total_state_cost + total_action_cost)
    
    return {
        "state_cost": total_state_cost,
        "action_cost": total_action_cost,
        "true_reward": true_reward,
        "rmse": float(np.sqrt(np.mean(np.array(pos_errors) ** 2))),
        "mean_pos_err": float(np.mean(pos_errors)),
        "max_pos_err": float(np.max(pos_errors)),
        "mean_vel_err": float(np.mean(vel_errors)),
        "mean_att_err": float(np.mean(att_errors)),
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
    """网格搜索 PID 最佳参数（深度搜索版）。"""
    print("\n[PID] 开始深度网格搜索...")
    
    # 扩大搜索范围（更密集的采样点）
    kp_xy_range = [3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0]  # 7个点
    kp_z_range = [6.0, 10.0, 15.0, 20.0, 28.0, 40.0]  # 6个点
    kd_xy_range = [1.5, 2.5, 4.0, 6.0, 9.0, 12.0]  # 6个点
    kd_z_range = [3.0, 5.0, 7.0, 10.0, 14.0]  # 5个点
    ki_xy_range = [0.0, 0.1, 0.5, 1.0]  # 积分项
    ki_z_range = [0.0, 0.2, 0.8, 1.5]
    kp_att_range = [8.0, 15.0, 25.0, 40.0, 60.0]  # 5个点
    kd_att_range = [0.5, 1.0, 2.0, 3.5, 5.0]  # 5个点
    att_scale_range = [0.08, 0.12, 0.18, 0.25, 0.35]  # 5个点
    
    best_params = PIDParams()
    best_reward = float("-inf")
    
    # ===== 阶段1: 粗搜索位置PD增益 =====
    print("  阶段1: 粗搜索位置PD增益 (7×6×3×3 = 378组)...")
    count = 0
    for kp_xy in kp_xy_range:
        for kp_z in kp_z_range:
            for kd_xy in [1.5, 4.0, 9.0]:  # 粗采样
                for kd_z in [3.0, 7.0, 14.0]:
                    count += 1
                    params = PIDParams(kp_xy=kp_xy, kp_z=kp_z, kd_xy=kd_xy, kd_z=kd_z)
                    reward = evaluate_pid_params(params, task, duration, n_episodes=2)
                    if reward > best_reward:
                        best_reward = reward
                        best_params = params
                    if count % 50 == 0:
                        print(f"    进度: {count}/378, 当前最佳={best_reward:.4f}")
    
    print(f"  阶段1完成: kp_xy={best_params.kp_xy}, kp_z={best_params.kp_z}, kd_xy={best_params.kd_xy}, kd_z={best_params.kd_z}, reward={best_reward:.4f}")
    
    # ===== 阶段2: 在最佳PD附近细搜索积分项 =====
    print("  阶段2: 细搜索积分项 (4×4 = 16组)...")
    base_kp_xy = best_params.kp_xy
    base_kp_z = best_params.kp_z
    base_kd_xy = best_params.kd_xy
    base_kd_z = best_params.kd_z
    
    for ki_xy in ki_xy_range:
        for ki_z in ki_z_range:
            params = PIDParams(
                kp_xy=base_kp_xy, kp_z=base_kp_z,
                kd_xy=base_kd_xy, kd_z=base_kd_z,
                ki_xy=ki_xy, ki_z=ki_z
            )
            reward = evaluate_pid_params(params, task, duration, n_episodes=3)
            if reward > best_reward:
                best_reward = reward
                best_params = params
    
    print(f"  阶段2完成: ki_xy={best_params.ki_xy}, ki_z={best_params.ki_z}, reward={best_reward:.4f}")
    
    # ===== 阶段3: 搜索姿态增益 =====
    print("  阶段3: 搜索姿态增益 (5×5 = 25组)...")
    for kp_att in kp_att_range:
        for kd_att in kd_att_range:
            params = PIDParams(
                kp_xy=best_params.kp_xy, kp_z=best_params.kp_z,
                kd_xy=best_params.kd_xy, kd_z=best_params.kd_z,
                ki_xy=best_params.ki_xy, ki_z=best_params.ki_z,
                kp_att=kp_att, kd_att=kd_att
            )
            reward = evaluate_pid_params(params, task, duration, n_episodes=3)
            if reward > best_reward:
                best_reward = reward
                best_params = params
    
    print(f"  阶段3完成: kp_att={best_params.kp_att}, kd_att={best_params.kd_att}, reward={best_reward:.4f}")
    
    # ===== 阶段4: 搜索姿态映射比例 =====
    print("  阶段4: 搜索姿态映射比例 (5组)...")
    for att_scale in att_scale_range:
        params = PIDParams(
            kp_xy=best_params.kp_xy, kp_z=best_params.kp_z,
            kd_xy=best_params.kd_xy, kd_z=best_params.kd_z,
            ki_xy=best_params.ki_xy, ki_z=best_params.ki_z,
            kp_att=best_params.kp_att, kd_att=best_params.kd_att,
            att_scale=att_scale
        )
        reward = evaluate_pid_params(params, task, duration, n_episodes=3)
        if reward > best_reward:
            best_reward = reward
            best_params = params
    
    print(f"  阶段4完成: att_scale={best_params.att_scale}, reward={best_reward:.4f}")
    
    # 最终评估
    final_reward = evaluate_pid_params(best_params, task, duration, n_episodes=8)
    print(f"  最终 PID 参数: {best_params.to_dict()}")
    print(f"  最终真实奖励: {final_reward:.4f}")
    
    return best_params, final_reward


def grid_search_lqr(task: str, duration: float) -> Tuple[LQRParams, float]:
    """网格搜索 LQR 最佳参数（深度搜索版）。"""
    print("\n[LQR] 开始深度网格搜索...")
    
    # 扩大搜索范围（更密集的采样点）
    k_pos_range = [1.5, 2.5, 4.0, 6.0, 9.0, 12.0, 18.0, 25.0]  # 8个点
    k_vel_range = [1.5, 2.5, 4.0, 6.0, 9.0, 12.0]  # 6个点
    k_att_range = [8.0, 15.0, 25.0, 40.0, 60.0, 80.0]  # 6个点
    k_omega_range = [1.5, 2.5, 4.0, 6.0, 9.0]  # 5个点
    att_scale_range = [0.08, 0.12, 0.18, 0.25, 0.35]  # 5个点
    
    best_params = LQRParams()
    best_reward = float("-inf")
    
    # ===== 阶段1: 粗搜索位置/速度增益 =====
    print("  阶段1: 粗搜索位置/速度增益 (8×6 = 48组)...")
    count = 0
    for k_pos in k_pos_range:
        for k_vel in k_vel_range:
            count += 1
            params = LQRParams(k_pos=k_pos, k_vel=k_vel)
            reward = evaluate_lqr_params(params, task, duration, n_episodes=2)
            if reward > best_reward:
                best_reward = reward
                best_params = params
            if count % 10 == 0:
                print(f"    进度: {count}/48, 当前最佳={best_reward:.4f}")
    
    print(f"  阶段1完成: k_pos={best_params.k_pos}, k_vel={best_params.k_vel}, reward={best_reward:.4f}")
    
    # ===== 阶段2: 细搜索姿态增益 =====
    print("  阶段2: 细搜索姿态增益 (6×5 = 30组)...")
    base_k_pos = best_params.k_pos
    base_k_vel = best_params.k_vel
    
    for k_att in k_att_range:
        for k_omega in k_omega_range:
            params = LQRParams(
                k_pos=base_k_pos, k_vel=base_k_vel,
                k_att=k_att, k_omega=k_omega
            )
            reward = evaluate_lqr_params(params, task, duration, n_episodes=3)
            if reward > best_reward:
                best_reward = reward
                best_params = params
    
    print(f"  阶段2完成: k_att={best_params.k_att}, k_omega={best_params.k_omega}, reward={best_reward:.4f}")
    
    # ===== 阶段3: 搜索姿态映射比例 =====
    print("  阶段3: 搜索姿态映射比例 (5组)...")
    for att_scale in att_scale_range:
        params = LQRParams(
            k_pos=best_params.k_pos, k_vel=best_params.k_vel,
            k_att=best_params.k_att, k_omega=best_params.k_omega,
            att_scale=att_scale
        )
        reward = evaluate_lqr_params(params, task, duration, n_episodes=3)
        if reward > best_reward:
            best_reward = reward
            best_params = params
    
    print(f"  阶段3完成: att_scale={best_params.att_scale}, reward={best_reward:.4f}")
    
    # ===== 阶段4: 在最佳参数附近精细搜索 =====
    print("  阶段4: 在最佳参数附近精细搜索...")
    # 在最佳k_pos, k_vel附近±20%范围内细搜
    fine_k_pos_range = [best_params.k_pos * f for f in [0.85, 0.92, 1.0, 1.08, 1.15]]
    fine_k_vel_range = [best_params.k_vel * f for f in [0.85, 0.92, 1.0, 1.08, 1.15]]
    
    for k_pos in fine_k_pos_range:
        for k_vel in fine_k_vel_range:
            if k_pos <= 0 or k_vel <= 0:
                continue
            params = LQRParams(
                k_pos=k_pos, k_vel=k_vel,
                k_att=best_params.k_att, k_omega=best_params.k_omega,
                att_scale=best_params.att_scale
            )
            reward = evaluate_lqr_params(params, task, duration, n_episodes=3)
            if reward > best_reward:
                best_reward = reward
                best_params = params
    
    print(f"  阶段4完成: k_pos={best_params.k_pos:.3f}, k_vel={best_params.k_vel:.3f}, reward={best_reward:.4f}")
    
    # 最终评估
    final_reward = evaluate_lqr_params(best_params, task, duration, n_episodes=8)
    print(f"  最终 LQR 参数: {best_params.to_dict()}")
    print(f"  最终真实奖励: {final_reward:.4f}")
    
    return best_params, final_reward


def local_refine_pid(params: PIDParams, task: str, duration: float, iterations: int = 50) -> Tuple[PIDParams, float]:
    """局部优化 PID 参数（深度优化版，坐标下降法）。"""
    print(f"\n[PID] 局部优化（{iterations}次迭代）...")
    
    current = params
    current_reward = evaluate_pid_params(current, task, duration, n_episodes=4)
    
    # 初始步长（相对大一些）
    step_sizes = {
        "kp_xy": 2.0, "kp_z": 2.5, "kd_xy": 0.8, "kd_z": 1.0,
        "ki_xy": 0.1, "ki_z": 0.15,
        "kp_att": 3.0, "kd_att": 0.4, "att_scale": 0.03
    }
    
    print(f"    初始奖励: {current_reward:.4f}")
    no_improve_count = 0
    
    for i in range(iterations):
        improved = False
        
        # 坐标下降：每次只调整一个参数
        for param_name in step_sizes:
            step = step_sizes[param_name]
            current_val = getattr(current, param_name)
            
            # 尝试 +step
            new_params = PIDParams(**current.to_dict())
            new_val = current_val + step
            if new_val > 0:
                setattr(new_params, param_name, new_val)
                new_reward = evaluate_pid_params(new_params, task, duration, n_episodes=3)
                if new_reward > current_reward + 1e-6:  # 小的容差避免噪声
                    current = new_params
                    current_reward = new_reward
                    improved = True
                    print(f"      迭代{i}: {param_name}↑ → {new_val:.4f}, reward={new_reward:.4f}")
                    continue
            
            # 尝试 -step
            new_params = PIDParams(**current.to_dict())
            new_val = current_val - step
            if new_val > 0:
                setattr(new_params, param_name, new_val)
                new_reward = evaluate_pid_params(new_params, task, duration, n_episodes=3)
                if new_reward > current_reward + 1e-6:
                    current = new_params
                    current_reward = new_reward
                    improved = True
                    print(f"      迭代{i}: {param_name}↓ → {new_val:.4f}, reward={new_reward:.4f}")
        
        if not improved:
            no_improve_count += 1
            # 减小步长
            for k in step_sizes:
                step_sizes[k] *= 0.7
            
            # 如果连续多次无改进且步长很小，提前终止
            if no_improve_count >= 5 and all(s < 0.05 for s in step_sizes.values()):
                print(f"    提前终止（连续{no_improve_count}次无改进，步长已很小）")
                break
        else:
            no_improve_count = 0
        
        if i % 10 == 0:
            print(f"    迭代 {i}: reward={current_reward:.4f}, 步长={list(step_sizes.values())[0]:.4f}")
    
    print(f"  最终优化奖励: {current_reward:.4f}")
    return current, current_reward


def local_refine_lqr(params: LQRParams, task: str, duration: float, iterations: int = 50) -> Tuple[LQRParams, float]:
    """局部优化 LQR 参数（深度优化版，坐标下降法）。"""
    print(f"\n[LQR] 局部优化（{iterations}次迭代）...")
    
    current = params
    current_reward = evaluate_lqr_params(current, task, duration, n_episodes=4)
    
    # 初始步长
    step_sizes = {
        "k_pos": 1.0, "k_vel": 1.0, "k_att": 3.0, "k_omega": 0.8, "att_scale": 0.03
    }
    
    print(f"    初始奖励: {current_reward:.4f}")
    no_improve_count = 0
    
    for i in range(iterations):
        improved = False
        
        # 坐标下降：每次只调整一个参数
        for param_name in step_sizes:
            step = step_sizes[param_name]
            current_val = getattr(current, param_name)
            
            # 尝试 +step
            new_params = LQRParams(**current.to_dict())
            new_val = current_val + step
            if new_val > 0:
                setattr(new_params, param_name, new_val)
                new_reward = evaluate_lqr_params(new_params, task, duration, n_episodes=3)
                if new_reward > current_reward + 1e-6:
                    current = new_params
                    current_reward = new_reward
                    improved = True
                    print(f"      迭代{i}: {param_name}↑ → {new_val:.4f}, reward={new_reward:.4f}")
                    continue
            
            # 尝试 -step
            new_params = LQRParams(**current.to_dict())
            new_val = current_val - step
            if new_val > 0:
                setattr(new_params, param_name, new_val)
                new_reward = evaluate_lqr_params(new_params, task, duration, n_episodes=3)
                if new_reward > current_reward + 1e-6:
                    current = new_params
                    current_reward = new_reward
                    improved = True
                    print(f"      迭代{i}: {param_name}↓ → {new_val:.4f}, reward={new_reward:.4f}")
        
        if not improved:
            no_improve_count += 1
            # 减小步长
            for k in step_sizes:
                step_sizes[k] *= 0.7
            
            # 如果连续多次无改进且步长很小，提前终止
            if no_improve_count >= 5 and all(s < 0.05 for s in step_sizes.values()):
                print(f"    提前终止（连续{no_improve_count}次无改进，步长已很小）")
                break
        else:
            no_improve_count = 0
        
        if i % 10 == 0:
            print(f"    迭代 {i}: reward={current_reward:.4f}, 步长={list(step_sizes.values())[0]:.4f}")
    
    print(f"  最终优化奖励: {current_reward:.4f}")
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
    parser = argparse.ArgumentParser(description="调优 PID/LQR 控制器（深度搜索版）")
    parser.add_argument("--task", default="figure8", choices=["hover", "figure8", "circle", "square", "helix"])
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--pid-refine-iters", type=int, default=50, help="PID 局部精调迭代次数（默认50）")
    parser.add_argument("--lqr-refine-iters", type=int, default=50, help="LQR 局部精调迭代次数（默认50）")
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

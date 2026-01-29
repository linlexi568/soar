#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCG-native PID/LQR reward evaluation script.

直接在 safe-control-gym 的 Quadrotor 环境上测试 PID 和 LQR 控制器，
输出每步奖励和累计奖励，用于与 Soar 训练结果比较。
"""
from __future__ import annotations
import sys
import os
import math
import numpy as np

# 确保 scripts/adapters 可被 import
sys.path.insert(0, os.path.dirname(__file__))

from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor


# ========== 简单 PID 控制器 ==========
class PIDController:
    """
    2D Quadrotor (x-z) PID controller.
    输入: [x, x_dot, z, z_dot, theta, theta_dot]
    输出: [T1, T2] — 两个推力 (safe-control-gym 默认 action)
    """
    def __init__(self):
        # Gains tuned for default SCG quadrotor
        self.kp_z = 8.0
        self.kd_z = 4.0
        self.kp_x = 4.0
        self.kd_x = 2.0
        self.kp_theta = 6.0
        self.kd_theta = 2.5
        # Nominal thrust per motor (hover ~ mg/2)
        self.m = 0.027
        self.g = 9.81
        self.T_hover = self.m * self.g / 2.0  # per motor

    def act(self, obs: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        """
        obs: [x, x_dot, z, z_dot, theta, theta_dot]
        target: [x_ref, z_ref] (default [0, 1])
        """
        if target is None:
            target = np.array([0.0, 1.0])  # hover at (0, 1)
        x, x_dot, z, z_dot, theta, theta_dot = obs
        x_ref, z_ref = target

        # Z control (altitude)
        ez = z_ref - z
        u_z = self.kp_z * ez - self.kd_z * z_dot

        # X control -> desired theta
        ex = x_ref - x
        theta_des = self.kp_x * ex - self.kd_x * x_dot
        theta_des = np.clip(theta_des, -0.5, 0.5)  # limit desired angle

        # Theta control
        e_theta = theta_des - theta
        u_theta = self.kp_theta * e_theta - self.kd_theta * theta_dot

        # Mix to thrusts
        T_base = self.T_hover + u_z / 2.0
        T1 = T_base - u_theta * 0.05
        T2 = T_base + u_theta * 0.05

        T1 = np.clip(T1, 0.0, 0.6)
        T2 = np.clip(T2, 0.0, 0.6)
        return np.array([T1, T2], dtype=np.float32)


# ========== 简单 LQR 控制器 ==========
class LQRController:
    """
    Cascaded PD-style LQR controller for 2D Quadrotor.
    使用经过手动调参的增益，模拟 LQR 的结构但避免模型不匹配问题。
    """
    def __init__(self):
        self.m = 0.027
        self.g = 9.81
        self.T_hover = self.m * self.g / 2.0

        # 高度控制增益 (优化后)
        self.kp_z = 10.0
        self.kd_z = 5.0

        # 水平位置控制增益 (优化后)
        self.kp_x = 4.5
        self.kd_x = 2.2

        # 姿态控制增益 (优化后)
        self.kp_theta = 7.0
        self.kd_theta = 2.8

    def act(self, obs: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        if target is None:
            target = np.array([0.0, 1.0])
        x, x_dot, z, z_dot, theta, theta_dot = obs
        x_ref, z_ref = target

        # Z control: total thrust adjustment
        z_err = z_ref - z
        u_z = self.kp_z * z_err - self.kd_z * z_dot

        # X control: compute desired theta
        x_err = x_ref - x
        theta_des = self.kp_x * x_err - self.kd_x * x_dot
        theta_des = np.clip(theta_des, -0.5, 0.5)

        # Theta control: differential thrust
        theta_err = theta_des - theta
        u_diff = self.kp_theta * theta_err - self.kd_theta * theta_dot

        # Compute individual thrusts
        T_base = self.T_hover + u_z / 2.0
        T1 = T_base - u_diff * 0.05
        T2 = T_base + u_diff * 0.05

        T1 = np.clip(T1, 0.0, 0.6)
        T2 = np.clip(T2, 0.0, 0.6)
        return np.array([T1, T2], dtype=np.float32)


def evaluate_controller(controller, env, episodes: int = 3, max_steps: int = 500, target=None):
    """
    Evaluate a controller on the given env.
    Returns per-episode and aggregate metrics.
    """
    all_rewards = []
    all_ep_returns = []
    all_pos_errors = []

    for ep in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = np.asarray(obs, dtype=np.float64)

        ep_reward = 0.0
        ep_pos_errors = []

        for step in range(max_steps):
            action = controller.act(obs, target)
            result = env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, done, truncated, info = result
                done = done or truncated
            obs = np.asarray(obs, dtype=np.float64)

            all_rewards.append(reward)
            ep_reward += reward

            # Position error (x, z)
            x, _, z, _, _, _ = obs
            if target is None:
                x_ref, z_ref = 0.0, 1.0
            else:
                x_ref, z_ref = target
            pos_err = math.sqrt((x - x_ref)**2 + (z - z_ref)**2)
            ep_pos_errors.append(pos_err)

            if done:
                break

        all_ep_returns.append(ep_reward)
        all_pos_errors.extend(ep_pos_errors)
        print(f"  Episode {ep+1}/{episodes}: Return={ep_reward:.2f}, "
              f"Steps={len(ep_pos_errors)}, RMSE={np.sqrt(np.mean(np.array(ep_pos_errors)**2)):.4f}")

    return {
        'mean_return': np.mean(all_ep_returns),
        'std_return': np.std(all_ep_returns),
        'mean_step_reward': np.mean(all_rewards),
        'rmse_pos': np.sqrt(np.mean(np.array(all_pos_errors)**2)),
        'max_pos_err': np.max(all_pos_errors),
    }


def main():
    print("=" * 60)
    print("SCG-native PID/LQR Reward Evaluation")
    print("=" * 60)

    # Create environment
    env = Quadrotor()
    target = np.array([0.0, 1.0])  # hover at x=0, z=1

    episodes = 5
    max_steps = 500

    # ===== PID =====
    print("\n[PID Controller]")
    pid = PIDController()
    pid_metrics = evaluate_controller(pid, env, episodes=episodes, max_steps=max_steps, target=target)
    print(f"  => Mean Return: {pid_metrics['mean_return']:.2f} ± {pid_metrics['std_return']:.2f}")
    print(f"  => Mean Step Reward: {pid_metrics['mean_step_reward']:.4f}")
    print(f"  => Position RMSE: {pid_metrics['rmse_pos']:.4f} m")
    print(f"  => Max Pos Error: {pid_metrics['max_pos_err']:.4f} m")

    # ===== LQR =====
    print("\n[LQR Controller]")
    lqr = LQRController()
    lqr_metrics = evaluate_controller(lqr, env, episodes=episodes, max_steps=max_steps, target=target)
    print(f"  => Mean Return: {lqr_metrics['mean_return']:.2f} ± {lqr_metrics['std_return']:.2f}")
    print(f"  => Mean Step Reward: {lqr_metrics['mean_step_reward']:.4f}")
    print(f"  => Position RMSE: {lqr_metrics['rmse_pos']:.4f} m")
    print(f"  => Max Pos Error: {lqr_metrics['max_pos_err']:.4f} m")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"{'Controller':<12} {'Mean Return':>12} {'Step Reward':>12} {'Pos RMSE':>10}")
    print("-" * 48)
    print(f"{'PID':<12} {pid_metrics['mean_return']:>12.2f} {pid_metrics['mean_step_reward']:>12.4f} {pid_metrics['rmse_pos']:>10.4f}")
    print(f"{'LQR':<12} {lqr_metrics['mean_return']:>12.2f} {lqr_metrics['mean_step_reward']:>12.4f} {lqr_metrics['rmse_pos']:>10.4f}")

    env.close()
    print("\nDone.")


if __name__ == '__main__':
    main()

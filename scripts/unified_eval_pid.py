#!/usr/bin/env python3
"""
统一评估脚本 - Isaac Gym必须最先导入
"""
# ⚠️ Isaac Gym 必须在 torch 之前导入！
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'isaacgym' / 'python'))

from isaacgym import gymapi  # noqa: F401 - 必须先导入

import json
import numpy as np
import torch

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '01_soar'))

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from utils.reward_scg_exact import SCGExactRewardCalculator
from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity

def quat_to_euler_np(q):
    """四元数转欧拉角 (roll, pitch, yaw)"""
    x, y, z, w = q[0], q[1], q[2], q[3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)

class SimplePID:
    """简化的PID控制器"""
    def __init__(self, kp_xy=5.0, kp_z=10.0, kd_xy=17.5, kd_z=1.0, 
                 kp_att=60.0, kd_att=5.0, att_scale=0.1,
                 mass=0.027, g=9.81):
        self.kp_xy = kp_xy
        self.kp_z = kp_z
        self.kd_xy = kd_xy
        self.kd_z = kd_z
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.att_scale = att_scale
        self.mass = mass
        self.g = g
        self.hover = mass * g
    
    def compute(self, pos, vel, quat, omega, target_pos, target_vel=None):
        if target_vel is None:
            target_vel = np.zeros(3)
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        
        # 期望加速度
        acc_des = np.array([
            self.kp_xy * pos_err[0] + self.kd_xy * vel_err[0],
            self.kp_xy * pos_err[1] + self.kd_xy * vel_err[1],
            self.kp_z * pos_err[2] + self.kd_z * vel_err[2] + self.g,
        ])
        
        # 推力
        fz = self.mass * acc_des[2]
        fz = float(np.clip(fz, 0.0, 2.0 * self.hover))
        
        # 期望姿态
        roll_des = -acc_des[1] / self.g * self.att_scale
        pitch_des = acc_des[0] / self.g * self.att_scale
        rpy_cur = quat_to_euler_np(quat)
        rpy_err = np.array([roll_des - rpy_cur[0], pitch_des - rpy_cur[1], -rpy_cur[2]])
        
        # 力矩
        torque = np.array([
            self.kp_att * rpy_err[0] - self.kd_att * omega[0],
            self.kp_att * rpy_err[1] - self.kd_att * omega[1],
            0.0,  # yaw
        ])
        torque = np.clip(torque, -0.1, 0.1)
        
        return np.array([fz, torque[0], torque[1], torque[2]], dtype=np.float32)


def evaluate_controller(controller, task: str, duration: float, episodes: int = 3):
    """评估控制器"""
    device = 'cuda:0'
    
    def make_targets(t):
        return scg_position_velocity(task, t)
    
    env = IsaacGymDroneEnv(num_envs=1, device=device, headless=True, duration_sec=duration)
    reward_calc = SCGExactRewardCalculator(num_envs=1, device=device)
    
    dt = 1.0 / 48.0
    steps = int(duration / dt)
    
    ep_rewards = []
    for ep in range(episodes):
        env.reset()
        reward_calc.reset()
        controller.reset() if hasattr(controller, 'reset') else None
        t = 0.0
        
        for s in range(steps):
            obs = env.get_obs()
            pos = np.asarray(obs['position'][0], dtype=np.float32)
            vel = np.asarray(obs['velocity'][0], dtype=np.float32)
            quat = np.asarray(obs['orientation'][0], dtype=np.float32)
            omega = np.asarray(obs['angular_velocity'][0], dtype=np.float32)
            tgt_pos, tgt_vel = make_targets(t)
            
            action4 = controller.compute(pos, vel, quat, omega, tgt_pos, tgt_vel)
            
            # 6D力格式
            forces = torch.zeros(1, 6, device=device)
            forces[0, 2] = float(action4[0])
            forces[0, 3] = float(action4[1])
            forces[0, 4] = float(action4[2])
            forces[0, 5] = float(action4[3])
            
            obs_next, _, done, _ = env.step(forces)
            
            # 奖励计算
            pos_t = torch.tensor(obs_next['position'], device=device)
            vel_t = torch.tensor(obs_next['velocity'], device=device)
            quat_t = torch.tensor(obs_next['orientation'], device=device)
            omega_t = torch.tensor(obs_next['angular_velocity'], device=device)
            target_pos_t = torch.tensor(tgt_pos, device=device).unsqueeze(0)
            reward_calc.compute_step(pos_t, vel_t, quat_t, omega_t, target_pos_t, forces[:, 2:6])
            
            t += dt
        
        comps = reward_calc.get_components()
        total_cost = float(comps['total_cost'].sum().item())
        ep_rewards.append(-total_cost)
    
    env.close()
    return {
        'mean_reward': float(np.mean(ep_rewards)),
        'std_reward': float(np.std(ep_rewards)),
        'rewards': ep_rewards,
    }


def main():
    print("=" * 70)
    print("🔬 统一评估：PID控制器 (Isaac Gym + SCG精确奖励)")
    print("=" * 70)
    
    # 配置
    task = 'square'
    duration = 5.0
    episodes = 5
    
    print(f"\n【配置】")
    print(f"  任务: {task}")
    print(f"  时长: {duration}s")
    print(f"  步数: {int(duration * 48)}")
    print(f"  评估轮数: {episodes}")
    
    # PID参数 (从baselines_retune.json)
    pid_params = {
        'kp_xy': 5.0,
        'kp_z': 10.0,
        'kd_xy': 17.5019,
        'kd_z': 1.0,
        'kp_att': 60.0,
        'kd_att': 5.0,
        'att_scale': 0.1,
    }
    
    print(f"\n【PID参数】")
    for k, v in pid_params.items():
        print(f"  {k}: {v}")
    
    # 创建控制器
    pid = SimplePID(**pid_params)
    
    # 评估
    print(f"\n【评估中...】")
    metrics = evaluate_controller(pid, task, duration, episodes)
    
    print(f"\n【PID评估结果】")
    print(f"  mean_reward: {metrics['mean_reward']:.2f}")
    print(f"  std_reward: {metrics['std_reward']:.2f}")
    print(f"  各轮奖励: {[f'{r:.2f}' for r in metrics['rewards']]}")
    
    # 对比
    print(f"\n【对比baselines_retune.json】")
    print(f"  报告奖励: -520.05")
    print(f"  当前评估: {metrics['mean_reward']:.2f}")
    print(f"  差异: {abs(metrics['mean_reward'] - (-520.05)):.2f}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()

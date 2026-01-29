"""Exact safe-control-gym quadrotor_3D_track reward implementation.

严格复现 SCG 论文的 reward 定义：
    r_t = -(x_err^T Q x_err + u^T R u)

其中:
- x_err = [x-x_ref, vx, y-y_ref, vy, z-z_ref, vz, φ, θ, ψ, ωx, ωy, ωz]  (12维)
- Q = diag(rew_state_weight)
- R = rew_act_weight * I_4

SCG quadrotor_3D_track 默认权重:
    rew_state_weight = [1, 0.01, 1, 0.01, 1, 0.01, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01]
    rew_act_weight = 0.0001

这是一个**确定性**的线性二次代价，没有任何额外 shaping。
"""
from __future__ import annotations

from typing import Dict, Optional
import torch
import math


# SCG quadrotor_3D_track 默认权重
SCG_STATE_WEIGHTS = torch.tensor([
    1.0,    # x
    0.01,   # vx
    1.0,    # y
    0.01,   # vy
    1.0,    # z
    0.01,   # vz
    0.5,    # φ (roll)
    0.5,    # θ (pitch)
    0.5,    # ψ (yaw)
    0.01,   # ωx
    0.01,   # ωy
    0.01,   # ωz
], dtype=torch.float32)

SCG_ACTION_WEIGHT = 0.0001


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """四元数 [qx, qy, qz, qw] -> 欧拉角 [roll, pitch, yaw] (弧度)
    
    Args:
        quat: [N, 4] 四元数张量，顺序为 [qx, qy, qz, qw]
    
    Returns:
        euler: [N, 3] 欧拉角张量 [roll, pitch, yaw]
    """
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = torch.clamp(sinp, -1.0, 1.0)  # 防止 asin 溢出
    pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw], dim=1)


class SCGExactRewardCalculator:
    """精确复现 safe-control-gym quadrotor_3D_track 的 reward 计算。
    
    reward = -(state_error^T @ Q @ state_error + action^T @ R @ action)
    
    其中 Q, R 是对角矩阵，权重来自 SCG 默认配置。
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str = 'cpu',
        state_weights: Optional[torch.Tensor] = None,
        action_weight: float = SCG_ACTION_WEIGHT,
    ):
        """
        Args:
            num_envs: 并行环境数量
            device: 计算设备
            state_weights: 12维状态权重，None则用SCG默认
            action_weight: 动作二次代价权重
        """
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # 状态权重 Q（对角元素）
        if state_weights is None:
            self.Q = SCG_STATE_WEIGHTS.clone().to(self.device)
        else:
            self.Q = state_weights.to(self.device)
        
        # 动作权重 R（标量，乘以单位阵）
        self.R = float(action_weight)
        
        # 累积统计（用于报告）
        self._total_state_cost = torch.zeros(num_envs, device=self.device)
        self._total_action_cost = torch.zeros(num_envs, device=self.device)
        self._steps = 0
        
        # 参考姿态（默认水平：roll=pitch=yaw=0）
        self.ref_euler = torch.zeros(3, device=self.device)
    
    def reset(self, num_envs: Optional[int] = None):
        """重置累积统计"""
        if num_envs is not None:
            self.num_envs = num_envs
        self._total_state_cost = torch.zeros(self.num_envs, device=self.device)
        self._total_action_cost = torch.zeros(self.num_envs, device=self.device)
        self._steps = 0
    
    def compute_step(
        self,
        pos: torch.Tensor,           # [N, 3] 当前位置 [x, y, z]
        vel: torch.Tensor,           # [N, 3] 当前速度 [vx, vy, vz]
        quat: torch.Tensor,          # [N, 4] 当前姿态四元数 [qx, qy, qz, qw]
        omega: torch.Tensor,         # [N, 3] 当前角速度 [ωx, ωy, ωz]
        target_pos: torch.Tensor,    # [N, 3] 或 [3] 目标位置 [x_ref, y_ref, z_ref]
        action: torch.Tensor,        # [N, 4] 或 [N, 6] 控制动作
        target_vel: Optional[torch.Tensor] = None,  # [N, 3] 目标速度（默认0）
        done_mask: Optional[torch.Tensor] = None,   # [N] bool，已结束的 env
    ) -> torch.Tensor:
        """计算单步 reward（精确 SCG 公式）
        
        Returns:
            reward: [N] 每个环境的即时 reward（负值）
        """
        N = pos.shape[0]
        
        # 目标速度默认为 0
        if target_vel is None:
            target_vel = torch.zeros(N, 3, device=self.device)
        
        # 确保 target_pos 是 [N, 3]
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0).expand(N, -1)
        
        # 四元数 -> 欧拉角
        euler = quat_to_euler(quat)  # [N, 3] = [roll, pitch, yaw]
        
        # 构建 12 维状态误差向量
        # SCG 顺序: [x, vx, y, vy, z, vz, φ, θ, ψ, ωx, ωy, ωz]
        pos_err = pos - target_pos                    # [N, 3]
        vel_err = vel - target_vel                    # [N, 3]
        euler_err = euler - self.ref_euler.unsqueeze(0)  # [N, 3]
        omega_err = omega                              # [N, 3]，参考角速度为 0
        
        # 交错排列成 SCG 顺序
        state_err = torch.zeros(N, 12, device=self.device)
        state_err[:, 0] = pos_err[:, 0]    # x
        state_err[:, 1] = vel_err[:, 0]    # vx
        state_err[:, 2] = pos_err[:, 1]    # y
        state_err[:, 3] = vel_err[:, 1]    # vy
        state_err[:, 4] = pos_err[:, 2]    # z
        state_err[:, 5] = vel_err[:, 2]    # vz
        state_err[:, 6] = euler_err[:, 0]  # φ (roll)
        state_err[:, 7] = euler_err[:, 1]  # θ (pitch)
        state_err[:, 8] = euler_err[:, 2]  # ψ (yaw)
        state_err[:, 9] = omega_err[:, 0]  # ωx
        state_err[:, 10] = omega_err[:, 1] # ωy
        state_err[:, 11] = omega_err[:, 2] # ωz
        
        # 状态代价: sum_i(Q_i * err_i^2)
        state_cost = (state_err ** 2) @ self.Q  # [N]
        
        # 动作代价: R * ||u||^2
        # 只取前 4 维（推力/力矩），忽略可能的 6 维扩展
        u = action[:, :4] if action.shape[1] >= 4 else action
        action_cost = self.R * (u ** 2).sum(dim=1)  # [N]
        
        # 总 reward（负值）
        reward = -(state_cost + action_cost)
        
        # 处理 done mask
        if done_mask is not None:
            active = (~done_mask).float()
            reward = reward * active
        
        # 累积统计
        self._total_state_cost += state_cost
        self._total_action_cost += action_cost
        self._steps += 1
        
        return reward
    
    def get_components(self) -> Dict[str, torch.Tensor]:
        """返回累积的各组件代价"""
        return {
            'state_cost': self._total_state_cost.clone(),
            'action_cost': self._total_action_cost.clone(),
            'total_cost': self._total_state_cost + self._total_action_cost,
            'steps': self._steps,
        }
    
    def get_mean_reward(self) -> torch.Tensor:
        """返回每步平均 reward"""
        if self._steps == 0:
            return torch.zeros(self.num_envs, device=self.device)
        total_cost = self._total_state_cost + self._total_action_cost
        return -total_cost / self._steps


__all__ = [
    'SCGExactRewardCalculator',
    'SCG_STATE_WEIGHTS',
    'SCG_ACTION_WEIGHT',
    'quat_to_euler',
]

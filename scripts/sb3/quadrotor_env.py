"""SB3 标准 Quadrotor 环境。

这是一个标准的 Gymnasium 接口四旋翼环境，遵循 SB3 的 reward 设计惯例：
    reward = -pos_error - ctrl_cost_weight * ||action||^2 + alive_bonus

设计理念：
1. 简洁透明：reward 公式简单，无隐藏 shaping
2. 标准接口：完全兼容 SB3 的 PPO/SAC/TD3
3. 可复现：与 MuJoCo 标准环境设计一致

使用示例:
    from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv
    from stable_baselines3 import PPO
    
    env = QuadrotorTrackingEnv(trajectory='figure8')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utilities.trajectory_presets import (
    get_scg_trajectory_config,
    scg_position_velocity,
)


class QuadrotorTrackingEnv(gym.Env):
    """SB3 标准四旋翼轨迹跟踪环境。
    
    观测空间 (18D):
        - pos_error: [3] 位置误差 (target - current)
        - velocity: [3] 线速度
        - euler: [3] 欧拉角 (roll, pitch, yaw)
        - omega: [3] 角速度
        - target_pos: [3] 目标位置（用于轨迹跟踪）
        - target_vel: [3] 目标速度（用于轨迹跟踪）
    
    动作空间 (4D):
        - [fz, tx, ty, tz]: 归一化推力和力矩 [-1, 1]
    
    Reward 设计（SB3 标准风格）:
        reward = alive_bonus - pos_cost - vel_cost - ctrl_cost - orientation_cost
        
        - alive_bonus: 每步存活奖励（默认 0.1）
        - pos_cost: pos_cost_weight * ||pos_error||  （位置误差）
        - vel_cost: vel_cost_weight * ||vel_error||  （速度误差）
        - ctrl_cost: ctrl_cost_weight * ||action||^2 （控制代价）
        - orientation_cost: orient_cost_weight * (|roll| + |pitch|) （姿态代价）
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    # 动作缩放参数
    ACTION_SCALE = np.array([3.5, 0.12, 0.12, 0.06], dtype=np.float32)
    ACTION_OFFSET = np.array([3.5, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # 物理参数
    MASS = 0.027  # kg (Crazyflie)
    GRAVITY = 9.81
    HOVER_THRUST = MASS * GRAVITY  # ~0.265 N
    
    # 默认 reward 权重（SB3 风格）
    DEFAULT_REWARD_WEIGHTS = {
        'alive_bonus': 0.1,
        'pos_cost_weight': 1.0,
        'vel_cost_weight': 0.1,
        'ctrl_cost_weight': 0.001,
        'orient_cost_weight': 0.1,
    }
    
    def __init__(
        self,
        trajectory: str = 'figure8',
        trajectory_params: Optional[Dict[str, Any]] = None,
        duration: float = 5.0,
        control_freq: int = 50,
        reward_weights: Optional[Dict[str, float]] = None,
        terminate_on_crash: bool = True,
        crash_height: float = 0.05,
        max_tilt: float = 1.0,  # ~57 degrees
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            trajectory: 轨迹类型 ('hover', 'figure8', 'circle', 'setpoint')
            trajectory_params: 轨迹参数
            duration: Episode 时长（秒）
            control_freq: 控制频率 (Hz)
            reward_weights: Reward 权重覆盖
            terminate_on_crash: 坠机时是否终止
            crash_height: 坠机高度阈值
            max_tilt: 最大倾斜角（弧度），超过则终止
            render_mode: 渲染模式
            seed: 随机种子
        """
        super().__init__()
        
        self.trajectory = trajectory
        self.trajectory_params = trajectory_params or {}
        self.duration = duration
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.max_steps = int(duration * control_freq)
        
        # Reward 权重
        self.reward_weights = {**self.DEFAULT_REWARD_WEIGHTS}
        if reward_weights:
            self.reward_weights.update(reward_weights)
        
        # 终止条件
        self.terminate_on_crash = terminate_on_crash
        self.crash_height = crash_height
        self.max_tilt = max_tilt
        
        self.render_mode = render_mode
        
        # 观测空间: [pos_err(3), vel(3), euler(3), omega(3), target_pos(3), target_vel(3)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        
        # 动作空间: [fz, tx, ty, tz] 归一化到 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # 状态
        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._euler = np.zeros(3, dtype=np.float32)  # roll, pitch, yaw
        self._omega = np.zeros(3, dtype=np.float32)
        
        # 目标
        self._target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._target_vel = np.zeros(3, dtype=np.float32)
        
        self._step_count = 0
        self._rng = np.random.default_rng(seed)
        
        # 轨迹配置
        self._setup_trajectory()
    
    def _setup_trajectory(self):
        """配置轨迹参数"""
        task = (self.trajectory or 'hover').lower()
        overrides = dict(self.trajectory_params or {})

        if task == 'setpoint':
            target = overrides.get('target') or overrides.get('center') or [0.0, 0.0, 1.0]
            self._traj_target = np.array(target, dtype=np.float32)
            self._traj_params = {}
            self._traj_center = self._traj_target.copy()
            self._traj_task = 'setpoint'
            return

        center_override = overrides.pop('center', None)
        height = overrides.pop('height', None)
        radius = overrides.pop('radius', None)
        if radius is not None and 'R' not in overrides:
            overrides['R'] = radius
        side_len = overrides.pop('side_len', None)
        if side_len is not None and 'scale' not in overrides:
            overrides['scale'] = side_len

        cfg = get_scg_trajectory_config(task, overrides=overrides)
        self._traj_params = cfg.params
        center = center_override
        if center is None and height is not None:
            center = (0.0, 0.0, float(height))
        self._traj_center = np.array(center if center is not None else cfg.center, dtype=np.float32)
        self._traj_task = cfg.task
    
    def _get_target(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算时刻 t 的目标位置和速度"""
        if getattr(self, '_traj_task', None) == 'setpoint':
            return self._traj_target.copy(), np.zeros(3, dtype=np.float32)

        pos, vel = scg_position_velocity(
            self._traj_task,
            t,
            params=self._traj_params,
            center=self._traj_center,
        )
        return pos.astype(np.float32), vel.astype(np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """获取观测"""
        pos_err = self._target_pos - self._pos
        return np.concatenate([
            pos_err,
            self._vel,
            self._euler,
            self._omega,
            self._target_pos,
            self._target_vel,
        ]).astype(np.float32)
    
    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """计算 reward（SB3 标准风格）"""
        w = self.reward_weights
        
        # 位置误差
        pos_err = self._target_pos - self._pos
        pos_cost = w['pos_cost_weight'] * np.linalg.norm(pos_err)
        
        # 速度误差
        vel_err = self._target_vel - self._vel
        vel_cost = w['vel_cost_weight'] * np.linalg.norm(vel_err)
        
        # 控制代价
        ctrl_cost = w['ctrl_cost_weight'] * np.sum(action ** 2)
        
        # 姿态代价（鼓励水平）
        orient_cost = w['orient_cost_weight'] * (
            abs(self._euler[0]) + abs(self._euler[1])
        )
        
        # 存活奖励
        alive_bonus = w['alive_bonus']
        
        # 总 reward
        reward = alive_bonus - pos_cost - vel_cost - ctrl_cost - orient_cost
        
        info = {
            'pos_cost': pos_cost,
            'vel_cost': vel_cost,
            'ctrl_cost': ctrl_cost,
            'orient_cost': orient_cost,
            'alive_bonus': alive_bonus,
            'pos_error': np.linalg.norm(pos_err),
        }
        
        return float(reward), info
    
    def _is_terminated(self) -> bool:
        """检查是否终止"""
        if not self.terminate_on_crash:
            return False
        
        # 坠机检测
        if self._pos[2] < self.crash_height:
            return True
        
        # 过度倾斜检测
        if abs(self._euler[0]) > self.max_tilt or abs(self._euler[1]) > self.max_tilt:
            return True
        
        return False
    
    def _simple_dynamics(self, action: np.ndarray):
        """简化的四旋翼动力学（用于快速仿真）
        
        这是一个简化模型，用于：
        1. 快速原型验证
        2. 与 Isaac Gym 训练结果对比
        
        真实测试应使用 Isaac Gym 或其他高保真仿真器。
        """
        # 反归一化动作
        real_action = action * self.ACTION_SCALE + self.ACTION_OFFSET
        fz, tx, ty, tz = real_action
        
        # 推力方向（body frame -> world frame）
        roll, pitch, yaw = self._euler
        
        # 简化旋转矩阵（小角度近似）
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # 推力在世界坐标系的分量
        thrust_world = np.array([
            fz * (sr * sy + cr * cy * sp),
            fz * (cr * sp * sy - cy * sr),
            fz * cr * cp,
        ], dtype=np.float32)
        
        # 线加速度
        acc = thrust_world / self.MASS - np.array([0, 0, self.GRAVITY])
        
        # 更新速度和位置
        self._vel += acc * self.dt
        self._pos += self._vel * self.dt
        
        # 角加速度（简化模型）
        # 假设力矩直接映射到角加速度
        inertia_approx = 1e-5  # 简化惯性
        alpha = np.array([tx, ty, tz]) / inertia_approx
        
        # 更新角速度和姿态
        self._omega += alpha * self.dt
        self._omega *= 0.98  # 阻尼
        self._euler += self._omega * self.dt
        
        # 限制姿态角
        self._euler = np.clip(self._euler, -np.pi/2, np.pi/2)
        
        # 地面碰撞
        if self._pos[2] < 0:
            self._pos[2] = 0
            self._vel[2] = 0
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # 初始化状态（略微随机扰动）
        noise_scale = 0.02
        self._pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._pos += self._rng.uniform(-noise_scale, noise_scale, 3).astype(np.float32)
        self._vel = self._rng.uniform(-0.01, 0.01, 3).astype(np.float32)
        self._euler = self._rng.uniform(-0.05, 0.05, 3).astype(np.float32)
        self._omega = self._rng.uniform(-0.1, 0.1, 3).astype(np.float32)
        
        self._step_count = 0
        
        # 更新目标
        self._target_pos, self._target_vel = self._get_target(0.0)
        
        obs = self._get_obs()
        info = {
            'pos': self._pos.copy(),
            'target_pos': self._target_pos.copy(),
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        action = np.asarray(action, dtype=np.float32).flatten()[:4]
        action = np.clip(action, -1.0, 1.0)
        
        # 动力学仿真
        self._simple_dynamics(action)
        
        # 更新目标
        t = (self._step_count + 1) * self.dt
        self._target_pos, self._target_vel = self._get_target(t)
        
        # 计算 reward
        reward, reward_info = self._compute_reward(action)
        
        self._step_count += 1
        
        # 检查终止
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_steps
        
        obs = self._get_obs()
        info = {
            'pos': self._pos.copy(),
            'vel': self._vel.copy(),
            'euler': self._euler.copy(),
            'target_pos': self._target_pos.copy(),
            'step': self._step_count,
            **reward_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """渲染（当前为占位符）"""
        if self.render_mode == "human":
            print(f"Step {self._step_count}: pos={self._pos}, target={self._target_pos}")
        return None
    
    def close(self):
        """关闭环境"""
        pass


# 注册环境
def register_quadrotor_envs():
    """注册四旋翼环境到 Gymnasium"""
    from gymnasium.envs.registration import register
    
    # 悬停任务
    register(
        id='QuadrotorHover-v0',
        entry_point='scripts.sb3.quadrotor_env:QuadrotorTrackingEnv',
        max_episode_steps=250,
        kwargs={'trajectory': 'hover', 'duration': 5.0},
    )
    
    # Figure-8 轨迹跟踪
    register(
        id='QuadrotorFigure8-v0',
        entry_point='scripts.sb3.quadrotor_env:QuadrotorTrackingEnv',
        max_episode_steps=250,
        kwargs={'trajectory': 'figure8', 'duration': 5.0},
    )
    
    # 圆形轨迹跟踪
    register(
        id='QuadrotorCircle-v0',
        entry_point='scripts.sb3.quadrotor_env:QuadrotorTrackingEnv',
        max_episode_steps=250,
        kwargs={'trajectory': 'circle', 'duration': 5.0},
    )


__all__ = [
    'QuadrotorTrackingEnv',
    'register_quadrotor_envs',
]

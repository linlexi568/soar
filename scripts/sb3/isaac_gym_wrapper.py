"""Isaac Gym 环境的 Gymnasium/SB3 封装器。

将 Soar 的 Isaac Gym 环境封装为标准 Gymnasium 接口，
支持 Stable-Baselines3 的 PPO/SAC/TD3 等算法。

使用方式:
    from scripts.sb3.isaac_gym_wrapper import make_isaac_vec_env
    
    vec_env = make_isaac_vec_env(
        trajectory_type='figure8',
        num_envs=64,
        reward_type='scg_exact',
    )
    
    from stable_baselines3 import PPO
    model = PPO('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=100000)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# 添加项目路径
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "01_soar"))

# Isaac Gym 环境
try:
    from envs.isaac_gym_drone_env import IsaacGymDroneEnv, ISAAC_GYM_AVAILABLE
except ImportError:
    try:
        from _01_soar.envs.isaac_gym_drone_env import IsaacGymDroneEnv, ISAAC_GYM_AVAILABLE
    except ImportError:
        ISAAC_GYM_AVAILABLE = False
        IsaacGymDroneEnv = None

# Reward 计算器
try:
    from utils.reward_scg_exact import SCGExactRewardCalculator
except ImportError:
    SCGExactRewardCalculator = None



class IsaacGymVecEnv(gym.Env):
    """Isaac Gym 向量化环境的 Gymnasium 封装。
    
    注意：这是一个"假"的单环境接口，内部实际运行多个并行环境。
    SB3 的 VecEnv 会进一步封装这个环境。
    
    观测空间 (12D):
        - pos_err: [3] 位置误差 (target - current)
        - vel: [3] 线速度
        - euler: [3] 欧拉角 (roll, pitch, yaw)
        - omega: [3] 角速度
    
    动作空间 (4D):
        - [fz, tx, ty, tz]: 推力和力矩（归一化到 [-1, 1]）
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    # 动作缩放参数（将 [-1, 1] 映射到实际范围）
    ACTION_SCALE = np.array([3.5, 0.1, 0.1, 0.05], dtype=np.float32)  # [fz, tx, ty, tz]
    ACTION_OFFSET = np.array([3.5, 0.0, 0.0, 0.0], dtype=np.float32)  # fz 偏移到 [0, 7]
    
    def __init__(
        self,
        trajectory_type: str = 'figure8',
        num_envs: int = 1,
        duration: float = 5.0,
        reward_type: str = 'scg_exact',
        trajectory_params: Optional[Dict[str, Any]] = None,
        device: str = 'cuda:0',
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if not ISAAC_GYM_AVAILABLE:
            raise RuntimeError("Isaac Gym 未安装，无法使用此环境")
        if reward_type != 'scg_exact':
            raise ValueError("IsaacGymVecEnv 现仅支持 'scg_exact' 奖励")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.render_mode = render_mode
        self.duration = float(duration)
        self.trajectory_type = trajectory_type
        self.reward_type = reward_type
        
        # 控制频率
        self.control_freq = 48  # Hz
        self.max_steps = int(duration * self.control_freq)
        self._step_count = 0
        
        # 轨迹配置
        self.trajectory_params = trajectory_params or {}
        self._trajectory_config = self._build_trajectory_config()
        
        # 观测空间: [pos_err(3), vel(3), euler(3), omega(3)] = 12D
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )
        
        # 动作空间: [fz, tx, ty, tz] 归一化到 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        
        # Isaac Gym 环境
        self._isaac_env: Optional[IsaacGymDroneEnv] = None
        
        # Reward 计算器
        self._reward_calc = None
        self._init_reward_calculator()
        
        # 当前目标位置
        self._target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
    def _build_trajectory_config(self) -> Dict[str, Any]:
        """构建轨迹配置"""
        if self.trajectory_type == 'hover':
            return {
                'type': 'hover',
                'height': self.trajectory_params.get('height', 1.0),
            }
        elif self.trajectory_type == 'figure8':
            return {
                'type': 'figure8',
                'params': {
                    'scale': self.trajectory_params.get('scale', 1.0),
                    'period': self.trajectory_params.get('period', 5.0),
                    'plane': self.trajectory_params.get('plane', 'xz'),
                    'center': self.trajectory_params.get('center', [0.0, 0.0, 1.0]),
                },
            }
        elif self.trajectory_type == 'circle':
            return {
                'type': 'circle',
                'params': {
                    'radius': self.trajectory_params.get('radius', 0.5),
                    'period': self.trajectory_params.get('period', 5.0),
                    'center': self.trajectory_params.get('center', [0.0, 0.0, 1.0]),
                },
            }
        else:
            return {'type': 'hover', 'height': 1.0}
    
    def _init_reward_calculator(self) -> None:
        """初始化 reward 计算器（仅支持 SCG 精确代价）。"""
        if SCGExactRewardCalculator is None:
            raise RuntimeError("SCGExactRewardCalculator 未安装，无法计算论文定义的奖励")
        self._reward_calc = SCGExactRewardCalculator(
            num_envs=self.num_envs,
            device=str(self.device),
        )
    
    def _get_isaac_env(self) -> IsaacGymDroneEnv:
        """延迟初始化 Isaac Gym 环境"""
        if self._isaac_env is None:
            self._isaac_env = IsaacGymDroneEnv(
                num_envs=self.num_envs,
                device=str(self.device),
                headless=True,
            )
        return self._isaac_env
    
    def _compute_target(self, t: float) -> np.ndarray:
        """计算时刻 t 的目标位置"""
        cfg = self._trajectory_config
        
        if cfg['type'] == 'hover':
            return np.array([0.0, 0.0, cfg.get('height', 1.0)], dtype=np.float32)
        
        elif cfg['type'] == 'figure8':
            params = cfg.get('params', {})
            scale = params.get('scale', 1.0)
            period = params.get('period', 5.0)
            plane = params.get('plane', 'xz')
            center = np.array(params.get('center', [0.0, 0.0, 1.0]), dtype=np.float32)
            
            omega = 2 * np.pi / period
            # Lemniscate of Bernoulli (figure-8)
            a = np.sin(omega * t)
            b = np.sin(omega * t) * np.cos(omega * t)
            
            if plane == 'xz':
                return center + np.array([scale * a, 0.0, scale * b], dtype=np.float32)
            elif plane == 'xy':
                return center + np.array([scale * a, scale * b, 0.0], dtype=np.float32)
            else:  # yz
                return center + np.array([0.0, scale * a, scale * b], dtype=np.float32)
        
        elif cfg['type'] == 'circle':
            params = cfg.get('params', {})
            radius = params.get('radius', 0.5)
            period = params.get('period', 5.0)
            center = np.array(params.get('center', [0.0, 0.0, 1.0]), dtype=np.float32)
            
            omega = 2 * np.pi / period
            return center + np.array([
                radius * np.cos(omega * t),
                radius * np.sin(omega * t),
                0.0,
            ], dtype=np.float32)
        
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """四元数 [qx, qy, qz, qw] -> 欧拉角 [roll, pitch, yaw]"""
        qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # Roll
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # Yaw
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.stack([roll, pitch, yaw], axis=-1)
    
    def _get_obs(self, env_idx: int = 0) -> np.ndarray:
        """获取单个环境的观测"""
        env = self._get_isaac_env()
        states = env.get_states_batch()
        
        pos = states['pos'][env_idx].cpu().numpy()
        vel = states['vel'][env_idx].cpu().numpy()
        quat = states['quat'][env_idx].cpu().numpy()
        omega = states['omega'][env_idx].cpu().numpy()
        
        euler = self._quat_to_euler(quat)
        pos_err = self._target_pos - pos
        
        obs = np.concatenate([pos_err, vel, euler, omega]).astype(np.float32)
        return obs
    
    def _get_all_obs(self) -> np.ndarray:
        """获取所有环境的观测 [num_envs, 12]"""
        env = self._get_isaac_env()
        states = env.get_states_batch()
        
        pos = states['pos'].cpu().numpy()  # [N, 3]
        vel = states['vel'].cpu().numpy()
        quat = states['quat'].cpu().numpy()
        omega = states['omega'].cpu().numpy()
        
        euler = self._quat_to_euler(quat)  # [N, 3]
        pos_err = self._target_pos.reshape(1, 3) - pos  # [N, 3]
        
        obs = np.concatenate([pos_err, vel, euler, omega], axis=1).astype(np.float32)
        return obs
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        env = self._get_isaac_env()
        env.reset()
        
        self._step_count = 0
        self._target_pos = self._compute_target(0.0)
        
        if self._reward_calc is not None:
            self._reward_calc.reset(self.num_envs)
        
        obs = self._get_obs(0)
        info = {'target_pos': self._target_pos.copy()}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作（单环境接口）"""
        # 反归一化动作
        action = np.asarray(action, dtype=np.float32).flatten()[:4]
        real_action = action * self.ACTION_SCALE + self.ACTION_OFFSET
        
        # 构建 Isaac Gym 动作格式 [N, 6]: [0, 0, fz, tx, ty, tz]
        env = self._get_isaac_env()
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        actions[:, 2] = float(real_action[0])  # fz
        actions[:, 3] = float(real_action[1])  # tx
        actions[:, 4] = float(real_action[2])  # ty
        actions[:, 5] = float(real_action[3])  # tz
        
        # 步进
        obs_dict, _, dones, infos = env.step(actions)
        
        # 更新目标
        t = (self._step_count + 1) / self.control_freq
        self._target_pos = self._compute_target(t)
        
        # 计算 reward
        states = env.get_states_batch()
        reward = self._compute_reward(states, actions, dones)
        
        self._step_count += 1
        
        # 检查终止
        terminated = bool(dones[0].item())
        truncated = self._step_count >= self.max_steps
        
        obs = self._get_obs(0)
        info = {
            'target_pos': self._target_pos.copy(),
            'pos': states['pos'][0].cpu().numpy(),
            'step': self._step_count,
        }
        
        return obs, float(reward), terminated, truncated, info
    
    def _compute_reward(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """计算 reward"""
        pos = states['pos']
        vel = states['vel']
        quat = states['quat']
        omega = states['omega']
        target = torch.tensor(self._target_pos, device=self.device)
        if self._reward_calc is None:
            pos_err = torch.norm(pos[0] - target)
            return float(-pos_err.item())

        target_pos = target.view(1, 3).expand(pos.shape[0], -1)
        reward = self._reward_calc.compute_step(
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            target_pos=target_pos,
            action=actions[:, 2:6],
            done_mask=dones.bool(),
        )
        return float(reward[0].item())
    
    def render(self):
        """渲染（当前不支持）"""
        pass
    
    def close(self):
        """关闭环境"""
        if self._isaac_env is not None:
            self._isaac_env.close()
            self._isaac_env = None


class IsaacGymSB3VecEnv:
    """原生支持 SB3 VecEnv 接口的 Isaac Gym 封装。
    
    这是一个真正的向量化环境，直接利用 Isaac Gym 的并行能力。
    不需要 SubprocVecEnv 或 DummyVecEnv 封装。
    """
    
    # 动作缩放
    ACTION_SCALE = np.array([3.5, 0.1, 0.1, 0.05], dtype=np.float32)
    ACTION_OFFSET = np.array([3.5, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def __init__(
        self,
        num_envs: int = 64,
        trajectory_type: str = 'figure8',
        trajectory_params: Optional[Dict[str, Any]] = None,
        duration: float = 5.0,
        reward_type: str = 'scg_exact',
        device: str = 'cuda:0',
    ):
        if reward_type != 'scg_exact':
            raise ValueError("IsaacGymSB3VecEnv 仅支持 'scg_exact' 奖励")
        if not ISAAC_GYM_AVAILABLE:
            raise RuntimeError("Isaac Gym 未安装，无法使用此环境")

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.duration = duration
        self.trajectory_type = trajectory_type
        self.reward_type = reward_type
        self.trajectory_params = trajectory_params or {}
        
        # 控制频率
        self.control_freq = 48
        self.max_steps = int(duration * self.control_freq)
        self._step_counts = np.zeros(num_envs, dtype=np.int32)
        
        # 空间定义
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Isaac Gym 环境
        self._isaac_env: Optional[IsaacGymDroneEnv] = None
        
        # Reward 计算器
        self._reward_calc = None
        self._init_reward_calculator()
        
        # 轨迹配置
        self._trajectory_config = self._build_trajectory_config()
        
        # 目标位置 [N, 3]
        self._target_pos = np.zeros((num_envs, 3), dtype=np.float32)
        self._target_pos[:, 2] = 1.0
    
    def _build_trajectory_config(self) -> Dict[str, Any]:
        """构建轨迹配置"""
        if self.trajectory_type == 'hover':
            return {'type': 'hover', 'height': self.trajectory_params.get('height', 1.0)}
        elif self.trajectory_type == 'figure8':
            return {
                'type': 'figure8',
                'params': {
                    'scale': self.trajectory_params.get('scale', 1.0),
                    'period': self.trajectory_params.get('period', 5.0),
                    'plane': self.trajectory_params.get('plane', 'xz'),
                    'center': self.trajectory_params.get('center', [0.0, 0.0, 1.0]),
                },
            }
        elif self.trajectory_type == 'circle':
            return {
                'type': 'circle',
                'params': {
                    'radius': self.trajectory_params.get('radius', 0.5),
                    'period': self.trajectory_params.get('period', 5.0),
                    'center': self.trajectory_params.get('center', [0.0, 0.0, 1.0]),
                },
            }
        return {'type': 'hover', 'height': 1.0}
    
    def _init_reward_calculator(self) -> None:
        if SCGExactRewardCalculator is None:
            raise RuntimeError("SCGExactRewardCalculator 未安装，无法计算论文奖励")
        self._reward_calc = SCGExactRewardCalculator(
            num_envs=self.num_envs,
            device=str(self.device),
        )
    
    def _get_isaac_env(self) -> IsaacGymDroneEnv:
        if self._isaac_env is None:
            self._isaac_env = IsaacGymDroneEnv(
                num_envs=self.num_envs, device=str(self.device), headless=True
            )
        return self._isaac_env
    
    def _compute_targets(self, step_counts: np.ndarray) -> np.ndarray:
        """批量计算目标位置 [N, 3]"""
        cfg = self._trajectory_config
        t = step_counts / self.control_freq
        N = len(step_counts)
        
        if cfg['type'] == 'hover':
            targets = np.zeros((N, 3), dtype=np.float32)
            targets[:, 2] = cfg.get('height', 1.0)
            return targets
        
        elif cfg['type'] == 'figure8':
            params = cfg.get('params', {})
            scale = params.get('scale', 1.0)
            period = params.get('period', 5.0)
            plane = params.get('plane', 'xz')
            center = np.array(params.get('center', [0.0, 0.0, 1.0]), dtype=np.float32)
            
            omega = 2 * np.pi / period
            a = np.sin(omega * t)
            b = np.sin(omega * t) * np.cos(omega * t)
            
            targets = np.tile(center, (N, 1))
            if plane == 'xz':
                targets[:, 0] += scale * a
                targets[:, 2] += scale * b
            elif plane == 'xy':
                targets[:, 0] += scale * a
                targets[:, 1] += scale * b
            else:
                targets[:, 1] += scale * a
                targets[:, 2] += scale * b
            return targets.astype(np.float32)
        
        elif cfg['type'] == 'circle':
            params = cfg.get('params', {})
            radius = params.get('radius', 0.5)
            period = params.get('period', 5.0)
            center = np.array(params.get('center', [0.0, 0.0, 1.0]), dtype=np.float32)
            
            omega = 2 * np.pi / period
            targets = np.tile(center, (N, 1))
            targets[:, 0] += radius * np.cos(omega * t)
            targets[:, 1] += radius * np.sin(omega * t)
            return targets.astype(np.float32)
        
        targets = np.zeros((N, 3), dtype=np.float32)
        targets[:, 2] = 1.0
        return targets
    
    def _quat_to_euler_batch(self, quat: np.ndarray) -> np.ndarray:
        """批量四元数转欧拉角"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.stack([roll, pitch, yaw], axis=1).astype(np.float32)
    
    def _get_all_obs(self) -> np.ndarray:
        """获取所有观测 [N, 12]"""
        env = self._get_isaac_env()
        states = env.get_states_batch()
        
        pos = states['pos'].cpu().numpy()
        vel = states['vel'].cpu().numpy()
        quat = states['quat'].cpu().numpy()
        omega = states['omega'].cpu().numpy()
        
        euler = self._quat_to_euler_batch(quat)
        pos_err = self._target_pos - pos
        
        return np.concatenate([pos_err, vel, euler, omega], axis=1).astype(np.float32)
    
    def reset(self) -> np.ndarray:
        """重置所有环境，返回观测 [N, 12]"""
        env = self._get_isaac_env()
        env.reset()
        
        self._step_counts[:] = 0
        self._target_pos = self._compute_targets(self._step_counts)
        
        if self._reward_calc is not None:
            self._reward_calc.reset(self.num_envs)
        
        return self._get_all_obs()
    
    def step(self, actions: np.ndarray):
        """向量化步进
        
        Args:
            actions: [N, 4] 归一化动作
        
        Returns:
            obs: [N, 12]
            rewards: [N]
            dones: [N]
            infos: [N] dicts
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        # 反归一化
        real_actions = actions * self.ACTION_SCALE + self.ACTION_OFFSET
        
        # 构建 Isaac 动作
        env = self._get_isaac_env()
        isaac_actions = torch.zeros((self.num_envs, 6), device=self.device)
        isaac_actions[:, 2] = torch.from_numpy(real_actions[:, 0]).to(self.device)
        isaac_actions[:, 3] = torch.from_numpy(real_actions[:, 1]).to(self.device)
        isaac_actions[:, 4] = torch.from_numpy(real_actions[:, 2]).to(self.device)
        isaac_actions[:, 5] = torch.from_numpy(real_actions[:, 3]).to(self.device)
        
        # 步进
        obs_dict, env_rewards, dones_tensor, infos = env.step(isaac_actions)
        
        # 更新步数和目标
        self._step_counts += 1
        self._target_pos = self._compute_targets(self._step_counts)
        
        # 计算 reward
        states = env.get_states_batch()
        rewards = self._compute_rewards_batch(states, isaac_actions, dones_tensor)
        
        # 检查截断
        truncated = self._step_counts >= self.max_steps
        dones = dones_tensor.cpu().numpy() | truncated
        
        # 处理 done 的环境（自动 reset）
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            env_ids_tensor = torch.tensor(done_indices, dtype=torch.long, device=self.device)
            env.reset(env_ids=env_ids_tensor)
            self._step_counts[done_indices] = 0
        
        obs = self._get_all_obs()
        
        # 构建 info
        infos_list = [{'TimeLimit.truncated': bool(truncated[i])} for i in range(self.num_envs)]
        
        return obs, rewards, dones, infos_list
    
    def _compute_rewards_batch(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> np.ndarray:
        """批量计算 reward"""
        pos = states['pos']
        vel = states['vel']
        quat = states['quat']
        omega = states['omega']
        target = torch.from_numpy(self._target_pos).to(self.device)
        if self._reward_calc is None:
            pos_err = torch.norm(pos - target, dim=1)
            return -pos_err.cpu().numpy()

        target_pos = target.view(self.num_envs, 3)
        rewards = self._reward_calc.compute_step(
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            target_pos=target_pos,
            action=actions[:, 2:6],
            done_mask=dones.bool(),
        )
        return rewards.cpu().numpy()
    
    def close(self):
        if self._isaac_env is not None:
            self._isaac_env.close()
            self._isaac_env = None
    
    # SB3 VecEnv 兼容属性
    @property
    def unwrapped(self):
        return self
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
    
    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs
    
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs
    
    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)
    
    def seed(self, seed=None):
        pass


def make_isaac_vec_env(
    trajectory_type: str = 'figure8',
    num_envs: int = 64,
    reward_type: str = 'scg_exact',
    duration: float = 5.0,
    device: str = 'cuda:0',
    **kwargs,
) -> IsaacGymSB3VecEnv:
    """创建 SB3 兼容的 Isaac Gym 向量化环境。
    
    Args:
        trajectory_type: 轨迹类型 ('hover', 'figure8', 'circle')
        num_envs: 并行环境数
        reward_type: 仅支持 'scg_exact'
        duration: Episode 时长（秒）
        device: GPU 设备
    
    Returns:
        SB3 兼容的向量化环境
    """
    return IsaacGymSB3VecEnv(
        num_envs=num_envs,
        trajectory_type=trajectory_type,
        duration=duration,
        reward_type=reward_type,
        device=device,
        **kwargs,
    )


__all__ = [
    'IsaacGymVecEnv',
    'IsaacGymSB3VecEnv',
    'make_isaac_vec_env',
]

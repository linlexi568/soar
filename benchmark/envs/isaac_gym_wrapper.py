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

# 添加项目路径 (必须在导入 torch/sb3 之前，以便优先导入 isaacgym)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "01_soar"))

# 尝试导入 Isaac Gym (必须在 torch 之前)
try:
    from envs.isaac_gym_drone_env import IsaacGymDroneEnv, ISAAC_GYM_AVAILABLE
except ImportError:
    try:
        from _01_soar.envs.isaac_gym_drone_env import IsaacGymDroneEnv, ISAAC_GYM_AVAILABLE
    except ImportError:
        ISAAC_GYM_AVAILABLE = False
        IsaacGymDroneEnv = None

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

# 微型无人机（Crazyflie 量级）动作范围
# 提升推力下限，防止初始策略给出接近0导致自由落体；收紧扭矩，减少翻滚爆炸
# [2025-12-13 FIX] 放宽限制以匹配 PID/LQR 基线的能力
THRUST_MIN = 0.0      # 允许自由落体 (PID/LQR 为 0.0)
THRUST_MAX = 1.5      # 允许更大推力 (PID/LQR 约为 2.0*mg ≈ 0.53，这里给足余量)
# 稍微放宽扭矩，便于姿态跟踪
# [2025-12-13 FIX] 从 0.008 提升到 0.05，否则无法跟踪 Figure8
TORQUE_XY_MAX = 0.05
TORQUE_Z_MAX = 0.01
ACTION_LOW = np.array([THRUST_MIN, -TORQUE_XY_MAX, -TORQUE_XY_MAX, -TORQUE_Z_MAX], dtype=np.float32)
ACTION_HIGH = np.array([THRUST_MAX, TORQUE_XY_MAX, TORQUE_XY_MAX, TORQUE_Z_MAX], dtype=np.float32)

# Reward 计算器
try:
    from utils.reward_scg_exact import SCGExactRewardCalculator
except ImportError:
    SCGExactRewardCalculator = None

try:
    from scripts.adapters.safecontrol_soar_adapter import SoarPolicyAdapter
except Exception:
    SoarPolicyAdapter = None

from utilities.trajectory_presets import scg_position_velocity, get_scg_trajectory_config


class IsaacGymVecEnv(gym.Env):
    """Isaac Gym 向量化环境的 Gymnasium 封装。
    
    注意：这是一个"假"的单环境接口，内部实际运行多个并行环境。
    SB3 的 VecEnv 会进一步封装这个环境。
    
    观测空间 (12D):
        - pos_err: [3] 位置误差 (target - current)
        - vel: [3] 线速度
        - euler: [3] 欧拉角 (roll, pitch, yaw)
        - omega: [3] 角速度

    动作空间 (4D) —— 已是物理值:
        - fz: 推力 [0, 1] N（≈3.7×重力上限）
        - tx: x 轴力矩 [-0.01, 0.01] N·m
        - ty: y 轴力矩 [-0.01, 0.01] N·m
        - tz: z 轴偏航力矩 [-0.005, 0.005] N·m
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
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
        
        # 轨迹配置
        self.trajectory_params = trajectory_params or {}
        
        # 终止惩罚基数（用于早死惩罚衰减）：越早死扣得越多，默认 20，对应 -20..0
        self.crash_base_penalty = float(self.trajectory_params.get('crash_base_penalty', 20.0))
        self._trajectory_config = self._build_trajectory_config()
        
        # 观测空间: [pos_err(3), vel(3), euler(3), omega(3)] = 12D
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )
        
        # 动作空间: [fz, tx, ty, tz] 直接输出物理值（不归一化）
        # fz: [0, 7] N, tx/ty: [-0.1, 0.1] N·m, tz: [-0.05, 0.05] N·m
        self.action_space = spaces.Box(
            low=ACTION_LOW.copy(),
            high=ACTION_HIGH.copy(),
            shape=(4,),
            dtype=np.float32,
        )
        
        # Isaac Gym 环境
        self._isaac_env: Optional[IsaacGymDroneEnv] = None
        
        # Reward 计算器
        self._reward_calc = None
        self._init_reward_calculator()
        
        # Residual Adapter
        self._residual_adapter = None
        
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
        elif self.trajectory_type == 'square':
            return {
                'type': 'square',
                'params': {
                    'side_length': self.trajectory_params.get('side_length', 1.0),
                    'period': self.trajectory_params.get('period', 8.0),
                    'center': self.trajectory_params.get('center', [0.0, 0.0, 1.0]),
                },
            }
        elif self.trajectory_type == 'helix':
            return {
                'type': 'helix',
                'params': {
                    'radius': self.trajectory_params.get('radius', 0.5),
                    'period': self.trajectory_params.get('period', 5.0),
                    'pitch_per_rev': self.trajectory_params.get('pitch_per_rev', 0.2),
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
            # 如果 render_mode 是 'human'，则开启有头模式
            headless = (self.render_mode != 'human')
            self._isaac_env = IsaacGymDroneEnv(
                num_envs=self.num_envs,
                device=str(self.device),
                headless=headless,
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
        elif cfg['type'] == 'square':
            params = cfg.get('params', {})
            side = params.get('side_length', 1.0)
            period = params.get('period', 8.0)
            center = np.array(params.get('center', [0.0, 0.0, 1.0]), dtype=np.float32)

            phase = (t % period) / period * 4.0
            seg = int(np.floor(phase))
            u = phase - seg
            half = side * 0.5
            if seg == 0:
                offset = np.array([-half + u * side, -half, 0.0], dtype=np.float32)
            elif seg == 1:
                offset = np.array([half, -half + u * side, 0.0], dtype=np.float32)
            elif seg == 2:
                offset = np.array([half - u * side, half, 0.0], dtype=np.float32)
            else:
                offset = np.array([-half, half - u * side, 0.0], dtype=np.float32)
            return center + offset
        elif cfg['type'] == 'helix':
            params = cfg.get('params', {})
            radius = params.get('radius', 0.5)
            period = params.get('period', 5.0)
            pitch_per_rev = params.get('pitch_per_rev', 0.2)
            center = np.array(params.get('center', [0.0, 0.0, 1.0]), dtype=np.float32)

            omega = 2 * np.pi / period
            angle = omega * t
            height = pitch_per_rev * (t / period)
            return center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height,
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
        
        # 计算轨迹起点 (t=0)
        self._step_count = 0
        self._target_pos = self._compute_target(0.0)
        
        # 重置环境，无人机spawn在轨迹起点
        env = self._get_isaac_env()
        env.reset(initial_pos=self._target_pos)
        
        if self._reward_calc is not None:
            self._reward_calc.reset(self.num_envs)
        
        obs = self._get_obs(0)
        info = {'target_pos': self._target_pos.copy()}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作（单环境接口）"""
        # 动作已经是物理值，直接使用，并在物理范围内裁剪，防止初期策略输出过大推力
        action = np.asarray(action, dtype=np.float32).flatten()[:4]
        real_action = np.clip(action, ACTION_LOW, ACTION_HIGH)
        
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
        if self._residual_adapter is not None:
            try:
                self._residual_adapter.close()
            except Exception:
                pass
            self._residual_adapter = None


class IsaacGymSB3VecEnv(VecEnv):
    """原生支持 SB3 VecEnv 接口的 Isaac Gym 封装。
    
    这是一个真正的向量化环境，直接利用 Isaac Gym 的并行能力。
    不需要 SubprocVecEnv 或 DummyVecEnv 封装。
    """
    
    # 不再需要动作缩放（直接使用物理值）
    
    def __init__(
        self,
        num_envs: int = 64,
        trajectory_type: str = 'figure8',
        trajectory_params: Optional[Dict[str, Any]] = None,
        duration: float = 5.0,
        reward_type: str = 'scg_exact',
        device: str = 'cuda:0',
        shaping_cfg: Optional[Dict[str, float]] = None,
        residual_cfg: Optional[Dict[str, Any]] = None,
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
        self.shaping_cfg = shaping_cfg or {}
        self.residual_cfg = residual_cfg or {}
        self.residual_scale = float(self.residual_cfg.get('residual_scale', 0.5))  # RL残差范围限制
        self._pending_actions = None  # for SB3 VecEnv API
        
        # 控制频率
        self.control_freq = 48
        self.max_steps = int(duration * self.control_freq)
        self._step_counts = np.zeros(num_envs, dtype=np.int32)
        
        # 空间定义（动作直接使用物理值）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=ACTION_LOW.copy(),
            high=ACTION_HIGH.copy(),
            shape=(4,),
            dtype=np.float32
        )

        super().__init__(num_envs=self.num_envs, observation_space=self.observation_space, action_space=self.action_space)
        
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
        
        # Episode 统计追踪
        self._episode_returns = np.zeros(num_envs, dtype=np.float32)          # shaped: two-track after episode end
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self._last_crash_penalty = np.zeros(num_envs, dtype=np.float32)
        self._episode_raw_returns = np.zeros(num_envs, dtype=np.float32)      # SCG 原始代价累计（未 clamp、未加存活奖、未加终止罚）
        self._episode_survival_bonus = np.zeros(num_envs, dtype=np.float32)   # 存活奖励累计（alive_bonus * steps）
        self._total_env_steps = 0  # 累计环境步数（用于调度）
        self.use_pure_scg = bool(self.shaping_cfg.get('use_pure_scg', False))

        # 奖励重加权/课程参数：以“活得久”为主导；若 use_pure_scg=True 则忽略这些 shaping
        self.crash_base_penalty = float(self.shaping_cfg.get('crash_base_penalty', 80.0))
        self.crash_slope = float(self.shaping_cfg.get('crash_slope', 80.0))
        self.survival_bonus_scale = float(self.shaping_cfg.get('survival_bonus_scale', 0.0))
        self.alive_bonus = float(self.shaping_cfg.get('alive_bonus', 3.5))
        self.completion_bonus = float(self.shaping_cfg.get('completion_bonus', 50.0))
        self.scg_pos_alpha = float(self.shaping_cfg.get('scg_pos_alpha', 100.0))
        self.scg_pos_eps = float(self.shaping_cfg.get('scg_pos_eps', 1e-3))
        # 旧的两赛道课程逻辑暂时关闭，仅保留占位
        self._per_step_bonus_lut = np.array([1.0], dtype=np.float32)
        self._per_step_bonus_scale = 0.0
        self._last_crash_info = {}
        self._residual_adapter = None
        self._init_residual_adapter()
    
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
        elif self.trajectory_type == 'square':
            return {
                'type': 'square',
                'params': {
                    'side_length': self.trajectory_params.get('side_length', 1.0),
                    'period': self.trajectory_params.get('period', 8.0),
                    'center': self.trajectory_params.get('center', [0.0, 0.0, 1.0]),
                },
            }
        elif self.trajectory_type == 'helix':
            return {
                'type': 'helix',
                'params': {
                    'radius': self.trajectory_params.get('radius', 0.5),
                    'period': self.trajectory_params.get('period', 5.0),
                    'pitch_per_rev': self.trajectory_params.get('pitch_per_rev', 0.2),
                    'center': self.trajectory_params.get('center', [0.0, 0.0, 1.0]),
                },
            }
        elif self.trajectory_type == 'waypoints':
            points = np.asarray(self.trajectory_params.get('points', [[0.0, 0.0, 1.0]]), dtype=np.float32)
            hold_steps = int(self.trajectory_params.get('hold_steps', self.control_freq))  # default 1s/point
            cycle = bool(self.trajectory_params.get('cycle', True))
            return {
                'type': 'waypoints',
                'points': points,
                'hold_steps': max(1, hold_steps),
                'cycle': cycle,
            }
        return {'type': 'hover', 'height': 1.0}
    
    def _init_reward_calculator(self) -> None:
        if SCGExactRewardCalculator is None:
            raise RuntimeError("SCGExactRewardCalculator 未安装，无法计算论文奖励")
        self._reward_calc = SCGExactRewardCalculator(
            num_envs=self.num_envs,
            device=str(self.device),
        )

    def _init_residual_adapter(self) -> None:
        """初始化残差学习基线控制器（强制使用 PID）。"""
        enabled = bool(self.residual_cfg.get('enabled'))
        if not enabled or SoarPolicyAdapter is None:
            self._residual_adapter = None
            return
        pid_params = self.residual_cfg.get('pid_params') if isinstance(self.residual_cfg, dict) else None
        adapter_kwargs: Dict[str, Any] = {
            'mode': 'pid',
            'action_space': 'thrust_torque',
            'device': str(self.device),
            'pid_params': pid_params,
        }
        try:
            self._residual_adapter = SoarPolicyAdapter(**adapter_kwargs)
            print("[Residual] PID 基线已启用")
        except Exception as exc:
            print(f"[Residual] 初始化 PID 基线失败: {exc}")
            self._residual_adapter = None

    # 允许外部在构建后注入/替换残差基线控制器
    def set_residual_adapter(self, adapter: Any):
        """手动设置残差基线控制器，用于 SB3 训练脚本注入 PID/CPID."""
        self._residual_adapter = adapter
    
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
        elif cfg['type'] == 'square':
            params = cfg.get('params', {})
            side = params.get('side_length', 1.0)
            period = params.get('period', 8.0)
            center = np.array(params.get('center', [0.0, 0.0, 1.0]), dtype=np.float32)

            phase = (np.mod(t, period) / period) * 4.0
            seg = np.floor(phase).astype(np.int64)
            u = phase - seg
            targets = np.tile(center, (N, 1))

            # Match scg_position_velocity: start at center [0,0], go +y, then -x, then -y, then +x
            # segment 0: along +y (from 0 to side)
            # segment 1: along -x (from 0 to -side)
            # segment 2: along -y (from side to 0)
            # segment 3: along +x (from -side to 0)
            mask0 = seg == 0
            mask1 = seg == 1
            mask2 = seg == 2
            mask3 = seg >= 3

            # seg 0: y goes from 0 to side
            targets[mask0, 1] += u[mask0] * side

            # seg 1: y = side, x goes from 0 to -side
            targets[mask1, 0] += -u[mask1] * side
            targets[mask1, 1] += side

            # seg 2: x = -side, y goes from side to 0
            targets[mask2, 0] += -side
            targets[mask2, 1] += side - u[mask2] * side

            # seg 3: y = 0, x goes from -side to 0
            targets[mask3, 0] += -side + u[mask3] * side

            return targets.astype(np.float32)
        elif cfg['type'] == 'helix':
            params = cfg.get('params', {})
            radius = params.get('radius', 0.5)
            period = params.get('period', 5.0)
            pitch_per_rev = params.get('pitch_per_rev', 0.2)
            center = np.array(params.get('center', [0.0, 0.0, 1.0]), dtype=np.float32)

            omega = 2 * np.pi / period
            angle = omega * t
            height = pitch_per_rev * (t / period)
            targets = np.tile(center, (N, 1))
            targets[:, 0] += radius * np.cos(angle)
            targets[:, 1] += radius * np.sin(angle)
            targets[:, 2] += height
            return targets.astype(np.float32)
        elif cfg['type'] == 'waypoints':
            points: np.ndarray = cfg.get('points', np.zeros((1, 3), dtype=np.float32))
            hold_steps: int = int(cfg.get('hold_steps', self.control_freq))
            cycle: bool = bool(cfg.get('cycle', True))
            if points.ndim != 2 or points.shape[1] != 3:
                points = np.zeros((1, 3), dtype=np.float32)
                points[0, 2] = 1.0
            idx = (step_counts // max(1, hold_steps)).astype(np.int64)
            if cycle:
                idx = np.mod(idx, len(points))
            else:
                idx = np.clip(idx, 0, len(points) - 1)
            targets = points[idx]
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

    def _compute_residual_actions(self, states: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
        """使用基线控制器计算当前状态的动作（返回物理空间的原始输出）。"""
        if self._residual_adapter is None:
            return None
        try:
            pos = states['pos'].cpu().numpy()
            vel = states['vel'].cpu().numpy()
            quat = states['quat'].cpu().numpy()
            omega = states['omega'].cpu().numpy()
        except Exception:
            return None
        euler = self._quat_to_euler_batch(quat)
        pos_err = self._target_pos - pos
        base_raw = np.zeros((self.num_envs, 4), dtype=np.float32)
        for i in range(self.num_envs):
            obs_dict = {
                'pos_err': pos_err[i],
                'vel': vel[i],
                'rpy': euler[i],
                'omega': omega[i],
                'rpy_err': euler[i],
            }
            try:
                base = self._residual_adapter.act(obs_dict)
            except Exception:
                base = np.zeros(4, dtype=np.float32)
            base_raw[i] = np.asarray(base, dtype=np.float32)
        # 直接返回物理空间的PID输出，不归一化！
        return base_raw
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """重置所有环境，返回观测 [N, 12]"""
        env = self._get_isaac_env()
        if seed is not None:
            try:
                env.seed(seed)
            except AttributeError:
                pass
        
        # 计算轨迹起点 (t=0)
        self._step_counts[:] = 0
        self._target_pos = self._compute_targets(self._step_counts)
        
        # 重置环境，所有无人机spawn在轨迹起点
        # 注意：向量化环境中所有环境共享同一轨迹起点
        initial_pos_np = self._target_pos[0].cpu().numpy() if isinstance(self._target_pos, torch.Tensor) else self._target_pos[0]

        # [FIX] Circle 任务强制确保从圆周开始（防止配置错误导致从圆心开始）
        if self.trajectory_type == 'circle':
            radius = float(self.trajectory_params.get('radius', 0.5))
            center = self.trajectory_params.get('center', [0.0, 0.0, 1.0])
            xy_dist = np.linalg.norm(initial_pos_np[:2] - np.array(center[:2]))
            if xy_dist < 0.1 * radius:
                # 计算异常，强制修正为圆周起点
                initial_pos_np = np.array([center[0] + radius, center[1], center[2]], dtype=np.float32)

        env.reset(initial_pos=initial_pos_np)
        
        # 重置 episode 统计
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        self._last_crash_penalty[:] = 0.0
        self._episode_raw_returns[:] = 0.0
        self._episode_survival_bonus[:] = 0.0
        self._total_env_steps = 0
        self._last_crash_info = {}
        
        if self._reward_calc is not None:
            self._reward_calc.reset(self.num_envs)

        obs = self._get_all_obs()
        self._reset_seeds()
        self._reset_options()
        return obs

    # SB3 VecEnv API: async step interface
    def step_async(self, actions: np.ndarray):
        self._pending_actions = actions

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_wait called before step_async")
        actions = self._pending_actions
        self._pending_actions = None
        return self.step(actions)
    
    def step(self, actions: np.ndarray):
        """向量化步进
        
        Args:
            actions: [N, 4] 物理动作 (fz/tx/ty/tz)
        
        Returns:
            obs: [N, 12]
            rewards: [N]
            dones: [N]
            infos: [N] dicts
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        env = self._get_isaac_env()

        # PID 残差（如有）
        if self._residual_adapter is not None:
            states_before = env.get_states_batch()
            pid_baseline = self._compute_residual_actions(states_before)  # 物理空间 [N, 4]
            if pid_baseline is not None:
                max_residual = np.abs(pid_baseline) * self.residual_scale
                rl_residual_physical = np.clip(actions, -max_residual, max_residual)
                real_actions = pid_baseline + rl_residual_physical
                self._pid_magnitude = float(np.mean(np.abs(pid_baseline)))
                self._rl_magnitude = float(np.mean(np.abs(rl_residual_physical)))
            else:
                real_actions = actions
                self._pid_magnitude = 0.0
                self._rl_magnitude = float(np.mean(np.abs(real_actions)))
        else:
            real_actions = actions
            self._pid_magnitude = 0.0
            self._rl_magnitude = float(np.mean(np.abs(real_actions)))

        # 裁剪到安全范围
        real_actions = np.clip(real_actions, ACTION_LOW, ACTION_HIGH)
        self._last_actions = np.asarray(real_actions, dtype=np.float32)

        # 构建 Isaac 动作
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
        self._total_env_steps += self.num_envs

        # 计算 reward（每步即时奖励）
        states = env.get_states_batch()
        _, raw_step_rewards = self._compute_rewards_batch(states, isaac_actions, dones_tensor)
        scg_step_rewards = raw_step_rewards.astype(np.float32)

        if self.use_pure_scg:
            # 纯 SCG 奖励：直接用 SCG 累积作为训练信号
            self._episode_raw_returns += scg_step_rewards
            self._episode_returns += scg_step_rewards
            self._episode_lengths += 1
            survival_step_rewards = scg_step_rewards  # for logging consistency
        else:
            # 赛道1：存活赛道，每步只加 alive_bonus（不依赖 SCG）
            survival_step_rewards = np.zeros_like(scg_step_rewards)
            if self.alive_bonus != 0.0:
                survival_step_rewards += self.alive_bonus
            self._episode_survival_bonus += survival_step_rewards

            # 赛道2：SCG 累积（原始负代价，不 clamp）
            self._episode_raw_returns += scg_step_rewards

            # 训练用 shaped_return 先跟随 survival 赛道，终止时再重写
            self._episode_returns += survival_step_rewards
            self._episode_lengths += 1

        # 检查截断
        truncated = self._step_counts >= self.max_steps
        crashed = dones_tensor.cpu().numpy() & (~truncated)
        dones = dones_tensor.cpu().numpy() | truncated

        if self.use_pure_scg:
            final_rewards = scg_step_rewards.copy()
        else:
            # 终止重写：两赛道
            final_rewards = survival_step_rewards.copy()
            if self.max_steps > 0 and (self.crash_base_penalty > 0 or self.crash_slope > 0):
                survival_ratio = np.clip(self._episode_lengths / self.max_steps, 0.0, 1.0)
                # 坠毁：覆盖为严格负值，不依赖存活奖励累积
                crash_penalty = -self.crash_base_penalty - self.crash_slope * (1.0 - survival_ratio)
                crash_penalty = np.minimum(crash_penalty, -1e-6)  # 保证严格为负
                final_rewards = np.where(crashed, crash_penalty, final_rewards)
                self._episode_returns = np.where(crashed, crash_penalty, self._episode_returns)

            # 满步：用 SCG 转正 + completion_bonus，覆盖 survival 赛道
            if truncated.any():
                # 对于 truncated 的环境，shaped_return = completion_bonus + alpha/(eps + |scg_return|)
                scg_abs = np.abs(self._episode_raw_returns)
                pos_scg = self.completion_bonus + self.scg_pos_alpha / (self.scg_pos_eps + scg_abs)
                final_rewards = np.where(truncated, pos_scg, final_rewards)
                self._episode_returns = np.where(truncated, pos_scg, self._episode_returns)

        # 构建 info（在 reset 之前记录 episode 统计）
        infos_list = []
        for i in range(self.num_envs):
            info = {'TimeLimit.truncated': bool(truncated[i])}
            if dones[i]:
                info['episode'] = {
                    'r': float(self._episode_returns[i]),
                    'l': int(self._episode_lengths[i])
                }
                info['raw_episode_r'] = float(self._episode_raw_returns[i])
                info['crashed'] = bool(crashed[i])
                if hasattr(self, '_pid_magnitude'):
                    info['pid_magnitude'] = self._pid_magnitude
                    info['rl_magnitude'] = self._rl_magnitude
                if hasattr(self, '_last_actions'):
                    info['last_action'] = self._last_actions[i].copy()
            infos_list.append(info)

        # 处理 done 的环境（自动 reset）
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            env_ids_tensor = torch.tensor(done_indices, dtype=torch.long, device=self.device)
            env.reset(env_ids=env_ids_tensor)
            self._step_counts[done_indices] = 0
            self._episode_returns[done_indices] = 0.0
            self._episode_lengths[done_indices] = 0
            self._episode_raw_returns[done_indices] = 0.0
            self._episode_survival_bonus[done_indices] = 0.0

        obs = self._get_all_obs()

        return obs, final_rewards, dones, infos_list
    
    def _compute_rewards_batch(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        raw_rewards = self._reward_calc.compute_step(
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            target_pos=target_pos,
            action=actions[:, 2:6],
            done_mask=None,
        )
        # 每步即时奖励直接返回（不做 shaping），shaping 在 episode 结束时处理
        return raw_rewards.cpu().numpy(), raw_rewards.cpu().numpy()

        # OLD CODE - 保留注释供参考
        # 课程/调度进度 [0,1]
        if False and self.shaping_schedule_steps > 0:
            progress = min(self._total_env_steps / self.shaping_schedule_steps, 1.0)
        else:
            progress = 1.0
        crash_scale = self.crash_penalty_scale_start + (self.crash_penalty_scale_end - self.crash_penalty_scale_start) * progress
        alive_bonus = self.alive_bonus_start + (self.alive_bonus_end - self.alive_bonus_start) * progress
        self._current_crash_scale = crash_scale
        self._last_alive_bonus = alive_bonus

        # 提前坠毁会得到较少的负代价，添加终止惩罚抵消“早死更优”的激励
        dones = dones.to(self.device)
        penalties = torch.zeros(self.num_envs, device=self.device)
        if dones.any():
            step_counts = torch.from_numpy(self._step_counts).to(self.device)
            remaining = (self.max_steps - step_counts).clamp(min=0).float()
            mean_step_cost = torch.clamp(raw_rewards.detach().abs().mean(), min=1.0)
            penalties = crash_scale * mean_step_cost * remaining * dones.float()
            rewards = rewards - penalties

        # 每步存活奖励（衰减到目标值）
        if alive_bonus != 0.0:
            rewards = rewards + alive_bonus

        self._last_crash_penalty = penalties.cpu().numpy()
        return rewards.cpu().numpy(), raw_rewards.cpu().numpy()
    
    def close(self):
        if self._isaac_env is not None:
            self._isaac_env.close()
            self._isaac_env = None
    
    # SB3 VecEnv 兼容属性
    @property
    def unwrapped(self):
        return self
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        idxs = list(self._get_indices(indices))
        return [False] * len(idxs)
    
    def env_method(self, method_name, *args, indices=None, **kwargs):
        idxs = list(self._get_indices(indices))
        if not hasattr(self, method_name):
            return [None] * len(idxs)
        method = getattr(self, method_name)
        result = method(*args, **kwargs)
        return [result] * len(idxs)
    
    def get_attr(self, attr_name, indices=None):
        idxs = list(self._get_indices(indices))
        value = getattr(self, attr_name, None)
        return [value] * len(idxs)
    
    def set_attr(self, attr_name, value, indices=None):
        for _ in self._get_indices(indices):
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
        trajectory_type: 轨迹类型 ('hover', 'figure8', 'circle', 'square', 'helix')
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

"""Soar 程序适配器：将 DSL 程序转换为 SB3 兼容的策略。

这个模块允许在 SB3 环境中测试 Soar 生成的控制程序。

使用示例:
    from scripts.sb3.soar_policy import SoarPolicy
    from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv
    
    # 加载 Soar 程序
    program = load_program('results/best_program.json')
    policy = SoarPolicy(program)
    
    # 在 SB3 环境中测试
    env = QuadrotorTrackingEnv(trajectory='figure8')
    obs, info = env.reset()
    
    for _ in range(250):
        action = policy.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 添加项目路径
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "01_soar"))


class SoarPolicy:
    """将 Soar DSL 程序转换为 SB3 兼容的策略。
    
    支持的 DSL 节点类型:
        - VAR: 状态变量 (pos_err_x, vel_z, roll, etc.)
        - CONST: 常量
        - BinOp: 二元运算 (+, -, *, /, max, min)
        - UnaryOp: 一元运算 (neg, abs, sign, sqrt, tanh, clip)
        - Conditional: 条件表达式 (if cond then a else b)
    
    观测映射 (18D -> 变量):
        obs[0:3]   -> pos_err_x, pos_err_y, pos_err_z
        obs[3:6]   -> vel_x, vel_y, vel_z
        obs[6:9]   -> roll, pitch, yaw
        obs[9:12]  -> omega_x, omega_y, omega_z
        obs[12:15] -> target_pos (可选)
        obs[15:18] -> target_vel (可选)
    """
    
    # 观测索引映射
    OBS_MAP = {
        # 位置误差
        'pos_err_x': 0, 'err_x': 0, 'e_x': 0,
        'pos_err_y': 1, 'err_y': 1, 'e_y': 1,
        'pos_err_z': 2, 'err_z': 2, 'e_z': 2,
        # 速度
        'vel_x': 3, 'vx': 3, 'v_x': 3,
        'vel_y': 4, 'vy': 4, 'v_y': 4,
        'vel_z': 5, 'vz': 5, 'v_z': 5,
        # 欧拉角
        'roll': 6, 'phi': 6,
        'pitch': 7, 'theta': 7,
        'yaw': 8, 'psi': 8,
        # 角速度
        'omega_x': 9, 'wx': 9, 'p': 9,
        'omega_y': 10, 'wy': 10, 'q': 10,
        'omega_z': 11, 'wz': 11, 'r': 11,
        # 积分项（需要内部维护）
        'err_i_x': -1, 'err_i_y': -2, 'err_i_z': -3,
        'err_i_roll': -4, 'err_i_pitch': -5, 'err_i_yaw': -6,
    }
    
    # 动作缩放（与 QuadrotorTrackingEnv 一致）
    ACTION_SCALE = np.array([3.5, 0.12, 0.12, 0.06], dtype=np.float32)
    ACTION_OFFSET = np.array([3.5, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def __init__(
        self,
        program: Union[List[Dict], str, Path],
        dt: float = 0.02,  # 控制周期（与 50Hz 对应）
    ):
        """
        Args:
            program: Soar DSL 程序（列表或 JSON 文件路径）
            dt: 控制周期（用于积分项计算）
        """
        # 加载程序
        if isinstance(program, (str, Path)):
            with open(program, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.program = data
                elif isinstance(data, dict) and 'program' in data:
                    self.program = data['program']
                else:
                    self.program = data
        else:
            self.program = program
        
        self.dt = dt
        
        # 积分状态
        self._integral_state = {
            'err_i_x': 0.0,
            'err_i_y': 0.0,
            'err_i_z': 0.0,
            'err_i_roll': 0.0,
            'err_i_pitch': 0.0,
            'err_i_yaw': 0.0,
        }
        
        # 解析程序规则
        self._rules = self._parse_program()
    
    def _parse_program(self) -> Dict[str, Any]:
        """解析程序规则"""
        rules = {
            'fz': None,  # 推力
            'tx': None,  # roll 力矩
            'ty': None,  # pitch 力矩
            'tz': None,  # yaw 力矩
        }
        
        # 输出名称映射
        output_map = {
            'fz': 'fz', 'f_z': 'fz', 'thrust': 'fz',
            'tx': 'tx', 't_x': 'tx', 'torque_x': 'tx', 'roll_torque': 'tx',
            'ty': 'ty', 't_y': 'ty', 'torque_y': 'ty', 'pitch_torque': 'ty',
            'tz': 'tz', 't_z': 'tz', 'torque_z': 'tz', 'yaw_torque': 'tz',
        }
        
        for rule in self.program:
            if not isinstance(rule, dict):
                continue
            
            output = rule.get('output', rule.get('out', '')).lower().replace(' ', '_')
            node = rule.get('node', rule.get('expr'))
            
            if output in output_map and node is not None:
                key = output_map[output]
                rules[key] = node
        
        return rules
    
    def reset(self):
        """重置积分状态"""
        for key in self._integral_state:
            self._integral_state[key] = 0.0
    
    def _get_var_value(self, var_name: str, obs: np.ndarray) -> float:
        """获取变量值"""
        name = var_name.lower().replace(' ', '_')
        
        # 检查是否是积分变量
        if name in self._integral_state:
            return self._integral_state[name]
        
        # 检查观测映射
        if name in self.OBS_MAP:
            idx = self.OBS_MAP[name]
            if idx >= 0 and idx < len(obs):
                return float(obs[idx])
        
        # 尝试常见别名
        aliases = {
            'altitude_error': 2,  # pos_err_z
            'height_error': 2,
            'lateral_error': 0,  # pos_err_x
        }
        if name in aliases:
            return float(obs[aliases[name]])
        
        # 默认返回 0
        return 0.0
    
    def _eval_node(self, node: Any, obs: np.ndarray) -> float:
        """递归求值 AST 节点"""
        if node is None:
            return 0.0
        
        # 常量
        if isinstance(node, (int, float)):
            return float(node)
        
        # 字符串变量
        if isinstance(node, str):
            return self._get_var_value(node, obs)
        
        # 列表形式的节点
        if isinstance(node, list):
            if len(node) == 0:
                return 0.0
            
            op = node[0] if isinstance(node[0], str) else None
            
            if op == 'VAR' and len(node) > 1:
                return self._get_var_value(node[1], obs)
            
            elif op == 'CONST' and len(node) > 1:
                return float(node[1])
            
            elif op in ('+', 'add', 'Add') and len(node) > 2:
                return self._eval_node(node[1], obs) + self._eval_node(node[2], obs)
            
            elif op in ('-', 'sub', 'Sub') and len(node) > 2:
                return self._eval_node(node[1], obs) - self._eval_node(node[2], obs)
            
            elif op in ('*', 'mul', 'Mul') and len(node) > 2:
                return self._eval_node(node[1], obs) * self._eval_node(node[2], obs)
            
            elif op in ('/', 'div', 'Div') and len(node) > 2:
                divisor = self._eval_node(node[2], obs)
                if abs(divisor) < 1e-10:
                    return 0.0
                return self._eval_node(node[1], obs) / divisor
            
            elif op in ('neg', 'Neg') and len(node) > 1:
                return -self._eval_node(node[1], obs)
            
            elif op in ('abs', 'Abs') and len(node) > 1:
                return abs(self._eval_node(node[1], obs))
            
            elif op in ('sign', 'Sign') and len(node) > 1:
                val = self._eval_node(node[1], obs)
                return 1.0 if val > 0 else (-1.0 if val < 0 else 0.0)
            
            elif op in ('sqrt', 'Sqrt') and len(node) > 1:
                val = self._eval_node(node[1], obs)
                return math.sqrt(max(0.0, val))
            
            elif op in ('tanh', 'Tanh') and len(node) > 1:
                return math.tanh(self._eval_node(node[1], obs))
            
            elif op in ('clip', 'Clip', 'clamp') and len(node) > 3:
                val = self._eval_node(node[1], obs)
                lo = self._eval_node(node[2], obs)
                hi = self._eval_node(node[3], obs)
                return max(lo, min(hi, val))
            
            elif op in ('max', 'Max') and len(node) > 2:
                return max(self._eval_node(node[1], obs), self._eval_node(node[2], obs))
            
            elif op in ('min', 'Min') and len(node) > 2:
                return min(self._eval_node(node[1], obs), self._eval_node(node[2], obs))
            
            elif op in ('if', 'If', 'cond', 'Cond') and len(node) > 3:
                cond = self._eval_node(node[1], obs)
                return self._eval_node(node[2], obs) if cond > 0 else self._eval_node(node[3], obs)
        
        # 字典形式的节点
        if isinstance(node, dict):
            node_type = node.get('type', node.get('op', ''))
            
            if node_type == 'VAR':
                return self._get_var_value(node.get('name', node.get('var', '')), obs)
            
            elif node_type == 'CONST':
                return float(node.get('value', node.get('val', 0)))
            
            elif node_type in ('BinOp', 'binop'):
                op = node.get('op', '+')
                left = self._eval_node(node.get('left', node.get('l')), obs)
                right = self._eval_node(node.get('right', node.get('r')), obs)
                
                if op in ('+', 'add'):
                    return left + right
                elif op in ('-', 'sub'):
                    return left - right
                elif op in ('*', 'mul'):
                    return left * right
                elif op in ('/', 'div'):
                    return left / right if abs(right) > 1e-10 else 0.0
                elif op in ('max', 'Max'):
                    return max(left, right)
                elif op in ('min', 'Min'):
                    return min(left, right)
            
            elif node_type in ('UnaryOp', 'unaryop'):
                op = node.get('op', 'neg')
                val = self._eval_node(node.get('arg', node.get('x')), obs)
                
                if op == 'neg':
                    return -val
                elif op == 'abs':
                    return abs(val)
                elif op == 'sqrt':
                    return math.sqrt(max(0.0, val))
                elif op == 'tanh':
                    return math.tanh(val)
            
            # 直接包含子节点的情况
            if 'left' in node or 'l' in node:
                return self._eval_node(node.get('left', node.get('l')), obs)
        
        return 0.0
    
    def _update_integral(self, obs: np.ndarray):
        """更新积分状态"""
        # 位置误差积分
        self._integral_state['err_i_x'] += obs[0] * self.dt
        self._integral_state['err_i_y'] += obs[1] * self.dt
        self._integral_state['err_i_z'] += obs[2] * self.dt
        
        # 姿态误差积分
        self._integral_state['err_i_roll'] += obs[6] * self.dt
        self._integral_state['err_i_pitch'] += obs[7] * self.dt
        self._integral_state['err_i_yaw'] += obs[8] * self.dt
        
        # 限制积分器防止 windup
        for key in self._integral_state:
            self._integral_state[key] = np.clip(self._integral_state[key], -10.0, 10.0)
    
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """预测动作（SB3 接口兼容）。
        
        Args:
            obs: 观测 [18] 或 [N, 18]
            deterministic: 是否确定性（对于 DSL 始终为 True）
        
        Returns:
            action: 归一化动作 [4] 或 [N, 4]
        """
        obs = np.asarray(obs, dtype=np.float32)
        
        # 处理批量观测
        if obs.ndim == 2:
            return np.array([self.predict(o) for o in obs])
        
        # 确保是 1D
        obs = obs.flatten()
        
        # 更新积分
        self._update_integral(obs)
        
        # 计算各输出
        fz = self._eval_node(self._rules['fz'], obs) if self._rules['fz'] else 0.0
        tx = self._eval_node(self._rules['tx'], obs) if self._rules['tx'] else 0.0
        ty = self._eval_node(self._rules['ty'], obs) if self._rules['ty'] else 0.0
        tz = self._eval_node(self._rules['tz'], obs) if self._rules['tz'] else 0.0
        
        # 物理输出
        raw_action = np.array([fz, tx, ty, tz], dtype=np.float32)
        
        # 归一化到 [-1, 1]
        normalized_action = (raw_action - self.ACTION_OFFSET) / self.ACTION_SCALE
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        
        return normalized_action


def load_soar_program(path: Union[str, Path]) -> List[Dict]:
    """加载 Soar 程序。
    
    支持的格式:
        - JSON 列表
        - JSON 对象 (带 'program' 键)
        - 训练结果文件
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    
    if isinstance(data, dict):
        if 'program' in data:
            return data['program']
        if 'best_program' in data:
            return data['best_program']
        if 'results' in data and isinstance(data['results'], list):
            # 选择最佳程序
            best = max(data['results'], key=lambda x: x.get('reward', float('-inf')))
            return best.get('program', [])
    
    raise ValueError(f"无法从 {path} 加载程序")


def evaluate_soar_in_sb3(
    program_path: Union[str, Path],
    trajectory: str = 'figure8',
    n_episodes: int = 10,
    render: bool = False,
) -> Dict[str, Any]:
    """在 SB3 环境中评估 Soar 程序。
    
    Args:
        program_path: 程序文件路径
        trajectory: 轨迹类型
        n_episodes: 评估 episode 数
        render: 是否渲染
    
    Returns:
        评估结果
    """
    from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv
    
    # 加载程序
    program = load_soar_program(program_path)
    policy = SoarPolicy(program)
    
    # 创建环境
    env = QuadrotorTrackingEnv(
        trajectory=trajectory,
        render_mode='human' if render else None,
    )
    
    episode_rewards = []
    episode_lengths = []
    episode_pos_errors = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        policy.reset()  # 重置积分状态
        
        total_reward = 0.0
        steps = 0
        pos_errors = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            action = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if 'pos_error' in info:
                pos_errors.append(info['pos_error'])
            
            if render:
                env.render()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if pos_errors:
            episode_pos_errors.append(np.mean(pos_errors))
    
    env.close()
    
    results = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_pos_error': float(np.mean(episode_pos_errors)) if episode_pos_errors else None,
        'n_episodes': n_episodes,
        'trajectory': trajectory,
    }
    
    print(f"\n📊 Soar 评估结果 ({n_episodes} episodes):")
    print(f"   Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   Mean Length: {results['mean_length']:.1f}")
    if results['mean_pos_error'] is not None:
        print(f"   Mean Pos Error: {results['mean_pos_error']:.4f} m")
    
    return results


__all__ = [
    'SoarPolicy',
    'load_soar_program',
    'evaluate_soar_in_sb3',
]

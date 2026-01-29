#!/usr/bin/env python3
"""
导出真实仿真数据：一个周期内的位置误差曲线

使用 Isaac Gym 环境 + manual.md 中的控制律，导出真实的 pos_err_x, pos_err_y, pos_err_z
"""

import sys
from pathlib import Path
import os
import math
import json

# 路径设置
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / '01_soar') not in sys.path:
    sys.path.insert(0, str(ROOT / '01_soar'))

# Isaac Gym 使用正确的路径 (soar 而非 soar-20251205)
_ISAAC_GYM_PY = Path('/home/linlexi/桌面/soar/isaacgym/python')
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

# Isaac Gym 必须在 torch 前导入
try:
    import isaacgym
except ImportError:
    pass

import torch
import numpy as np
import pandas as pd

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from utils.reward_scg_exact import SCGExactRewardCalculator
from core.dsl import BinaryOpNode, UnaryOpNode, TerminalNode, ConstantNode

# ============================================================================
#                         轨迹定义
# ============================================================================

from utilities.trajectory_presets import scg_position, get_scg_trajectory_config

def get_square_sign_target(t: float) -> np.ndarray:
    """手工实现的 Sign 控制器专用 square 轨迹（与 analyze_square_sign.py 一致）"""
    period = 5.0
    scale = 1.0
    cycle = t % period
    seg_period = period / 4.0
    seg_idx = int(cycle // seg_period)
    seg_time = cycle - seg_idx * seg_period
    speed = scale / seg_period
    dist = speed * seg_time
    x = 0.0
    y = 0.0
    if seg_idx == 0:
        y = dist
    elif seg_idx == 1:
        x = -dist
        y = scale
    elif seg_idx == 2:
        x = -scale
        y = scale - dist
    else:
        x = -scale + dist
    return np.array([x, y, 1.0], dtype=np.float32)


def get_target_fn(traj_type):
    """获取轨迹目标函数"""
    if traj_type == 'square':
        return get_square_sign_target
    cfg = get_scg_trajectory_config(traj_type)
    
    def target_fn(t):
        pos = scg_position(traj_type, t, params=cfg.params, center=cfg.center)
        return pos
        
    return target_fn

# ============================================================================
#                         控制律 (来自 manual.md)
# ============================================================================

def smooth(val, s):
    """smooth 算子: s * tanh(x/s)"""
    return s * math.tanh(val / s)

def sign(val):
    """sign 算子"""
    if val > 0:
        return 1.0
    elif val < 0:
        return -1.0
    return 0.0

def create_square_sign_programs(k_p=0.6, k_d=1.766, k_w=0.773):
    """使用 DSL AST 构造 square sign 控制器（保证与 analyze_square_sign.py 完全一致）"""
    sign_err_y = UnaryOpNode('sign', TerminalNode('pos_err_y'))
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), sign_err_y)
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    prog_tx = BinaryOpNode('-', BinaryOpNode('+', term1_tx, term2_tx), term3_tx)

    sign_err_x = UnaryOpNode('sign', TerminalNode('pos_err_x'))
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), sign_err_x)
    term2_ty = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_x'))
    term3_ty = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_y'))
    prog_ty = BinaryOpNode('-', BinaryOpNode('-', term1_ty, term2_ty), term3_ty)

    prog_tz = BinaryOpNode('-',
        BinaryOpNode('*', ConstantNode(4.0), TerminalNode('err_p_yaw')),
        BinaryOpNode('*', ConstantNode(0.8), TerminalNode('ang_vel_z'))
    )
    prog_fz = BinaryOpNode('+',
        BinaryOpNode('-',
            BinaryOpNode('*', ConstantNode(0.5), TerminalNode('pos_err_z')),
            BinaryOpNode('*', ConstantNode(0.2), TerminalNode('vel_z'))
        ),
        ConstantNode(0.65)
    )
    return prog_tx, prog_ty, prog_tz, prog_fz


def get_controller(traj_type):
    """返回对应轨迹的控制律函数"""
    
    if traj_type == 'figure8':
        # u_tx = ((-0.489 * smooth(pos_err_y, s=1.285)) + (1.062 * vel_y)) - (0.731 * ang_vel_x)
        def u_tx(sd):
            return ((-0.489 * smooth(sd['pos_err_y'], 1.285)) + (1.062 * sd['vel_y'])) - (0.731 * sd['ang_vel_x'])
        k_p, k_d, k_w, k_s = 0.489, 1.062, 0.731, 1.285
        
    elif traj_type == 'circle':
        # u_tx = ((-2.104 * smooth(pos_err_y, s=0.296)) + (1.111 * vel_y)) - (0.727 * ang_vel_x)
        def u_tx(sd):
            return ((-2.104 * smooth(sd['pos_err_y'], 0.296)) + (1.111 * sd['vel_y'])) - (0.727 * sd['ang_vel_x'])
        k_p, k_d, k_w, k_s = 2.104, 1.111, 0.727, 0.296
        
    elif traj_type == 'square':
        prog_tx, prog_ty, prog_tz, prog_fz = create_square_sign_programs()
        def u_tx(sd):
            return prog_tx.evaluate(sd)
        def u_ty(sd):
            return prog_ty.evaluate(sd)
        def u_tz(sd):
            return prog_tz.evaluate(sd)
        def u_fz(sd):
            return prog_fz.evaluate(sd)
        return u_tx, u_ty, u_tz, u_fz
    
    # u_ty (镜像 u_tx)
    def u_ty(sd, k_p=k_p, k_d=k_d, k_w=k_w, k_s=k_s):
        if k_s is not None:
            return ((k_p * smooth(sd['pos_err_x'], k_s)) - (k_d * sd['vel_x'])) - (k_w * sd['ang_vel_y'])
        else:
            return ((k_p * sign(sd['pos_err_x'])) - (k_d * sd['vel_x'])) - (k_w * sd['ang_vel_y'])
    
    # u_tz (yaw 控制)
    def u_tz(sd):
        return 4.0 * sd['err_p_yaw'] - 0.8 * sd['ang_vel_z']
    
    # u_fz (高度控制)
    if traj_type == 'square':
        # Match sign controller config used during manual cost estimation
        def u_fz(sd):
            return 0.65 - 0.5 * sd['pos_err_z'] - 0.2 * sd['vel_z']
    else:
        def u_fz(sd):
            return 1.0 * sd['pos_err_z'] - 0.5 * sd['vel_z'] + 0.65
    
    return u_tx, u_ty, u_tz, u_fz

# ============================================================================
#                         主仿真循环
# ============================================================================

def run_simulation(env, scg_calc, traj_type):
    """运行一个周期的真实仿真"""

    # 获取控制器
    u_tx_fn, u_ty_fn, u_tz_fn, u_fz_fn = get_controller(traj_type)
    
    # 获取轨迹函数
    target_fn = get_target_fn(traj_type)
    
    # 重置到轨迹起点
    initial_target = target_fn(0.0)
    initial_pos = torch.tensor([initial_target], device=env.device, dtype=torch.float32)
    env.reset(initial_pos=initial_pos)
    scg_calc.reset()
    
    dt = 1.0 / 48.0  # 48Hz 控制频率
    cfg = get_scg_trajectory_config(traj_type)
    period = cfg.params.get('period', 5.0)

    print(f"\n{'='*60}")
    print(f"轨迹: {traj_type}, 周期: {period:.3f}s")
    print('='*60)

    num_steps = int(period / dt)
    
    data = []
    cumulative_reward = 0.0
    
    for step in range(num_steps):
        t = step * dt
        target = target_fn(t)
        target_tensor = torch.tensor(target, device=env.device, dtype=torch.float32)
        
        # 获取状态
        pos = env.pos[0]
        vel = env.lin_vel[0]
        quat = env.quat[0]
        omega = env.ang_vel[0]
        
        # 位置误差
        pos_err = target_tensor - pos
        
        # 四元数转欧拉角
        qx, qy, qz, qw = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
        
        # Yaw
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Pitch
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(np.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Roll
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Yaw 误差
        err_p_yaw = 0.0 - yaw
        while err_p_yaw > np.pi: err_p_yaw -= 2*np.pi
        while err_p_yaw < -np.pi: err_p_yaw += 2*np.pi
        
        # 构建状态字典
        state_dict = {
            'pos_err_x': pos_err[0].item(),
            'pos_err_y': pos_err[1].item(),
            'pos_err_z': pos_err[2].item(),
            'vel_x': vel[0].item(),
            'vel_y': vel[1].item(),
            'vel_z': vel[2].item(),
            'ang_vel_x': omega[0].item(),
            'ang_vel_y': omega[1].item(),
            'ang_vel_z': omega[2].item(),
            'err_p_yaw': err_p_yaw,
        }
        
        # 计算控制量
        u_tx = u_tx_fn(state_dict)
        u_ty = u_ty_fn(state_dict)
        u_tz = u_tz_fn(state_dict)
        u_fz = u_fz_fn(state_dict)
        
        # 限幅
        u_tx = max(-0.4, min(0.4, u_tx))
        u_ty = max(-0.4, min(0.4, u_ty))
        u_tz = max(-0.5, min(0.5, u_tz))
        u_fz = max(0.0, min(1.3, u_fz))
        
        # SCG 代价计算
        scg_action = torch.tensor([[u_fz, u_tx, u_ty, u_tz]], device=env.device)
        step_reward_tensor = scg_calc.compute_step(
            env.pos, env.lin_vel, env.quat, env.ang_vel, target_tensor, scg_action
        )
        step_reward = step_reward_tensor[0].item()
        cumulative_reward += step_reward
        
        # 执行动作
        actions = torch.tensor([[0.0, 0.0, u_fz, u_tx, u_ty, u_tz]], device=env.device)
        env.step(actions)
        
        # 记录数据
        data.append({
            'time': t,
            'traj': traj_type,
            'ref_x': target[0],
            'ref_y': target[1],
            'ref_z': target[2],
            'actual_x': pos[0].item(),
            'actual_y': pos[1].item(),
            'actual_z': pos[2].item(),
            'pos_err_x': state_dict['pos_err_x'],
            'pos_err_y': state_dict['pos_err_y'],
            'pos_err_z': state_dict['pos_err_z'],
            'u_tx': u_tx,
            'u_ty': u_ty,
            'u_tz': u_tz,
            'u_fz': u_fz,
            'step_reward': step_reward,
            'cumulative_reward': cumulative_reward,
        })
    
    # 获取总代价
    components = scg_calc.get_components()
    total_cost = components["total_cost"][0].item()
    total_reward = -total_cost
    if not math.isclose(total_reward, cumulative_reward, rel_tol=1e-5, abs_tol=1e-5):
        print(f"  ⚠️ reward mismatch: cumulative={cumulative_reward:.6f}, from cost={total_reward:.6f}")
    
    print(f"  Total SCG Cost: {total_cost:.4f}")
    print(f"  Total SCG Reward: {total_reward:.4f}")
    
    # 统计误差
    err_x = [d['pos_err_x'] for d in data]
    err_y = [d['pos_err_y'] for d in data]
    err_z = [d['pos_err_z'] for d in data]
    
    print(f"  pos_err_x: max={max(err_x):.4f}, min={min(err_x):.4f}, rms={np.sqrt(np.mean(np.array(err_x)**2)):.4f}")
    print(f"  pos_err_y: max={max(err_y):.4f}, min={min(err_y):.4f}, rms={np.sqrt(np.mean(np.array(err_y)**2)):.4f}")
    print(f"  pos_err_z: max={max(err_z):.4f}, min={min(err_z):.4f}, rms={np.sqrt(np.mean(np.array(err_z)**2)):.4f}")
    
    return data, total_cost, total_reward, period

def main():
    print("="*60)
    print("导出真实 Isaac Gym 仿真数据")
    print("="*60)
    
    all_data = []
    costs = {}
    rewards = {}
    periods = {}
    
    trajectories = ['figure8', 'circle', 'square']
    
    for traj_type in trajectories:
        env = IsaacGymDroneEnv(num_envs=1, device='cuda:0', headless=True, duration_sec=5.0)
        scg_calc = SCGExactRewardCalculator(num_envs=1, device='cuda:0')
        data, cost, reward, period = run_simulation(env, scg_calc, traj_type)
        env.close()
        all_data.extend(data)
        costs[traj_type] = cost
        rewards[traj_type] = reward
        periods[traj_type] = period
    
    # 保存数据
    df = pd.DataFrame(all_data)
    df.to_csv('trajectory_error_data.csv', index=False)
    
    print("\n" + "="*60)
    print("数据导出完成")
    print("="*60)
    print(f"文件: trajectory_error_data.csv")
    print(f"总行数: {len(df)}")
    print("\n各轨迹 SCG Cost/Reward:")
    for traj, cost in costs.items():
        reward = rewards[traj]
        period = periods[traj]
        print(f"  {traj}: period={period:.3f}s, cost={cost:.4f}, reward={reward:.4f}")
    
    # 保存元数据
    meta = {
        'trajectories': {
            traj: {
                'period': periods[traj],
                'cost': costs[traj],
                'reward': rewards[traj],
            }
            for traj in trajectories
        },
        'control_laws': {
            'figure8': 'u_tx = ((-0.489 * smooth(pos_err_y, s=1.285)) + (1.062 * vel_y)) - (0.731 * ang_vel_x)',
            'circle': 'u_tx = ((-2.104 * smooth(pos_err_y, s=0.296)) + (1.111 * vel_y)) - (0.727 * ang_vel_x)',
            'square': 'u_tx = ((-0.972 * smooth(pos_err_y, s=1.988)) + (1.591 * vel_y)) - (0.660 * ang_vel_x)',
        },
        'dt': 1.0/48.0,
        'columns': ['time', 'traj', 'ref_x', 'ref_y', 'ref_z', 'actual_x', 'actual_y', 'actual_z', 
                   'pos_err_x', 'pos_err_y', 'pos_err_z', 'u_tx', 'u_ty', 'u_tz', 'u_fz', 'step_reward', 'cumulative_reward'],
    }
    
    with open('trajectory_error_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n元数据已保存到: trajectory_error_meta.json")

if __name__ == "__main__":
    main()

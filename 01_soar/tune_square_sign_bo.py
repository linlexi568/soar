#!/usr/bin/env python3
"""
使用 Optuna 对 Square 轨迹的 Sign 控制律进行贝叶斯调参。
目标：最小化 SCG Cost
控制律结构：
u_tx = ((-k_p * sign(pos_err_y)) + (k_d * vel_y)) - (k_w * ang_vel_x)
u_ty = (( k_p * sign(pos_err_x)) - (k_d * vel_x)) - (k_w * ang_vel_y)
"""

import os
import sys
import math
import numpy as np

# 路径设置
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(os.path.join(ROOT, '01_soar')) not in sys.path:
    sys.path.insert(0, str(os.path.join(ROOT, '01_soar')))

# Isaac Gym 路径
_ISAAC_GYM_PY = os.path.join(os.path.dirname(ROOT), 'soar', 'isaacgym', 'python')
if os.path.exists(_ISAAC_GYM_PY) and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

try:
    import isaacgym
except ImportError:
    pass

import torch
import optuna

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from utils.reward_scg_exact import SCGExactRewardCalculator
from utilities.trajectory_presets import scg_position, get_scg_trajectory_config

# ============================================================================
#                         辅助函数
# ============================================================================

def sign(val):
    if val > 0:
        return 1.0
    elif val < 0:
        return -1.0
    return 0.0

def get_target_fn(traj_type):
    cfg = get_scg_trajectory_config(traj_type)
    def target_fn(t):
        pos = scg_position(traj_type, t, params=cfg.params, center=cfg.center)
        return pos
    return target_fn

# ============================================================================
#                         评估函数
# ============================================================================

def objective(trial):
    # 1. 采样参数
    k_p = trial.suggest_float('k_p', 0.1, 5.0)
    k_d = trial.suggest_float('k_d', 0.1, 5.0)
    k_w = trial.suggest_float('k_w', 0.1, 2.0)
    
    # 2. 定义控制器
    def u_tx_fn(sd):
        return ((-k_p * sign(sd['pos_err_y'])) + (k_d * sd['vel_y'])) - (k_w * sd['ang_vel_x'])
    
    def u_ty_fn(sd):
        return ((k_p * sign(sd['pos_err_x'])) - (k_d * sd['vel_x'])) - (k_w * sd['ang_vel_y'])
        
    def u_tz_fn(sd):
        return 4.0 * sd['err_p_yaw'] - 0.8 * sd['ang_vel_z']
    
    def u_fz_fn(sd):
        return 1.0 * sd['pos_err_z'] - 0.5 * sd['vel_z'] + 0.65

    # 3. 初始化环境
    # 注意：每次 trial 都需要重置环境，但为了效率，我们可以复用 env 对象吗？
    # IsaacGymDroneEnv 通常比较重，最好在外部创建 env，这里只 reset。
    # 但 Optuna 可能并行运行。这里假设单线程运行。
    
    global env, scg_calc
    
    traj_type = 'square'
    target_fn = get_target_fn(traj_type)
    
    # 重置到起点
    initial_target = target_fn(0.0)
    # initial_pos = torch.tensor([initial_target], device=env.device, dtype=torch.float32)
    initial_pos = torch.from_numpy(initial_target).unsqueeze(0).to(env.device, dtype=torch.float32)
    env.reset(initial_pos=initial_pos)
    scg_calc.reset()
    
    dt = 1.0 / 48.0
    cfg = get_scg_trajectory_config(traj_type)
    period = cfg.params.get('period', 5.0)
    num_steps = int(period / dt)
    
    total_cost = 0.0
    crashed = False
    
    for step in range(num_steps):
        t = step * dt
        target = target_fn(t)
        # target_tensor = torch.tensor(target, device=env.device, dtype=torch.float32)
        target_tensor = torch.from_numpy(target).to(env.device, dtype=torch.float32)
        
        # 获取状态
        pos = env.pos[0]
        vel = env.lin_vel[0]
        quat = env.quat[0]
        omega = env.ang_vel[0]
        
        # 检查是否坠毁 (高度过低或发散)
        if pos[2] < 0.1 or torch.any(torch.abs(pos) > 5.0):
            crashed = True
            break
            
        pos_err = target_tensor - pos
        
        # 四元数转欧拉角
        qx, qy, qz, qw = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        err_p_yaw = 0.0 - yaw
        while err_p_yaw > np.pi: err_p_yaw -= 2*np.pi
        while err_p_yaw < -np.pi: err_p_yaw += 2*np.pi
        
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
        
        # 计算控制
        u_tx = u_tx_fn(state_dict)
        u_ty = u_ty_fn(state_dict)
        u_tz = u_tz_fn(state_dict)
        u_fz = u_fz_fn(state_dict)
        
        # 限幅
        u_tx = max(-0.4, min(0.4, u_tx))
        u_ty = max(-0.4, min(0.4, u_ty))
        u_tz = max(-0.5, min(0.5, u_tz))
        u_fz = max(0.0, min(1.3, u_fz))
        
        # SCG Cost
        scg_action = torch.tensor([[u_fz, u_tx, u_ty, u_tz]], device=env.device)
        scg_calc.compute_step(env.pos, env.lin_vel, env.quat, env.ang_vel, target_tensor, scg_action)
        
        # Step
        actions = torch.tensor([[0.0, 0.0, u_fz, u_tx, u_ty, u_tz]], device=env.device)
        env.step(actions)
        
    if crashed:
        return 2000.0 + (num_steps - step) # 惩罚坠毁
        
    components = scg_calc.get_components()
    total_cost = components["total_cost"][0].item()
    return total_cost

# ============================================================================
#                         主程序
# ============================================================================

if __name__ == "__main__":
    # 初始化环境 (全局变量)
    env = IsaacGymDroneEnv(
        num_envs=1,
        headless=True,
        control_freq_hz=48,
        physics_freq_hz=240,
        device='cuda:0'
    )
    scg_calc = SCGExactRewardCalculator(env.num_envs, env.device)
    
    print("开始 Square (Sign) 轨迹贝叶斯调参...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print("\n调参完成!")
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
    
    # 验证最佳参数
    print("\n验证最佳参数...")
    best_params = study.best_params
    # 重新运行一次以展示详细信息 (可选)

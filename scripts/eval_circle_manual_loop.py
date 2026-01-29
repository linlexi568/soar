#!/usr/bin/env python3
"""
手动循环评估 Circle 程序并记录详细轨迹
直接操作Isaac Gym环境，逐步执行程序
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_soar'))

# 导入顺序很重要: Isaac Gym必须在torch之前导入
import isaacgym  # 必须先导入
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# 自定义模块
from core.dsl import load_program, reset_program_state
from core.isaac_env_pool import IsaacEnvPool

print("正在加载模块...")

# 配置参数
PROGRAM_PATH = "results/soar_train/circle/best_programs/program_1.txt"
OUTPUT_CSV = "results/circle_trajectory_manual.csv"
PLOT_DIR = Path("results/plots/circle_manual")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 仿真参数
NUM_ENVS = 1
DURATION = 5.0  # 秒
CONTROL_FREQ = 48  # Hz
MAX_STEPS = int(DURATION * CONTROL_FREQ)

# 轨迹参数（圆形）
RADIUS = 1.0
OMEGA_TRAJ = 2.0 * np.pi / 8.0  # 8秒一圈
HEIGHT = 1.0

print("="*70)
print("Circle 程序手动评估与轨迹记录")
print("="*70)

# 加载程序
program = load_program(PROGRAM_PATH)
print(f"✓ 程序加载: {len(program)} 条规则\n")

# 打印规则
for i, rule in enumerate(program):
    print(f"规则 {i+1}:")
    print(f"  条件: {rule.condition}")
    print(f"  动作: {rule.action}")

print("\n初始化Isaac Gym环境...")

# 创建环境池
env_pool = IsaacEnvPool(
    num_envs=NUM_ENVS,
    aggregate_mode=True,
    device='cuda:0',
    control_freq_hz=CONTROL_FREQ,
    physics_freq_hz=240,
)

print(f"[Isaac Gym] ✅ 环境池就绪（{NUM_ENVS} 环境）")
print(f"  - 控制频率: {CONTROL_FREQ} Hz")
print(f"  - 物理频率: 240 Hz")
print(f"  - 最大步数: {MAX_STEPS}\n")

# 重置环境
obs = env_pool.reset()
print("✓ 环境已重置\n")

# 重置程序状态
reset_program_state(program)

# 轨迹记录
trajectory_data = []

print(f"开始评估（{MAX_STEPS} 步）...")

from scipy.spatial.transform import Rotation

# 评估程序
def eval_program(program, state):
    """评估程序返回控制输出"""
    actions = {'u_fz': 0.0, 'u_tx': 0.0, 'u_ty': 0.0, 'u_tz': 0.0}
    
    for rule in program:
        # 评估条件
        condition_met = True
        if hasattr(rule, 'condition') and rule.condition is not None:
            try:
                # 简单评估：检查condition字符串
                cond_str = str(rule.condition)
                # 这里需要用state替换变量并eval
                # 为简化，暂时假设条件都满足
                condition_met = True
            except:
                condition_met = False
        
        if condition_met and hasattr(rule, 'action'):
            for action in rule.action:
                try:
                    if hasattr(action, 'target') and hasattr(action, 'value'):
                        target_name = str(action.target)
                        
                        # 评估value表达式
                        if hasattr(action.value, 'value'):  # 常量
                            val = float(action.value.value)
                        elif hasattr(action.value, 'name'):  # 变量
                            var_name = str(action.value.name)
                            val = state.get(var_name, 0.0)
                        elif hasattr(action.value, 'op'):  # 二元操作
                            # 递归评估（简化版）
                            val = 0.0
                        elif hasattr(action.value, 'operator'):  # 一元操作
                            val = 0.0
                        else:
                            val = 0.0
                        
                        if target_name in actions:
                            actions[target_name] = val
                except Exception as e:
                    pass
    
    return actions['u_fz'], actions['u_tx'], actions['u_ty'], actions['u_tz']

# 目标位置函数
def target_pos(t):
    """计算圆形轨迹目标位置"""
    angle = OMEGA_TRAJ * t
    x = RADIUS * np.cos(angle)
    y = RADIUS * np.sin(angle)
    z = HEIGHT
    return np.array([x, y, z], dtype=np.float32)

# 主循环
for step in range(MAX_STEPS):
    t = step / CONTROL_FREQ
    
    # 计算目标位置
    tgt = target_pos(t)
    
    # 获取观测
    pos = obs['position'][0].cpu().numpy() if torch.is_tensor(obs['position']) else obs['position'][0]
    vel = obs['velocity'][0].cpu().numpy() if torch.is_tensor(obs['velocity']) else obs['velocity'][0]
    quat = obs['orientation'][0].cpu().numpy() if torch.is_tensor(obs['orientation']) else obs['orientation'][0]
    omega = obs['angular_velocity'][0].cpu().numpy() if torch.is_tensor(obs['angular_velocity']) else obs['angular_velocity'][0]
    
    # 计算误差
    pos_err = tgt - pos
    
    # 计算RPY
    rpy = Rotation.from_quat(quat).as_euler('XYZ', degrees=False)
    
    # 构建状态字典（完整版）
    state_dict = {
        'pos_err_x': float(pos_err[0]),
        'pos_err_y': float(pos_err[1]),
        'pos_err_z': float(pos_err[2]),
        'pos_err': float(np.linalg.norm(pos_err)),
        'pos_err_xy': float(np.linalg.norm(pos_err[:2])),
        'pos_err_z_abs': float(abs(pos_err[2])),
        'vel_x': float(vel[0]),
        'vel_y': float(vel[1]),
        'vel_z': float(vel[2]),
        'vel_err': float(np.linalg.norm(vel)),
        'err_p_roll': float(rpy[0]),
        'err_p_pitch': float(rpy[1]),
        'err_p_yaw': float(rpy[2]),
        'ang_err': float(np.linalg.norm(rpy)),
        'rpy_err_mag': float(np.linalg.norm(rpy)),
        'ang_vel_x': float(omega[0]),
        'ang_vel_y': float(omega[1]),
        'ang_vel_z': float(omega[2]),
        'ang_vel': float(np.linalg.norm(omega)),
        'ang_vel_mag': float(np.linalg.norm(omega)),
        # 积分项（简化为0）
        'err_i_x': 0.0,
        'err_i_y': 0.0,
        'err_i_z': 0.0,
        'err_i_roll': 0.0,
        'err_i_pitch': 0.0,
        'err_i_yaw': 0.0,
        # 微分项
        'err_d_x': float(-vel[0]),
        'err_d_y': float(-vel[1]),
        'err_d_z': float(-vel[2]),
        'err_d_roll': float(-omega[0]),
        'err_d_pitch': float(-omega[1]),
        'err_d_yaw': float(-omega[2]),
    }
    
    # 评估程序
    u_fz, u_tx, u_ty, u_tz = eval_program(program, state_dict)
    
    # 记录轨迹
    trajectory_data.append({
        'time': t,
        'pos_x': float(pos[0]),
        'pos_y': float(pos[1]),
        'pos_z': float(pos[2]),
        'pos_err_x': state_dict['pos_err_x'],
        'pos_err_y': state_dict['pos_err_y'],
        'pos_err_z': state_dict['pos_err_z'],
        'vel_x': state_dict['vel_x'],
        'vel_y': state_dict['vel_y'],
        'vel_z': state_dict['vel_z'],
        'ang_vel_x': state_dict['ang_vel_x'],
        'ang_vel_y': state_dict['ang_vel_y'],
        'ang_vel_z': state_dict['ang_vel_z'],
        'u_fz': u_fz,
        'u_tx': u_tx,
        'u_ty': u_ty,
        'u_tz': u_tz,
        'tgt_x': float(tgt[0]),
        'tgt_y': float(tgt[1]),
        'tgt_z': float(tgt[2]),
    })
    
    # 构建动作张量
    action_tensor = torch.zeros((NUM_ENVS, 6), device='cuda:0', dtype=torch.float32)
    action_tensor[0, 2] = u_fz
    action_tensor[0, 3] = u_tx
    action_tensor[0, 4] = u_ty
    action_tensor[0, 5] = u_tz
    
    # 执行步进
    obs, rewards, dones, info = env_pool.step(action_tensor)
    
    # 周期性打印
    if step % 50 == 0:
        print(f"  步骤 {step}/{MAX_STEPS}, pos_err={np.linalg.norm(pos_err):.3f}m, u_fz={u_fz:.3f}")

print(f"\n✓ 评估完成！记录了 {len(trajectory_data)} 个数据点\n")

# 保存轨迹
df = pd.DataFrame(trajectory_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✓ 轨迹已保存: {OUTPUT_CSV}")

# 统计分析
print("\n" + "="*70)
print("统计分析")
print("="*70)

pos_err_x_rmse = np.sqrt(np.mean(df['pos_err_x']**2))
pos_err_y_rmse = np.sqrt(np.mean(df['pos_err_y']**2))
pos_err_z_rmse = np.sqrt(np.mean(df['pos_err_z']**2))

print("位置误差 RMSE:")
print(f"  X: {pos_err_x_rmse:.4f} m")
print(f"  Y: {pos_err_y_rmse:.4f} m")
print(f"  Z: {pos_err_z_rmse:.4f} m\n")

print("控制输出统计:")
for ctrl in ['u_fz', 'u_tx', 'u_ty', 'u_tz']:
    mean_val = df[ctrl].mean()
    std_val = df[ctrl].std()
    min_val = df[ctrl].min()
    max_val = df[ctrl].max()
    print(f"  {ctrl}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 范围=[{min_val:.4f}, {max_val:.4f}]")

# 饱和分析
u_fz_saturated = ((df['u_fz'] <= -3.0) | (df['u_fz'] >= 3.0)).sum()
vel_y_zero = (df['vel_y'].abs() < 0.01).sum()

print(f"\nu_fz 饱和情况: {u_fz_saturated}/{len(df)} 步 ({100*u_fz_saturated/len(df):.1f}%)")
print(f"vel_y 接近零点: {vel_y_zero}/{len(df)} 步 ({100*vel_y_zero/len(df):.1f}%)")

print(f"\n✓ 分析完成！")

# 关闭环境
env_pool.close()
print("[Isaac Gym] 环境已关闭")

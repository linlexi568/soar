#!/usr/bin/env python3
"""快速测试当前best程序的实际控制效果"""
import numpy as np

# 模拟程序
def compute_control(state):
    """
    u_tx = 0.979 - err_d_roll*ang_vel_x - err_d_roll*err_p_roll
    u_ty = (err_p_pitch - 1.0448) * err_i_x * vel_x * pos_err_x
    """
    # 假设典型误差值
    pos_err_x = state.get('pos_err_x', 0.1)
    pos_err_y = state.get('pos_err_y', 0.1)
    vel_x = state.get('vel_x', 0.05)
    vel_y = state.get('vel_y', 0.05)
    err_p_roll = state.get('err_p_roll', 0.02)
    err_p_pitch = state.get('err_p_pitch', 0.02)
    err_d_roll = state.get('err_d_roll', 0.01)
    err_i_x = state.get('err_i_x', 0.0)
    ang_vel_x = state.get('ang_vel_x', 0.01)
    ang_vel_y = state.get('ang_vel_y', 0.01)
    
    u_tx = 0.979 - err_d_roll * ang_vel_x - err_d_roll * err_p_roll
    u_ty = (err_p_pitch - 1.0448) * err_i_x * vel_x * pos_err_x
    
    return u_tx, u_ty

# 测试场景
print("="*80)
print("测试当前最佳程序的控制输出")
print("="*80)

# 场景1：初始小误差
print("\n【场景1：初始小误差】")
state1 = {
    'pos_err_x': 0.1, 'pos_err_y': 0.1,
    'vel_x': 0.05, 'vel_y': 0.05,
    'err_p_roll': 0.02, 'err_p_pitch': 0.02,
    'err_d_roll': 0.01, 'err_i_x': 0.0,
    'ang_vel_x': 0.01, 'ang_vel_y': 0.01
}
u_tx, u_ty = compute_control(state1)
print(f"pos_err=(0.1, 0.1), vel=(0.05, 0.05)")
print(f"u_tx = {u_tx:.6f}")
print(f"u_ty = {u_ty:.6f}")
print(f"问题：u_tx是常数，完全不响应y轴误差！")
print(f"问题：u_ty≈0，因为err_i_x初始为0，乘积归零！")

# 场景2：大误差
print("\n【场景2：大位置误差】")
state2 = {
    'pos_err_x': 1.0, 'pos_err_y': 1.0,
    'vel_x': 0.2, 'vel_y': 0.2,
    'err_p_roll': 0.1, 'err_p_pitch': 0.1,
    'err_d_roll': 0.05, 'err_i_x': 0.5,
    'ang_vel_x': 0.05, 'ang_vel_y': 0.05
}
u_tx, u_ty = compute_control(state2)
print(f"pos_err=(1.0, 1.0), vel=(0.2, 0.2), err_i_x=0.5")
print(f"u_tx = {u_tx:.6f}")
print(f"u_ty = {u_ty:.6f}")
print(f"问题：u_tx仍是常数，不响应y轴大误差！")
print(f"问题：u_ty乘积发散，绝对值可能很大！")

# 场景3：积分累积
print("\n【场景3：积分项累积】")
state3 = {
    'pos_err_x': 0.5, 'pos_err_y': 0.5,
    'vel_x': 0.1, 'vel_y': 0.1,
    'err_p_roll': 0.05, 'err_p_pitch': 0.05,
    'err_d_roll': 0.02, 'err_i_x': 2.0,  # 积分饱和
    'ang_vel_x': 0.02, 'ang_vel_y': 0.02
}
u_tx, u_ty = compute_control(state3)
print(f"pos_err=(0.5, 0.5), vel=(0.1, 0.1), err_i_x=2.0")
print(f"u_tx = {u_tx:.6f}")
print(f"u_ty = {u_ty:.6f}")
print(f"问题：u_ty爆炸，因为四重乘积！")

# 对比：理想PD控制器
print("\n" + "="*80)
print("【理想PD控制器对比】")
print("="*80)
kp, kd = 2.0, 1.0
u_tx_ideal = kp * state1['pos_err_y'] + kd * state1['vel_y']
u_ty_ideal = kp * state1['pos_err_x'] + kd * state1['vel_x']
print(f"场景1下，理想PD (kp=2.0, kd=1.0):")
print(f"u_tx_ideal = {u_tx_ideal:.6f} (响应y轴误差)")
print(f"u_ty_ideal = {u_ty_ideal:.6f} (响应x轴误差)")

print("\n" + "="*80)
print("【结论】")
print("="*80)
print("1. u_tx ≈ 0.979 几乎是常数，完全不响应y轴位置误差")
print("2. u_ty 是四重乘积，在err_i_x=0时归零，在err_i_x大时爆炸")
print("3. 缺少核心的比例-微分(PD)反馈：kp*pos_err ± kd*vel")
print("4. 这解释了为什么reward = -7.37e5 (state_cost巨大)")
print("5. 无人机基本无法跟踪轨迹，只能靠悬停推力保持不坠落")
print("="*80)

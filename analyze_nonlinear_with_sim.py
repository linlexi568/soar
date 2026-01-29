#!/usr/bin/env python3
"""
非线性控制律分析: smooth vs sign
使用 Isaac Gym 真实仿真数据

图1: 非线性项响应特性对比
图2: 相平面分析 (真实仿真轨迹)
"""

import sys
from pathlib import Path
import math

# 路径设置
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / '01_soar') not in sys.path:
    sys.path.insert(0, str(ROOT / '01_soar'))

# Isaac Gym 路径 (使用正确的路径)
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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================================
#                         控制律参数 (来自 manual.md)
# ============================================================================

# Figure8 参数
FIGURE8_PARAMS = {
    'k_p': 0.489,
    'k_s': 1.285,
    'k_d': 1.062,
    'k_w': 0.731,
    'cost': 81.78,
}

# Square (sign) 参数
SQUARE_PARAMS = {
    'k_p': 0.600,
    'k_d': 1.766,
    'k_w': 0.773,
    'cost': 52.99,
}

# ============================================================================
#                         字体设置
# ============================================================================

LABEL_SIZE = 22
TITLE_SIZE = 24
TICK_SIZE = 18
LEGEND_SIZE = 16
ANNOT_SIZE = 14

# ============================================================================
#                         非线性函数
# ============================================================================

def smooth(val, s):
    """smooth 算子: s * tanh(x/s)"""
    if isinstance(val, np.ndarray):
        return s * np.tanh(val / s)
    return s * math.tanh(val / s)

def sign_func(val):
    """sign 算子"""
    if isinstance(val, np.ndarray):
        return np.sign(val)
    if val > 0:
        return 1.0
    elif val < 0:
        return -1.0
    return 0.0

# ============================================================================
#                         仿真函数
# ============================================================================

def run_simulation(traj_type: str, device: str = "cuda:0"):
    """运行 Isaac Gym 仿真，返回轨迹数据"""
    
    from envs.isaac_gym_drone_env import IsaacGymDroneEnv
    from utils.reward_scg_exact import SCGExactRewardCalculator
    from utilities.trajectory_presets import scg_position, get_scg_trajectory_config
    
    print(f"\n{'='*50}")
    print(f"Running simulation: {traj_type}")
    print('='*50)
    
    # 创建环境
    env = IsaacGymDroneEnv(
        num_envs=1,
        device=device,
        headless=True,
    )
    
    # SCG 代价计算器
    scg_calc = SCGExactRewardCalculator(
        num_envs=1,
        device=device,
    )
    
    # 获取轨迹配置
    cfg = get_scg_trajectory_config(traj_type)
    period = cfg.params.get('period', 5.0)
    
    def target_fn(t):
        return scg_position(traj_type, t, params=cfg.params, center=cfg.center)
    
    # 控制律
    if traj_type == 'figure8':
        k_p, k_s, k_d, k_w = 0.489, 1.285, 1.062, 0.731
        def u_tx_fn(sd):
            return ((-k_p * smooth(sd['pos_err_y'], k_s)) + (k_d * sd['vel_y'])) - (k_w * sd['ang_vel_x'])
    elif traj_type == 'square':
        k_p, k_d, k_w = 0.600, 1.766, 0.773
        def u_tx_fn(sd):
            return ((-k_p * sign_func(sd['pos_err_y'])) + (k_d * sd['vel_y'])) - (k_w * sd['ang_vel_x'])
    else:
        raise ValueError(f"Unknown trajectory: {traj_type}")
    
    # u_ty (镜像)
    if traj_type == 'figure8':
        def u_ty_fn(sd):
            return ((k_p * smooth(sd['pos_err_x'], k_s)) - (k_d * sd['vel_x'])) - (k_w * sd['ang_vel_y'])
    else:
        def u_ty_fn(sd):
            return ((k_p * sign_func(sd['pos_err_x'])) - (k_d * sd['vel_x'])) - (k_w * sd['ang_vel_y'])
    
    def u_tz_fn(sd):
        return 4.0 * sd['err_p_yaw'] - 0.8 * sd['ang_vel_z']
    
    def u_fz_fn(sd):
        return 1.0 * sd['pos_err_z'] - 0.5 * sd['vel_z'] + 0.65
    
    # 重置到起点
    initial_target = target_fn(0.0)
    initial_pos = torch.tensor([initial_target], device=env.device, dtype=torch.float32)
    env.reset(initial_pos=initial_pos)
    scg_calc.reset()
    
    dt = 1.0 / 48.0
    num_steps = int(period / dt)
    
    data = []
    
    for step in range(num_steps):
        t = step * dt
        target = target_fn(t)
        target_tensor = torch.tensor(target, device=env.device, dtype=torch.float32)
        
        pos = env.pos[0]
        vel = env.lin_vel[0]
        quat = env.quat[0]
        omega = env.ang_vel[0]
        
        pos_err = target_tensor - pos
        
        # 四元数转欧拉角
        qx, qy, qz, qw = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        err_p_yaw = -yaw
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
        
        # 记录数据
        data.append({
            'time': t,
            'pos_err_y': state_dict['pos_err_y'],
            'vel_y': state_dict['vel_y'],
            'ang_vel_x': state_dict['ang_vel_x'],
            'u_tx': u_tx,
        })
        
        # 执行动作
        actions = torch.tensor([[0.0, 0.0, u_fz, u_tx, u_ty, u_tz]], device=env.device)
        env.step(actions)
    
    env.close()
    
    print(f"  Collected {len(data)} data points")
    return data

# ============================================================================
#                     图1: 非线性项响应特性对比
# ============================================================================

def plot_nonlinear_response(save_path=None):
    """绘制 smooth vs sign 非线性项在不同 pos_err_y 下的响应"""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    pos_err = np.linspace(-2, 2, 1000)
    
    # ================== (a) 原始非线性函数 ==================
    ax = axes[0]
    
    smooth_out = smooth(pos_err, FIGURE8_PARAMS['k_s'])
    ax.plot(pos_err, smooth_out, 'b-', linewidth=2.5, label=f"smooth(e, s={FIGURE8_PARAMS['k_s']:.3f})")
    
    sign_out = sign_func(pos_err)
    ax.plot(pos_err, sign_out, 'r-', linewidth=2.5, label="sign(e)")
    
    ax.plot(pos_err, pos_err, 'k--', alpha=0.3, linewidth=1.5, label="linear (y=e)")
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Position Error $e$ (m)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Function Output', fontsize=LABEL_SIZE)
    ax.set_title('(a) Nonlinear Functions', fontsize=TITLE_SIZE)
    ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # ================== (b) 控制律比例项 ==================
    ax = axes[1]
    
    fig8_prop = -FIGURE8_PARAMS['k_p'] * smooth(pos_err, FIGURE8_PARAMS['k_s'])
    ax.plot(pos_err, fig8_prop, 'b-', linewidth=2.5, 
            label=f"Figure8: $-{FIGURE8_PARAMS['k_p']:.2f} \\cdot smooth$")
    
    sq_prop = -SQUARE_PARAMS['k_p'] * sign_func(pos_err)
    ax.plot(pos_err, sq_prop, 'r-', linewidth=2.5, 
            label=f"Square: $-{SQUARE_PARAMS['k_p']:.2f} \\cdot sign$")
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=LABEL_SIZE)
    ax.set_ylabel('P-term Output $u_p$', fontsize=LABEL_SIZE)
    ax.set_title('(b) Control Law P-term', fontsize=TITLE_SIZE)
    ax.legend(loc='upper right', fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    
    # ================== (c) 有效增益 ==================
    ax = axes[2]
    
    smooth_gain = 1.0 / np.cosh(pos_err / FIGURE8_PARAMS['k_s'])**2
    ax.plot(pos_err, FIGURE8_PARAMS['k_p'] * smooth_gain, 'b-', linewidth=2.5, 
            label="Figure8 (smooth)")
    
    eps = 0.05
    dirac_approx = (1.0 / (eps * np.sqrt(2*np.pi))) * np.exp(-pos_err**2 / (2*eps**2))
    dirac_approx = dirac_approx / np.max(dirac_approx) * 2
    ax.plot(pos_err, dirac_approx, 'r-', linewidth=2.5, alpha=0.6, 
            label="Square (sign)")
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Gain $\\partial u_p / \\partial e$', fontsize=LABEL_SIZE)
    ax.set_title('(c) Local Gain', fontsize=TITLE_SIZE)
    ax.legend(loc='upper right', fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.1, 2.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig

# ============================================================================
#                     图2: 相平面分析 (真实仿真数据)
# ============================================================================

def plot_phase_plane_with_sim(fig8_data, square_data, save_path=None):
    """使用真实仿真数据绘制相平面图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 网格 (背景)
    e_range = np.linspace(-0.5, 0.5, 80)
    v_range = np.linspace(-1.0, 1.0, 80)
    E, V = np.meshgrid(e_range, v_range)
    ang_vel_x = 0.0
    
    cmap = plt.cm.RdBu_r
    norm = Normalize(vmin=-0.4, vmax=0.4)
    
    # ================== (a) Figure8 ==================
    ax = axes[0]
    
    # 背景: 控制场
    U_fig8 = ((-FIGURE8_PARAMS['k_p'] * smooth(E, FIGURE8_PARAMS['k_s'])) + 
              (FIGURE8_PARAMS['k_d'] * V)) - (FIGURE8_PARAMS['k_w'] * ang_vel_x)
    c1 = ax.contourf(E, V, U_fig8, levels=50, cmap=cmap, norm=norm, alpha=0.7)
    
    # 真实轨迹
    pos_err_y = [d['pos_err_y'] for d in fig8_data]
    vel_y = [d['vel_y'] for d in fig8_data]
    u_tx = [d['u_tx'] for d in fig8_data]
    
    scatter = ax.scatter(pos_err_y, vel_y, c=u_tx, cmap=cmap, norm=norm, 
                         s=15, edgecolors='k', linewidths=0.3, zorder=5)
    ax.plot(pos_err_y, vel_y, 'k-', linewidth=0.8, alpha=0.5, zorder=4)
    ax.plot(pos_err_y[0], vel_y[0], 'go', markersize=12, label='Start', zorder=6)
    
    cbar = plt.colorbar(c1, ax=ax)
    cbar.set_label('$u_{tx}$', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Velocity $v_y$ (m/s)', fontsize=LABEL_SIZE)
    ax.set_title(f'(a) Figure8 (smooth)', fontsize=TITLE_SIZE)
    ax.legend(loc='upper right', fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    
    # ================== (b) Square ==================
    ax = axes[1]
    
    U_sq = ((-SQUARE_PARAMS['k_p'] * sign_func(E)) + 
            (SQUARE_PARAMS['k_d'] * V)) - (SQUARE_PARAMS['k_w'] * ang_vel_x)
    c2 = ax.contourf(E, V, U_sq, levels=50, cmap=cmap, norm=norm, alpha=0.7)
    
    pos_err_y = [d['pos_err_y'] for d in square_data]
    vel_y = [d['vel_y'] for d in square_data]
    u_tx = [d['u_tx'] for d in square_data]
    
    scatter = ax.scatter(pos_err_y, vel_y, c=u_tx, cmap=cmap, norm=norm, 
                         s=15, edgecolors='k', linewidths=0.3, zorder=5)
    ax.plot(pos_err_y, vel_y, 'k-', linewidth=0.8, alpha=0.5, zorder=4)
    ax.plot(pos_err_y[0], vel_y[0], 'go', markersize=12, label='Start', zorder=6)
    
    cbar = plt.colorbar(c2, ax=ax)
    cbar.set_label('$u_{tx}$', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Velocity $v_y$ (m/s)', fontsize=LABEL_SIZE)
    ax.set_title(f'(b) Square (sign)', fontsize=TITLE_SIZE)
    ax.legend(loc='upper right', fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig

# ============================================================================
#                              主函数
# ============================================================================

def main():
    output_dir = ROOT / 'results' / 'nonlinear_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Nonlinear Control Law Analysis: smooth vs sign")
    print("Using Isaac Gym Real Simulation Data")
    print("=" * 60)
    
    # 图1: 非线性响应特性 (无需仿真)
    print("\n[1/3] Plotting nonlinear response characteristics...")
    plot_nonlinear_response(save_path=output_dir / 'nonlinear_response.png')
    
    # 运行仿真
    print("\n[2/3] Running Isaac Gym simulations...")
    fig8_data = run_simulation('figure8')
    square_data = run_simulation('square')
    
    # 图2: 相平面 (使用真实数据)
    print("\n[3/3] Plotting phase plane with real simulation data...")
    plot_phase_plane_with_sim(fig8_data, square_data, 
                              save_path=output_dir / 'phase_plane_real.png')
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()

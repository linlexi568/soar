"""
相平面向量场分析 (Phase Portrait with Vector Field)

展示 Figure8 (smooth) 和 Square (sign) 控制器的闭环动态行为
在 (e_y, v_y) 平面上绘制向量场，直观显示系统状态如何演化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --- 字体设置 ---
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
#                         控制律参数 (来自 manual.md)
# ============================================================================

# Figure8 (Smooth): u_tx = ((-0.489 * smooth(pos_err_y, s=1.285)) + (1.062 * vel_y)) - (0.731 * ang_vel_x)
FIGURE8_PARAMS = {
    'k_p': 0.489,
    'k_s': 1.285,
    'k_d': 1.062,
    'k_w': 0.731
}

# Square (Sign): u_tx = ((-0.600 * sign(pos_err_y)) + (1.766 * vel_y)) - (0.773 * ang_vel_x)
SQUARE_PARAMS = {
    'k_p': 0.600,
    'k_d': 1.766,
    'k_w': 0.773
}

# PID Baseline (用于对比)
PID_PARAMS = {
    'k_p': 0.5,  # 等效线性增益
    'k_d': 1.0
}

# ============================================================================
#                         控制函数定义
# ============================================================================

def smooth(val, s):
    """smooth 算子: s * tanh(x/s)"""
    return s * np.tanh(val / s)

def ctrl_figure8(e_y, v_y, omega_x=0):
    """Figure8 控制律 (smooth)"""
    k_p, k_s, k_d, k_w = FIGURE8_PARAMS['k_p'], FIGURE8_PARAMS['k_s'], FIGURE8_PARAMS['k_d'], FIGURE8_PARAMS['k_w']
    u = (-k_p * smooth(e_y, k_s)) + (k_d * v_y) - (k_w * omega_x)
    return u

def ctrl_square(e_y, v_y, omega_x=0):
    """Square 控制律 (sign)"""
    k_p, k_d, k_w = SQUARE_PARAMS['k_p'], SQUARE_PARAMS['k_d'], SQUARE_PARAMS['k_w']
    u = (-k_p * np.sign(e_y)) + (k_d * v_y) - (k_w * omega_x)
    return u

def ctrl_pid(e_y, v_y):
    """线性 PID 控制律 (用于对比)"""
    k_p, k_d = PID_PARAMS['k_p'], PID_PARAMS['k_d']
    u = -k_p * e_y + k_d * v_y
    return u

# ============================================================================
#                         简化动力学模型
# ============================================================================
# 假设简化为二阶系统: e_y'' = -u (单位质量)
# 状态空间: x1 = e_y, x2 = v_y = e_y'
# dx1/dt = x2
# dx2/dt = -u(x1, x2)

def dynamics_figure8(e_y, v_y):
    """Figure8 闭环动力学"""
    u = ctrl_figure8(e_y, v_y)
    de_dt = v_y
    dv_dt = -u  # 简化: 加速度 = -控制量
    return de_dt, dv_dt

def dynamics_square(e_y, v_y):
    """Square 闭环动力学"""
    u = ctrl_square(e_y, v_y)
    de_dt = v_y
    dv_dt = -u
    return de_dt, dv_dt

def dynamics_pid(e_y, v_y):
    """PID 闭环动力学"""
    u = ctrl_pid(e_y, v_y)
    de_dt = v_y
    dv_dt = -u
    return de_dt, dv_dt

# ============================================================================
#                         轨迹仿真
# ============================================================================

def simulate_trajectory(dynamics_func, e0, v0, dt=0.01, t_max=10):
    """从初始状态仿真轨迹"""
    steps = int(t_max / dt)
    e_hist, v_hist = [e0], [v0]
    e, v = e0, v0
    
    for _ in range(steps):
        de, dv = dynamics_func(e, v)
        e += de * dt
        v += dv * dt
        e_hist.append(e)
        v_hist.append(v)
        
        # 收敛检测
        if abs(e) < 0.01 and abs(v) < 0.01:
            break
    
    return np.array(e_hist), np.array(v_hist)

# ============================================================================
#                         绘图
# ============================================================================

def plot_phase_portrait():
    """绘制相平面向量场"""
    
    # 定义网格
    e_range = np.linspace(-2, 2, 20)
    v_range = np.linspace(-2, 2, 20)
    E, V = np.meshgrid(e_range, v_range)
    
    # 计算向量场
    dE_fig8 = np.zeros_like(E)
    dV_fig8 = np.zeros_like(V)
    dE_square = np.zeros_like(E)
    dV_square = np.zeros_like(V)
    dE_pid = np.zeros_like(E)
    dV_pid = np.zeros_like(V)
    
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            dE_fig8[i,j], dV_fig8[i,j] = dynamics_figure8(E[i,j], V[i,j])
            dE_square[i,j], dV_square[i,j] = dynamics_square(E[i,j], V[i,j])
            dE_pid[i,j], dV_pid[i,j] = dynamics_pid(E[i,j], V[i,j])
    
    # 归一化向量长度 (便于可视化)
    def normalize_vectors(dE, dV):
        magnitude = np.sqrt(dE**2 + dV**2)
        magnitude[magnitude == 0] = 1
        return dE / magnitude, dV / magnitude, magnitude
    
    dE_fig8_n, dV_fig8_n, mag_fig8 = normalize_vectors(dE_fig8, dV_fig8)
    dE_square_n, dV_square_n, mag_square = normalize_vectors(dE_square, dV_square)
    dE_pid_n, dV_pid_n, mag_pid = normalize_vectors(dE_pid, dV_pid)
    
    # 仿真几条代表性轨迹
    initial_conditions = [
        (-1.5, 0),    # 左侧静止
        (1.5, 0),     # 右侧静止
        (0, 1.5),     # 中心向上运动
        (0, -1.5),    # 中心向下运动
        (-1.5, 1.0),  # 左上
        (1.5, -1.0),  # 右下
    ]
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    configs = [
        ('Figure8 (Smooth)', dE_fig8_n, dV_fig8_n, mag_fig8, dynamics_figure8, 'Blues'),
        ('Square (Sign)', dE_square_n, dV_square_n, mag_square, dynamics_square, 'Oranges'),
        ('Linear PID', dE_pid_n, dV_pid_n, mag_pid, dynamics_pid, 'Greens'),
    ]
    
    for ax, (title, dE_n, dV_n, mag, dyn_func, cmap) in zip(axes, configs):
        # 绘制向量场 (用颜色表示速度大小)
        quiver = ax.quiver(E, V, dE_n, dV_n, mag, 
                          cmap=cmap, alpha=0.7, scale=25)
        
        # 绘制轨迹
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
        for (e0, v0), color in zip(initial_conditions, colors):
            e_traj, v_traj = simulate_trajectory(dyn_func, e0, v0, dt=0.01, t_max=15)
            ax.plot(e_traj, v_traj, color=color, linewidth=1.5, alpha=0.8)
            ax.plot(e0, v0, 'o', color=color, markersize=6)  # 起点
        
        # 标记原点 (平衡点)
        ax.plot(0, 0, 'k*', markersize=15, label='Equilibrium')
        
        # 对于 Square，标记切换面 e_y = 0
        if 'Sign' in title:
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                      alpha=0.5, label='Switching surface')
        
        ax.set_xlim([-2.2, 2.2])
        ax.set_ylim([-2.2, 2.2])
        ax.set_xlabel('Position Error $e_y$ (m)', fontsize=14)
        ax.set_ylabel('Velocity $v_y$ (m/s)', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        
        if 'Sign' in title:
            ax.legend(loc='upper right', fontsize=10)
    
    # 添加 (a), (b), (c) 标签
    labels = ['(a)', '(b)', '(c)']
    for i, ax in enumerate(axes):
        ax.text(0.5, -0.12, labels[i], transform=ax.transAxes, 
                fontsize=16, ha='center', va='top')
    
    plt.tight_layout()
    
    # 保存
    output_path = 'phase_portrait_vector_field.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {output_path}")
    
    # 额外：绘制速度场强度对比图
    plot_velocity_magnitude_comparison()
    
    return fig

def plot_velocity_magnitude_comparison():
    """绘制收敛速度对比热力图"""
    
    e_range = np.linspace(-2, 2, 100)
    v_range = np.linspace(-2, 2, 100)
    E, V = np.meshgrid(e_range, v_range)
    
    # 计算每点的"收敛趋势" = -d(e^2 + v^2)/dt = -2(e*de/dt + v*dv/dt)
    def convergence_rate(dynamics_func, E, V):
        rate = np.zeros_like(E)
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                e, v = E[i,j], V[i,j]
                de, dv = dynamics_func(e, v)
                # Lyapunov derivative: dV/dt where V = 0.5*(e^2 + v^2)
                rate[i,j] = -(e * de + v * dv)
        return rate
    
    rate_fig8 = convergence_rate(dynamics_figure8, E, V)
    rate_square = convergence_rate(dynamics_square, E, V)
    rate_pid = convergence_rate(dynamics_pid, E, V)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [
        ('Figure8 (Smooth)', rate_fig8),
        ('Square (Sign)', rate_square),
        ('Linear PID', rate_pid),
    ]
    
    vmin = min(rate_fig8.min(), rate_square.min(), rate_pid.min())
    vmax = max(rate_fig8.max(), rate_square.max(), rate_pid.max())
    
    for ax, (title, rate) in zip(axes, datasets):
        im = ax.contourf(E, V, rate, levels=50, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        ax.contour(E, V, rate, levels=[0], colors='black', linewidths=2)
        ax.set_xlabel('Position Error $e_y$ (m)', fontsize=14)
        ax.set_ylabel('Velocity $v_y$ (m/s)', fontsize=14)
        ax.set_title(f'{title}\n$-\\dot{{V}}$ (Convergence Rate)', fontsize=14)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 添加 (a), (b), (c) 标签
    labels = ['(a)', '(b)', '(c)']
    for i, ax in enumerate(axes):
        ax.text(0.5, -0.12, labels[i], transform=ax.transAxes, 
                fontsize=16, ha='center', va='top')
    
    plt.tight_layout()
    output_path = 'convergence_rate_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"收敛速度对比图已保存到: {output_path}")

if __name__ == "__main__":
    plot_phase_portrait()
    print("\n分析完成!")

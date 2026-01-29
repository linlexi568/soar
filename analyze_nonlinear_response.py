#!/usr/bin/env python3
"""
非线性控制律分析: smooth vs sign 响应特性对比 + 相平面分析

分析来自 manual.md 的两个控制律:
- Figure8 (smooth): u_tx = ((-0.489 * smooth(pos_err_y, s=1.285)) + (1.062 * vel_y)) - (0.731 * ang_vel_x)
- Square (sign):    u_tx = ((-0.600 * sign(pos_err_y)) + (1.766 * vel_y)) - (0.773 * ang_vel_x)

生成图表:
1. 非线性项的响应特性对比 (smooth vs sign 在不同 pos_err_y 下)
2. 相平面分析 (pos_err_y vs vel_y，颜色标注控制输入)
"""

import sys
from pathlib import Path
import math
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

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

# Square 参数
SQUARE_PARAMS = {
    'k_p': 0.600,
    'k_d': 1.766,
    'k_w': 0.773,
    'cost': 52.99,
}

# ============================================================================
#                         非线性函数定义
# ============================================================================

def smooth(val, s):
    """smooth 算子: s * tanh(x/s)
    
    特点:
    - 小 x: 近似线性 ≈ x
    - 大 x: 饱和到 ±s
    - 处处可微
    """
    return s * np.tanh(val / s)

def sign_func(val):
    """sign 算子
    
    特点:
    - bang-bang 控制
    - 只有 {-1, 0, +1} 三个输出
    - 在 0 处不可微
    """
    return np.sign(val)

# ============================================================================
#                         控制律计算
# ============================================================================

def compute_figure8_utx(pos_err_y, vel_y, ang_vel_x):
    """Figure8 控制律 (使用 smooth)"""
    p = FIGURE8_PARAMS
    return ((-p['k_p'] * smooth(pos_err_y, p['k_s'])) + (p['k_d'] * vel_y)) - (p['k_w'] * ang_vel_x)

def compute_square_utx(pos_err_y, vel_y, ang_vel_x):
    """Square 控制律 (使用 sign)"""
    p = SQUARE_PARAMS
    return ((-p['k_p'] * sign_func(pos_err_y)) + (p['k_d'] * vel_y)) - (p['k_w'] * ang_vel_x)

# ============================================================================
#                     图1: 非线性项响应特性对比
# ============================================================================

def plot_nonlinear_response(save_path=None):
    """绘制 smooth vs sign 非线性项在不同 pos_err_y 下的响应"""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 误差范围
    pos_err = np.linspace(-2, 2, 1000)
    
    # ================== 子图1: 原始非线性函数 ==================
    ax = axes[0]
    
    # Smooth
    smooth_out = smooth(pos_err, FIGURE8_PARAMS['k_s'])
    ax.plot(pos_err, smooth_out, 'b-', linewidth=2, label=f"smooth(e, s={FIGURE8_PARAMS['k_s']:.3f})")
    
    # Sign
    sign_out = sign_func(pos_err)
    ax.plot(pos_err, sign_out, 'r-', linewidth=2, label="sign(e)")
    
    # 线性参考
    ax.plot(pos_err, pos_err, 'k--', alpha=0.3, linewidth=1, label="线性 (y=e)")
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Position Error $e$ (m)', fontsize=11)
    ax.set_ylabel('Function Output', fontsize=11)
    ax.set_title('(a) Nonlinear Function Characteristics', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # ================== Subplot 2: Proportional Term ==================
    ax = axes[1]
    
    # Figure8: -k_p * smooth(e, s)
    fig8_prop = -FIGURE8_PARAMS['k_p'] * smooth(pos_err, FIGURE8_PARAMS['k_s'])
    ax.plot(pos_err, fig8_prop, 'b-', linewidth=2, 
            label=f"Figure8: $-{FIGURE8_PARAMS['k_p']:.3f} \\cdot smooth(e)$")
    
    # Square: -k_p * sign(e)
    sq_prop = -SQUARE_PARAMS['k_p'] * sign_func(pos_err)
    ax.plot(pos_err, sq_prop, 'r-', linewidth=2, 
            label=f"Square: $-{SQUARE_PARAMS['k_p']:.3f} \\cdot sign(e)$")
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=11)
    ax.set_ylabel('Proportional Output $u_p$', fontsize=11)
    ax.set_title('(b) Control Law P-term Comparison', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    
    # ================== Subplot 3: Effective Gain ==================
    ax = axes[2]
    
    # Smooth derivative: d/de [s*tanh(e/s)] = sech^2(e/s)
    smooth_gain = 1.0 / np.cosh(pos_err / FIGURE8_PARAMS['k_s'])**2
    ax.plot(pos_err, FIGURE8_PARAMS['k_p'] * smooth_gain, 'b-', linewidth=2, 
            label="Figure8 (smooth)")
    
    # Sign "effective gain" - Dirac delta approximation
    eps = 0.05
    dirac_approx = (1.0 / (eps * np.sqrt(2*np.pi))) * np.exp(-pos_err**2 / (2*eps**2))
    dirac_approx = dirac_approx / np.max(dirac_approx) * 2
    ax.plot(pos_err, dirac_approx, 'r-', linewidth=2, alpha=0.6, 
            label="Square (sign) - impulse")
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=11)
    ax.set_ylabel('Effective Gain $\\partial u_p / \\partial e$', fontsize=11)
    ax.set_title('(c) Local Gain Characteristics', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.1, 2.5)
    
    # Annotation
    ax.annotate('smooth: high gain at small error\nsaturates at large error', xy=(0.5, 0.3), fontsize=9,
                xycoords='axes fraction', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {save_path}")
    
    plt.show()
    return fig

# ============================================================================
#                     图2: 相平面分析
# ============================================================================

def plot_phase_plane(save_path=None):
    """绘制 pos_err_y vs vel_y 相平面图，用颜色标注控制输入"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 网格范围
    e_range = np.linspace(-1.5, 1.5, 100)
    v_range = np.linspace(-2, 2, 100)
    E, V = np.meshgrid(e_range, v_range)
    
    # 固定 ang_vel_x = 0
    ang_vel_x = 0.0
    
    # ================== 子图1: Figure8 (smooth) ==================
    ax = axes[0]
    
    U_fig8 = compute_figure8_utx(E, V, ang_vel_x)
    
    # 热力图
    cmap = plt.cm.RdBu_r
    norm = Normalize(vmin=-0.5, vmax=0.5)
    c1 = ax.contourf(E, V, U_fig8, levels=50, cmap=cmap, norm=norm)
    
    # 等高线
    contours = ax.contour(E, V, U_fig8, levels=[-0.3, -0.1, 0, 0.1, 0.3], 
                          colors='k', linewidths=0.5, alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    # 零控制线 (u_tx = 0)
    ax.contour(E, V, U_fig8, levels=[0], colors='white', linewidths=2)
    
    plt.colorbar(c1, ax=ax, label='$u_{tx}$')
    
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=11)
    ax.set_ylabel('Velocity $v_y$ (m/s)', fontsize=11)
    ax.set_title(f'(a) Figure8 Phase Plane\n(smooth, Cost={FIGURE8_PARAMS["cost"]})', fontsize=12)
    
    # ================== Subplot 2: Square (sign) ==================
    ax = axes[1]
    
    U_sq = compute_square_utx(E, V, ang_vel_x)
    
    c2 = ax.contourf(E, V, U_sq, levels=50, cmap=cmap, norm=norm)
    contours = ax.contour(E, V, U_sq, levels=[-0.3, -0.1, 0, 0.1, 0.3], 
                          colors='k', linewidths=0.5, alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    ax.contour(E, V, U_sq, levels=[0], colors='white', linewidths=2)
    
    plt.colorbar(c2, ax=ax, label='$u_{tx}$')
    
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=11)
    ax.set_ylabel('Velocity $v_y$ (m/s)', fontsize=11)
    ax.set_title(f'(b) Square Phase Plane\n(sign, Cost={SQUARE_PARAMS["cost"]})', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {save_path}")
    
    plt.show()
    return fig

# ============================================================================
#                     图3: 相平面流线图 (矢量场)
# ============================================================================

def plot_phase_plane_streamlines(save_path=None):
    """绘制相平面流线图 - 展示闭环系统行为"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 网格
    e_range = np.linspace(-1.5, 1.5, 30)
    v_range = np.linspace(-2, 2, 30)
    E, V = np.meshgrid(e_range, v_range)
    
    # 简化动力学: de/dt = -v, dv/dt ≈ k * u_tx (忽略高阶动力学)
    # 这里假设 u_tx 近似正比于加速度
    k_accel = 5.0  # 响应增益
    ang_vel_x = 0.0
    
    # ================== 子图1: Figure8 ==================
    ax = axes[0]
    
    U_fig8 = compute_figure8_utx(E, V, ang_vel_x)
    
    # State derivatives
    dE_dt = -V  # de/dt = -v
    dV_dt = k_accel * U_fig8  # simplified accel response
    
    # Streamlines
    speed = np.sqrt(dE_dt**2 + dV_dt**2)
    lw = 2 * speed / speed.max()
    
    ax.streamplot(E, V, dE_dt, dV_dt, color=U_fig8, cmap='RdBu_r', 
                  linewidth=1.5, density=1.5, arrowsize=1.5)
    
    # Equilibrium point
    ax.plot(0, 0, 'ko', markersize=10, markerfacecolor='yellow', 
            markeredgewidth=2, label='Equilibrium')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=11)
    ax.set_ylabel('Velocity $v_y$ (m/s)', fontsize=11)
    ax.set_title('(a) Figure8 Phase Portrait (smooth)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    
    # ================== Subplot 2: Square ==================
    ax = axes[1]
    
    U_sq = compute_square_utx(E, V, ang_vel_x)
    
    dE_dt = -V
    dV_dt = k_accel * U_sq
    
    ax.streamplot(E, V, dE_dt, dV_dt, color=U_sq, cmap='RdBu_r', 
                  linewidth=1.5, density=1.5, arrowsize=1.5)
    
    ax.plot(0, 0, 'ko', markersize=10, markerfacecolor='yellow', 
            markeredgewidth=2, label='Equilibrium')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position Error $e_y$ (m)', fontsize=11)
    ax.set_ylabel('Velocity $v_y$ (m/s)', fontsize=11)
    ax.set_title('(b) Square Phase Portrait (sign)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {save_path}")
    
    plt.show()
    return fig

# ============================================================================
#                     图4: 不同 smooth 参数 s 的影响
# ============================================================================

def plot_smooth_parameter_study(save_path=None):
    """研究 smooth 参数 s 对控制特性的影响"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    pos_err = np.linspace(-2, 2, 500)
    
    # ================== 子图1: 不同 s 值的 smooth 函数 ==================
    ax = axes[0]
    
    s_values = [0.296, 0.5, 1.0, 1.285, 1.988, 3.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(s_values)))
    
    for s, c in zip(s_values, colors):
        y = smooth(pos_err, s)
        label = f's={s:.3f}'
        if s == 0.296:
            label += ' (circle)'
        elif s == 1.285:
            label += ' (figure8)'
        elif s == 1.988:
            label += ' (square)'
        ax.plot(pos_err, y, color=c, linewidth=2, label=label)
    
    # 参考线
    ax.plot(pos_err, sign_func(pos_err), 'r--', linewidth=1.5, alpha=0.5, label='sign(e)')
    ax.plot(pos_err, pos_err, 'k--', linewidth=1, alpha=0.3, label='linear')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Position Error $e$ (m)', fontsize=11)
    ax.set_ylabel('smooth(e, s)', fontsize=11)
    ax.set_title('(a) smooth Function with Different s', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2.5, 2.5)
    
    # ================== Subplot 2: Physical Meaning of s ==================
    ax = axes[1]
    
    s_range = np.linspace(0.1, 3.0, 100)
    
    # Saturation = s (smooth approaches +/-s for large errors)
    ax.plot(s_range, s_range, 'b-', linewidth=2, label='Saturation = s')
    
    # Slope at origin = 1 (for any s)
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Slope at origin = 1')
    
    # Mark s values for three trajectories
    for s, traj, color in [(0.296, 'circle', 'green'), 
                            (1.285, 'figure8', 'orange'), 
                            (1.988, 'square', 'purple')]:
        ax.axvline(x=s, color=color, linestyle=':', linewidth=2, alpha=0.7)
        ax.annotate(f'{traj}\ns={s}', xy=(s, 0.5), fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Parameter s', fontsize=11)
    ax.set_ylabel('Characteristic Value', fontsize=11)
    ax.set_title('(b) Physical Meaning of s in smooth(e,s)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 3.5)
    
    # Explanation
    ax.text(1.5, 2.8, "smaller s -> closer to sign (bang-bang)\nlarger s -> closer to linear", 
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {save_path}")
    
    plt.show()
    return fig

# ============================================================================
#                              主函数
# ============================================================================

def main():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 输出目录
    output_dir = Path(__file__).resolve().parent / 'results' / 'nonlinear_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("非线性控制律分析: smooth vs sign")
    print("=" * 60)
    print("\nFigure8 控制律 (smooth):")
    print(f"  u_tx = -k_p·smooth(e_y, s) + k_d·v_y - k_w·ω_x")
    print(f"  k_p={FIGURE8_PARAMS['k_p']}, s={FIGURE8_PARAMS['k_s']}, k_d={FIGURE8_PARAMS['k_d']}, k_w={FIGURE8_PARAMS['k_w']}")
    print(f"  Cost: {FIGURE8_PARAMS['cost']}")
    
    print("\nSquare 控制律 (sign):")
    print(f"  u_tx = -k_p·sign(e_y) + k_d·v_y - k_w·ω_x")
    print(f"  k_p={SQUARE_PARAMS['k_p']}, k_d={SQUARE_PARAMS['k_d']}, k_w={SQUARE_PARAMS['k_w']}")
    print(f"  Cost: {SQUARE_PARAMS['cost']}")
    print()
    
    # 生成图表
    print("\n[1/4] 非线性响应特性对比...")
    plot_nonlinear_response(save_path=output_dir / 'nonlinear_response.png')
    
    print("\n[2/4] 相平面分析 (热力图)...")
    plot_phase_plane(save_path=output_dir / 'phase_plane_heatmap.png')
    
    print("\n[3/4] 相平面流线图...")
    plot_phase_plane_streamlines(save_path=output_dir / 'phase_plane_streamlines.png')
    
    print("\n[4/4] smooth 参数研究...")
    plot_smooth_parameter_study(save_path=output_dir / 'smooth_parameter_study.png')
    
    print("\n" + "=" * 60)
    print(f"✓ 所有图表已保存到: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()

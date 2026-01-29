#!/usr/bin/env python3
"""
Square Trajectory Analysis: PID vs LQR vs Smooth (Soar)
Comparison using real benchmark results

Comparison of:
1. PID Controller (from benchmark/results/pid/pid_square.json)
   Cost = 81.99, RMSE = 0.437m

2. LQR Controller (from benchmark/results/lqr/lqr_square.json)
   Cost = 61.35, RMSE = 0.322m

3. Smooth (Soar) Controller (from benchmark/results/soar/square_soar.json)
   k_p = 1.643, s = 0.786, k_d = 1.633
   Cost = 43.34, RMSE = 0.376m
   ✅ Best Performance!

Plots:
(a) Control Output vs Error (P-Term)
(b) Effective Stiffness (Gain)
(c) Phase Plane (Step Response - Double Integrator Model)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --- 字体设置 ---
# 尝试加载系统中的 Times New Roman
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    # 如果找不到绝对路径，尝试使用通用名称，或者回退到 serif
    print(f"Warning: Font file not found at {font_path}, trying generic family.")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# ============================================================================
#                         Parameters (from benchmark results)
# ============================================================================

# PID Parameters
PID_PARAMS = {
    'kp': 8.826,  # kp_xy from benchmark
    'kd': 5.752,  # kd_xy
    'ki': 0.0,
    'cost': 81.99,
    'rmse': 0.437,
    'name': 'PID'
}

# LQR Parameters
LQR_PARAMS = {
    'kp': 10.349,  # k_pos from benchmark
    'kd': 5.384,   # k_vel
    'cost': 61.35,
    'rmse': 0.322,
    'name': 'LQR'
}

# Smooth (Soar) Parameters
SMOOTH_PARAMS = {
    'kp': 1.643,
    's': 0.786,
    'kd': 1.633,
    'cost': 43.34,
    'rmse': 0.376,
    'name': 'Smooth'
}

# 饱和限制 (Normalized Torque)
U_MAX = 1.0

# ============================================================================
#                         Functions
# ============================================================================

def ctrl_smooth(e, v):
    # P term: kp * s * tanh(e/s)
    u_p = SMOOTH_PARAMS['kp'] * SMOOTH_PARAMS['s'] * np.tanh(e / SMOOTH_PARAMS['s'])
    u_d = SMOOTH_PARAMS['kd'] * v
    return np.clip(u_p + u_d, -U_MAX, U_MAX)

def ctrl_pid(e, v):
    u_p = PID_PARAMS['kp'] * e
    u_d = PID_PARAMS['kd'] * v
    return np.clip(u_p + u_d, -U_MAX, U_MAX)

def ctrl_lqr(e, v):
    u_p = LQR_PARAMS['kp'] * e
    u_d = LQR_PARAMS['kd'] * v
    return np.clip(u_p + u_d, -U_MAX, U_MAX)

# ============================================================================
#                         Simulation (Phase Plane)
# ============================================================================

# --- Phase Plane Simulation ---
dt = 0.005
steps = 2400  # 12s duration

def run_sim(ctrl_func):
    x = -2.0  # Start at -2m
    v = 0.0
    hist_x = [x]
    hist_v = [v]
    for _ in range(steps):
        # Error definition: e = Target - x = 0 - x = -x
        e = -x
        u = ctrl_func(e, -v)  # D term acts on error_dot = -v
        
        accel = u
        v += accel * dt
        x += v * dt
        hist_x.append(x)
        hist_v.append(v)
    return hist_x, hist_v

# ============================================================================
#                         Plotting
# ============================================================================

def plot_analysis():
    # --- 1. Static P-Curve Data (v=0) ---
    e_range = np.linspace(-2, 2, 200)
    p_smooth = [ctrl_smooth(e, 0) for e in e_range]
    p_pid = [ctrl_pid(e, 0) for e in e_range]
    p_lqr = [ctrl_lqr(e, 0) for e in e_range]

    # --- 2. Effective Stiffness Data (u_p / e) ---
    e_abs = np.linspace(0.01, 2, 200)
    k_eff_smooth = [abs(ctrl_smooth(e, 0)/e) for e in e_abs]
    k_eff_pid = [abs(ctrl_pid(e, 0)/e) for e in e_abs]
    k_eff_lqr = [abs(ctrl_lqr(e, 0)/e) for e in e_abs]

    # --- 3. Phase Plane Simulation ---
    xs, vs = run_sim(ctrl_smooth)
    xp, vp = run_sim(ctrl_pid)
    xl, vl = run_sim(ctrl_lqr)

    # --- Plotting ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: P-Curve
    # Note: For SOAR, K shown is the linearized gain (kp), not kp*s
    axs[0].plot(e_range, p_pid, 'r--', label=f'PID (K={PID_PARAMS["kp"]:.1f})', alpha=0.6)
    axs[0].plot(e_range, p_lqr, 'g-.', label=f'LQR (K={LQR_PARAMS["kp"]:.1f})', alpha=0.6)
    axs[0].plot(e_range, p_smooth, 'b-', linewidth=2.5, label=f'SOAR (K={SMOOTH_PARAMS["kp"]:.2f})')
    axs[0].set_title('Control Output vs Error (P-Term)', fontsize=20)
    axs[0].set_xlabel('Position Error (m)', fontsize=20)
    axs[0].set_ylabel('Control Output (Normalized)', fontsize=20)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='lower right', fontsize=13)
    axs[0].set_ylim([-1.2, 1.2])

    # Plot 2: Effective Stiffness
    axs[1].plot(e_abs, k_eff_pid, 'r--', label='PID', alpha=0.6)
    axs[1].plot(e_abs, k_eff_lqr, 'g-.', label='LQR', alpha=0.6)
    axs[1].plot(e_abs, k_eff_smooth, 'b-', linewidth=2.5, label='SOAR')
    axs[1].set_title('Effective Stiffness (Gain)', fontsize=20)
    axs[1].set_xlabel('Error Magnitude |e| (m)', fontsize=20)
    axs[1].set_ylabel('Effective Gain (u/e)', fontsize=20)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=13)
    axs[1].set_yscale('log')  # Log scale to show the huge difference

    # Plot 3: Phase Plane
    axs[2].plot(xp, vp, 'r--', label='PID', alpha=0.6)
    axs[2].plot(xl, vl, 'g-.', label='LQR', alpha=0.6)
    axs[2].plot(xs, vs, 'b-', linewidth=2.5, label='SOAR')
    axs[2].set_title('Phase Plane (Step Response)', fontsize=20)
    axs[2].set_xlabel('Position x (m)', fontsize=20)
    axs[2].set_ylabel('Velocity v (m/s)', fontsize=20)
    axs[2].grid(True, alpha=0.3)
    axs[2].plot([-2], [0], 'ko', label='Start', markersize=8, zorder=10)  # Start
    axs[2].plot([0], [0], 'rx', markersize=12, label='Target', markeredgewidth=2, zorder=10)  # Target
    # Move legend to lower center
    axs[2].legend(loc='lower center', framealpha=0.9, fontsize=13)
    axs[2].set_ylim(bottom=-1)

    # Add (a), (b), (c) labels
    labels = ['(a)', '(b)', '(c)']
    for i, ax in enumerate(axs):
        ax.text(0.5, -0.25, labels[i], transform=ax.transAxes, 
                fontsize=20, fontweight='normal', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('square_analysis_plots.png', dpi=300)
    print("\n" + "="*70)
    print("SQUARE TRAJECTORY ANALYSIS - BENCHMARK COMPARISON")
    print("="*70)
    print(f"{'Controller':<12} {'Cost':<10} {'RMSE (m)':<12} {'Improvement':<15}")
    print("-"*70)
    
    pid_cost = PID_PARAMS['cost']
    lqr_cost = LQR_PARAMS['cost']
    smooth_cost = SMOOTH_PARAMS['cost']
    
    lqr_improve = (1 - lqr_cost/pid_cost) * 100
    smooth_improve = (1 - smooth_cost/pid_cost) * 100
    
    print(f"{'PID':<12} {pid_cost:<10.2f} {PID_PARAMS['rmse']:<12.3f} {'Baseline':<15}")
    print(f"{'LQR':<12} {lqr_cost:<10.2f} {LQR_PARAMS['rmse']:<12.3f} {lqr_improve:.1f}%")
    print(f"{'SOAR':<12} {smooth_cost:<10.2f} {SMOOTH_PARAMS['rmse']:<12.3f} {smooth_improve:.1f}% ✅ BEST")
    print("="*70)
    print("✅ Saved: square_analysis_plots.png")
    print("="*70)

if __name__ == '__main__':
    plot_analysis()

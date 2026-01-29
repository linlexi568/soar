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

# --- 参数定义 (Square Task) ---

# 1. Sign Controller (Manual)
# u_tx = ((-0.600 * sign(pos_err_y)) + (1.766 * vel_y))
k_p_sign = 0.600
k_d_sign = 1.766

# 2. PID Baseline (Square)
# kp_xy=8.826, kd_xy=5.752, kp_att=9.566, att_scale=0.278
k_p_pid = 8.826 * 0.278 * 9.566   # ~23.47
k_d_pid = 5.752 * 0.278 * 9.566   # ~15.30

# 3. LQR Baseline (Square)
# k_pos=7.638, k_vel=4.810, k_att=14.722, att_scale=0.276
k_p_lqr = 7.638 * 0.276 * 14.722  # ~31.03
k_d_lqr = 4.810 * 0.276 * 14.722  # ~19.54

# 饱和限制
U_MAX = 1.0

# --- 定义控制函数 ---

def ctrl_sign(e, v):
    # u = -0.6 * sign(e) + 1.766 * v
    # In our sim logic: accel = -u.
    # So we return u exactly as formula.
    # Note: v in formula is velocity. In sim we pass v.
    # e in formula is pos_err.
    
    # Avoid chattering in static plot by using sign
    u_p = -k_p_sign * np.sign(e)
    u_d = k_d_sign * v
    
    # For plotting P-curve, we usually plot the magnitude of restoring force.
    # Formula: u_tx.
    # If e>0, u_tx = -0.6. This is restoring force (negative).
    # We will plot the raw output u_tx.
    return np.clip(u_p + u_d, -U_MAX, U_MAX)

def ctrl_pid(e, v):
    # PID formula: u = -Kp * e + Kd * v
    # (Matches manual structure: -Kp*smooth(e) + Kd*v)
    u_p = -k_p_pid * e
    u_d = k_d_pid * v
    return np.clip(u_p + u_d, -U_MAX, U_MAX)

def ctrl_lqr(e, v):
    u_p = -k_p_lqr * e
    u_d = k_d_lqr * v
    return np.clip(u_p + u_d, -U_MAX, U_MAX)

# --- 1. Static P-Curve Data (v=0) ---
# Zoom in to show the difference between High Gain (PID) and Sign
# PID is linear but saturates quickly. Sign is a step.
e_range = np.linspace(-0.2, 0.2, 400)
p_sign = [ctrl_sign(e, 0) for e in e_range]
p_pid = [ctrl_pid(e, 0) for e in e_range]
p_lqr = [ctrl_lqr(e, 0) for e in e_range]

# --- 2. Effective Stiffness Data (|u_p| / |e|) ---
e_abs = np.linspace(0.001, 2, 400)
k_eff_sign = [abs(ctrl_sign(e, 0)/e) for e in e_abs]
k_eff_pid = [abs(ctrl_pid(e, 0)/e) for e in e_abs]
k_eff_lqr = [abs(ctrl_lqr(e, 0)/e) for e in e_abs]

# --- 3. Phase Plane Simulation ---
dt = 0.001 # Smaller dt for sign controller to reduce numerical chatter
steps = 8160 # Increase to 12s to ensure Sign controller reaches target

def run_sim(ctrl_func):
    x = -2.0 
    v = 0.0
    hist_x = [x]
    hist_v = [v]
    for _ in range(steps):
        e = -x # Target 0
        u = ctrl_func(e, v)
        accel = -u 
        
        v += accel * dt
        x += v * dt
        hist_x.append(x)
        hist_v.append(v)
    return hist_x, hist_v

xs, vs = run_sim(ctrl_sign)
xp, vp = run_sim(ctrl_pid)
xl, vl = run_sim(ctrl_lqr)

# --- Plotting ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: P-Curve
axs[0].plot(e_range, p_pid, 'r--', label=f'PID (K={k_p_pid:.1f})', alpha=0.6)
axs[0].plot(e_range, p_lqr, 'g-.', label=f'LQR (K={k_p_lqr:.1f})', alpha=0.6)
axs[0].plot(e_range, p_sign, 'b-', linewidth=2.5, label=f'Sign (Mag={k_p_sign:.2f})')
axs[0].set_title('Control Output vs Error (Zoomed)', fontsize=20)
axs[0].set_xlabel('Position Error (m)', fontsize=20)
axs[0].set_ylabel('Control Output (Normalized)', fontsize=20)
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc='upper right', fontsize=13)
axs[0].set_ylim([-1.2, 1.2])

# Plot 2: Effective Stiffness
axs[1].plot(e_abs, k_eff_pid, 'r--', label='PID', alpha=0.6)
axs[1].plot(e_abs, k_eff_lqr, 'g-.', label='LQR', alpha=0.6)
axs[1].plot(e_abs, k_eff_sign, 'b-', linewidth=2.5, label='Sign')
axs[1].set_title('Effective Stiffness (Gain)', fontsize=20)
axs[1].set_xlabel('Error Magnitude |e| (m)', fontsize=20)
axs[1].set_ylabel('Effective Gain (u/e)', fontsize=20)
axs[1].grid(True, alpha=0.3)
axs[1].legend(fontsize=13)
axs[1].set_yscale('log') 

# Plot 3: Phase Plane
axs[2].plot(xp, vp, 'r--', label='PID', alpha=0.6)
axs[2].plot(xl, vl, 'g-.', label='LQR', alpha=0.6)
axs[2].plot(xs, vs, 'b-', linewidth=2.5, label='Sign')
axs[2].set_title('Phase Plane (Step Response)', fontsize=20)
axs[2].set_xlabel('Position x (m)', fontsize=20)
axs[2].set_ylabel('Velocity v (m/s)', fontsize=20)
axs[2].grid(True, alpha=0.3)
axs[2].plot([-2], [0], 'ko', label='Start', markersize=8, zorder=10) 
axs[2].plot([0], [0], 'rx', markersize=12, label='Target', markeredgewidth=2, zorder=10)
axs[2].legend(loc='lower center', framealpha=0.9, fontsize=13)
axs[2].set_ylim(bottom=-0.75)

# Add (a), (b), (c) labels
labels = ['(a)', '(b)', '(c)']
for i, ax in enumerate(axs):
    ax.text(0.5, -0.25, labels[i], transform=ax.transAxes, fontsize=20, ha='center', va='top')

plt.tight_layout()
plt.savefig('square_analysis_plots.png', dpi=300)
print("Plots saved to square_analysis_plots.png")

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

# --- 参数定义 ---

# 1. SOAR Controller (Manual; uses smooth nonlinearity in P-term)
# u = -0.489 * smooth(e, 1.285) + 1.062 * v
# smooth(e, s) = s * tanh(e/s)
# u_p = 0.489 * 1.285 * tanh(e/1.285)
k_p_smooth = 0.489
s_smooth = 1.285
k_d_smooth = 1.062

# 2. PID Baseline (Calculated Equivalent Gains)
# K_eq = kp_xy * att_scale * kp_att
# D_eq = kd_xy * att_scale * kp_att (approx)
k_p_pid = 11.01 * 0.219 * 16.07  # ~38.75
k_d_pid = 4.88 * 0.219 * 16.07   # ~17.17

# 3. LQR Baseline
k_p_lqr = 9.76 * 0.256 * 15.34   # ~38.32
k_d_lqr = 5.29 * 0.256 * 15.34   # ~20.77

# 饱和限制 (Normalized Torque)
U_MAX = 1.0

# --- 定义控制函数 ---

def ctrl_smooth(e, v):
    # P term: 0.489 * s * tanh(e/s)
    # Note: manual formula is -0.489 * smooth... we use positive K for restoring force logic in plot
    u_p = k_p_smooth * s_smooth * np.tanh(e / s_smooth)
    u_d = k_d_smooth * v
    return np.clip(u_p + u_d, -U_MAX, U_MAX)

def ctrl_pid(e, v):
    u_p = k_p_pid * e
    u_d = k_d_pid * v
    return np.clip(u_p + u_d, -U_MAX, U_MAX)

def ctrl_lqr(e, v):
    u_p = k_p_lqr * e
    u_d = k_d_lqr * v
    return np.clip(u_p + u_d, -U_MAX, U_MAX)

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
dt = 0.005
steps = 2400  # 12s duration

def run_sim(ctrl_func):
    x = -2.0 # Start at -2m
    v = 0.0
    hist_x = [x]
    hist_v = [v]
    for _ in range(steps):
        # Error definition: e = Target - x = 0 - x = -x
        # Control force u tries to reduce error.
        # Dynamics: x'' = u (Simplified unit mass)
        # If x < 0, e > 0. u should be > 0 to push x up.
        e = -x 
        u = ctrl_func(e, -v) # D term acts on error_dot = -v
        
        accel = u 
        v += accel * dt
        x += v * dt
        hist_x.append(x)
        hist_v.append(v)
    return hist_x, hist_v

xs, vs = run_sim(ctrl_smooth)
xp, vp = run_sim(ctrl_pid)
xl, vl = run_sim(ctrl_lqr)

# --- Plotting ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: P-Curve
axs[0].plot(e_range, p_pid, 'r--', label=f'PID (K={k_p_pid:.1f})', alpha=0.6)
axs[0].plot(e_range, p_lqr, 'g-.', label=f'LQR (K={k_p_lqr:.1f})', alpha=0.6)
axs[0].plot(e_range, p_smooth, 'b-', linewidth=2.5, label=f'SOAR (K={k_p_smooth:.2f})')
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
axs[1].set_yscale('log') # Log scale to show the huge difference

# Plot 3: Phase Plane
axs[2].plot(xp, vp, 'r--', label='PID', alpha=0.6)
axs[2].plot(xl, vl, 'g-.', label='LQR', alpha=0.6)
axs[2].plot(xs, vs, 'b-', linewidth=2.5, label='SOAR')
axs[2].set_title('Phase Plane (Step Response)', fontsize=20)
axs[2].set_xlabel('Position x (m)', fontsize=20)
axs[2].set_ylabel('Velocity v (m/s)', fontsize=20)
axs[2].grid(True, alpha=0.3)
axs[2].plot([-2], [0], 'ko', label='Start', markersize=8, zorder=10) # Start
axs[2].plot([0], [0], 'rx', markersize=12, label='Target', markeredgewidth=2, zorder=10) # Target
# Move legend to lower center as requested
axs[2].legend(loc='lower center', framealpha=0.9, fontsize=13)
axs[2].set_ylim(bottom=-1)

# Add (a), (b), (c) labels
labels = ['(a)', '(b)', '(c)']
for i, ax in enumerate(axs):
    ax.text(0.5, -0.25, labels[i], transform=ax.transAxes, 
            fontsize=20, fontweight='normal', ha='center', va='center')

plt.tight_layout()
plt.savefig('nonlinear_analysis_plots_v2.png', dpi=300)
print("Plots saved to nonlinear_analysis_plots_v2.png")

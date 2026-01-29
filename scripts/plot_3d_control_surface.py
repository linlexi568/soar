#!/usr/bin/env python3
"""
Generate 3D Control Surface plots for PID, LQR, and Soar.
Visualizes the control law u = f(e, v).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

def smooth(val, s):
    return s * np.tanh(val / s)

def plot_surface_3d(ax, X, Y, Z, title, color_map=cm.coolwarm):
    surf = ax.plot_surface(X, Y, Z, cmap=color_map, linewidth=0, antialiased=False, alpha=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Error $e$ [m]', fontsize=8)
    ax.set_ylabel('Velocity $v$ [m/s]', fontsize=8)
    ax.set_zlabel('Control $u$', fontsize=8)
    return surf

def main():
    # Grid setup
    e_range = np.linspace(-2.0, 2.0, 50)
    v_range = np.linspace(-2.0, 2.0, 50)
    E, V = np.meshgrid(e_range, v_range)

    # Parameters
    # Square
    # PID: kp=8.826, kd=5.752 (Pos loop)
    pid_sq_kp = 8.826
    pid_sq_kd = 5.752
    # LQR: k_pos=3.547, k_vel=5.174
    lqr_sq_kp = 3.547
    lqr_sq_kd = 5.174
    # Soar: k_p=0.972, k_d=1.591, k_s=1.988
    pi_sq_kp = 0.972
    pi_sq_kd = 1.591
    pi_sq_ks = 1.988

    # Figure8
    # PID: kp=13.00, kd=3.138
    pid_f8_kp = 13.00
    pid_f8_kd = 3.138
    # LQR: Using Square params as baseline (common practice if not retuned)
    lqr_f8_kp = 3.547
    lqr_f8_kd = 5.174
    # Soar: k_p=0.489, k_d=1.062, k_s=1.285
    pi_f8_kp = 0.489
    pi_f8_kd = 1.062
    pi_f8_ks = 1.285

    # Create figure
    fig = plt.figure(figsize=(16, 8))
    
    # --- Subplot 1: Square ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # PID (Blue)
    Z_pid_sq = -pid_sq_kp * E - pid_sq_kd * V
    surf1 = ax1.plot_surface(E, V, Z_pid_sq, color='blue', alpha=0.3, linewidth=0, antialiased=False)
    # LQR (Green)
    Z_lqr_sq = -lqr_sq_kp * E - lqr_sq_kd * V
    surf2 = ax1.plot_surface(E, V, Z_lqr_sq, color='green', alpha=0.3, linewidth=0, antialiased=False)
    # Soar (Red)
    Z_pi_sq = -pi_sq_kp * smooth(E, pi_sq_ks) - pi_sq_kd * V
    surf3 = ax1.plot_surface(E, V, Z_pi_sq, color='red', alpha=0.6, linewidth=0, antialiased=False)

    ax1.set_title('Square Trajectory Control Surfaces\n(Blue: PID, Green: LQR, Red: Soar)', fontsize=12)
    ax1.set_xlabel('Error $e$ [m]')
    ax1.set_ylabel('Velocity $v$ [m/s]')
    ax1.set_zlabel('Control Output $u$')
    
    # Fake legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=4, label='PID'),
                       Line2D([0], [0], color='green', lw=4, label='LQR'),
                       Line2D([0], [0], color='red', lw=4, label='Soar')]
    ax1.legend(handles=legend_elements, loc='upper left')

    # --- Subplot 2: Figure8 ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # PID (Blue)
    Z_pid_f8 = -pid_f8_kp * E - pid_f8_kd * V
    ax2.plot_surface(E, V, Z_pid_f8, color='blue', alpha=0.3, linewidth=0, antialiased=False)
    # LQR (Green)
    Z_lqr_f8 = -lqr_f8_kp * E - lqr_f8_kd * V
    ax2.plot_surface(E, V, Z_lqr_f8, color='green', alpha=0.3, linewidth=0, antialiased=False)
    # Soar (Red)
    Z_pi_f8 = -pi_f8_kp * smooth(E, pi_f8_ks) - pi_f8_kd * V
    ax2.plot_surface(E, V, Z_pi_f8, color='red', alpha=0.6, linewidth=0, antialiased=False)

    ax2.set_title('Figure8 Trajectory Control Surfaces\n(Blue: PID, Green: LQR, Red: Soar)', fontsize=12)
    ax2.set_xlabel('Error $e$ [m]')
    ax2.set_ylabel('Velocity $v$ [m/s]')
    ax2.set_zlabel('Control Output $u$')
    ax2.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig('control_surface_3d.png', dpi=300)
    print("Saved 3D control surface plot to control_surface_3d.png")

if __name__ == "__main__":
    main()

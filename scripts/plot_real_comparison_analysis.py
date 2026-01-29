#!/usr/bin/env python3
"""
Generate comparison plots (like the theoretical ones) using REAL Isaac Gym simulation data.
Plots: 
(a) Control Output vs Error (P-Term)
(b) Effective Stiffness (Gain) 
(c) Phase Plane (Step Response)

Runs simulations directly and plots the results for both Square and Figure8 trajectories.
"""

import sys
from pathlib import Path
import os
import math

# Path setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / '01_soar') not in sys.path:
    sys.path.insert(0, str(ROOT / '01_soar'))

_ISAAC_GYM_PY = Path('/home/linlexi/桌面/soar/isaacgym/python')
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

# Ensure LD_LIBRARY_PATH is set for Isaac Gym
_ISAAC_BINDINGS = _ISAAC_GYM_PY / 'isaacgym' / '_bindings' / 'linux-x86_64'
if _ISAAC_BINDINGS.exists():
    os.environ['LD_LIBRARY_PATH'] = str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import isaacgym
except ImportError as e:
    print(f"Failed to import isaacgym: {e}")
    pass

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from scripts.baselines.tune_pid_lqr_isaac import IsaacPIDController, IsaacLQRController
from utilities.trajectory_presets import scg_position, get_scg_trajectory_config

# --- Font setup ---
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# --- Soar Controller (Smooth) ---
def smooth(val, s):
    return s * math.tanh(val / s)

def soar_control(sd, params):
    k_p, k_d, k_w, k_s = params
    
    # u_tx
    u_tx = ((-k_p * smooth(sd['pos_err_y'], k_s)) + (k_d * sd['vel_y'])) - (k_w * sd['ang_vel_x'])
    
    # u_ty (mirrored)
    u_ty = ((k_p * smooth(sd['pos_err_x'], k_s)) - (k_d * sd['vel_x'])) - (k_w * sd['ang_vel_y'])
    
    # u_tz (yaw) - simple P controller
    u_tz = -0.1 * sd['ang_vel_z']
    
    # u_fz (altitude) - simple PD
    u_fz = 1.0 + (10.0 * -sd['pos_err_z']) - (5.0 * sd['vel_z'])
    
    return np.array([u_fz, u_tx, u_ty, u_tz], dtype=np.float32)

def run_simulation(env, controller_type, trajectory_type, duration=4.0):
    print(f"Running simulation for: {controller_type} on {trajectory_type}")
    
    # Reset environment
    cfg = get_scg_trajectory_config(trajectory_type)
    initial_target = scg_position(trajectory_type, 0.0, params=cfg.params, center=cfg.center)
    initial_pos_tensor = torch.tensor([initial_target], device=env.device, dtype=torch.float32)
    
    env.reset(initial_pos=initial_pos_tensor)
    
    # Initialize controllers
    ctrl = None
    pi_params = None
    
    if controller_type == 'PID':
        if trajectory_type == 'square':
            # Params from plot_square_analysis.py (Hardcoded baseline)
            ctrl = IsaacPIDController(
                kp_xy=8.826, kd_xy=5.752, kp_att=9.566, att_scale=0.278,
                kp_z=14.0, kd_z=6.0,
                output_normalized=True
            )
        elif trajectory_type == 'figure8':
            # Params from cpid_figure8.json
            ctrl = IsaacPIDController(
                kp_xy=13.00, kd_xy=3.138, kp_att=16.05, att_scale=0.376,
                kp_z=12.0, kd_z=9.07,
                output_normalized=True
            )
            
    elif controller_type == 'LQR':
        # Using Square params for both as baseline
        # IsaacLQRController does not accept output_normalized in init
        ctrl = IsaacLQRController(
            k_pos=3.547, k_vel=5.174, k_att=12.68, k_omega=2.324, 
            k_yaw=0.803, k_yaw_rate=0.464, att_scale=0.200
        )
        
    elif controller_type == 'Soar':
        if trajectory_type == 'square':
            # Cost: 47.11
            pi_params = (0.972, 1.591, 0.660, 1.988)
        elif trajectory_type == 'figure8':
            # Cost: 81.78
            pi_params = (0.489, 1.062, 0.731, 1.285)

    # Simulation loop
    data = []
    dt = 0.02  # 50Hz
    steps = int(duration / dt)
    
    if ctrl is not None:
        ctrl.set_dt(dt)
    
    for i in range(steps):
        t = i * dt
        
        # Get state
        state = env.get_obs()
        # Extract first environment's state
        pos = state['position'][0]
        quat = state['orientation'][0]
        vel = state['velocity'][0]
        ang_vel = state['angular_velocity'][0]
        
        # Get target
        target_pos = scg_position(trajectory_type, t, params=cfg.params, center=cfg.center)
        
        # Calculate errors (simplified for plotting)
        pos_err = pos - target_pos
        
        # State dict for Soar
        sd = {
            'pos_err_x': pos_err[0], 'pos_err_y': pos_err[1], 'pos_err_z': pos_err[2],
            'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
            'ang_vel_x': ang_vel[0], 'ang_vel_y': ang_vel[1], 'ang_vel_z': ang_vel[2]
        }
        
        # Compute Control
        if controller_type == 'Soar':
            # Soar P-term: -k_p * smooth(e, k_s)
            # Note: The controller returns u_tx, u_ty. We focus on Y axis (u_tx controls Y error in body frame roughly, 
            # but let's stick to the definition in soar_control: u_tx = -kp*smooth(pos_err_y)...)
            # Actually u_tx is torque around X, which produces acceleration in Y.
            # Let's extract the exact P-term used for Y-axis control.
            # In soar_control: u_tx = ((-k_p * smooth(sd['pos_err_y'], k_s)) + ...)
            # So u_p_y = -k_p * smooth(sd['pos_err_y'], k_s)
            k_p, k_d, k_w, k_s = pi_params
            u_p_y = -k_p * smooth(sd['pos_err_y'], k_s)
            
            action = soar_control(sd, pi_params)
            
        else:
            # PID/LQR
            # compute(pos, vel, quat, omega, target_pos, target_vel=None)
            action = ctrl.compute(pos, vel, quat, ang_vel, target_pos, np.zeros(3))
            
            # Extract P-term for Y axis
            # PID: kp_xy * pos_err[1]
            # LQR: k_pos * pos_err[1]
            if controller_type == 'PID':
                u_p_y = ctrl.kp_xy * pos_err[1]
            elif controller_type == 'LQR':
                u_p_y = ctrl.k_pos * pos_err[1]

        # Step environment
        # Broadcast action to all envs if needed, but we only care about the first one
        # However, env.step expects [num_envs, 4]
        # Since we set num_envs=1, we just need [1, 4]
        action_tensor = torch.tensor([action], device=env.device, dtype=torch.float32)
        env.step(action_tensor)
        
        # Record data
        # We focus on Y-axis for analysis as per previous plots
        # u_y = action[2] # u_ty is usually pitch torque -> x motion? 
        # Wait, let's check coordinate system.
        # In Quadcopter: Pitch (Torque Y) -> X motion. Roll (Torque X) -> -Y motion.
        # Soar: u_tx = ... pos_err_y ... -> Roll Torque
        # PID/LQR: torque[0] is Roll.
        # So we should look at Torque X (action[1]) vs Error Y.
        
        u_total = action[1] 
        e_y = pos_err[1]
        
        data.append({
            'Time': t,
            'Error': e_y,
            'Control_Total': u_total,
            'Control_P': u_p_y,
            'Controller': controller_type
        })
        
    return pd.DataFrame(data)

def plot_analysis(df_all, trajectory_type):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'PID': 'blue', 'LQR': 'green', 'Soar': 'red'}
    styles = {'PID': '--', 'LQR': ':', 'Soar': '-'}
    
    # (a) Control Surface Cross-section (P-Term vs Error)
    # This visualizes the "Stiffness" of the controller
    ax = axes[0]
    for name, group in df_all.groupby('Controller'):
        group = group.sort_values('Error')
        ax.plot(group['Error'], group['Control_P'], label=name, 
                color=colors[name], linestyle=styles[name], linewidth=2, alpha=0.8)
    
    ax.set_title('(a) Control Stiffness (P-Term)', fontsize=16)
    ax.set_xlabel('Position Error $e$ [m]', fontsize=14)
    ax.set_ylabel('Control Output $u_p$ (P-Term)', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=12)

    # (b) Phase Plane (Error Rate vs Error)
    ax = axes[1]
    for name, group in df_all.groupby('Controller'):
        group = group.sort_values('Time')
        e = group['Error'].values
        t = group['Time'].values
        # Calculate e_dot
        e_dot = np.gradient(e, t)
        
        ax.plot(e, e_dot, label=name, color=colors[name], linestyle=styles[name], linewidth=2)
        
    ax.set_title('(b) Phase Plane Trajectory', fontsize=16)
    ax.set_xlabel('Position Error $e$ [m]', fontsize=14)
    ax.set_ylabel('Error Rate $\dot{e}$ [m/s]', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    # Add start/end markers
    for name, group in df_all.groupby('Controller'):
        group = group.sort_values('Time')
        e = group['Error'].values
        e_dot = np.gradient(e, group['Time'].values)
        # Start
        ax.plot(e[0], e_dot[0], 'o', color=colors[name], markersize=6, alpha=0.5)
        # End
        ax.plot(e[-1], e_dot[-1], 'x', color=colors[name], markersize=8, markeredgewidth=2)

    plt.tight_layout()
    filename = f'real_comparison_analysis_{trajectory_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filename}")

def main():
    print("Initializing Isaac Gym environment...")
    # Set num_envs=1 to avoid shape mismatch and speed up
    env = IsaacGymDroneEnv(headless=True, num_envs=1)
    
    for traj in ['square', 'figure8']:
        print(f"--- Processing {traj} ---")
        dfs = []
        for ctrl in ['PID', 'LQR', 'Soar']:
            df = run_simulation(env, ctrl, traj)
            dfs.append(df)
            
        df_all = pd.concat(dfs)
        plot_analysis(df_all, traj)
        
    print("Done.")

if __name__ == "__main__":
    main()

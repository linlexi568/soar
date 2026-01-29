#!/usr/bin/env python3
"""
Compare Math Expressions for Figure8 (Smooth) and Square (Sign) controllers using REAL Isaac Gym data.
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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from utilities.trajectory_presets import scg_position, get_scg_trajectory_config

# --- Controllers ---
def smooth(val, s):
    return s * math.tanh(val / s)

def sign(val):
    return math.copysign(1.0, val)

def soar_smooth_control(sd, params):
    # Figure8 Smooth Params: k_p=0.489, k_s=1.285, k_d=1.062, k_w=0.731
    k_p, k_s, k_d, k_w = params
    
    # u_tx
    u_tx = ((-k_p * smooth(sd['pos_err_y'], k_s)) + (k_d * sd['vel_y'])) - (k_w * sd['ang_vel_x'])
    
    # u_ty (mirrored)
    u_ty = ((k_p * smooth(sd['pos_err_x'], k_s)) - (k_d * sd['vel_x'])) - (k_w * sd['ang_vel_y'])
    
    # u_tz (yaw)
    u_tz = -0.1 * sd['ang_vel_z']
    
    # u_fz (altitude)
    u_fz = 1.0 + (10.0 * -sd['pos_err_z']) - (5.0 * sd['vel_z'])
    
    return np.array([u_fz, u_tx, u_ty, u_tz], dtype=np.float32)

def soar_sign_control(sd, params):
    # Square Sign Params: k_p=0.600, k_d=1.766, k_w=0.773
    k_p, k_d, k_w = params
    
    # u_tx
    u_tx = ((-k_p * sign(sd['pos_err_y'])) + (k_d * sd['vel_y'])) - (k_w * sd['ang_vel_x'])
    
    # u_ty (mirrored)
    u_ty = ((k_p * sign(sd['pos_err_x'])) - (k_d * sd['vel_x'])) - (k_w * sd['ang_vel_y'])
    
    # u_tz (yaw)
    u_tz = -0.1 * sd['ang_vel_z']
    
    # u_fz (altitude)
    u_fz = 1.0 + (10.0 * -sd['pos_err_z']) - (5.0 * sd['vel_z'])
    
    return np.array([u_fz, u_tx, u_ty, u_tz], dtype=np.float32)

def run_sweep(env, controller_type):
    print(f"Running sweep for: {controller_type}")
    data = []
    # Sweep range from -2.0 to 2.0
    errors = np.linspace(-2.0, 2.0, 100)
    
    # Params
    if controller_type == 'Smooth':
        params = (0.489, 1.285, 1.062, 0.731)
    elif controller_type == 'Sign':
        params = (0.600, 1.766, 0.773)
        
    for e_val in errors:
        # Reset env to specific position: y = e_val
        # Target is (0,0,1), so pos_err_y = e_val
        initial_pos = torch.tensor([[0.0, e_val, 1.0]], device=env.device, dtype=torch.float32)
        env.reset(initial_pos=initial_pos)
        
        # Get state
        state = env.get_obs()
        # state values are already numpy arrays if get_obs handles it, or tensors.
        # Based on error, they are numpy arrays.
        pos = state['position'][0]
        vel = state['velocity'][0]
        ang_vel = state['angular_velocity'][0]
        
        # Ensure they are numpy arrays (if they were tensors, this would fail, but they are arrays)
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        if isinstance(vel, torch.Tensor):
            vel = vel.cpu().numpy()
        if isinstance(ang_vel, torch.Tensor):
            ang_vel = ang_vel.cpu().numpy()
        
        # Target (0,0,1)
        target_pos = np.array([0.0, 0.0, 1.0])
        pos_err = pos - target_pos
        
        sd = {
            'pos_err_x': pos_err[0], 'pos_err_y': pos_err[1], 'pos_err_z': pos_err[2],
            'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
            'ang_vel_x': ang_vel[0], 'ang_vel_y': ang_vel[1], 'ang_vel_z': ang_vel[2]
        }
        
        # Compute Control P-term
        if controller_type == 'Smooth':
            u_p = -params[0] * smooth(sd['pos_err_y'], params[1])
        elif controller_type == 'Sign':
            u_p = -params[0] * sign(sd['pos_err_y'])
            
        data.append({'Error': sd['pos_err_y'], 'Control_P': u_p})
        
    return pd.DataFrame(data)

def run_recovery_simulation(env, controller_type, initial_error=2.0, duration=6.0):
    """Run a recovery simulation from a large initial error to a static target point."""
    print(f"Running recovery simulation for: {controller_type} (initial error={initial_error}m)")
    
    # Static target at origin (0, 0, 1)
    target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    # Start with initial error in Y axis
    initial_pos = np.array([0.0, initial_error, 1.0], dtype=np.float32)
    initial_pos_tensor = torch.tensor([initial_pos], device=env.device, dtype=torch.float32)
    
    env.reset(initial_pos=initial_pos_tensor)
    
    # Params
    if controller_type == 'Smooth':
        params = (0.489, 1.285, 1.062, 0.731)
    elif controller_type == 'Sign':
        params = (0.600, 1.766, 0.773)

    # Simulation loop
    data = []
    dt = 0.02  # 50Hz
    steps = int(duration / dt)
    
    for i in range(steps):
        t = i * dt
        
        # Get state
        state = env.get_obs()
        pos = state['position'][0]
        vel = state['velocity'][0]
        ang_vel = state['angular_velocity'][0]
        
        # Calculate errors (static target)
        pos_err = pos - target_pos
        
        # State dict
        sd = {
            'pos_err_x': pos_err[0], 'pos_err_y': pos_err[1], 'pos_err_z': pos_err[2],
            'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
            'ang_vel_x': ang_vel[0], 'ang_vel_y': ang_vel[1], 'ang_vel_z': ang_vel[2]
        }
        
        # Compute Control
        if controller_type == 'Smooth':
            action = soar_smooth_control(sd, params)
            u_p = -params[0] * smooth(sd['pos_err_y'], params[1])
        elif controller_type == 'Sign':
            action = soar_sign_control(sd, params)
            u_p = -params[0] * sign(sd['pos_err_y'])
            
        # Step environment
        action_tensor = torch.tensor([action], device=env.device, dtype=torch.float32)
        env.step(action_tensor)
        
        # Record data
        u_total = action[1]
        e_y = pos_err[1]
        v_y = vel[1]
        
        data.append({
            'Time': t,
            'Error': e_y,
            'Velocity': v_y,
            'Control_Total': u_total,
            'Control_P': u_p,
            'Controller': controller_type
        })
        
    return pd.DataFrame(data)

def run_simulation(env, controller_type, trajectory_type, duration=4.0, initial_offset=None):
    print(f"Running simulation for: {controller_type} on {trajectory_type}")
    
    # Reset environment
    cfg = get_scg_trajectory_config(trajectory_type)
    initial_target = scg_position(trajectory_type, 0.0, params=cfg.params, center=cfg.center)
    
    # Apply offset if provided (e.g., [0, 1.0, 0] to start 1m away in Y)
    if initial_offset is not None:
        initial_pos_val = initial_target + np.array(initial_offset, dtype=np.float32)
    else:
        initial_pos_val = initial_target
        
    initial_pos_tensor = torch.tensor([initial_pos_val], device=env.device, dtype=torch.float32)
    
    env.reset(initial_pos=initial_pos_tensor)
    
    # Params
    if controller_type == 'Smooth':
        # Figure8 Smooth: k_p=0.489, k_s=1.285, k_d=1.062, k_w=0.731
        params = (0.489, 1.285, 1.062, 0.731)
    elif controller_type == 'Sign':
        # Square Sign: k_p=0.600, k_d=1.766, k_w=0.773
        params = (0.600, 1.766, 0.773)

    # Simulation loop
    data = []
    dt = 0.02  # 50Hz
    steps = int(duration / dt)
    
    for i in range(steps):
        t = i * dt
        
        # Get state
        state = env.get_obs()
        pos = state['position'][0]
        vel = state['velocity'][0]
        ang_vel = state['angular_velocity'][0]
        
        # Get target
        target_pos = scg_position(trajectory_type, t, params=cfg.params, center=cfg.center)
        
        # Calculate errors
        pos_err = pos - target_pos
        
        # State dict
        sd = {
            'pos_err_x': pos_err[0], 'pos_err_y': pos_err[1], 'pos_err_z': pos_err[2],
            'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
            'ang_vel_x': ang_vel[0], 'ang_vel_y': ang_vel[1], 'ang_vel_z': ang_vel[2]
        }
        
        # Compute Control
        if controller_type == 'Smooth':
            action = soar_smooth_control(sd, params)
            # P-term for analysis: -k_p * smooth(e, k_s)
            u_p = -params[0] * smooth(sd['pos_err_y'], params[1])
        elif controller_type == 'Sign':
            action = soar_sign_control(sd, params)
            # P-term for analysis: -k_p * sign(e)
            u_p = -params[0] * sign(sd['pos_err_y'])
            
        # Step environment
        action_tensor = torch.tensor([action], device=env.device, dtype=torch.float32)
        env.step(action_tensor)
        
        # Record data (Focus on Y axis: u_tx controls Y motion via Roll)
        # u_tx is action[1]
        u_total = action[1]
        e_y = pos_err[1]
        v_y = vel[1]
        
        data.append({
            'Time': t,
            'Error': e_y,
            'Velocity': v_y,
            'Control_Total': u_total,
            'Control_P': u_p,
            'Controller': controller_type
        })
        
    return pd.DataFrame(data)

def plot_real_comparison(df_smooth, df_sign, df_smooth_sweep, df_sign_sweep):
    fig = plt.figure(figsize=(15, 6))

    # --- Plot 1: 2D Stiffness Comparison (Real Data Scatter) ---
    ax1 = fig.add_subplot(1, 3, 1)
    
    # Theoretical Lines for reference
    e_ref = np.linspace(-2, 2, 100)
    # Smooth: k_p=0.489, k_s=1.285
    u_smooth_ref = -0.489 * np.tanh(e_ref / 1.285) * 1.285
    # Sign: k_p=0.600
    u_sign_ref = -0.600 * np.sign(e_ref)
    
    ax1.plot(e_ref, u_smooth_ref, 'r--', alpha=0.4, linewidth=1, label='Theory (Smooth)')
    ax1.plot(e_ref, u_sign_ref, 'b--', alpha=0.4, linewidth=1, label='Theory (Sign)')

    # Sweep Data (The full curve from real simulation state)
    ax1.scatter(df_smooth_sweep['Error'], df_smooth_sweep['Control_P'], c='r', s=20, marker='x', alpha=0.6, label='Sweep Data (Smooth)')
    ax1.scatter(df_sign_sweep['Error'], df_sign_sweep['Control_P'], c='b', s=20, marker='x', alpha=0.6, label='Sweep Data (Sign)')

    # Trajectory Data (Where the drone actually flew)
    ax1.scatter(df_smooth['Error'], df_smooth['Control_P'], c='darkred', s=10, alpha=0.8, label='Flight Data (Smooth)')
    ax1.scatter(df_sign['Error'], df_sign['Control_P'], c='darkblue', s=10, alpha=0.8, label='Flight Data (Sign)')
    
    ax1.set_title('(a) Real Stiffness Response\n(Sweep vs Flight)', fontsize=12)
    ax1.set_xlabel('Error $e$ [m]')
    ax1.set_ylabel('P-Term Output $u_p$')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=8)

    # --- Plot 2: 3D Phase Portrait (Smooth) ---
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # Plot trajectory in e-v-u space
    ax2.plot(df_smooth['Error'], df_smooth['Velocity'], df_smooth['Control_Total'], 
             c='r', linewidth=2, alpha=0.8)
    ax2.set_title('(b) Figure8 (Smooth)\nReal Trajectory', fontsize=12)
    ax2.set_xlabel('Error $e$')
    ax2.set_ylabel('Velocity $v$')
    ax2.set_zlabel('Total Control $u$')

    # --- Plot 3: 3D Phase Portrait (Sign) ---
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(df_sign['Error'], df_sign['Velocity'], df_sign['Control_Total'], 
             c='b', linewidth=2, alpha=0.8)
    ax3.set_title('(c) Square (Sign)\nReal Trajectory', fontsize=12)
    ax3.set_xlabel('Error $e$')
    ax3.set_ylabel('Velocity $v$')
    ax3.set_zlabel('Total Control $u$')

    plt.tight_layout()
    plt.savefig('real_math_comparison.png', dpi=300)
    print("Saved real data comparison plot to real_math_comparison.png")

def main():
    print("Initializing Isaac Gym environment...")
    env = IsaacGymDroneEnv(headless=True, num_envs=1)
    
    # Run Sweeps (to get the full curve)
    df_smooth_sweep = run_sweep(env, 'Smooth')
    df_sign_sweep = run_sweep(env, 'Sign')

    # Run Recovery Simulations (from large initial error to static target)
    # This will show the full trajectory from error=2.0 to error=0
    df_smooth = run_recovery_simulation(env, 'Smooth', initial_error=2.0, duration=6.0)
    df_sign = run_recovery_simulation(env, 'Sign', initial_error=2.0, duration=6.0)
    
    plot_real_comparison(df_smooth, df_sign, df_smooth_sweep, df_sign_sweep)
    print("Done.")

if __name__ == "__main__":
    main()

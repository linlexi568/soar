#!/usr/bin/env python3
"""
Export comparison data for PID, LQR, and Soar (Smooth) on the Square trajectory.
"""

import sys
from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
# import torch # Moved down

# Path setup
ROOT = Path(__file__).resolve().parent
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
    # Do not pass, let it fail later or try to continue if it's just a warning
    pass

import torch # Import torch AFTER isaacgym

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from scripts.baselines.tune_pid_lqr_isaac import IsaacPIDController, IsaacLQRController
from utilities.trajectory_presets import scg_position, get_scg_trajectory_config

# --- Soar Controller (Smooth) ---
def smooth(val, s):
    return s * math.tanh(val / s)

def soar_control(sd):
    # Square parameters: k_p=0.972, k_d=1.591, k_w=0.660, k_s=1.988
    k_p, k_d, k_w, k_s = 0.972, 1.591, 0.660, 1.988
    
    # u_tx
    u_tx = ((-k_p * smooth(sd['pos_err_y'], k_s)) + (k_d * sd['vel_y'])) - (k_w * sd['ang_vel_x'])
    
    # u_ty (mirrored)
    u_ty = ((k_p * smooth(sd['pos_err_x'], k_s)) - (k_d * sd['vel_x'])) - (k_w * sd['ang_vel_y'])
    
    # u_tz (yaw) - simple P controller
    u_tz = -0.1 * sd['ang_vel_z']
    
    # u_fz (altitude) - simple PD
    u_fz = 1.0 + (10.0 * -sd['pos_err_z']) - (5.0 * sd['vel_z'])
    
    return np.array([u_fz, u_tx, u_ty, u_tz], dtype=np.float32)

def run_simulation(env, controller_type, duration=4.0):
    print(f"Running simulation for: {controller_type}")
    
    # Reset environment
    # Set initial position to trajectory start
    traj_type = 'square'
    cfg = get_scg_trajectory_config(traj_type)
    initial_target = scg_position(traj_type, 0.0, params=cfg.params, center=cfg.center)
    initial_pos_tensor = torch.tensor([initial_target], device=env.device, dtype=torch.float32)
    
    env.reset(initial_pos=initial_pos_tensor)
    
    # Trajectory config
    
    # Initialize controllers
    if controller_type == 'PID':
        # Params from plot_square_analysis.py
        ctrl = IsaacPIDController(
            kp_xy=8.826, kd_xy=5.752, kp_att=9.566, att_scale=0.278,
            kp_z=14.0, kd_z=6.0,  # Defaults
            output_normalized=True
        )
    elif controller_type == 'LQR':
        # Params from plot_square_analysis.py
        ctrl = IsaacLQRController(
            k_pos=7.638, k_vel=4.810, k_att=14.722, att_scale=0.276,
            torque_clip=0.5 # Match PID clip
        )
    
    data = []
    
    steps = int(duration * 48) # 48Hz
    
    for i in range(steps):
        t = i / 48.0
        
        # Get state
        # env.pos is (num_envs, 3)
        pos = env.pos[0].cpu().numpy()
        quat = env.quat[0].cpu().numpy()
        vel = env.lin_vel[0].cpu().numpy()
        ang_vel = env.ang_vel[0].cpu().numpy()
        
        # Get target
        target_pos = scg_position(traj_type, t, params=cfg.params, center=cfg.center)
        target_vel = np.zeros(3) # Simplified, assume 0 vel target for step-like behavior
        
        # Calculate errors
        pos_err = pos - target_pos
        
        # Compute control
        if controller_type == 'Soar':
            # Prepare state dict for DSL controller
            sd = {
                'pos_err_x': pos_err[0], 'pos_err_y': pos_err[1], 'pos_err_z': pos_err[2],
                'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
                'ang_vel_x': ang_vel[0], 'ang_vel_y': ang_vel[1], 'ang_vel_z': ang_vel[2]
            }
            action = soar_control(sd)
            
            # For plotting, we want the "effective" control output related to X/Y error
            # Soar outputs normalized torque directly.
            # u_tx controls rotation around X, which affects Y motion.
            # u_ty controls rotation around Y, which affects X motion.
            u_plot = action[2] # u_ty (affects X)
            
        elif controller_type in ['PID', 'LQR']:
            action = ctrl.compute(pos, vel, quat, ang_vel, target_pos, target_vel)
            # PID/LQR output: [fz, tx, ty, tz]
            # u_ty affects X motion
            u_plot = action[2]
            
        # Step environment
        # Action needs to be (num_envs, 4)
        action_tensor = torch.tensor([action], device=env.device, dtype=torch.float32)
        env.step(action_tensor)
        
        # Record data
        # We focus on X-axis behavior for the "Step Response" analysis
        # Square trajectory moves in Y first, then X. 
        # Let's record everything, but for phase plane we might look at a specific segment.
        
        data.append({
            'time': t,
            'method': controller_type,
            'pos_x': pos[0],
            'pos_y': pos[1],
            'pos_z': pos[2],
            'ref_x': target_pos[0],
            'ref_y': target_pos[1],
            'ref_z': target_pos[2],
            'err_x': pos_err[0],
            'err_y': pos_err[1],
            'vel_x': vel[0],
            'vel_y': vel[1],
            'u_plot': u_plot # Torque Y (controlling X)
        })
        
    return data

def main():
    env = IsaacGymDroneEnv(num_envs=1, device='cuda:0', headless=True, duration_sec=4.0)
    
    all_data = []
    
    for method in ['PID', 'LQR', 'Soar']:
        method_data = run_simulation(env, method)
        all_data.extend(method_data)
        
    df = pd.DataFrame(all_data)
    df.to_csv('comparison_data.csv', index=False)
    print("Data exported to comparison_data.csv")

if __name__ == "__main__":
    main()

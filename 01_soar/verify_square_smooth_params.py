#!/usr/bin/env python3
"""
Verify Square trajectory with new smooth parameters from BO tuning

Best params from BO:
  k_p: 1.6434643124552815
  k_s: 0.7861879055222127
  k_d: 1.632741597362691
  k_w: 0.6294447587302711

This script runs a full evaluation to verify the cost.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import isaacgym
except ImportError:
    pass

import torch
import numpy as np
import math

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from utils.reward_scg_exact import SCGExactRewardCalculator
from utilities.trajectory_presets import scg_position, get_scg_trajectory_config

# Best parameters from BO
K_P = 1.6434643124552815
K_S = 0.7861879055222127
K_D = 1.632741597362691
K_W = 0.6294447587302711

def smooth(val, s):
    """smooth operator: s * tanh(x/s)"""
    return s * math.tanh(val / s)

def verify_parameters(num_episodes=10):
    """Run multiple episodes to verify cost"""
    print("="*70)
    print("VERIFYING SQUARE TRAJECTORY WITH NEW SMOOTH PARAMETERS")
    print("="*70)
    print(f"Parameters:")
    print(f"  k_p = {K_P:.6f}")
    print(f"  k_s = {K_S:.6f}")
    print(f"  k_d = {K_D:.6f}")
    print(f"  k_w = {K_W:.6f}")
    print(f"\nRunning {num_episodes} episodes...")
    print("="*70)
    
    env = IsaacGymDroneEnv(
        num_envs=1,
        initial_height=1.0,
        spacing=5.0,
        headless=True,
        control_freq_hz=48,
        physics_freq_hz=240,
        device='cuda:0'
    )
    
    scg_calc = SCGExactRewardCalculator(env.num_envs, env.device)
    
    cfg = get_scg_trajectory_config('square')
    
    def get_target(t):
        return scg_position('square', t, params=cfg.params, center=cfg.center)
    
    dt = 1.0 / 48.0
    period = cfg.params.get('period', 5.0)
    num_steps = int(period / dt)
    
    all_costs = []
    
    for ep in range(num_episodes):
        # Reset
        initial_target = get_target(0.0)
        initial_pos = torch.from_numpy(initial_target).unsqueeze(0).to(env.device, dtype=torch.float32)
        env.reset(initial_pos=initial_pos)
        scg_calc.reset()
        
        crashed = False
        
        for step in range(num_steps):
            t = step * dt
            target = get_target(t)
            target_tensor = torch.from_numpy(target).to(env.device, dtype=torch.float32)
            
            # Get state
            pos = env.pos[0]
            vel = env.lin_vel[0]
            quat = env.quat[0]
            omega = env.ang_vel[0]
            
            # Check crash
            if pos[2] < 0.1 or torch.any(torch.abs(pos) > 5.0):
                crashed = True
                break
            
            pos_err = target_tensor - pos
            
            # Compute yaw
            qx, qy, qz, qw = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            err_p_yaw = 0.0 - yaw
            while err_p_yaw > np.pi: err_p_yaw -= 2*np.pi
            while err_p_yaw < -np.pi: err_p_yaw += 2*np.pi
            
            # Control with new parameters
            u_tx = -K_P * smooth(pos_err[1].item(), K_S) + K_D * vel[1].item() - K_W * omega[0].item()
            u_ty = K_P * smooth(pos_err[0].item(), K_S) - K_D * vel[0].item() - K_W * omega[1].item()
            u_tz = 4.0 * err_p_yaw - 0.8 * omega[2].item()
            u_fz = 1.0 * pos_err[2].item() - 0.5 * vel[2].item() + 0.65
            
            # Clip
            u_tx = max(-1.0, min(1.0, u_tx))
            u_ty = max(-1.0, min(1.0, u_ty))
            u_tz = max(-1.0, min(1.0, u_tz))
            u_fz = max(0.0, min(1.5, u_fz))
            
            # Compute reward
            scg_action = torch.tensor([[u_fz, u_tx, u_ty, u_tz]], device=env.device)
            scg_calc.compute_step(env.pos, env.lin_vel, env.quat, env.ang_vel, target_tensor, scg_action)
            
            # Step
            actions = torch.tensor([[0.0, 0.0, u_fz, u_tx, u_ty, u_tz]], device=env.device)
            env.step(actions)
        
        if crashed:
            cost = 2000.0
            print(f"  Episode {ep+1}: CRASHED")
        else:
            components = scg_calc.get_components()
            cost = components["total_cost"][0].item()
            print(f"  Episode {ep+1}: Cost = {cost:.2f}")
        
        all_costs.append(cost)
    
    env.close()
    
    # Statistics
    mean_cost = np.mean(all_costs)
    std_cost = np.std(all_costs)
    min_cost = np.min(all_costs)
    max_cost = np.max(all_costs)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"Mean Cost:   {mean_cost:.3f} ± {std_cost:.3f}")
    print(f"Min Cost:    {min_cost:.3f}")
    print(f"Max Cost:    {max_cost:.3f}")
    print(f"Episodes:    {num_episodes}")
    print("="*70)
    
    # Compare with original sign-based
    print("\nComparison:")
    print(f"  Original (sign):  ~52.99")
    print(f"  New (smooth):     {mean_cost:.2f}")
    if mean_cost < 60:
        print("  ✅ Comparable or better performance!")
    else:
        print("  ⚠️  Performance decreased, may need further tuning")
    
    return mean_cost

if __name__ == "__main__":
    verify_parameters(num_episodes=10)

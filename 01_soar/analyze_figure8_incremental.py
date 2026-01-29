import os
import sys
import time
import numpy as np
# torch import moved down
import copy
import random
import math

# Add path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    import isaacgym
except ImportError:
    pass

import torch

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from utils.reward_scg_exact import SCGExactRewardCalculator
from core.dsl import ProgramNode, BinaryOpNode, UnaryOpNode, TerminalNode, ConstantNode

def figure8_target_xy(t):
    # Figure8 trajectory: Period=5.0, Scale=0.8
    period = 5.0
    omega = 2.0 * np.pi / period
    A = 0.8
    B = 0.8
    
    x = A * np.sin(omega * t)
    y = B * np.sin(omega * t) * np.cos(omega * t)
    z = 1.0
    return np.array([x, y, z])

class DSLController:
    def __init__(self, program_tx, program_ty, program_tz, program_fz):
        self.program_tx = program_tx
        self.program_ty = program_ty
        self.program_tz = program_tz
        self.program_fz = program_fz
    
    def get_action(self, state_dict):
        u_tx = self.program_tx.evaluate(state_dict)
        u_ty = self.program_ty.evaluate(state_dict)
        u_tz = self.program_tz.evaluate(state_dict)
        u_fz = self.program_fz.evaluate(state_dict)
        # Convert to 6D [fx, fy, fz, tx, ty, tz]
        # Assuming u_fz is thrust, u_tx/ty/tz are torques
        return np.array([0.0, 0.0, u_fz, u_tx, u_ty, u_tz])

# --- Structure 1: Damping Only (Inner Loop) ---
# u_tx = -k_w * ang_vel_x
def create_damping_only_programs(k_w, k_z_p=2.0, k_z_d=1.0):
    # u_tx = -k_w * ang_vel_x
    prog_tx = BinaryOpNode('*', ConstantNode(-k_w), TerminalNode('ang_vel_x'))

    # u_ty = -k_w * ang_vel_y
    prog_ty = BinaryOpNode('*', ConstantNode(-k_w), TerminalNode('ang_vel_y'))
    
    # u_tz (Yaw damping + holding)
    prog_tz = BinaryOpNode('-', 
                BinaryOpNode('*', ConstantNode(4.0), TerminalNode('err_p_yaw')),
                BinaryOpNode('*', ConstantNode(0.8), TerminalNode('ang_vel_z'))
              )
    # u_fz (Height hold)
    prog_fz = BinaryOpNode('+',
                BinaryOpNode('-',
                    BinaryOpNode('*', ConstantNode(k_z_p), TerminalNode('pos_err_z')),
                    BinaryOpNode('*', ConstantNode(k_z_d), TerminalNode('vel_z'))
                ),
                ConstantNode(0.65)
              )
    return prog_tx, prog_ty, prog_tz, prog_fz

# --- Structure 2: Damping + Velocity (Inner + Mid Loop) ---
# u_tx = -k_w * ang_vel_x + k_d * vel_y
def create_damping_velocity_programs(k_w, k_d, k_z_p=2.0, k_z_d=1.0):
    # u_tx
    term1 = BinaryOpNode('*', ConstantNode(-k_w), TerminalNode('ang_vel_x'))
    term2 = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    prog_tx = BinaryOpNode('+', term1, term2)

    # u_ty
    term1 = BinaryOpNode('*', ConstantNode(-k_w), TerminalNode('ang_vel_y'))
    term2 = BinaryOpNode('*', ConstantNode(-k_d), TerminalNode('vel_x')) # Note sign
    prog_ty = BinaryOpNode('+', term1, term2)
    
    # u_tz
    prog_tz = BinaryOpNode('-', 
                BinaryOpNode('*', ConstantNode(4.0), TerminalNode('err_p_yaw')),
                BinaryOpNode('*', ConstantNode(0.8), TerminalNode('ang_vel_z'))
              )
    # u_fz
    prog_fz = BinaryOpNode('+',
                BinaryOpNode('-',
                    BinaryOpNode('*', ConstantNode(k_z_p), TerminalNode('pos_err_z')),
                    BinaryOpNode('*', ConstantNode(k_z_d), TerminalNode('vel_z'))
                ),
                ConstantNode(0.65)
              )
    return prog_tx, prog_ty, prog_tz, prog_fz

# --- Structure 3: Full (P + D + Damping) ---
# u_tx = ((-k_p * smooth(pos_err_y, s=k_s)) + (k_d * vel_y)) - (k_w * ang_vel_x)
def create_full_programs(k_p, k_s, k_d, k_w, k_z_p=2.0, k_z_d=1.0):
    # u_tx
    smooth_err_y = UnaryOpNode('smooth', TerminalNode('pos_err_y'), params={'s': ConstantNode(k_s)})
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), smooth_err_y)
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    
    # (term1 + term2) - term3
    prog_tx = BinaryOpNode('-', 
                BinaryOpNode('+', term1_tx, term2_tx), 
                term3_tx
              )

    # u_ty
    smooth_err_x = UnaryOpNode('smooth', TerminalNode('pos_err_x'), params={'s': ConstantNode(k_s)})
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), smooth_err_x)
    term2_ty = BinaryOpNode('*', ConstantNode(-k_d), TerminalNode('vel_x'))
    term3_ty = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_y'))
    
    prog_ty = BinaryOpNode('-', 
                BinaryOpNode('+', term1_ty, term2_ty), 
                term3_ty
              )
    
    # u_tz
    prog_tz = BinaryOpNode('-', 
                BinaryOpNode('*', ConstantNode(4.0), TerminalNode('err_p_yaw')),
                BinaryOpNode('*', ConstantNode(0.8), TerminalNode('ang_vel_z'))
              )
              
    # u_fz
    prog_fz = BinaryOpNode('+',
                BinaryOpNode('-',
                    BinaryOpNode('*', ConstantNode(k_z_p), TerminalNode('pos_err_z')),
                    BinaryOpNode('*', ConstantNode(k_z_d), TerminalNode('vel_z'))
                ),
                ConstantNode(0.65)
              )
    return prog_tx, prog_ty, prog_tz, prog_fz

def evaluate_params(env, scg_calc, create_fn, params, num_steps=240):
    prog_tx, prog_ty, prog_tz, prog_fz = create_fn(*params)
    controller = DSLController(prog_tx, prog_ty, prog_tz, prog_fz)
    
    initial_pos = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
    env.reset(initial_pos=initial_pos)
    scg_calc.reset()
    
    dt = 1.0 / 48.0
    total_cost = 0.0
    failed = False
    
    for step in range(num_steps):
        t = step * dt
        target = figure8_target_xy(t)
        target_tensor = torch.tensor(target, device=env.device, dtype=torch.float32)
        
        pos = env.pos[0]
        vel = env.lin_vel[0]
        quat = env.quat[0]
        omega = env.ang_vel[0]
        
        # Check divergence
        if torch.norm(pos) > 10.0 or torch.isnan(pos).any():
            failed = True
            break
            
        pos_err = target_tensor - pos
        
        qx, qy, qz, qw = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        err_p_yaw = 0.0 - yaw
        while err_p_yaw > np.pi: err_p_yaw -= 2*np.pi
        while err_p_yaw < -np.pi: err_p_yaw += 2*np.pi

        state_dict = {
            'pos_err_x': pos_err[0].item(),
            'pos_err_y': pos_err[1].item(),
            'pos_err_z': pos_err[2].item(),
            'vel_x': vel[0].item(),
            'vel_y': vel[1].item(),
            'vel_z': vel[2].item(),
            'ang_vel_x': omega[0].item(),
            'ang_vel_y': omega[1].item(),
            'ang_vel_z': omega[2].item(),
            'err_p_yaw': err_p_yaw
        }
        
        action = controller.get_action(state_dict)
        action_tensor = torch.tensor(action, device=env.device, dtype=torch.float32).unsqueeze(0)
        
        env.step(action_tensor)
        
        # Reward calculation
        # obs_dict = { ... }  <-- Removed
        # ref_dict = { ... }  <-- Removed
        
        # Use compute_step directly
        reward = scg_calc.compute_step(
            pos=pos.unsqueeze(0),
            vel=vel.unsqueeze(0),
            quat=quat.unsqueeze(0),
            omega=omega.unsqueeze(0),
            target_pos=target_tensor.unsqueeze(0),
            action=action_tensor,
            target_vel=torch.zeros_like(vel).unsqueeze(0)
        )
        
        # Cost = -Reward
        total_cost += -reward.item()
        
    if failed:
        return 10000.0
    return total_cost

def optimize_damping_only(env, scg_calc):
    print("\n=== Optimizing Damping Only (Inner Loop) ===")
    print("Structure: u_tx = -k_w * ang_vel_x")
    
    best_cost = float('inf')
    best_params = None
    
    # Random Search
    for i in range(50):
        k_w = random.uniform(0.1, 2.0)
        
        params = (k_w,)
        cost = evaluate_params(env, scg_calc, create_damping_only_programs, params)
        
        if cost < best_cost:
            best_cost = cost
            best_params = params
            print(f"Iter {i}: New Best Cost: {best_cost:.2f} | k_w={k_w:.3f}")
            
    return best_cost, best_params

def optimize_damping_velocity(env, scg_calc):
    print("\n=== Optimizing Damping + Velocity (Inner + Mid Loop) ===")
    print("Structure: u_tx = -k_w * ang_vel_x + k_d * vel_y")
    
    best_cost = float('inf')
    best_params = None
    
    # Random Search
    for i in range(50):
        k_w = random.uniform(0.1, 2.0)
        k_d = random.uniform(0.1, 2.0)
        
        params = (k_w, k_d)
        cost = evaluate_params(env, scg_calc, create_damping_velocity_programs, params)
        
        if cost < best_cost:
            best_cost = cost
            best_params = params
            print(f"Iter {i}: New Best Cost: {best_cost:.2f} | k_w={k_w:.3f}, k_d={k_d:.3f}")
            
    return best_cost, best_params

def optimize_full_structure(env, scg_calc):
    print("\n=== Optimizing Full Structure (P + D + Damping) ===")
    print("Structure: u_tx = ((-k_p * smooth(pos_err_y)) + (k_d * vel_y)) - (k_w * ang_vel_x)")
    
    best_cost = float('inf')
    best_params = None
    
    # Random Search
    for i in range(50):
        k_p = random.uniform(0.1, 3.0)
        k_s = random.uniform(0.1, 2.0)
        k_d = random.uniform(0.1, 2.0)
        k_w = random.uniform(0.1, 1.5) # Damping gain
        
        params = (k_p, k_s, k_d, k_w)
        cost = evaluate_params(env, scg_calc, create_full_programs, params)
        
        if cost < best_cost:
            best_cost = cost
            best_params = params
            print(f"Iter {i}: New Best Cost: {best_cost:.2f} | k_p={k_p:.3f}, k_s={k_s:.3f}, k_d={k_d:.3f}, k_w={k_w:.3f}")
            
    # Local Refinement
    print("--- Local Refinement ---")
    if best_params is None:
        print("Failed to find any stable parameters.")
        return 10000.0, (0,0,0,0)

    for i in range(30):
        k_p = best_params[0] + random.gauss(0, 0.2)
        k_s = best_params[1] + random.gauss(0, 0.1)
        k_d = best_params[2] + random.gauss(0, 0.1)
        k_w = best_params[3] + random.gauss(0, 0.1)
        
        k_p = max(0.1, k_p)
        k_s = max(0.05, k_s)
        k_d = max(0.0, k_d)
        k_w = max(0.0, k_w)
        
        params = (k_p, k_s, k_d, k_w)
        cost = evaluate_params(env, scg_calc, create_full_programs, params)
        
        if cost < best_cost:
            best_cost = cost
            best_params = params
            print(f"Refine {i}: New Best Cost: {best_cost:.2f} | k_p={k_p:.3f}, k_s={k_s:.3f}, k_d={k_d:.3f}, k_w={k_w:.3f}")
            
    return best_cost, best_params

def main():
    env = IsaacGymDroneEnv(num_envs=1, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    scg_calc = SCGExactRewardCalculator(env.num_envs, env.device)
    
    # 1. Test Damping Only
    cost_damp, params_damp = optimize_damping_only(env, scg_calc)
    
    # 2. Test Damping + Velocity
    cost_dv, params_dv = optimize_damping_velocity(env, scg_calc)
    
    # 3. Test Full
    cost_full, params_full = optimize_full_structure(env, scg_calc)
    
    print("\n\n================ SUMMARY ================")
    print(f"1. Damping Only:     Cost = {cost_damp:.2f}")
    print(f"   Params: k_w={params_damp[0]:.3f}")
    print(f"2. Damping + Vel:    Cost = {cost_dv:.2f}")
    print(f"   Params: k_w={params_dv[0]:.3f}, k_d={params_dv[1]:.3f}")
    print(f"3. Full Structure:   Cost = {cost_full:.2f}")
    print(f"   Params: k_p={params_full[0]:.3f}, k_s={params_full[1]:.3f}, k_d={params_full[2]:.3f}, k_w={params_full[3]:.3f}")
    print(f"   (Reference Cost: 83.06)")
    print("=========================================")

    # 4. Verify Manual Params
    print("\n=== Verifying New Best Params ===")
    # k_p=0.489, k_s=1.285, k_d=1.062, k_w=0.731
    manual_params = (0.489, 1.285, 1.062, 0.731)
    cost_manual = evaluate_params(env, scg_calc, create_full_programs, manual_params)
    print(f"New Best Params Cost: {cost_manual:.4f}")

if __name__ == "__main__":
    main()

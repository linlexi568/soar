
import os
import sys
import time
import numpy as np
# import torch  # Moved down

# Add path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    import isaacgym
except ImportError:
    pass

import torch
import copy
import random

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from utils.reward_scg_exact import SCGExactRewardCalculator
from core.dsl import ProgramNode, BinaryOpNode, UnaryOpNode, TerminalNode, ConstantNode

def figure8_target_xy(t):
    A, B, T = 1.0, 1.0, 5.0
    omega = 2 * np.pi / T
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
        return np.array([u_fz, u_tx, u_ty, u_tz])

def create_programs(k_p, k_d, k_w):
    # u_tx = -k_p * pos_err_y + k_d * vel_y - k_w * ang_vel_x
    # Constructing: ((-k_p * pos_err_y) + (k_d * vel_y)) - (k_w * ang_vel_x)
    
    # Term 1: -k_p * pos_err_y
    # Note: DSL doesn't have unary '-', so use 0 - ... or -1 * ...
    # Or just ConstantNode(-k_p)
    
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), TerminalNode('pos_err_y'))
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    
    prog_tx = BinaryOpNode('-', 
                BinaryOpNode('+', term1_tx, term2_tx), 
                term3_tx
              )

    # u_ty = k_p * pos_err_x - k_d * vel_x - k_w * ang_vel_y
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), TerminalNode('pos_err_x'))
    term2_ty = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_x'))
    term3_ty = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_y'))
    
    prog_ty = BinaryOpNode('-', 
                BinaryOpNode('-', term1_ty, term2_ty), 
                term3_ty
              )
              
    # u_tz = 2.0 * (-yaw) - 0.5 * ang_vel_z
    # Simplified: -2.0 * yaw - 0.5 * ang_vel_z
    # Note: 'yaw' might not be in state_dict directly as 'yaw', usually 'err_p_yaw' is available or we compute it.
    # IsaacGymDroneEnv state_dict usually has 'err_p_yaw' if we set it up, or we can use 'yaw' if we add it.
    # Let's stick to standard variables. 'err_p_yaw' is standard for tracking.
    # u_tz = 4.0 * err_p_yaw - 0.8 * ang_vel_z (from previous successful configs)
    prog_tz = BinaryOpNode('-', 
                BinaryOpNode('*', ConstantNode(4.0), TerminalNode('err_p_yaw')),
                BinaryOpNode('*', ConstantNode(0.8), TerminalNode('ang_vel_z'))
              )
              
    # u_fz = 0.5 * pos_err_z - 0.2 * vel_z + 0.65
    prog_fz = BinaryOpNode('+',
                BinaryOpNode('-',
                    BinaryOpNode('*', ConstantNode(0.5), TerminalNode('pos_err_z')),
                    BinaryOpNode('*', ConstantNode(0.2), TerminalNode('vel_z'))
                ),
                ConstantNode(0.65)
              )
              
    return prog_tx, prog_ty, prog_tz, prog_fz

def evaluate_params(env, scg_calc, k_p, k_d, k_w, num_steps=240):
    prog_tx, prog_ty, prog_tz, prog_fz = create_programs(k_p, k_d, k_w)
    controller = DSLController(prog_tx, prog_ty, prog_tz, prog_fz)
    
    initial_pos = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
    env.reset(initial_pos=initial_pos)
    scg_calc.reset()
    
    dt = 1.0 / 48.0
    
    total_reward = 0.0
    
    for step in range(num_steps):
        t = step * dt
        target = figure8_target_xy(t)
        target_tensor = torch.tensor(target, device=env.device, dtype=torch.float32)
        
        # Construct state dict for DSL
        # We need to manually populate the state dict as the DSL expects
        pos = env.pos[0]
        vel = env.lin_vel[0]
        quat = env.quat[0]
        omega = env.ang_vel[0]
        
        # Calculate derived variables
        pos_err = target_tensor - pos
        
        # Euler angles
        qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp).item()
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0)).item()
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp).item()
        
        # Yaw error (assuming target yaw = 0 for simplicity, or tangent to path)
        # For Figure8, we usually just keep yaw=0
        err_p_yaw = 0.0 - yaw
        # Normalize angle
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
            'err_p_yaw': err_p_yaw,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }
        
        # Get action from DSL
        action_vals = controller.get_action(state_dict)
        u_fz, u_tx, u_ty, u_tz = action_vals
        
        # Clamp
        u_tx = max(-0.4, min(0.4, u_tx))
        u_ty = max(-0.4, min(0.4, u_ty))
        u_tz = max(-0.5, min(0.5, u_tz))
        u_fz = max(0.0, min(1.3, u_fz))
        
        actions = torch.tensor([[0.0, 0.0, u_fz, u_tx, u_ty, u_tz]], device=env.device)
        
        # SCG Reward
        scg_action = torch.tensor([[u_fz, u_tx, u_ty, u_tz]], device=env.device)
        scg_calc.compute_step(env.pos, env.lin_vel, env.quat, env.ang_vel, target_tensor, scg_action)
        
        env.step(actions)
        
    components = scg_calc.get_components()
    return components["total_cost"][0].item()

def main():
    print("初始化 Isaac Gym 环境...")
    env = IsaacGymDroneEnv(num_envs=1, device='cuda:0', headless=True, duration_sec=5.0)
    scg_calc = SCGExactRewardCalculator(num_envs=1, device='cuda:0')
    
    print("开始 BO (Bayesian Optimization) 风格的参数搜索...")
    print("目标: 最小化 Total Cost (State Cost + Action Cost)")
    print("-" * 60)
    
    # Search space
    # k_p: [0.1, 1.0]
    # k_d: [0.0, 0.5]
    # k_w: [0.0, 0.5]
    
    best_cost = float('inf')
    best_params = None
    
    # 1. Random Search (Exploration)
    num_random = 20
    print(f"阶段 1: 随机搜索 ({num_random} 次迭代)")
    
    candidates = []
    
    for i in range(num_random):
        k_p = random.uniform(0.1, 0.8)
        k_d = random.uniform(0.05, 0.4)
        k_w = random.uniform(0.05, 0.3)
        
        cost = evaluate_params(env, scg_calc, k_p, k_d, k_w)
        print(f"Iter {i+1:02d}: k_p={k_p:.3f}, k_d={k_d:.3f}, k_w={k_w:.3f} => Cost={cost:.2f}")
        
        candidates.append((cost, (k_p, k_d, k_w)))
        if cost < best_cost:
            best_cost = cost
            best_params = (k_p, k_d, k_w)
            
    # 2. Local Refinement (Exploitation)
    print("-" * 60)
    print(f"阶段 2: 局部精细化搜索 (基于最佳参数)")
    print(f"当前最佳: {best_params} (Cost: {best_cost:.2f})")
    
    center_k_p, center_k_d, center_k_w = best_params
    num_local = 10
    sigma = 0.05
    
    for i in range(num_local):
        k_p = center_k_p + random.gauss(0, sigma)
        k_d = center_k_d + random.gauss(0, sigma)
        k_w = center_k_w + random.gauss(0, sigma)
        
        # Bounds
        k_p = max(0.1, min(1.0, k_p))
        k_d = max(0.0, min(0.5, k_d))
        k_w = max(0.0, min(0.5, k_w))
        
        cost = evaluate_params(env, scg_calc, k_p, k_d, k_w)
        print(f"Refine {i+1:02d}: k_p={k_p:.3f}, k_d={k_d:.3f}, k_w={k_w:.3f} => Cost={cost:.2f}")
        
        if cost < best_cost:
            best_cost = cost
            best_params = (k_p, k_d, k_w)
            
    print("=" * 60)
    print("搜索完成")
    print(f"最佳参数: k_p={best_params[0]:.4f}, k_d={best_params[1]:.4f}, k_w={best_params[2]:.4f}")
    print(f"最佳 Cost: {best_cost:.4f}")
    print("=" * 60)
    
    # Print the final DSL expression
    prog_tx, prog_ty, _, _ = create_programs(*best_params)
    print("生成的 DSL 表达式 (u_tx):")
    print(prog_tx)
    print("生成的 DSL 表达式 (u_ty):")
    print(prog_ty)
    
    env.close()

if __name__ == "__main__":
    main()

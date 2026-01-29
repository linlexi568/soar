
import os
import sys
import time
import numpy as np
# import torch

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

def circle_target_xy(t):
    # Circle trajectory: R=1.0, T=5.0
    R = 1.0
    period = 5.0
    omega = 2.0 * np.pi / period
    
    # x = R * cos(wt)
    # y = R * sin(wt)
    # z = 1.0
    
    x = R * np.cos(omega * t)
    y = R * np.sin(omega * t)
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

def create_nonlinear_programs(k_p, k_d, k_w, k_s, k_i, k_z_p, k_z_d):
    # Nonlinear Control (4 terms with restricted Integral):
    # u_tx = -k_p * smooth(pos_err_y, s=k_s) + k_d * vel_y - k_w * ang_vel_x - k_i * int_err_y
    
    smooth_err_y = UnaryOpNode('smooth', TerminalNode('pos_err_y'), params={'s': ConstantNode(k_s)})
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), smooth_err_y)
    
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    term4_tx = BinaryOpNode('*', ConstantNode(k_i), TerminalNode('pos_err_integ_y'))
    
    # ((term1 + term2) - term3) - term4
    prog_tx = BinaryOpNode('-',
                BinaryOpNode('-', 
                    BinaryOpNode('+', term1_tx, term2_tx), 
                    term3_tx
                ),
                term4_tx
              )

    # u_ty = k_p * smooth(pos_err_x) - k_d * vel_x - k_w * ang_vel_y + k_i * int_err_x
    smooth_err_x = UnaryOpNode('smooth', TerminalNode('pos_err_x'), params={'s': ConstantNode(k_s)})
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), smooth_err_x)
    
    term2_ty = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_x'))
    term3_ty = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_y'))
    term4_ty = BinaryOpNode('*', ConstantNode(k_i), TerminalNode('pos_err_integ_x'))
    
    prog_ty = BinaryOpNode('+', 
                BinaryOpNode('-', 
                    BinaryOpNode('-', term1_ty, term2_ty), 
                    term3_ty
                ),
                term4_ty
              )
              
    # u_tz = 4.0 * err_p_yaw - 0.8 * ang_vel_z
    prog_tz = BinaryOpNode('-', 
                BinaryOpNode('*', ConstantNode(4.0), TerminalNode('err_p_yaw')),
                BinaryOpNode('*', ConstantNode(0.8), TerminalNode('ang_vel_z'))
              )
              
    # u_fz = k_z_p * pos_err_z - k_z_d * vel_z + 0.65
    prog_fz = BinaryOpNode('+',
                BinaryOpNode('-',
                    BinaryOpNode('*', ConstantNode(k_z_p), TerminalNode('pos_err_z')),
                    BinaryOpNode('*', ConstantNode(k_z_d), TerminalNode('vel_z'))
                ),
                ConstantNode(0.65)
              )
              
    return prog_tx, prog_ty, prog_tz, prog_fz

def evaluate_params(env, scg_calc, k_p, k_d, k_w, k_s, k_i, k_z_p, k_z_d, num_steps=240):
    prog_tx, prog_ty, prog_tz, prog_fz = create_nonlinear_programs(k_p, k_d, k_w, k_s, k_i, k_z_p, k_z_d)
    controller = DSLController(prog_tx, prog_ty, prog_tz, prog_fz)
    
    # Circle starts at (1, 0, 1)
    initial_pos = torch.tensor([[1.0, 0.0, 1.0]], device=env.device)
    env.reset(initial_pos=initial_pos)
    scg_calc.reset()
    
    dt = 1.0 / 48.0
    pos_err_integ = torch.zeros(3, device=env.device)
    
    for step in range(num_steps):
        t = step * dt
        target = circle_target_xy(t)
        target_tensor = torch.tensor(target, device=env.device, dtype=torch.float32)
        
        pos = env.pos[0]
        vel = env.lin_vel[0]
        quat = env.quat[0]
        omega = env.ang_vel[0]
        
        pos_err = target_tensor - pos
        pos_err_integ += pos_err * dt
        # Anti-windup
        pos_err_integ = torch.clamp(pos_err_integ, -2.0, 2.0)
        
        qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp).item()
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0)).item()
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp).item()
        
        err_p_yaw = 0.0 - yaw
        while err_p_yaw > np.pi: err_p_yaw -= 2*np.pi
        while err_p_yaw < -np.pi: err_p_yaw += 2*np.pi
        
        state_dict = {
            'pos_err_x': pos_err[0].item(),
            'pos_err_y': pos_err[1].item(),
            'pos_err_z': pos_err[2].item(),
            'pos_err_integ_x': pos_err_integ[0].item(),
            'pos_err_integ_y': pos_err_integ[1].item(),
            'pos_err_integ_z': pos_err_integ[2].item(),
            'vel_x': vel[0].item(),
            'vel_y': vel[1].item(),
            'vel_z': vel[2].item(),
            'ang_vel_x': omega[0].item(),
            'ang_vel_y': omega[1].item(),
            'ang_vel_z': omega[2].item(),
            'err_p_yaw': err_p_yaw,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'target_x': target[0],
            'target_y': target[1]
        }
        
        action_vals = controller.get_action(state_dict)
        u_fz, u_tx, u_ty, u_tz = action_vals
        
        u_tx = max(-0.4, min(0.4, u_tx))
        u_ty = max(-0.4, min(0.4, u_ty))
        u_tz = max(-0.5, min(0.5, u_tz))
        u_fz = max(0.0, min(1.3, u_fz))
        
        actions = torch.tensor([[0.0, 0.0, u_fz, u_tx, u_ty, u_tz]], device=env.device)
        
        scg_action = torch.tensor([[u_fz, u_tx, u_ty, u_tz]], device=env.device)
        scg_calc.compute_step(env.pos, env.lin_vel, env.quat, env.ang_vel, target_tensor, scg_action)
        
        env.step(actions)
        
    components = scg_calc.get_components()
    return components["total_cost"][0].item()

def main():
    print("初始化 Isaac Gym 环境 (Circle Trajectory)...")
    env = IsaacGymDroneEnv(num_envs=1, device='cuda:0', headless=True, duration_sec=5.0)
    scg_calc = SCGExactRewardCalculator(num_envs=1, device='cuda:0')
    
    print("开始 Nonlinear DSL 参数搜索 (Circle - 4 Terms with Integral)...")
    print("目标: Total Cost < 100")
    print("-" * 60)
    
    best_cost = float('inf')
    best_params = None
    
    # 1. Random Search
    num_random = 60
    print(f"阶段 1: 随机搜索 ({num_random} 次迭代)")
    
    for i in range(num_random):
        k_p = random.uniform(0.5, 3.0)
        k_d = random.uniform(0.5, 2.5)
        k_w = random.uniform(0.1, 1.0)
        k_s = random.uniform(0.1, 2.0)
        k_i = random.uniform(0.0, 1.5) # Integral gain
        k_z_p = random.uniform(0.5, 2.0)
        k_z_d = random.uniform(0.5, 1.5)
        
        cost = evaluate_params(env, scg_calc, k_p, k_d, k_w, k_s, k_i, k_z_p, k_z_d)
        print(f"Iter {i+1:02d}: k_p={k_p:.3f}, k_d={k_d:.3f}, k_w={k_w:.3f}, k_s={k_s:.3f}, k_i={k_i:.3f}, k_z_p={k_z_p:.3f}, k_z_d={k_z_d:.3f} => Cost={cost:.2f}")
        
        if cost < best_cost:
            best_cost = cost
            best_params = (k_p, k_d, k_w, k_s, k_i, k_z_p, k_z_d)
            
    # 2. Local Refinement
    print("-" * 60)
    print(f"阶段 2: 局部精细化搜索")
    print(f"当前最佳: {best_params} (Cost: {best_cost:.2f})")
    
    if best_params is None:
        best_params = (1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0)

    center_k_p, center_k_d, center_k_w, center_k_s, center_k_i, center_k_z_p, center_k_z_d = best_params
    num_local = 40
    sigma = 0.15
    
    for i in range(num_local):
        k_p = center_k_p + random.gauss(0, sigma)
        k_d = center_k_d + random.gauss(0, sigma)
        k_w = center_k_w + random.gauss(0, sigma)
        k_s = center_k_s + random.gauss(0, sigma)
        k_i = center_k_i + random.gauss(0, sigma)
        k_z_p = center_k_z_p + random.gauss(0, sigma)
        k_z_d = center_k_z_d + random.gauss(0, sigma)
        
        k_p = max(0.1, min(4.0, k_p))
        k_d = max(0.1, min(3.0, k_d))
        k_w = max(0.0, min(2.0, k_w))
        k_s = max(0.01, min(3.0, k_s))
        k_i = max(0.0, min(3.0, k_i))
        k_z_p = max(0.1, min(3.0, k_z_p))
        k_z_d = max(0.1, min(2.0, k_z_d))
        
        cost = evaluate_params(env, scg_calc, k_p, k_d, k_w, k_s, k_i, k_z_p, k_z_d)
        print(f"Refine {i+1:02d}: k_p={k_p:.3f}, k_d={k_d:.3f}, k_w={k_w:.3f}, k_s={k_s:.3f}, k_i={k_i:.3f}, k_z_p={k_z_p:.3f}, k_z_d={k_z_d:.3f} => Cost={cost:.2f}")
        
        if cost < best_cost:
            best_cost = cost
            best_params = (k_p, k_d, k_w, k_s, k_i, k_z_p, k_z_d)
            
    print("=" * 60)
    print("搜索完成")
    print(f"最佳参数: k_p={best_params[0]:.4f}, k_d={best_params[1]:.4f}, k_w={best_params[2]:.4f}, k_s={best_params[3]:.4f}, k_i={best_params[4]:.4f}, k_z_p={best_params[5]:.4f}, k_z_d={best_params[6]:.4f}")
    print(f"最佳 Cost: {best_cost:.4f}")
    print("=" * 60)
    
    prog_tx, prog_ty, _, _ = create_nonlinear_programs(*best_params)
    print("生成的 Nonlinear DSL 表达式 (u_tx):")
    print(prog_tx)
    print("生成的 Nonlinear DSL 表达式 (u_ty):")
    print(prog_ty)
    
    env.close()

if __name__ == "__main__":
    main()

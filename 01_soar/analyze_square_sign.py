
import os, sys

# Add path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    import isaacgym
except ImportError:
    pass

import torch
import numpy as np
from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from utils.reward_scg_exact import SCGExactRewardCalculator
from core.dsl import ProgramNode, BinaryOpNode, UnaryOpNode, TerminalNode, ConstantNode

def square_target_xy(t):
    period = 5.0
    scale = 1.0
    cycle = t % period
    seg_period = period / 4.0
    seg_idx = int(cycle // seg_period)
    seg_time = cycle - seg_idx * seg_period
    speed = scale / seg_period
    dist = speed * seg_time
    x, y = 0.0, 0.0
    if seg_idx == 0:
        x = 0.0; y = dist
    elif seg_idx == 1:
        x = -dist; y = scale
    elif seg_idx == 2:
        x = -scale; y = scale - dist
    else:
        x = -scale + dist; y = 0.0
    return np.array([x, y, 1.0])

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

def create_sign_programs(k_p, k_d, k_w):
    # Replace smooth with sign
    # u_tx = -k_p * sign(pos_err_y) + k_d * vel_y - k_w * ang_vel_x
    
    sign_err_y = UnaryOpNode('sign', TerminalNode('pos_err_y'))
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), sign_err_y)
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    prog_tx = BinaryOpNode('-', BinaryOpNode('+', term1_tx, term2_tx), term3_tx)

    # u_ty = k_p * sign(pos_err_x) - k_d * vel_x - k_w * ang_vel_y
    sign_err_x = UnaryOpNode('sign', TerminalNode('pos_err_x'))
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), sign_err_x)
    term2_ty = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_x'))
    term3_ty = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_y'))
    prog_ty = BinaryOpNode('-', BinaryOpNode('-', term1_ty, term2_ty), term3_ty)
              
    prog_tz = BinaryOpNode('-', 
                BinaryOpNode('*', ConstantNode(4.0), TerminalNode('err_p_yaw')),
                BinaryOpNode('*', ConstantNode(0.8), TerminalNode('ang_vel_z'))
              )
    prog_fz = BinaryOpNode('+',
                BinaryOpNode('-',
                    BinaryOpNode('*', ConstantNode(0.5), TerminalNode('pos_err_z')),
                    BinaryOpNode('*', ConstantNode(0.2), TerminalNode('vel_z'))
                ),
                ConstantNode(0.65)
              )
    return prog_tx, prog_ty, prog_tz, prog_fz

def run_test(k_p, k_d, k_w):
    print(f"\nTesting with k_p={k_p}, k_d={k_d}, k_w={k_w}")
    
    try:
        env = IsaacGymDroneEnv(num_envs=1, device='cuda:0', headless=True, duration_sec=5.0)
        scg_calc = SCGExactRewardCalculator(num_envs=1, device='cuda:0')
        
        prog_tx, prog_ty, prog_tz, prog_fz = create_sign_programs(k_p, k_d, k_w)
        controller = DSLController(prog_tx, prog_ty, prog_tz, prog_fz)
        
        initial_pos = torch.tensor([[0.0, 0.0, 1.0]], device='cuda:0')
        env.reset(initial_pos=initial_pos)
        scg_calc.reset()
        
        dt = 1.0 / 48.0
        
        for step in range(240):
            t = step * dt
            target = square_target_xy(t)
            target_tensor = torch.tensor(target, device='cuda:0', dtype=torch.float32)
            
            pos = env.pos[0]
            vel = env.lin_vel[0]
            quat = env.quat[0]
            omega = env.ang_vel[0]
            
            pos_err = target_tensor - pos
            
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
            
            action_vals = controller.get_action(state_dict)
            u_fz, u_tx, u_ty, u_tz = action_vals
            
            u_tx = max(-0.4, min(0.4, u_tx))
            u_ty = max(-0.4, min(0.4, u_ty))
            u_tz = max(-0.5, min(0.5, u_tz))
            u_fz = max(0.0, min(1.3, u_fz))
            
            actions = torch.tensor([[0.0, 0.0, u_fz, u_tx, u_ty, u_tz]], device='cuda:0')
            scg_action = torch.tensor([[u_fz, u_tx, u_ty, u_tz]], device='cuda:0')
            scg_calc.compute_step(env.pos, env.lin_vel, env.quat, env.ang_vel, target_tensor, scg_action)
            
            env.step(actions)
            
        env.close()
        components = scg_calc.get_components()
        cost = components["total_cost"][0].item()
        print(f'Final Cost: {cost:.2f}')
        return cost
    except Exception as e:
        print(f"Error: {e}")
        return 10000.0

# Original params from manual.md for square
# k_p=1.990, k_d=1.766, k_w=0.773
k_p_orig = 1.990
k_d_orig = 1.766
k_w_orig = 0.773

print("Testing SIGN function for Square Trajectory")

# Further refinement
run_test(0.6, k_d_orig, k_w_orig)
run_test(0.7, k_d_orig, k_w_orig)
run_test(0.8, k_d_orig, k_w_orig)
run_test(0.9, k_d_orig, k_w_orig)
run_test(1.0, k_d_orig, k_w_orig)

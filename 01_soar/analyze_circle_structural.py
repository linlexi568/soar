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

def circle_target_xy(t):
    radius = 1.0
    period = 5.0
    omega = 2.0 * np.pi / period
    x = radius * np.cos(omega * t)
    y = radius * np.sin(omega * t)
    
    # Velocity (derivative)
    vx = -radius * omega * np.sin(omega * t)
    vy = radius * omega * np.cos(omega * t)
    
    # Acceleration (derivative of velocity)
    ax = -radius * omega * omega * np.cos(omega * t)
    ay = -radius * omega * omega * np.sin(omega * t)
    
    return np.array([x, y, 1.0]), np.array([vx, vy, 0.0]), np.array([ax, ay, 0.0])

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

def create_baseline_programs(k_p, k_d, k_w, k_s):
    # u_tx = -k_p * smooth(pos_err_y) + k_d * vel_y - k_w * ang_vel_x
    smooth_err_y = UnaryOpNode('smooth', TerminalNode('pos_err_y'), params={'s': ConstantNode(k_s)})
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), smooth_err_y)
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    prog_tx = BinaryOpNode('-', BinaryOpNode('+', term1_tx, term2_tx), term3_tx)

    # u_ty = k_p * smooth(pos_err_x) - k_d * vel_x - k_w * ang_vel_y
    smooth_err_x = UnaryOpNode('smooth', TerminalNode('pos_err_x'), params={'s': ConstantNode(k_s)})
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), smooth_err_x)
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

def create_gain_scheduling_programs(k_p, k_d, k_w):
    # u_tx = -k_p * (pos_err_y / (1 + abs(pos_err_y))) + k_d * vel_y - k_w * ang_vel_x
    
    # Term: pos_err_y / (1 + abs(pos_err_y))
    # Denom: 1 + abs(pos_err_y)
    abs_err_y = UnaryOpNode('abs', TerminalNode('pos_err_y'))
    denom_y = BinaryOpNode('+', ConstantNode(1.0), abs_err_y)
    sched_err_y = BinaryOpNode('/', TerminalNode('pos_err_y'), denom_y)
    
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), sched_err_y)
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    prog_tx = BinaryOpNode('-', BinaryOpNode('+', term1_tx, term2_tx), term3_tx)

    # u_ty = k_p * (pos_err_x / (1 + abs(pos_err_x))) - k_d * vel_x - k_w * ang_vel_y
    abs_err_x = UnaryOpNode('abs', TerminalNode('pos_err_x'))
    denom_x = BinaryOpNode('+', ConstantNode(1.0), abs_err_x)
    sched_err_x = BinaryOpNode('/', TerminalNode('pos_err_x'), denom_x)
    
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), sched_err_x)
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

def create_cubic_programs(k_p, k_d, k_w):
    # u_tx = -k_p * (pos_err_y * abs(pos_err_y)) + k_d * vel_y - k_w * ang_vel_x
    
    # Term: pos_err_y * abs(pos_err_y)
    abs_err_y = UnaryOpNode('abs', TerminalNode('pos_err_y'))
    cubic_err_y = BinaryOpNode('*', TerminalNode('pos_err_y'), abs_err_y)
    
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), cubic_err_y)
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    prog_tx = BinaryOpNode('-', BinaryOpNode('+', term1_tx, term2_tx), term3_tx)

    # u_ty = k_p * (pos_err_x * abs(pos_err_x)) - k_d * vel_x - k_w * ang_vel_y
    abs_err_x = UnaryOpNode('abs', TerminalNode('pos_err_x'))
    cubic_err_x = BinaryOpNode('*', TerminalNode('pos_err_x'), abs_err_x)
    
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), cubic_err_x)
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

def create_pid_programs(k_p, k_d, k_w, k_s, k_i):
    # u_tx = -k_p * smooth(pos_err_y) + k_d * vel_y - k_w * ang_vel_x - k_i * integral(pos_err_y)
    # Note: For integral, we need time-series operator in DSL, but we don't have that.
    # Workaround: Use "ema" with very slow alpha (accumulates slowly over time).
    # Better: Add a manual integrator state.
    
    # Let's use: integral ~ ema with small alpha (0.1)
    smooth_err_y = UnaryOpNode('smooth', TerminalNode('pos_err_y'), params={'s': ConstantNode(k_s)})
    ema_err_y = UnaryOpNode('ema', TerminalNode('pos_err_y'), params={'alpha': ConstantNode(0.05)})
    
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), smooth_err_y)
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    term4_tx = BinaryOpNode('*', ConstantNode(k_i), ema_err_y)
    
    base_tx = BinaryOpNode('-', BinaryOpNode('+', term1_tx, term2_tx), term3_tx)
    prog_tx = BinaryOpNode('-', base_tx, term4_tx)

    # u_ty
    smooth_err_x = UnaryOpNode('smooth', TerminalNode('pos_err_x'), params={'s': ConstantNode(k_s)})
    ema_err_x = UnaryOpNode('ema', TerminalNode('pos_err_x'), params={'alpha': ConstantNode(0.05)})
    
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), smooth_err_x)
    term2_ty = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_x'))
    term3_ty = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_y'))
    term4_ty = BinaryOpNode('*', ConstantNode(k_i), ema_err_x)
    
    base_ty = BinaryOpNode('-', BinaryOpNode('-', term1_ty, term2_ty), term3_ty)
    prog_ty = BinaryOpNode('-', base_ty, term4_ty)
    
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

def create_linear_programs(k_p, k_d, k_w):
    # u_tx = -k_p * pos_err_y + k_d * vel_y - k_w * ang_vel_x
    
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), TerminalNode('pos_err_y'))
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    prog_tx = BinaryOpNode('-', BinaryOpNode('+', term1_tx, term2_tx), term3_tx)

    # u_ty = k_p * pos_err_x - k_d * vel_x - k_w * ang_vel_y
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), TerminalNode('pos_err_x'))
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

def create_feedforward_programs(k_p, k_d, k_w, k_s, k_ff):
    # u_tx = -k_p * smooth(pos_err_y) + k_d * vel_y - k_w * ang_vel_x - k_ff * target_acc_y
    # Note: target_acc_y is the required centripetal force direction
    # Roll creates force in -Y direction (approx), so to get +AccY, we need -Roll.
    # So we subtract target_acc_y.
    
    smooth_err_y = UnaryOpNode('smooth', TerminalNode('pos_err_y'), params={'s': ConstantNode(k_s)})
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), smooth_err_y)
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    term3_tx = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_x'))
    term4_tx = BinaryOpNode('*', ConstantNode(k_ff), TerminalNode('target_acc_y'))
    
    base_tx = BinaryOpNode('-', BinaryOpNode('+', term1_tx, term2_tx), term3_tx)
    prog_tx = BinaryOpNode('-', base_tx, term4_tx)

    # u_ty = k_p * smooth(pos_err_x) - k_d * vel_x - k_w * ang_vel_y + k_ff * target_acc_x
    # Pitch creates force in +X direction (approx).
    smooth_err_x = UnaryOpNode('smooth', TerminalNode('pos_err_x'), params={'s': ConstantNode(k_s)})
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), smooth_err_x)
    term2_ty = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_x'))
    term3_ty = BinaryOpNode('*', ConstantNode(k_w), TerminalNode('ang_vel_y'))
    term4_ty = BinaryOpNode('*', ConstantNode(k_ff), TerminalNode('target_acc_x'))
    
    base_ty = BinaryOpNode('-', BinaryOpNode('-', term1_ty, term2_ty), term3_ty)
    prog_ty = BinaryOpNode('+', base_ty, term4_ty)
    
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

def run_test(mode, params):
    print(f"\nTesting {mode} with params={params}")
    
    try:
        env = IsaacGymDroneEnv(num_envs=1, device='cuda:0', headless=True, duration_sec=5.0)
        scg_calc = SCGExactRewardCalculator(num_envs=1, device='cuda:0')
        
        if mode == 'baseline':
            prog_tx, prog_ty, prog_tz, prog_fz = create_baseline_programs(
                params['k_p'], params['k_d'], params['k_w'], params['k_s'])
        elif mode == 'feedforward':
            prog_tx, prog_ty, prog_tz, prog_fz = create_feedforward_programs(
                params['k_p'], params['k_d'], params['k_w'], params['k_s'], params['k_ff'])
        elif mode == 'gain_scheduling':
            prog_tx, prog_ty, prog_tz, prog_fz = create_gain_scheduling_programs(
                params['k_p'], params['k_d'], params['k_w'])
        elif mode == 'cubic':
            prog_tx, prog_ty, prog_tz, prog_fz = create_cubic_programs(
                params['k_p'], params['k_d'], params['k_w'])
        elif mode == 'linear':
            prog_tx, prog_ty, prog_tz, prog_fz = create_linear_programs(
                params['k_p'], params['k_d'], params['k_w'])
        elif mode == 'pid':
            prog_tx, prog_ty, prog_tz, prog_fz = create_pid_programs(
                params['k_p'], params['k_d'], params['k_w'], params['k_s'], params['k_i'])
        
        controller = DSLController(prog_tx, prog_ty, prog_tz, prog_fz)


        
        # Start at (1, 0, 1) for Circle
        initial_pos = torch.tensor([[1.0, 0.0, 1.0]], device='cuda:0')
        env.reset(initial_pos=initial_pos)
        scg_calc.reset()
        
        dt = 1.0 / 48.0
        
        for step in range(240):
            t = step * dt
            target_pos, target_vel, target_acc = circle_target_xy(t)
            target_tensor = torch.tensor(target_pos, device='cuda:0', dtype=torch.float32)
            
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
                'target_vel_x': target_vel[0],
                'target_vel_y': target_vel[1],
                'target_acc_x': target_acc[0],
                'target_acc_y': target_acc[1],
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

# Baseline Params (from manual.md)
# k_p=0.489, k_s=1.285, k_d=1.062, k_w=0.731 (Figure8 params, used as starting point)
# Actually manual.md says for circle:
# u_tx = ((-1.308 * smooth(pos_err_y, s=0.179)) + (0.477 * vel_y)) - (0.373 * ang_vel_x)
base_params = {
    'k_p': 1.308,
    'k_s': 0.179,
    'k_d': 0.477,
    'k_w': 0.373
}

print("Testing Circle Trajectory Structures")

# 1. Baseline
run_test('baseline', base_params)

# 3. PID (add integral term)
# u_tx = -k_p * smooth(pos_err_y) + k_d * vel_y - k_w * ang_vel_x - k_i * integral(pos_err_y)
# print("--- Testing Gain Scheduling ---")
# Previous tests showed high cost for Kp >= 1.0.
# Try smaller Kp values.
# for k_p in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
#     gs_params = base_params.copy()
#     gs_params['k_p'] = k_p
#     run_test('gain_scheduling', gs_params)

# 3. PID (add integral term)
# print("\n--- Testing PID (Integral) ---")
# # Try small Ki values.
# for k_i in [0.01, 0.05, 0.1, 0.2, 0.5]:
#     pid_params = base_params.copy()
#     pid_params['k_i'] = k_i
#     run_test('pid', pid_params)

# 4. Feedforward (add target velocity term)
print("\n--- Testing Feedforward (Acceleration) ---")
# Best params found via grid search: k_p=1.4, k_ff=0.03 -> Cost 157.37
best_ff_params = base_params.copy()
best_ff_params['k_p'] = 1.4
best_ff_params['k_ff'] = 0.03
run_test('feedforward', best_ff_params)

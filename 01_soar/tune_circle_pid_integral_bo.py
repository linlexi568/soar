"""
贝叶斯优化 - Circle 轨迹 非线性 PID 控制器
比例项和积分项都使用 smooth() 非线性变换
"""
import os, sys
import optuna

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

def create_nonlinear_pid_programs(k_p, k_d, k_w, k_s, k_i, s_i):
    """
    创建非线性 PID 控制器
    比例项用 smooth(err, s=k_s)
    积分项用 smooth(err_integral, s=s_i) 实现非线性
    
    u_tx = -k_p * smooth(pos_err_y, s=k_s) + k_d * vel_y - k_w * ang_vel_x - k_i * smooth(err_i_y, s=s_i)
    u_ty = k_p * smooth(pos_err_x, s=k_s) - k_d * vel_x + k_w * ang_vel_y + k_i * smooth(err_i_x, s=s_i)
    
    非线性特性:
    - smooth(x, s) = tanh(s * x) / s  -- 饱和非线性（可微软边界）
    """
    # u_tx
    # Term 1: -k_p * smooth(pos_err_y)
    smooth_err_y = UnaryOpNode('smooth', TerminalNode('pos_err_y'), params={'s': ConstantNode(k_s)})
    term1_tx = BinaryOpNode('*', ConstantNode(-k_p), smooth_err_y)
    
    # Term 2: k_d * vel_y
    term2_tx = BinaryOpNode('*', ConstantNode(k_d), TerminalNode('vel_y'))
    
    # Term 3: -k_w * ang_vel_x
    term3_tx = BinaryOpNode('*', ConstantNode(-k_w), TerminalNode('ang_vel_x'))
    
    # Term 4: -k_i * smooth(err_i_y)  -- 非线性积分项
    smooth_err_i_y = UnaryOpNode('smooth', TerminalNode('err_i_y'), params={'s': ConstantNode(s_i)})
    term4_tx = BinaryOpNode('*', ConstantNode(-k_i), smooth_err_i_y)
    
    # Combine: (Term1 + Term2) + (Term3 + Term4)
    prog_tx = BinaryOpNode('+', 
                BinaryOpNode('+', term1_tx, term2_tx),
                BinaryOpNode('+', term3_tx, term4_tx)
              )

    # u_ty
    # Term 1: k_p * smooth(pos_err_x)
    smooth_err_x = UnaryOpNode('smooth', TerminalNode('pos_err_x'), params={'s': ConstantNode(k_s)})
    term1_ty = BinaryOpNode('*', ConstantNode(k_p), smooth_err_x)
    
    # Term 2: -k_d * vel_x
    term2_ty = BinaryOpNode('*', ConstantNode(-k_d), TerminalNode('vel_x'))
    
    # Term 3: -k_w * ang_vel_y  (注意：负号！)
    term3_ty = BinaryOpNode('*', ConstantNode(-k_w), TerminalNode('ang_vel_y'))
    
    # Term 4: k_i * smooth(err_i_x)  -- 非线性积分项
    smooth_err_i_x = UnaryOpNode('smooth', TerminalNode('err_i_x'), params={'s': ConstantNode(s_i)})
    term4_ty = BinaryOpNode('*', ConstantNode(k_i), smooth_err_i_x)
    
    # Combine
    prog_ty = BinaryOpNode('+', 
                BinaryOpNode('+', term1_ty, term2_ty),
                BinaryOpNode('+', term3_ty, term4_ty)
              )
    
    # u_tz (yaw control)
    prog_tz = BinaryOpNode('-', 
                BinaryOpNode('*', ConstantNode(4.0), TerminalNode('err_p_yaw')),
                BinaryOpNode('*', ConstantNode(0.8), TerminalNode('ang_vel_z'))
              )
    
    # u_fz (thrust)
    prog_fz = BinaryOpNode('+',
                BinaryOpNode('-',
                    BinaryOpNode('*', ConstantNode(0.5), TerminalNode('pos_err_z')),
                    BinaryOpNode('*', ConstantNode(0.2), TerminalNode('vel_z'))
                ),
                ConstantNode(0.65)
              )
    return prog_tx, prog_ty, prog_tz, prog_fz

def objective(trial):
    # 搜索空间
    k_p = trial.suggest_float('k_p', 0.5, 3.0)
    k_d = trial.suggest_float('k_d', 0.2, 1.0)
    k_w = trial.suggest_float('k_w', 0.1, 0.8)
    k_s = trial.suggest_float('k_s', 0.05, 0.5)
    k_i = trial.suggest_float('k_i', 0.0, 0.5)  # 积分增益
    s_i = trial.suggest_float('s_i', 0.1, 2.0)  # 积分项的 smooth 参数
    
    try:
        env = IsaacGymDroneEnv(num_envs=1, device='cuda:0', headless=True, duration_sec=5.0)
        scg_calc = SCGExactRewardCalculator(num_envs=1, device='cuda:0')
        
        prog_tx, prog_ty, prog_tz, prog_fz = create_nonlinear_pid_programs(k_p, k_d, k_w, k_s, k_i, s_i)
        controller = DSLController(prog_tx, prog_ty, prog_tz, prog_fz)
        
        # Start at (1, 0, 1) for Circle
        initial_pos = torch.tensor([[1.0, 0.0, 1.0]], device='cuda:0')
        env.reset(initial_pos=initial_pos)
        scg_calc.reset()
        
        dt = 1.0 / 48.0
        
        # 线性积分累积，然后在控制器中通过 smooth() 应用非线性变换
        err_i_x = 0.0
        err_i_y = 0.0
        
        for step in range(240):
            t = step * dt
            target_pos = circle_target_xy(t)
            target_tensor = torch.tensor(target_pos, device='cuda:0', dtype=torch.float32)
            
            pos = env.pos[0]
            vel = env.lin_vel[0]
            quat = env.quat[0]
            omega = env.ang_vel[0]
            
            pos_err = target_tensor - pos
            
            # 累积积分
            err_i_x += pos_err[0].item() * dt
            err_i_y += pos_err[1].item() * dt
            # Anti-windup clamp
            err_i_x = max(-2.0, min(2.0, err_i_x))
            err_i_y = max(-2.0, min(2.0, err_i_y))
            
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
                'err_i_x': err_i_x,
                'err_i_y': err_i_y,
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
        
        return cost
    except Exception as e:
        print(f"Error: {e}")
        return 10000.0

if __name__ == "__main__":
    print("Starting Bayesian Optimization for Circle Trajectory (Nonlinear PID with smooth(integral))...")
    study = optuna.create_study(direction='minimize')
    
    # Enqueue known good params as starting point (k_i=0 for baseline)
    study.enqueue_trial({
        'k_p': 1.4,
        'k_d': 0.477,
        'k_w': 0.373,
        'k_s': 0.179,
        'k_i': 0.0,
        's_i': 1.0  # doesn't matter when k_i=0
    })
    
    # Try with nonlinear integral term
    study.enqueue_trial({
        'k_p': 1.4,
        'k_d': 0.477,
        'k_w': 0.373,
        'k_s': 0.179,
        'k_i': 0.1,
        's_i': 0.5  # lower s_i = more linear in small range
    })
    
    # Try stronger integral with harder saturation
    study.enqueue_trial({
        'k_p': 1.4,
        'k_d': 0.477,
        'k_w': 0.373,
        'k_s': 0.179,
        'k_i': 0.3,
        's_i': 1.5  # higher s_i = earlier saturation
    })
    
    study.optimize(objective, n_trials=100)
    
    print("\nOptimization finished!")
    print(f"Best Cost: {study.best_value}")
    print(f"Best Params: {study.best_params}")

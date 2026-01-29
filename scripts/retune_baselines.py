#!/usr/bin/env python3
"""
传统基线调参脚本 - 避免Isaac Gym重复初始化
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'isaacgym' / 'python'))

# 先导入Isaac Gym
from isaacgym import gymapi

import json
import math
import numpy as np
import torch
from typing import Dict, Any, Tuple

# 导入环境和奖励计算器
from scripts.baselines.tune_pid_lqr_isaac import (
    IsaacPIDController, IsaacLQRController,
    quat_to_euler_np, IsaacGymDroneEnv, SCGExactRewardCalculator
)
from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity

# ============================================================================
#                    配置参数 (修改这里)
# ============================================================================

TASKS = ['circle', 'square', 'helix', 'figure8']
TRIALS = 30          # 随机搜索次数
EPISODES = 3         # 每次评估的episode数
DURATION = 5         # episode时长(秒)
OUTPUT_DIR = ROOT / 'results' / 'baselines_unified'

# PID参数
PID_BOUNDS = {
    'kp_xy': (1.0, 50.0), 'kd_xy': (0.5, 30.0),
    'kp_z': (1.0, 50.0), 'kd_z': (0.5, 20.0),
    'kp_att': (5.0, 80.0), 'kd_att': (0.1, 10.0),
    'att_scale': (0.05, 0.5),
}
PID_INIT = {
    'kp_xy': 8.0, 'kd_xy': 4.0, 'kp_z': 14.0, 'kd_z': 6.0,
    'kp_att': 12.0, 'kd_att': 2.0, 'att_scale': 0.2,
}

# LQR参数
LQR_BOUNDS = {
    'q_pos': (0.1, 100.0), 'q_vel': (0.1, 50.0),
    'q_att': (0.1, 100.0), 'q_omega': (0.1, 20.0),
    'r_fz': (0.001, 10.0), 'r_torque': (0.001, 10.0),
}
LQR_INIT = {
    'q_pos': 10.0, 'q_vel': 2.0, 'q_att': 20.0, 'q_omega': 5.0,
    'r_fz': 1.0, 'r_torque': 1.0,
}

# CPID参数
CPID_BOUNDS = {
    'kp_pos_xy': (0.5, 20.0), 'kp_pos_z': (0.5, 10.0),
    'kd_pos_xy': (0.1, 10.0), 'kd_pos_z': (0.1, 5.0),
    'kp_vel_xy': (0.5, 20.0), 'kp_vel_z': (0.5, 20.0),
    'kp_att': (5.0, 80.0), 'kd_att': (0.1, 10.0),
    'att_scale': (0.05, 0.5),
}
CPID_INIT = {
    'kp_pos_xy': 5.0, 'kp_pos_z': 2.0, 'kd_pos_xy': 2.0, 'kd_pos_z': 1.0,
    'kp_vel_xy': 8.0, 'kp_vel_z': 4.0, 'kp_att': 20.0, 'kd_att': 2.0,
    'att_scale': 0.2,
}


def evaluate_controller(env, reward_calc, controller, task: str, duration: float, episodes: int = 3):
    """在单个环境实例中评估控制器"""
    device = env.device
    ctrl_freq = getattr(env, 'control_freq', 48.0)
    dt = 1.0 / float(ctrl_freq)
    steps = int(duration * ctrl_freq)
    
    cfg = get_scg_trajectory_config(task)
    center = np.array(cfg.center, dtype=np.float32)
    
    def make_targets(t: float):
        pos, vel = scg_position_velocity(task, t, params=cfg.params, center=center)
        return np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32)
    
    ep_rewards = []
    rmses = []
    
    for _ in range(episodes):
        env.reset()
        reward_calc.reset(1)
        if hasattr(controller, 'set_dt'):
            controller.set_dt(dt)
        if hasattr(controller, 'reset'):
            controller.reset()
        
        t = 0.0
        pos_errs = []
        
        for s in range(steps):
            obs = env.get_obs()
            pos = np.asarray(obs['position'][0], dtype=np.float32)
            vel = np.asarray(obs['velocity'][0], dtype=np.float32)
            quat = np.asarray(obs['orientation'][0], dtype=np.float32)
            omega = np.asarray(obs['angular_velocity'][0], dtype=np.float32)
            
            tgt_pos, tgt_vel = make_targets(t)
            action4 = controller.compute(pos, vel, quat, omega, tgt_pos, tgt_vel)
            
            forces = torch.zeros(1, 6, device=device)
            forces[0, 2] = float(action4[0])  # fz
            forces[0, 3] = float(action4[1])  # tx
            forces[0, 4] = float(action4[2])  # ty
            forces[0, 5] = float(action4[3])  # tz
            
            obs_next, _, done, _ = env.step(forces)
            
            # 计算奖励
            pos_t = torch.tensor(obs_next['position'], device=device)
            vel_t = torch.tensor(obs_next['velocity'], device=device)
            quat_t = torch.tensor(obs_next['orientation'], device=device)
            omega_t = torch.tensor(obs_next['angular_velocity'], device=device)
            target_pos_t = torch.tensor(tgt_pos, device=device)
            reward_calc.compute_step(pos_t, vel_t, quat_t, omega_t, target_pos_t, forces)
            
            pos_errs.append(float(np.linalg.norm(pos - tgt_pos)))
            t += dt
        
        comps = reward_calc.get_components()
        total_cost = float(comps['total_cost'].sum().item())
        ep_rewards.append(-total_cost)
        rmses.append(float(np.sqrt(np.mean(np.array(pos_errs)**2))))
    
    return {
        'mean_true_reward': float(np.mean(ep_rewards)),
        'std_true_reward': float(np.std(ep_rewards)),
        'rmse_pos': float(np.mean(rmses)),
    }


def random_search(env, reward_calc, ctrl_class, init_params, bounds, task, trials, episodes):
    """随机搜索调参"""
    best_params = dict(init_params)
    ctrl = ctrl_class(**best_params)
    best_metrics = evaluate_controller(env, reward_calc, ctrl, task, DURATION, episodes)
    best_reward = best_metrics['mean_true_reward']
    
    print(f"  初始奖励: {best_reward:.2f}")
    
    for i in range(trials):
        # 生成候选参数
        ratio = max(0.2, 1.0 - i / trials)  # 退火
        proposal = {}
        for k, v in init_params.items():
            lo, hi = bounds[k]
            span = (hi - lo) * 0.5 * ratio
            proposal[k] = float(np.clip(v + np.random.uniform(-span, span), lo, hi))
        
        ctrl = ctrl_class(**proposal)
        metrics = evaluate_controller(env, reward_calc, ctrl, task, DURATION, episodes)
        
        if metrics['mean_true_reward'] > best_reward:
            best_reward = metrics['mean_true_reward']
            best_params = proposal
            best_metrics = metrics
            print(f"  [更新] 轮次{i}: {best_reward:.2f}")
    
    return best_params, best_metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("传统基线调参 (统一SCG奖励)")
    print("=" * 70)
    print(f"  任务: {TASKS}")
    print(f"  调参轮数: {TRIALS}")
    print(f"  评估episodes: {EPISODES}")
    print(f"  时长: {DURATION}s")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建一个环境实例（整个过程复用）
    print("初始化Isaac Gym环境...")
    env = IsaacGymDroneEnv(num_envs=1, device=device, headless=True, duration_sec=DURATION)
    reward_calc = SCGExactRewardCalculator(num_envs=1, device=device)
    
    results = {}
    
    for task in TASKS:
        print(f"\n{'='*70}")
        print(f"任务: {task}")
        print(f"{'='*70}")
        
        results[task] = {}
        
        # PID
        print(f"\n[PID] 调参...")
        best_params, best_metrics = random_search(
            env, reward_calc, IsaacPIDController, 
            PID_INIT, PID_BOUNDS, task, TRIALS, EPISODES
        )
        results[task]['PID'] = {
            'params': best_params,
            'reward': best_metrics['mean_true_reward'],
            'std': best_metrics['std_true_reward'],
            'rmse': best_metrics['rmse_pos'],
            'trials': TRIALS,
        }
        print(f"[PID] 最佳: {best_metrics['mean_true_reward']:.2f}")
        
        # LQR
        print(f"\n[LQR] 调参...")
        best_params, best_metrics = random_search(
            env, reward_calc, IsaacLQRController,
            LQR_INIT, LQR_BOUNDS, task, TRIALS, EPISODES
        )
        results[task]['LQR'] = {
            'params': best_params,
            'reward': best_metrics['mean_true_reward'],
            'std': best_metrics['std_true_reward'],
            'rmse': best_metrics['rmse_pos'],
            'trials': TRIALS,
        }
        print(f"[LQR] 最佳: {best_metrics['mean_true_reward']:.2f}")
        
        # 保存中间结果
        with open(OUTPUT_DIR / f'{task}_baselines.json', 'w') as f:
            json.dump({task: results[task]}, f, indent=2)
    
    # 保存汇总
    output_file = OUTPUT_DIR / 'baselines_unified.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印汇总
    print(f"\n{'='*70}")
    print("调参完成!")
    print(f"{'='*70}")
    
    print("\n| 方法 | circle | square | helix | figure8 |")
    print("|------|--------|--------|-------|---------|")
    for method in ['PID', 'LQR']:
        row = [method]
        for task in ['circle', 'square', 'helix', 'figure8']:
            if task in results and method in results[task]:
                r = results[task][method]['reward']
                row.append(f"{r:.2f}")
            else:
                row.append("N/A")
        print("| " + " | ".join(row) + " |")
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
全方法统一对比脚本 - 使用 SCG 精确奖励
评估 Soar / SAC / TD3 / PID / CPID / LQR 在同一任务上的表现

所有参数写在脚本顶部，直接修改即可

奖励函数: r_t = -(x_err^T Q x_err + u^T R u)
  Q = diag([1, 0.01, 1, 0.01, 1, 0.01, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
  R = 0.0001
"""

import sys
from pathlib import Path
import os

# ============================================================================
# 路径设置 (Isaac Gym 必须在 torch 之前导入)
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / '01_soar') not in sys.path:
    sys.path.insert(0, str(ROOT / '01_soar'))

_ISAAC_GYM_PY = ROOT / 'isaacgym' / 'python'
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

# ============================================================================
#                    ★★★ 对比参数配置 (修改这里) ★★★
# ============================================================================

# --------------------- 任务配置 ---------------------
TASK = "figure8"          # 选择: circle, square, helix, figure8, hover
DURATION = 5.0            # 每个episode时长(秒)

# --------------------- 评估配置 ---------------------
NUM_ENVS = 1024           # 并行环境数
EPISODES = 10             # 评估episode数 (RL方法) / 基线也用10
DEVICE = "cuda:0"         # 设备

# --------------------- 要评估的方法 ---------------------
# True = 评估该方法, False = 跳过
EVAL_SOAR = True      # Soar DSL 程序
EVAL_SAC = False          # SAC (需要先训练)
EVAL_TD3 = False          # TD3 (需要先训练)
EVAL_PID = True           # PID 控制器
EVAL_CPID = True          # 级联 PID 控制器
EVAL_LQR = True           # LQR 控制器

# --------------------- 模型/程序路径 ---------------------
SOAR_PATH = ROOT / "results" / "scg_aligned" / f"{TASK}_safe_control_tracking_best.json"
SAC_MODEL_PATH = ROOT / "03_SAC" / "checkpoints" / TASK / "sac_final.zip"
TD3_MODEL_PATH = ROOT / "04_TD3" / "checkpoints" / TASK / "td3_final.zip"

# 基线参数文件 (如果不存在则用默认参数)
PID_PARAMS_PATH = ROOT / "results" / "aligned_baselines" / f"pid_{TASK}.json"
CPID_PARAMS_PATH = ROOT / "results" / "aligned_baselines" / f"cpid_{TASK}.json"
LQR_PARAMS_PATH = ROOT / "results" / "aligned_baselines" / f"lqr_{TASK}.json"

# --------------------- 输出配置 ---------------------
OUTPUT_PATH = ROOT / "results" / "comparison" / f"all_methods_{TASK}.json"

# ============================================================================
#                         对比代码 (不需要修改)
# ============================================================================

def eval_soar(program_path, task, duration, num_envs):
    """评估 Soar 程序"""
    import json
    from utils.batch_evaluation import BatchEvaluator
    from utilities.trajectory_presets import get_scg_trajectory_config
    
    if not program_path.exists():
        return None
    
    with open(program_path, 'r') as f:
        data = json.load(f)
    program = data['rules'] if isinstance(data, dict) and 'rules' in data else data
    
    traj_cfg = get_scg_trajectory_config(task)
    # 计算 t=0 时刻轨迹上的位置作为初始位置
    from utilities.trajectory_presets import scg_position
    initial_pos = scg_position(traj_cfg.task, t=0.0, params=traj_cfg.params, center=traj_cfg.center)
    trajectory_config = {
        'type': traj_cfg.task,
        'params': dict(traj_cfg.params),
        'initial_xyz': initial_pos.tolist()
    }
    
    evaluator = BatchEvaluator(
        isaac_num_envs=num_envs,
        reward_profile='safe_control_tracking',
        trajectory_config=trajectory_config,
        duration=int(duration),
        device=DEVICE,
        use_fast_path=True,
        strict_no_prior=True,
        reward_reduction='sum',
        zero_action_penalty=0.0,
        replicas_per_program=1,
        enable_output_mad=False,
    )
    
    rewards_train, rewards_true, metrics_list = evaluator.evaluate_batch_with_metrics([program])
    
    # 关闭环境防止资源泄漏
    evaluator.close()
    
    return {
        'mean_reward': float(rewards_true[0]),
        'state_cost': float(metrics_list[0].get('state_cost', 0)) if metrics_list else 0,
        'action_cost': float(metrics_list[0].get('action_cost', 0)) if metrics_list else 0,
    }


def eval_rl_model(model_path, algo, task, duration, num_envs, episodes):
    """评估 SAC/TD3 模型"""
    import numpy as np
    from scg_vec_env import IsaacSCGVecEnv
    
    if not model_path.exists():
        return None
    
    if algo == 'SAC':
        from stable_baselines3 import SAC
        model = SAC.load(str(model_path), device=DEVICE)
    else:
        from stable_baselines3 import TD3
        model = TD3.load(str(model_path), device=DEVICE)
    
    env = IsaacSCGVecEnv(num_envs=num_envs, task=task, duration=duration, device=DEVICE)
    
    episode_rewards = []
    for ep in range(episodes):
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += float(reward.sum())
        episode_rewards.append(total_reward / num_envs)
    
    env.close()
    
    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'episodes': episodes,
    }


def eval_baseline(controller_type, params_path, task, duration, num_envs, episodes):
    """评估 PID/CPID/LQR 基线"""
    import json
    from scripts.baselines.tune_pid_lqr_isaac import (
        IsaacPIDController,
        IsaacLQRController,
        evaluate_params,
    )
    
    # 默认参数
    default_pid = {
        'kp_xy': 10.0, 'kd_xy': 5.0, 'ki_xy': 0.05,
        'kp_z': 18.0, 'kd_z': 8.0, 'ki_z': 0.1,
        'kp_att': 15.0, 'kd_att': 2.5,
        'kp_yaw': 5.0, 'kd_yaw': 1.0,
        'att_scale': 0.25,
    }
    default_lqr = {
        'k_pos': 4.0, 'k_vel': 4.0, 'k_att': 12.0, 'k_omega': 3.0,
        'k_yaw': 0.0, 'k_yaw_rate': 0.0, 'att_scale': 0.2,
    }
    
    # 加载参数
    params = None
    if params_path.exists():
        with open(params_path, 'r') as f:
            data = json.load(f)
        params = data.get('best_params')
    
    if params is None:
        params = default_lqr if controller_type == 'lqr' else default_pid
    
    # 创建控制器
    if controller_type == 'lqr':
        controller = IsaacLQRController(**params)
    else:
        controller = IsaacPIDController(**params)
    
    metrics = evaluate_params(controller, task, duration, episodes=episodes, num_envs=num_envs)
    
    return {
        'mean_reward': float(metrics['mean_true_reward']),
        'std_reward': float(metrics['std_true_reward']),
        'rmse_pos': float(metrics['rmse_pos']),
    }


def main():
    import json
    from datetime import datetime
    
    print("=" * 70)
    print(f"全方法统一对比 - 任务: {TASK}")
    print("=" * 70)
    print(f"  并行环境: {NUM_ENVS}")
    print(f"  时长: {DURATION}s")
    print(f"  设备: {DEVICE}")
    print()
    
    results = {
        'task': TASK,
        'duration_sec': DURATION,
        'num_envs': NUM_ENVS,
        'episodes': EPISODES,
        'reward_type': 'scg_exact',
        'timestamp': datetime.now().isoformat(),
        'methods': {},
    }
    
    # 1. Soar
    if EVAL_SOAR:
        print("[1/6] 评估 Soar...")
        pf_result = eval_soar(SOAR_PATH, TASK, DURATION, NUM_ENVS)
        if pf_result:
            results['methods']['Soar'] = pf_result
            print(f"      ✓ 奖励: {pf_result['mean_reward']:.2f}")
        else:
            print(f"      ✗ 程序不存在: {SOAR_PATH}")
    
    # 2. SAC
    if EVAL_SAC:
        print("[2/6] 评估 SAC...")
        sac_result = eval_rl_model(SAC_MODEL_PATH, 'SAC', TASK, DURATION, NUM_ENVS, EPISODES)
        if sac_result:
            results['methods']['SAC'] = sac_result
            print(f"      ✓ 奖励: {sac_result['mean_reward']:.2f} ± {sac_result['std_reward']:.2f}")
        else:
            print(f"      ✗ 模型不存在: {SAC_MODEL_PATH}")
    
    # 3. TD3
    if EVAL_TD3:
        print("[3/6] 评估 TD3...")
        td3_result = eval_rl_model(TD3_MODEL_PATH, 'TD3', TASK, DURATION, NUM_ENVS, EPISODES)
        if td3_result:
            results['methods']['TD3'] = td3_result
            print(f"      ✓ 奖励: {td3_result['mean_reward']:.2f} ± {td3_result['std_reward']:.2f}")
        else:
            print(f"      ✗ 模型不存在: {TD3_MODEL_PATH}")
    
    # 4. PID
    if EVAL_PID:
        print("[4/6] 评估 PID...")
        pid_result = eval_baseline('pid', PID_PARAMS_PATH, TASK, DURATION, NUM_ENVS, EPISODES)
        if pid_result:
            results['methods']['PID'] = pid_result
            print(f"      ✓ 奖励: {pid_result['mean_reward']:.2f} | RMSE: {pid_result['rmse_pos']:.3f}m")
    
    # 5. CPID
    if EVAL_CPID:
        print("[5/6] 评估 CPID...")
        cpid_result = eval_baseline('cpid', CPID_PARAMS_PATH, TASK, DURATION, NUM_ENVS, EPISODES)
        if cpid_result:
            results['methods']['CPID'] = cpid_result
            print(f"      ✓ 奖励: {cpid_result['mean_reward']:.2f} | RMSE: {cpid_result['rmse_pos']:.3f}m")
    
    # 6. LQR
    if EVAL_LQR:
        print("[6/6] 评估 LQR...")
        lqr_result = eval_baseline('lqr', LQR_PARAMS_PATH, TASK, DURATION, NUM_ENVS, EPISODES)
        if lqr_result:
            results['methods']['LQR'] = lqr_result
            print(f"      ✓ 奖励: {lqr_result['mean_reward']:.2f} | RMSE: {lqr_result['rmse_pos']:.3f}m")
    
    # 汇总
    print()
    print("=" * 70)
    print("对比结果汇总")
    print("=" * 70)
    print(f"{'方法':<15} {'奖励 (↑ 越大越好)':<25} {'备注':<20}")
    print("-" * 70)
    
    for method, data in results['methods'].items():
        reward = data.get('mean_reward', 0)
        std = data.get('std_reward', 0)
        rmse = data.get('rmse_pos')
        
        if std > 0:
            reward_str = f"{reward:.2f} ± {std:.2f}"
        else:
            reward_str = f"{reward:.2f}"
        
        note = f"RMSE={rmse:.3f}m" if rmse else ""
        print(f"{method:<15} {reward_str:<25} {note:<20}")
    
    print("=" * 70)
    
    # 保存结果
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 对比结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

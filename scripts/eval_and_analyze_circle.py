#!/usr/bin/env python3
"""
评估程序并记录详细轨迹，然后进行非线性分析

可通过命令行指定任务类型和程序路径，默认使用 figure8 任务。
"""
import sys
import argparse
from pathlib import Path

# 添加路径（在导入任何其他模块前）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '01_soar'))
sys.path.insert(0, str(ROOT))

# Isaac Gym: 先尝试系统已安装版本，失败再尝试 vendor 路径（避免用 py36 绑定污染 py38）
_ISAAC_GYM_PY = ROOT / 'isaacgym' / 'python'
gym_loaded = False
try:
    from isaacgym import gymapi  # type: ignore
    gym_loaded = True
except Exception:
    if _ISAAC_GYM_PY.exists():
        sys.path.insert(0, str(_ISAAC_GYM_PY))
        try:
            from isaacgym import gymapi  # type: ignore
            gym_loaded = True
        except Exception:
            gym_loaded = False

# 现在可以导入其他模块
import json
import numpy as np
import torch

print("正在加载模块...")
from core.serialization import deserialize_program
from utils.batch_evaluation import BatchEvaluator
from utilities.trajectory_presets import get_scg_trajectory_config

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate program and plot trajectory")
    parser.add_argument("--task", default="figure8", help="trajectory task (figure8/circle/square/helix/hover)")
    parser.add_argument("--program", default=None, help="path to saved program json; default results/<task>_safe_control_tracking_best.json")
    parser.add_argument("--duration", type=float, default=5.0, help="episode duration in seconds")
    parser.add_argument("--device", default="cuda:0", help="compute device for Isaac Gym")
    parser.add_argument("--outdir", default=None, help="output directory for plots; default results/plots/<task>")
    return parser.parse_args()


def main():
    args = parse_args()
    TASK = args.task
    DURATION = float(args.duration)
    NUM_ENVS = 1  # 只用1个环境以便详细记录
    DEVICE = args.device
    program_path = Path(args.program) if args.program else ROOT / "results" / "soar_train" / f"{TASK}_safe_control_tracking_best.json"
    LOG_PATH = ROOT / "results" / f"{TASK}_trajectory.csv"
    PLOT_DIR = Path(args.outdir) if args.outdir else (ROOT / "results" / "plots" / TASK)

    print("=" * 70)
    print(f"{TASK} 程序评估与非线性分析")
    print("=" * 70)
    
    # 加载程序
    with open(program_path, 'r') as f:
        data = json.load(f)
    program = deserialize_program(data if isinstance(data, dict) and 'rules' in data else {'rules': data})
    print(f"✓ 程序加载: {len(program)} 条规则")
    
    # 构建轨迹配置
    traj_cfg = get_scg_trajectory_config(TASK)
    # 计算 t=0 时刻轨迹上的位置作为初始位置
    from utilities.trajectory_presets import scg_position
    initial_pos = scg_position(traj_cfg.task, t=0.0, params=traj_cfg.params, center=traj_cfg.center)
    trajectory_config = {
        'type': traj_cfg.task,
        'params': dict(traj_cfg.params),
        'initial_xyz': initial_pos.tolist()
    }
    
    # 创建评估器（单环境详细记录模式）
    print("\n初始化评估器...")
    evaluator = BatchEvaluator(
        isaac_num_envs=NUM_ENVS,
        reward_profile='safe_control_tracking',
        trajectory_config=trajectory_config,
        duration=DURATION,
        device=DEVICE,
        use_fast_path=False,  # 关闭快速路径以便记录
        strict_no_prior=True,
        reward_reduction='sum',
        zero_action_penalty=0.0,
        replicas_per_program=1,
        enable_output_mad=False,
        min_steps_frac=1.0,  # ⚠️ 强制运行完整episode（240步），禁用early stopping
    )
    
    # 使用evaluator内部方法记录轨迹（hook到evaluate_batch）
    print("\n开始评估并记录轨迹...")
    
    # Monkey patch记录hook
    trajectory_data = []
    original_step = None
    
    def hook_step(self_env, actions):
        """Hook step方法记录数据"""
        # 调用原始step获取结果
        result = original_step(actions)
        
        # 在step后记录状态和动作
        if isinstance(result, tuple) and len(result) >= 4:
            obs_dict, rewards, dones, infos = result
            
            if len(trajectory_data) < 1000:  # 限制记录数量
                pos = obs_dict['position'][0].cpu().numpy() if torch.is_tensor(obs_dict['position']) else obs_dict['position'][0]
                vel = obs_dict['velocity'][0].cpu().numpy() if torch.is_tensor(obs_dict['velocity']) else obs_dict['velocity'][0]
                omega = obs_dict['angular_velocity'][0].cpu().numpy() if torch.is_tensor(obs_dict['angular_velocity']) else obs_dict['angular_velocity'][0]
                
                # 计算目标和误差
                t = len(trajectory_data) / 48.0
                tgt = evaluator._target_pos(t)
                pos_err = tgt - pos
                
                # 从动作张量提取
                act = actions[0].cpu().numpy() if torch.is_tensor(actions) else actions[0]
                
                trajectory_data.append({
                    'time': t,
                    'pos_x': float(pos[0]),
                    'pos_y': float(pos[1]),
                    'pos_z': float(pos[2]),
                    'pos_err_x': float(pos_err[0]),
                    'pos_err_y': float(pos_err[1]),
                    'pos_err_z': float(pos_err[2]),
                    'vel_x': float(vel[0]),
                    'vel_y': float(vel[1]),
                    'vel_z': float(vel[2]),
                    'ang_vel_x': float(omega[0]),
                    'ang_vel_y': float(omega[1]),
                    'ang_vel_z': float(omega[2]),
                    'u_fz': float(act[2]) if len(act) > 2 else 0.0,
                    'u_tx': float(act[3]) if len(act) > 3 else 0.0,
                    'u_ty': float(act[4]) if len(act) > 4 else 0.0,
                    'u_tz': float(act[5]) if len(act) > 5 else 0.0,
                    'tgt_x': float(tgt[0]),
                    'tgt_y': float(tgt[1]),
                    'tgt_z': float(tgt[2]),
                })
        
        return result
    
    # 安装hook
    evaluator._init_isaac_gym_pool()
    env_pool = evaluator._isaac_env_pool
    original_step = env_pool.step
    env_pool.step = lambda actions: hook_step(env_pool, actions)
    
    # 运行评估（这会触发hook记录）
    print(f"运行评估...")
    rewards = evaluator.evaluate_batch([program])
    print(f"  奖励: {rewards[0]:.4f}")
    
    # 恢复原始step
    env_pool.step = original_step
    
    # 保存轨迹数据
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(trajectory_data)
    df.to_csv(LOG_PATH, index=False)
    print(f"\n✓ 轨迹数据已保存: {LOG_PATH}")
    print(f"  共 {len(df)} 个数据点")
    
    # 运行非线性分析
    print("\n" + "=" * 70)
    print("非线性分析")
    print("=" * 70)
    
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    
    # 1. 相平面：高度通道
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(df['pos_err_z'], df['vel_z'], linewidth=0.8, alpha=0.8)
    ax.set_xlabel('pos_err_z (m)')
    ax.set_ylabel('vel_z (m/s)')
    ax.set_title('Phase Plane: Z-axis')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'phase_z.png', dpi=150)
    plt.close()
    print(f"✓ 相平面图 (Z轴): {PLOT_DIR}/phase_z.png")
    
    # 2. 相平面：Y轴
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(df['pos_err_y'], df['vel_y'], linewidth=0.8, alpha=0.8)
    ax.set_xlabel('pos_err_y (m)')
    ax.set_ylabel('vel_y (m/s)')
    ax.set_title('Phase Plane: Y-axis')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'phase_y.png', dpi=150)
    plt.close()
    print(f"✓ 相平面图 (Y轴): {PLOT_DIR}/phase_y.png")
    
    # 3. u_fz 饱和特性
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['pos_err_z'], df['u_fz'], s=5, alpha=0.5)
    ax.axhline(0.265 - 2.0, color='r', linestyle='--', linewidth=1, label='下饱和限 (0.265-2)')
    ax.axhline(0.265 + 2.0, color='r', linestyle='--', linewidth=1, label='上饱和限 (0.265+2)')
    ax.axhline(0.265, color='g', linestyle=':', linewidth=1, label='悬停推力')
    ax.set_xlabel('pos_err_z (m)')
    ax.set_ylabel('u_fz (N)')
    ax.set_title('Thrust Saturation: u_fz vs pos_err_z')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'ufz_saturation.png', dpi=150)
    plt.close()
    print(f"✓ 推力饱和图: {PLOT_DIR}/ufz_saturation.png")
    
    # 4. u_ty 非线性（平方项）
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['ang_vel_y'], df['u_ty'], s=5, alpha=0.5)
    ax.set_xlabel('ang_vel_y (rad/s)')
    ax.set_ylabel('u_ty (rad/s)')
    ax.set_title('Pitch Nonlinearity: u_ty vs ang_vel_y (square term)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'uty_nonlinear.png', dpi=150)
    plt.close()
    print(f"✓ 俯仰非线性图: {PLOT_DIR}/uty_nonlinear.png")
    
    # 5. u_tz 除法奇异性
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['vel_y'], df['u_tz'], s=5, alpha=0.5, c=np.abs(df['pos_err_y']), cmap='viridis')
    ax.axvline(0, color='r', linestyle='--', linewidth=1, label='vel_y=0 (奇异面)')
    ax.set_xlabel('vel_y (m/s)')
    ax.set_ylabel('u_tz (rad/s)')
    ax.set_title('Yaw Division Singularity: u_tz vs vel_y')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('|pos_err_y|', fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'utz_singularity.png', dpi=150)
    plt.close()
    print(f"✓ 偏航奇异性图: {PLOT_DIR}/utz_singularity.png")
    
    # 6. 时域波形：控制输出
    fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(df['time'], df['u_fz'], linewidth=0.8)
    axes[0].set_ylabel('u_fz (N)')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df['time'], df['u_tx'], linewidth=0.8)
    axes[1].set_ylabel('u_tx (rad/s)')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(df['time'], df['u_ty'], linewidth=0.8)
    axes[2].set_ylabel('u_ty (rad/s)')
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(df['time'], df['u_tz'], linewidth=0.8)
    axes[3].set_ylabel('u_tz (rad/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)
    fig.suptitle('Control Signals Time Series')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'controls_time.png', dpi=150)
    plt.close()
    print(f"✓ 控制信号时域图: {PLOT_DIR}/controls_time.png")
    
    # 7. 位置误差
    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(df['time'], df['pos_err_x'], linewidth=0.8)
    axes[0].set_ylabel('pos_err_x (m)')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df['time'], df['pos_err_y'], linewidth=0.8)
    axes[1].set_ylabel('pos_err_y (m)')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(df['time'], df['pos_err_z'], linewidth=0.8)
    axes[2].set_ylabel('pos_err_z (m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)
    fig.suptitle('Position Errors Time Series')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'pos_err_time.png', dpi=150)
    plt.close()
    print(f"✓ 位置误差时域图: {PLOT_DIR}/pos_err_time.png")
    
    # 8. 3D 轨迹
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['pos_x'], df['pos_y'], df['pos_z'], linewidth=1, label='Actual', alpha=0.8)
    ax.plot(df['tgt_x'], df['tgt_y'], df['tgt_z'], linewidth=1, linestyle='--', label='Target', alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'trajectory_3d.png', dpi=150)
    plt.close()
    print(f"✓ 3D轨迹图: {PLOT_DIR}/trajectory_3d.png")
    
    # 统计分析
    print("\n" + "=" * 70)
    print("统计分析")
    print("=" * 70)
    
    print(f"位置误差 RMSE:")
    print(f"  X: {np.sqrt(np.mean(df['pos_err_x']**2)):.4f} m")
    print(f"  Y: {np.sqrt(np.mean(df['pos_err_y']**2)):.4f} m")
    print(f"  Z: {np.sqrt(np.mean(df['pos_err_z']**2)):.4f} m")
    
    print(f"\n控制输出统计:")
    for u in ['u_fz', 'u_tx', 'u_ty', 'u_tz']:
        print(f"  {u}: 均值={df[u].mean():.4f}, 标准差={df[u].std():.4f}, 范围=[{df[u].min():.4f}, {df[u].max():.4f}]")
    
    # 检查饱和
    fz_saturated = np.sum((df['u_fz'] <= 0.265 - 1.99) | (df['u_fz'] >= 0.265 + 1.99))
    print(f"\nu_fz 饱和情况: {fz_saturated}/{len(df)} 步 ({fz_saturated/len(df)*100:.1f}%)")
    
    # 检查奇异点
    vel_y_near_zero = np.sum(np.abs(df['vel_y']) < 0.1)
    print(f"vel_y 接近零点: {vel_y_near_zero}/{len(df)} 步 ({vel_y_near_zero/len(df)*100:.1f}%)")
    
    print("\n✓ 分析完成！所有图表已保存到:", PLOT_DIR.resolve())
    
    # 清理
    env_pool.close()

if __name__ == "__main__":
    main()

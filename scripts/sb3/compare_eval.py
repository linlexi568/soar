"""对比评估脚本：SB3 模型 vs Soar 程序。

在相同的 SB3 环境中对比评估不同方法的性能。

使用示例:
    # 对比 PPO 和 Soar
    python scripts/sb3/compare_eval.py \
        --sb3-model results/sb3/ppo_figure8_final.zip \
        --soar-program results/figure8-safe_control_tracking.json \
        --trajectory figure8 \
        --episodes 20
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 添加项目路径
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "01_soar"))

from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv
from scripts.sb3.soar_policy import SoarPolicy, load_soar_program


def evaluate_policy(
    policy,
    env: QuadrotorTrackingEnv,
    n_episodes: int = 10,
    policy_name: str = "Policy",
    is_sb3: bool = False,
) -> Dict[str, Any]:
    """通用策略评估函数。
    
    Args:
        policy: 策略对象（SB3 模型或 SoarPolicy）
        env: 评估环境
        n_episodes: 评估 episode 数
        policy_name: 策略名称（用于显示）
        is_sb3: 是否是 SB3 模型
    
    Returns:
        评估结果
    """
    episode_rewards = []
    episode_lengths = []
    episode_pos_errors = []
    episode_final_pos_errors = []
    episode_trajectories = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        if hasattr(policy, 'reset'):
            policy.reset()
        
        total_reward = 0.0
        steps = 0
        pos_errors = []
        trajectory = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # 获取动作
            if is_sb3:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = policy.predict(obs)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if 'pos_error' in info:
                pos_errors.append(info['pos_error'])
            
            if 'pos' in info:
                trajectory.append(info['pos'].copy())
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if pos_errors:
            episode_pos_errors.append(np.mean(pos_errors))
            episode_final_pos_errors.append(pos_errors[-1])
        episode_trajectories.append(trajectory)
    
    results = {
        'name': policy_name,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_pos_error': float(np.mean(episode_pos_errors)) if episode_pos_errors else None,
        'mean_final_pos_error': float(np.mean(episode_final_pos_errors)) if episode_final_pos_errors else None,
        'n_episodes': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }
    
    return results


def compare_methods(
    sb3_model_path: Optional[str] = None,
    soar_program_path: Optional[str] = None,
    trajectory: str = 'figure8',
    n_episodes: int = 10,
    save_results: bool = True,
    output_dir: str = 'results/compare',
) -> Dict[str, Any]:
    """对比评估 SB3 模型和 Soar 程序。
    
    Args:
        sb3_model_path: SB3 模型路径
        soar_program_path: Soar 程序路径
        trajectory: 轨迹类型
        n_episodes: 评估 episode 数
        save_results: 是否保存结果
        output_dir: 输出目录
    
    Returns:
        对比结果
    """
    # 创建环境
    env = QuadrotorTrackingEnv(
        trajectory=trajectory,
        duration=5.0,
        control_freq=50,
    )
    
    results = {
        'trajectory': trajectory,
        'n_episodes': n_episodes,
        'timestamp': datetime.now().isoformat(),
        'methods': {},
    }
    
    # 评估 SB3 模型
    if sb3_model_path:
        try:
            from stable_baselines3 import PPO, SAC, TD3, A2C
            
            # 根据文件名猜测算法
            model_name = Path(sb3_model_path).stem.lower()
            if 'ppo' in model_name:
                AlgoClass = PPO
            elif 'sac' in model_name:
                AlgoClass = SAC
            elif 'td3' in model_name:
                AlgoClass = TD3
            elif 'a2c' in model_name:
                AlgoClass = A2C
            else:
                AlgoClass = PPO
            
            print(f"\n📦 加载 SB3 模型: {sb3_model_path}")
            sb3_model = AlgoClass.load(sb3_model_path)
            
            print(f"🔄 评估 SB3 模型...")
            sb3_results = evaluate_policy(
                sb3_model, env, n_episodes,
                policy_name=f"SB3-{AlgoClass.__name__}",
                is_sb3=True,
            )
            results['methods']['sb3'] = sb3_results
            
        except ImportError:
            print("⚠️ stable-baselines3 未安装，跳过 SB3 评估")
        except Exception as e:
            print(f"⚠️ SB3 模型加载失败: {e}")
    
    # 评估 Soar 程序
    if soar_program_path:
        try:
            print(f"\n📦 加载 Soar 程序: {soar_program_path}")
            program = load_soar_program(soar_program_path)
            policy = SoarPolicy(program, dt=1.0/50.0)
            
            print(f"🔄 评估 Soar 程序...")
            soar_results = evaluate_policy(
                policy, env, n_episodes,
                policy_name="Soar",
                is_sb3=False,
            )
            results['methods']['soar'] = soar_results
            
        except Exception as e:
            print(f"⚠️ Soar 程序加载失败: {e}")
    
    # 评估 PID 基线
    print(f"\n📦 评估 PID 基线...")
    pid_policy = SimplePIDPolicy()
    pid_results = evaluate_policy(
        pid_policy, env, n_episodes,
        policy_name="PID-Baseline",
        is_sb3=False,
    )
    results['methods']['pid'] = pid_results
    
    env.close()
    
    # 打印对比结果
    print_comparison(results)
    
    # 保存结果
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"compare_{trajectory}_{timestamp}.json"
        
        # 移除不可序列化的数据
        save_results_data = {
            k: {kk: vv for kk, vv in v.items() if kk != 'episode_rewards' and kk != 'episode_lengths'}
            if isinstance(v, dict) else v
            for k, v in results.items()
        }
        
        with open(result_file, 'w') as f:
            json.dump(save_results_data, f, indent=2, default=str)
        print(f"\n✅ 结果已保存: {result_file}")
    
    return results


def print_comparison(results: Dict[str, Any]):
    """打印对比结果表格"""
    print("\n" + "=" * 70)
    print(f"📊 对比评估结果 - {results['trajectory']} 轨迹")
    print("=" * 70)
    
    methods = results.get('methods', {})
    if not methods:
        print("没有可用的评估结果")
        return
    
    # 表头
    print(f"\n{'方法':<20} {'Mean Reward':>12} {'Std':>8} {'Pos Error':>12} {'Final Err':>12}")
    print("-" * 70)
    
    # 按 reward 排序
    sorted_methods = sorted(
        methods.items(),
        key=lambda x: x[1].get('mean_reward', float('-inf')),
        reverse=True,
    )
    
    for name, data in sorted_methods:
        mean_r = data.get('mean_reward', 0)
        std_r = data.get('std_reward', 0)
        pos_err = data.get('mean_pos_error')
        final_err = data.get('mean_final_pos_error')
        
        pos_err_str = f"{pos_err:.4f}" if pos_err is not None else "N/A"
        final_err_str = f"{final_err:.4f}" if final_err is not None else "N/A"
        
        print(f"{data['name']:<20} {mean_r:>12.2f} {std_r:>8.2f} {pos_err_str:>12} {final_err_str:>12}")
    
    print("-" * 70)
    
    # 计算相对性能
    if len(sorted_methods) > 1:
        best_name, best_data = sorted_methods[0]
        print(f"\n🏆 最佳方法: {best_data['name']}")
        
        for name, data in sorted_methods[1:]:
            diff = best_data['mean_reward'] - data['mean_reward']
            pct = (diff / abs(data['mean_reward'])) * 100 if data['mean_reward'] != 0 else 0
            print(f"   vs {data['name']}: +{diff:.2f} ({pct:+.1f}%)")


class SimplePIDPolicy:
    """简单 PID 控制器作为基线。"""
    
    def __init__(
        self,
        kp_pos: float = 2.0,
        kd_pos: float = 1.0,
        kp_att: float = 5.0,
        kd_att: float = 1.0,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_att = kp_att
        self.kd_att = kd_att
        
        # 动作缩放
        self.ACTION_SCALE = np.array([3.5, 0.12, 0.12, 0.06], dtype=np.float32)
        self.ACTION_OFFSET = np.array([3.5, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def reset(self):
        pass
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).flatten()
        
        # 解析观测
        pos_err = obs[0:3]    # 位置误差
        vel = obs[3:6]        # 速度
        euler = obs[6:9]      # 欧拉角
        omega = obs[9:12]     # 角速度
        
        # 高度控制 (fz)
        fz = self.kp_pos * pos_err[2] - self.kd_pos * vel[2]
        fz += 0.265  # 悬停补偿
        
        # 姿态控制
        # 期望姿态来自位置误差
        desired_roll = -self.kp_pos * pos_err[1]
        desired_pitch = self.kp_pos * pos_err[0]
        
        roll_err = desired_roll - euler[0]
        pitch_err = desired_pitch - euler[1]
        yaw_err = -euler[2]  # 保持零偏航
        
        tx = self.kp_att * roll_err - self.kd_att * omega[0]
        ty = self.kp_att * pitch_err - self.kd_att * omega[1]
        tz = 0.5 * yaw_err - 0.1 * omega[2]
        
        # 物理输出
        raw_action = np.array([fz, tx, ty, tz], dtype=np.float32)
        
        # 归一化
        normalized = (raw_action - self.ACTION_OFFSET) / self.ACTION_SCALE
        return np.clip(normalized, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description='对比评估 SB3 和 Soar')
    
    parser.add_argument('--sb3-model', type=str, help='SB3 模型路径')
    parser.add_argument('--soar-program', type=str, help='Soar 程序路径')
    parser.add_argument('--trajectory', type=str, default='figure8',
                       choices=['hover', 'figure8', 'circle'],
                       help='轨迹类型')
    parser.add_argument('--episodes', type=int, default=10, help='评估 episode 数')
    parser.add_argument('--output-dir', type=str, default='results/compare',
                       help='输出目录')
    parser.add_argument('--no-save', action='store_true', help='不保存结果')
    
    args = parser.parse_args()
    
    compare_methods(
        sb3_model_path=args.sb3_model,
        soar_program_path=args.soar_program,
        trajectory=args.trajectory,
        n_episodes=args.episodes,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()

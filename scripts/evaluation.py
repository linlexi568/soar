"""Soar 标准评估框架。

提供与 SB3 Zoo / OpenAI Spinning Up 一致的评估方法，
确保论文中的 baseline 对比公平且可复现。

评估标准（对齐 OpenAI / SB3）：
- 10 random seeds
- 确定性评估（no exploration noise）
- 报告 mean ± std

物理指标（Quadrotor 专用）：
- Position RMSE (m)
- Max Deviation (m)
- Velocity RMSE (m/s)
- Control Effort (∑|u|²)
- Success Rate (%)
- Settling Time (s)

使用方式:
    from scripts.evaluation import StandardEvaluator
    
    evaluator = StandardEvaluator(trajectory='figure8', n_seeds=10)
    
    # 评估 SB3 模型
    sb3_results = evaluator.evaluate_sb3_model('results/sb3/ppo_figure8.zip')
    
    # 评估 Soar 程序
    soar_results = evaluator.evaluate_soar_program('results/best_program.json')
    
    # 生成对比表格
    evaluator.generate_comparison_table([sb3_results, soar_results])
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 添加项目路径
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "01_soar"))


@dataclass
class EvalMetrics:
    """评估指标数据类"""
    # 物理指标
    position_rmse: float = 0.0       # 位置 RMSE (m)
    position_rmse_std: float = 0.0
    max_deviation: float = 0.0       # 最大偏差 (m)
    max_deviation_std: float = 0.0
    velocity_rmse: float = 0.0       # 速度 RMSE (m/s)
    velocity_rmse_std: float = 0.0
    
    # 控制指标
    control_effort: float = 0.0      # 控制代价 (∑|u|²)
    control_effort_std: float = 0.0
    smoothness: float = 0.0          # 平滑度 (jerk)
    smoothness_std: float = 0.0
    
    # 任务指标
    success_rate: float = 0.0        # 成功率 (%)
    settling_time: float = 0.0       # 稳定时间 (s)
    settling_time_std: float = 0.0
    episode_return: float = 0.0      # Episode Return (用于 SB3 对比)
    episode_return_std: float = 0.0
    
    # 元信息
    n_episodes: int = 0
    n_seeds: int = 0
    method_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_rmse': f"{self.position_rmse:.4f} ± {self.position_rmse_std:.4f}",
            'max_deviation': f"{self.max_deviation:.4f} ± {self.max_deviation_std:.4f}",
            'velocity_rmse': f"{self.velocity_rmse:.4f} ± {self.velocity_rmse_std:.4f}",
            'control_effort': f"{self.control_effort:.2f} ± {self.control_effort_std:.2f}",
            'smoothness': f"{self.smoothness:.4f} ± {self.smoothness_std:.4f}",
            'success_rate': f"{self.success_rate:.1f}%",
            'settling_time': f"{self.settling_time:.3f} ± {self.settling_time_std:.3f}",
            'episode_return': f"{self.episode_return:.1f} ± {self.episode_return_std:.1f}",
            'n_episodes': self.n_episodes,
            'n_seeds': self.n_seeds,
            'method': self.method_name,
        }


# =============================================================================
# OpenAI / SB3 标准评估配置
# =============================================================================

OPENAI_SPINUP_CONFIG = {
    'n_seeds': 10,
    'total_timesteps': 3_000_000,
    'eval_freq': 10_000,
    'n_eval_episodes': 10,
    'deterministic': True,  # 评估时不加噪声
    'network_on_policy': [64, 32],   # PPO/A2C
    'network_off_policy': [256, 256],  # SAC/TD3
    'activation_on_policy': 'tanh',
    'activation_off_policy': 'relu',
}

SB3_ZOO_CONFIG = {
    'n_seeds': 5,
    'total_timesteps': 1_000_000,
    'eval_freq': 10_000,
    'n_eval_episodes': 50,
    'deterministic': True,
}

# 论文推荐配置（平衡精度和计算成本）
PAPER_CONFIG = {
    'n_seeds': 10,
    'n_eval_episodes': 20,
    'deterministic': True,
    'report_mean_std': True,
}


# =============================================================================
# SB3 Zoo Benchmark 参考数据
# =============================================================================

SB3_ZOO_BENCHMARKS = {
    # MuJoCo 环境 (训练 1M steps)
    'mujoco': {
        'HalfCheetah-v3': {
            'PPO': (5819, 664),
            'SAC': (9535, 100),
            'TD3': (9656, 970),
            'TQC': (12090, 127),
        },
        'Hopper-v3': {
            'PPO': (2410, 10),
            'SAC': (2326, 1130),
            'TD3': (3606, 4),
            'TQC': (3754, 8),
        },
        'Walker2d-v3': {
            'PPO': (3479, 822),
            'SAC': (3863, 254),
            'TD3': (4718, 46),
            'TQC': (4381, 500),
        },
        'Ant-v3': {
            'PPO': (1327, 452),
            'SAC': (4616, 1354),
            'TD3': (5813, 590),
        },
    },
    # PyBullet 环境（免费 MuJoCo 替代）
    'pybullet': {
        'HalfCheetahBulletEnv-v0': {
            'PPO': (2925, 64),
            'SAC': (2792, 12),
            'TD3': (2822, 20),
        },
        'HopperBulletEnv-v0': {
            'PPO': (2575, 223),
            'SAC': (2603, 164),
            'TD3': (2682, 28),
        },
        'Walker2DBulletEnv-v0': {
            'PPO': (2110, 14),
            'SAC': (2292, 14),
            'TD3': (2214, 231),
        },
        'AntBulletEnv-v0': {
            'PPO': (2866, 56),
            'SAC': (3073, 175),
            'TD3': (3300, 55),
        },
    },
}


class StandardEvaluator:
    """标准评估器：对齐 OpenAI / SB3 评估方法。
    
    提供公平、可复现的 baseline 对比。
    """
    
    def __init__(
        self,
        trajectory: str = 'figure8',
        duration: float = 5.0,
        n_seeds: int = 10,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        reward_type: str = 'pybullet_drones',
        device: str = 'cuda:0',
    ):
        """
        Args:
            trajectory: 轨迹类型
            duration: Episode 时长
            n_seeds: 随机种子数量（对齐 OpenAI 标准）
            n_eval_episodes: 每个 seed 的评估 episode 数
            deterministic: 是否使用确定性策略（无探索噪声）
            reward_type: Reward 类型
            device: 计算设备
        """
        self.trajectory = trajectory
        self.duration = duration
        self.n_seeds = n_seeds
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.reward_type = reward_type
        self.device = device
        
        # 控制频率
        self.control_freq = 50
        self.dt = 1.0 / self.control_freq
        
        # 成功判定阈值
        self.success_threshold = 0.1  # 位置误差 < 0.1m 视为成功
        self.settling_threshold = 0.05  # 稳定判定阈值
    
    def evaluate_sb3_model(
        self,
        model_path: str,
        algo: str = 'ppo',
    ) -> EvalMetrics:
        """评估 SB3 训练的模型。
        
        Args:
            model_path: 模型文件路径 (.zip)
            algo: 算法类型
        
        Returns:
            评估指标
        """
        try:
            from stable_baselines3 import PPO, SAC, TD3
            from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv
        except ImportError as e:
            print(f"❌ SB3 未安装: {e}")
            return EvalMetrics(method_name=f"SB3-{algo.upper()}")
        
        algo_map = {'ppo': PPO, 'sac': SAC, 'td3': TD3}
        if algo.lower() not in algo_map:
            print(f"❌ 不支持的算法: {algo}")
            return EvalMetrics(method_name=f"SB3-{algo.upper()}")
        
        AlgoClass = algo_map[algo.lower()]
        
        # 收集所有 seed 的结果
        all_results = []
        
        for seed in range(self.n_seeds):
            # 创建环境
            env = QuadrotorTrackingEnv(
                trajectory=self.trajectory,
                duration=self.duration,
                control_freq=self.control_freq,
            )
            
            # 加载模型
            model = AlgoClass.load(model_path, env=env)
            
            # 评估
            seed_results = self._evaluate_policy(
                env=env,
                predict_fn=lambda obs: model.predict(obs, deterministic=self.deterministic)[0],
                seed=seed,
            )
            all_results.append(seed_results)
            env.close()
        
        # 聚合结果
        metrics = self._aggregate_results(all_results, f"SB3-{algo.upper()}")
        return metrics
    
    def evaluate_soar_program(
        self,
        program_path: str,
    ) -> EvalMetrics:
        """评估 Soar 生成的控制程序。
        
        Args:
            program_path: 程序 JSON 文件路径
        
        Returns:
            评估指标
        """
        try:
            from scripts.sb3.soar_policy import SoarPolicy
            from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            return EvalMetrics(method_name="Soar")
        
        # 加载程序
        policy = SoarPolicy(program_path)
        
        # 收集所有 seed 的结果
        all_results = []
        
        for seed in range(self.n_seeds):
            env = QuadrotorTrackingEnv(
                trajectory=self.trajectory,
                duration=self.duration,
                control_freq=self.control_freq,
            )
            
            seed_results = self._evaluate_policy(
                env=env,
                predict_fn=policy.predict,
                seed=seed,
                reset_fn=policy.reset,
            )
            all_results.append(seed_results)
            env.close()
        
        metrics = self._aggregate_results(all_results, "Soar")
        return metrics
    
    def evaluate_pid_baseline(self) -> EvalMetrics:
        """评估 PID baseline。"""
        try:
            from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            return EvalMetrics(method_name="PID")
        
        # 简单 PD 控制器
        kp_pos = 2.0
        kd_pos = 0.5
        
        def pid_predict(obs):
            pos_err = obs[:3]
            vel = obs[3:6]
            
            # PD 控制
            fz = 0.265 + kp_pos * pos_err[2] - kd_pos * vel[2]  # hover + z control
            tx = kp_pos * pos_err[1] - kd_pos * vel[1]  # roll -> y
            ty = -kp_pos * pos_err[0] + kd_pos * vel[0]  # pitch -> x
            tz = 0.0  # yaw
            
            # 归一化
            action = np.array([
                (fz - 3.5) / 3.5,
                tx / 0.12,
                ty / 0.12,
                tz / 0.06,
            ], dtype=np.float32)
            return np.clip(action, -1, 1)
        
        all_results = []
        for seed in range(self.n_seeds):
            env = QuadrotorTrackingEnv(
                trajectory=self.trajectory,
                duration=self.duration,
                control_freq=self.control_freq,
            )
            seed_results = self._evaluate_policy(env, pid_predict, seed)
            all_results.append(seed_results)
            env.close()
        
        return self._aggregate_results(all_results, "PID")
    
    def _evaluate_policy(
        self,
        env,
        predict_fn,
        seed: int,
        reset_fn=None,
    ) -> Dict[str, List[float]]:
        """评估单个 policy 在一个 seed 下的表现。
        
        Returns:
            各指标的列表（每个 episode 一个值）
        """
        np.random.seed(seed)
        
        results = {
            'position_rmse': [],
            'max_deviation': [],
            'velocity_rmse': [],
            'control_effort': [],
            'smoothness': [],
            'episode_return': [],
            'success': [],
            'settling_time': [],
        }
        
        for ep in range(self.n_eval_episodes):
            obs, info = env.reset(seed=seed * 1000 + ep)
            if reset_fn:
                reset_fn()
            
            pos_errors = []
            vel_errors = []
            actions = []
            rewards = []
            done = False
            truncated = False
            
            while not (done or truncated):
                action = predict_fn(obs)
                obs, reward, done, truncated, info = env.step(action)
                
                pos_errors.append(np.linalg.norm(obs[:3]))
                vel_errors.append(np.linalg.norm(obs[3:6]))
                actions.append(action)
                rewards.append(reward)
            
            # 计算指标
            pos_errors = np.array(pos_errors)
            vel_errors = np.array(vel_errors)
            actions = np.array(actions)
            
            results['position_rmse'].append(np.sqrt(np.mean(pos_errors ** 2)))
            results['max_deviation'].append(np.max(pos_errors))
            results['velocity_rmse'].append(np.sqrt(np.mean(vel_errors ** 2)))
            results['control_effort'].append(np.sum(actions ** 2))
            results['episode_return'].append(np.sum(rewards))
            results['success'].append(float(pos_errors[-1] < self.success_threshold))
            
            # 计算 settling time
            settling_idx = len(pos_errors)
            for i in range(len(pos_errors) - 1, -1, -1):
                if pos_errors[i] > self.settling_threshold:
                    settling_idx = i + 1
                    break
            results['settling_time'].append(settling_idx * self.dt)
            
            # 计算 smoothness (jerk)
            if len(actions) > 2:
                jerk = np.diff(actions, axis=0, n=2)
                results['smoothness'].append(np.mean(np.abs(jerk)))
            else:
                results['smoothness'].append(0.0)
        
        return results
    
    def _aggregate_results(
        self,
        all_results: List[Dict[str, List[float]]],
        method_name: str,
    ) -> EvalMetrics:
        """聚合多个 seed 的结果。"""
        # 合并所有结果
        merged = {}
        for key in all_results[0].keys():
            merged[key] = []
            for seed_results in all_results:
                merged[key].extend(seed_results[key])
        
        # 计算 mean ± std
        metrics = EvalMetrics(
            position_rmse=np.mean(merged['position_rmse']),
            position_rmse_std=np.std(merged['position_rmse']),
            max_deviation=np.mean(merged['max_deviation']),
            max_deviation_std=np.std(merged['max_deviation']),
            velocity_rmse=np.mean(merged['velocity_rmse']),
            velocity_rmse_std=np.std(merged['velocity_rmse']),
            control_effort=np.mean(merged['control_effort']),
            control_effort_std=np.std(merged['control_effort']),
            smoothness=np.mean(merged['smoothness']),
            smoothness_std=np.std(merged['smoothness']),
            success_rate=100.0 * np.mean(merged['success']),
            settling_time=np.mean(merged['settling_time']),
            settling_time_std=np.std(merged['settling_time']),
            episode_return=np.mean(merged['episode_return']),
            episode_return_std=np.std(merged['episode_return']),
            n_episodes=len(merged['position_rmse']),
            n_seeds=len(all_results),
            method_name=method_name,
        )
        
        return metrics
    
    def generate_comparison_table(
        self,
        results: List[EvalMetrics],
        output_format: str = 'markdown',
    ) -> str:
        """生成对比表格。
        
        Args:
            results: 各方法的评估结果
            output_format: 输出格式 ('markdown', 'latex', 'csv')
        
        Returns:
            格式化的表格字符串
        """
        if output_format == 'markdown':
            return self._generate_markdown_table(results)
        elif output_format == 'latex':
            return self._generate_latex_table(results)
        else:
            return self._generate_csv_table(results)
    
    def _generate_markdown_table(self, results: List[EvalMetrics]) -> str:
        """生成 Markdown 表格"""
        lines = [
            f"## 评估结果对比 ({self.trajectory})",
            "",
            f"评估设置: {self.n_seeds} seeds × {self.n_eval_episodes} episodes, deterministic={self.deterministic}",
            "",
            "| Method | Pos RMSE (m) | Max Dev (m) | Ctrl Effort | Success (%) | Return |",
            "|--------|--------------|-------------|-------------|-------------|--------|",
        ]
        
        for r in results:
            lines.append(
                f"| {r.method_name} | "
                f"{r.position_rmse:.4f}±{r.position_rmse_std:.4f} | "
                f"{r.max_deviation:.4f}±{r.max_deviation_std:.4f} | "
                f"{r.control_effort:.1f}±{r.control_effort_std:.1f} | "
                f"{r.success_rate:.1f} | "
                f"{r.episode_return:.1f}±{r.episode_return_std:.1f} |"
            )
        
        return "\n".join(lines)
    
    def _generate_latex_table(self, results: List[EvalMetrics]) -> str:
        """生成 LaTeX 表格"""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            f"\\caption{{Evaluation Results on {self.trajectory} Trajectory}}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Method & Pos RMSE (m) & Max Dev (m) & Ctrl Effort & Success (\%) & Return \\",
            r"\midrule",
        ]
        
        for r in results:
            lines.append(
                f"{r.method_name} & "
                f"${r.position_rmse:.4f} \\pm {r.position_rmse_std:.4f}$ & "
                f"${r.max_deviation:.4f} \\pm {r.max_deviation_std:.4f}$ & "
                f"${r.control_effort:.1f} \\pm {r.control_effort_std:.1f}$ & "
                f"{r.success_rate:.1f} & "
                f"${r.episode_return:.1f} \\pm {r.episode_return_std:.1f}$ \\\\"
            )
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def _generate_csv_table(self, results: List[EvalMetrics]) -> str:
        """生成 CSV 表格"""
        lines = ["method,pos_rmse,pos_rmse_std,max_dev,max_dev_std,ctrl_effort,ctrl_effort_std,success_rate,return,return_std"]
        
        for r in results:
            lines.append(
                f"{r.method_name},{r.position_rmse:.6f},{r.position_rmse_std:.6f},"
                f"{r.max_deviation:.6f},{r.max_deviation_std:.6f},"
                f"{r.control_effort:.2f},{r.control_effort_std:.2f},"
                f"{r.success_rate:.2f},{r.episode_return:.2f},{r.episode_return_std:.2f}"
            )
        
        return "\n".join(lines)


def run_full_evaluation(
    trajectory: str = 'figure8',
    sb3_model_path: Optional[str] = None,
    soar_program_path: Optional[str] = None,
    include_pid: bool = True,
    output_dir: str = 'results/evaluation',
):
    """运行完整的对比评估。
    
    Args:
        trajectory: 轨迹类型
        sb3_model_path: SB3 模型路径
        soar_program_path: Soar 程序路径
        include_pid: 是否包含 PID baseline
        output_dir: 输出目录
    """
    print("=" * 60)
    print(f"标准评估框架 (OpenAI / SB3 对齐)")
    print(f"轨迹: {trajectory}")
    print("=" * 60)
    
    evaluator = StandardEvaluator(
        trajectory=trajectory,
        n_seeds=10,
        n_eval_episodes=20,
    )
    
    results = []
    
    # 评估 PID baseline
    if include_pid:
        print("\n📊 评估 PID baseline...")
        pid_results = evaluator.evaluate_pid_baseline()
        results.append(pid_results)
        print(f"   Position RMSE: {pid_results.position_rmse:.4f} ± {pid_results.position_rmse_std:.4f}")
    
    # 评估 SB3 模型
    if sb3_model_path and os.path.exists(sb3_model_path):
        print(f"\n📊 评估 SB3 模型: {sb3_model_path}")
        sb3_results = evaluator.evaluate_sb3_model(sb3_model_path)
        results.append(sb3_results)
        print(f"   Position RMSE: {sb3_results.position_rmse:.4f} ± {sb3_results.position_rmse_std:.4f}")
    
    # 评估 Soar 程序
    if soar_program_path and os.path.exists(soar_program_path):
        print(f"\n📊 评估 Soar 程序: {soar_program_path}")
        soar_results = evaluator.evaluate_soar_program(soar_program_path)
        results.append(soar_results)
        print(f"   Position RMSE: {soar_results.position_rmse:.4f} ± {soar_results.position_rmse_std:.4f}")
    
    # 生成报告
    if results:
        print("\n" + "=" * 60)
        print(evaluator.generate_comparison_table(results, 'markdown'))
        
        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Markdown
        with open(output_path / f"eval_{trajectory}.md", 'w') as f:
            f.write(evaluator.generate_comparison_table(results, 'markdown'))
        
        # LaTeX
        with open(output_path / f"eval_{trajectory}.tex", 'w') as f:
            f.write(evaluator.generate_comparison_table(results, 'latex'))
        
        # CSV
        with open(output_path / f"eval_{trajectory}.csv", 'w') as f:
            f.write(evaluator.generate_comparison_table(results, 'csv'))
        
        # JSON (完整数据)
        with open(output_path / f"eval_{trajectory}.json", 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        print(f"\n📁 结果已保存到: {output_path}")
    
    return results


# =============================================================================
# 学术引用格式
# =============================================================================

CITATIONS = {
    'sb3_zoo': r"""
@misc{rl-zoo3,
  author = {Raffin, Antonin},
  title = {RL Baselines3 Zoo},
  year = {2020},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/DLR-RM/rl-baselines3-zoo}},
}
""",
    'stable_baselines3': r"""
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
""",
    'ppo': r"""
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
""",
    'sac': r"""
@inproceedings{haarnoja2018soft,
  title={Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle={International conference on machine learning},
  pages={1861--1870},
  year={2018},
  organization={PMLR}
}
""",
    'gym_pybullet_drones': r"""
@misc{panerati2021learning,
  title={Learning to Fly -- a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
  author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
  year={2021},
  eprint={2103.02142},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
""",
}


def print_citations():
    """打印论文引用格式"""
    print("=" * 60)
    print("学术引用格式")
    print("=" * 60)
    for name, citation in CITATIONS.items():
        print(f"\n### {name}")
        print(citation)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Soar 标准评估框架")
    parser.add_argument("--trajectory", type=str, default="figure8",
                        choices=["hover", "figure8", "circle"])
    parser.add_argument("--sb3-model", type=str, default=None,
                        help="SB3 模型路径")
    parser.add_argument("--soar-program", type=str, default=None,
                        help="Soar 程序路径")
    parser.add_argument("--no-pid", action="store_true",
                        help="不评估 PID baseline")
    parser.add_argument("--output-dir", type=str, default="results/evaluation")
    parser.add_argument("--citations", action="store_true",
                        help="打印引用格式")
    
    args = parser.parse_args()
    
    if args.citations:
        print_citations()
    else:
        run_full_evaluation(
            trajectory=args.trajectory,
            sb3_model_path=args.sb3_model,
            soar_program_path=args.soar_program,
            include_pid=not args.no_pid,
            output_dir=args.output_dir,
        )

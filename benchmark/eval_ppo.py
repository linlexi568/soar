import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Isaac Gym 路径优先配置
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

_ISAAC_GYM_PY = ROOT / "isaacgym" / "python"
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

_ISAAC_BINDINGS = _ISAAC_GYM_PY / "isaacgym" / "_bindings" / "linux-x86_64"
if _ISAAC_BINDINGS.exists():
    os.environ.setdefault("LD_LIBRARY_PATH", str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))

try:
    from isaacgym import gymapi  # type: ignore
except Exception:
    pass

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from scripts.sb3.isaac_gym_wrapper import IsaacGymSB3VecEnv



TRAJECTORY_TYPE = "circle"  # 可选: "figure8" | "circle" | "square" | "helix" | "hover"
# 使用哪个模型：True=best，False=final
USE_BEST = True
# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
RESULTS_DIR = ROOT / "05_PPO" / "results" / TRAJECTORY_TYPE
BEST_DIR = RESULTS_DIR / "best"

# 模型/归一化路径（可切换 best 或 final）
MODEL_PATH = (BEST_DIR / "ppo_best_scg.zip") if USE_BEST else (RESULTS_DIR / "ppo_final.zip")
MODEL_FALLBACK = RESULTS_DIR / "ppo_final.zip"

VECNORM_BEST = BEST_DIR / "vec_normalize_best.pkl"
VECNORM_FINAL = RESULTS_DIR / "vec_normalize.pkl"

NUM_ENVS_EVAL = 256  # 评测用少一点并行，省显存
EPISODES = 20

# 轨迹配置（与训练一致）
# 可选: "figure8" | "circle" | "square" | "helix" | "hover"


# figure8 配置 (项目标准: xy 平面)
FIGURE8_PERIOD = 5.0
FIGURE8_SCALE = 0.8
FIGURE8_PLANE = "xy"
FIGURE8_CENTER = [0.0, 0.0, 1.0]

# circle 配置
CIRCLE_PERIOD = 5.0
CIRCLE_RADIUS = 0.9
CIRCLE_CENTER = [0.0, 0.0, 1.0]

# square 配置
SQUARE_PERIOD = 5.0
SQUARE_SIDE = 0.8
SQUARE_CENTER = [0.0, 0.0, 1.0]

# helix 配置
HELIX_PERIOD = 8.0
HELIX_RADIUS = 0.7
HELIX_PITCH_PER_REV = 0.1
HELIX_CENTER = [0.0, 0.0, 1.0]

# hover 配置
HOVER_HEIGHT = 1.0


def _resolve_paths():
    model_path = MODEL_PATH
    vec_path = VECNORM_BEST if USE_BEST else VECNORM_FINAL

    if not model_path.exists():
        alt_best = BEST_DIR / "ppo_best_scg.zip"
        if alt_best.exists():
            model_path = alt_best
        elif MODEL_FALLBACK.exists():
            model_path = MODEL_FALLBACK
        else:
            raise FileNotFoundError(f"模型不存在: {model_path} 或 {MODEL_FALLBACK}")

    if not vec_path.exists():
        alt_vec = BEST_DIR / "vec_normalize_best.pkl"
        if alt_vec.exists():
            vec_path = alt_vec
        elif VECNORM_FINAL.exists():
            vec_path = VECNORM_FINAL
        else:
            raise FileNotFoundError(f"缺少 VecNormalize 统计: {vec_path} 以及备选 {alt_vec} / {VECNORM_FINAL}")

    return model_path, vec_path


def make_eval_env():
    if TRAJECTORY_TYPE == "figure8":
        trajectory_params = {
            "trajectory_type": "figure8",
            "period": FIGURE8_PERIOD,
            "scale": FIGURE8_SCALE,
            "plane": FIGURE8_PLANE,
            "center": FIGURE8_CENTER,
        }
    elif TRAJECTORY_TYPE == "circle":
        trajectory_params = {
            "trajectory_type": "circle",
            "period": CIRCLE_PERIOD,
            "radius": CIRCLE_RADIUS,
            "center": CIRCLE_CENTER,
        }
    elif TRAJECTORY_TYPE == "square":
        trajectory_params = {
            "trajectory_type": "square",
            "period": SQUARE_PERIOD,
            "side_length": SQUARE_SIDE,
            "center": SQUARE_CENTER,
        }
    elif TRAJECTORY_TYPE == "helix":
        trajectory_params = {
            "trajectory_type": "helix",
            "period": HELIX_PERIOD,
            "radius": HELIX_RADIUS,
            "pitch_per_rev": HELIX_PITCH_PER_REV,
            "center": HELIX_CENTER,
        }
    elif TRAJECTORY_TYPE == "hover":
        trajectory_params = {
            "trajectory_type": "hover",
            "height": HOVER_HEIGHT,
        }
    else:
        raise ValueError(f"Unsupported trajectory type: {TRAJECTORY_TYPE}")

    # 与训练对齐：仅 use_pure_scg=True
    shaping_cfg = {"use_pure_scg": True}
    env = IsaacGymSB3VecEnv(
        num_envs=NUM_ENVS_EVAL,
        trajectory_type=TRAJECTORY_TYPE,
        trajectory_params=trajectory_params,
        reward_type="scg_exact",
        shaping_cfg=shaping_cfg,
        residual_cfg={},
        device="cuda:0",
    )
    _, vec_path = _resolve_paths()
    setattr(env, "render_mode", None)
    env = VecNormalize.load(str(vec_path), env)
    env.training = False      # eval 模式，冻结统计
    env.norm_reward = False   # 输出原始 SCG 奖励，便于与 PID 直接对比
    return env


def evaluate():
    model_path, _ = _resolve_paths()
    env = make_eval_env()
    model = PPO.load(str(model_path), env=env, device="cuda:0")

    episode_rewards = []      # VecNormalize 关闭 reward 归一化后，直接为原始 SCG
    episode_rewards_raw = []  # 冗余存 raw_episode_r（与上相同，便于核对）
    episode_lengths = []

    # Data collection for plotting (Env 0 only)
    traj_x, traj_y, traj_z = [], [], []
    ref_x, ref_y, ref_z = [], [], []
    
    # Access underlying envs
    # env is VecNormalize -> env.venv is IsaacGymSB3VecEnv
    sb3_env = env.venv

    obs = env.reset() 
    isaac_env = sb3_env._isaac_env 
    dones = np.zeros(env.num_envs, dtype=bool)
    env_0_done = False

    # [FIX] 在 reset 后立即记录初始位置
    pos_init = isaac_env.pos[0].cpu().numpy()
    traj_x.append(pos_init[0])
    traj_y.append(pos_init[1])
    traj_z.append(pos_init[2])
    
    target_init = sb3_env._target_pos[0].copy()
    ref_x.append(target_init[0])
    ref_y.append(target_init[1])
    ref_z.append(target_init[2])

    while len(episode_rewards) < EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones_step, infos = env.step(action)
        dones = dones_step

        # Collect trajectory for env 0
        if not env_0_done:
            # Position from physics state (GPU tensor -> CPU numpy)
            pos = isaac_env.pos[0].cpu().numpy()
            traj_x.append(pos[0])
            traj_y.append(pos[1])
            traj_z.append(pos[2])
            
            # Target from env state
            target = sb3_env._target_pos[0].copy()
            ref_x.append(target[0])
            ref_y.append(target[1])
            ref_z.append(target[2])
            
            if dones[0]:
                env_0_done = True

        # 收集已结束 episode 的统计
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
                if "raw_episode_r" in info:
                    episode_rewards_raw.append(info["raw_episode_r"])
                if len(episode_rewards) >= EPISODES:
                    break
    env.close()

    mean_r = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_r = float(np.std(episode_rewards)) if episode_rewards else 0.0
    mean_raw = float(np.mean(episode_rewards_raw)) if episode_rewards_raw else mean_r
    std_raw = float(np.std(episode_rewards_raw)) if episode_rewards_raw else std_r
    mean_l = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    print("================ EVAL (PPO, Pure SCG, no norm) ================")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Raw SCG mean/std: {mean_r:.4f} / {std_r:.4f}")
    print(f"Length mean: {mean_l:.4f}")
    print("==============================================================")

    # Plotting
    print("Plotting trajectory...")
    plt.figure(figsize=(8, 8))
    plt.plot(ref_x, ref_y, 'r--', label='Reference')
    plt.plot(traj_x, traj_y, 'b-', label='PPO')
    plt.title(f'PPO Trajectory Tracking ({TRAJECTORY_TYPE})')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    save_path = RESULTS_DIR / f"ppo_{TRAJECTORY_TYPE}_topdown.png"
    plt.savefig(str(save_path))
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    evaluate()

#!/bin/bash
#
# SAC 训练脚本 - 使用 SCG 精确奖励
# 修改参数: 直接编辑这个脚本
#

set -e  # 遇到错误立即退出

# ============================================================================
#                    ★★★ 超参数配置 (修改这里) ★★★
# ============================================================================

# --------------------- 任务配置 ---------------------
TASK="circle"              # 选择: circle, square, helix, figure8
SEED=42                    # 随机种子
DURATION=5.0               # 每个episode时长(秒)

# --------------------- 训练配置 ---------------------
TOTAL_TIMESTEPS=500000     # 总训练步数
NUM_ENVS=256               # 并行环境数
DEVICE="cuda:0"            # 设备

# --------------------- SAC 超参数 ---------------------
LEARNING_RATE=3e-4         # 学习率
BUFFER_SIZE=100000         # 回放缓冲区大小
BATCH_SIZE=256             # 批次大小
GAMMA=0.99                 # 折扣因子
TAU=0.005                  # 软更新系数
ENT_COEF="auto"            # 熵系数 (auto=自动调整)
LEARNING_STARTS=1000       # 开始学习前的步数
TRAIN_FREQ=1               # 每N步训练一次
GRADIENT_STEPS=1           # 每次训练的梯度步数

# --------------------- 网络结构 ---------------------
NET_ARCH="256,256"         # MLP隐藏层 (逗号分隔)

# ============================================================================

# 进入项目根目录
cd "$(dirname "$0")/.."

# 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "错误: 未找到虚拟环境 .venv"
    exit 1
fi

# 检查依赖
python3 -c "import stable_baselines3" 2>/dev/null || {
    echo "错误: 未安装 stable-baselines3"
    echo "运行: pip install stable-baselines3[extra]"
    exit 1
}

echo "======================================================================"
echo "SAC 训练 - 任务: $TASK"
echo "======================================================================"
echo "  总步数: $TOTAL_TIMESTEPS"
echo "  并行环境: $NUM_ENVS"
echo "  学习率: $LEARNING_RATE"
echo "  熵系数: $ENT_COEF"
echo "  网络: [$NET_ARCH]"
echo ""

# 创建输出目录
mkdir -p "03_SAC/checkpoints/$TASK"
mkdir -p "03_SAC/logs/$TASK"

# 运行训练
python3 << PYEOF
import sys
from pathlib import Path

# 路径设置
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / '02_PPO'))
sys.path.insert(0, str(ROOT / '01_soar'))

from scg_vec_env import IsaacSCGVecEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import json
from datetime import datetime

# 参数
TASK = "$TASK"
SEED = $SEED
DURATION = $DURATION
TOTAL_TIMESTEPS = $TOTAL_TIMESTEPS
NUM_ENVS = $NUM_ENVS
DEVICE = "$DEVICE"
LEARNING_RATE = $LEARNING_RATE
BUFFER_SIZE = $BUFFER_SIZE
BATCH_SIZE = $BATCH_SIZE
GAMMA = $GAMMA
TAU = $TAU
ENT_COEF = "$ENT_COEF"
LEARNING_STARTS = $LEARNING_STARTS
TRAIN_FREQ = $TRAIN_FREQ
GRADIENT_STEPS = $GRADIENT_STEPS
NET_ARCH = [int(x) for x in "$NET_ARCH".split(',')]

SAVE_DIR = ROOT / "03_SAC" / "checkpoints"
LOG_DIR = ROOT / "03_SAC" / "logs"

# 创建环境
train_env = IsaacSCGVecEnv(
    num_envs=NUM_ENVS,
    device=DEVICE,
    task=TASK,
    duration=DURATION,
)

eval_env = IsaacSCGVecEnv(
    num_envs=64,
    device=DEVICE,
    task=TASK,
    duration=DURATION,
)

# 回调
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=str(SAVE_DIR / TASK),
    name_prefix="sac",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(SAVE_DIR / TASK),
    log_path=str(LOG_DIR / TASK),
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
)

# 创建模型
policy_kwargs = dict(net_arch=dict(pi=NET_ARCH, qf=NET_ARCH))

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=LEARNING_RATE,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    tau=TAU,
    ent_coef=ENT_COEF if ENT_COEF != "auto" else "auto",
    learning_starts=LEARNING_STARTS,
    train_freq=TRAIN_FREQ,
    gradient_steps=GRADIENT_STEPS,
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    device=DEVICE,
    tensorboard_log=str(LOG_DIR),
)

# 训练
print("\n开始训练...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True,
)

# 保存最终模型
final_path = SAVE_DIR / TASK / "sac_final.zip"
model.save(final_path)
print(f"\n模型已保存到: {final_path}")

# 测试
print(f"\n测试 SAC on {TASK}...")
test_env = IsaacSCGVecEnv(
    num_envs=1024,
    device=DEVICE,
    task=TASK,
    duration=DURATION,
)

episode_rewards = []
for ep in range(10):
    obs = test_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = test_env.step(action)
        total_reward += reward.sum()
        if dones.any():
            done = True
    ep_reward = total_reward / 1024
    episode_rewards.append(ep_reward)
    print(f"  Episode {ep+1}/10: {ep_reward:.2f}")

mean_reward = sum(episode_rewards) / len(episode_rewards)
std_reward = (sum((r - mean_reward)**2 for r in episode_rewards) / len(episode_rewards)) ** 0.5
print(f"\n✓ SAC on {TASK}: {mean_reward:.2f} ± {std_reward:.2f}")

# 保存结果
result = {
    "task": TASK,
    "algorithm": "SAC",
    "mean_reward": float(mean_reward),
    "std_reward": float(std_reward),
    "episodes": [float(r) for r in episode_rewards],
    "hyperparameters": {
        "total_timesteps": TOTAL_TIMESTEPS,
        "num_envs": NUM_ENVS,
        "learning_rate": LEARNING_RATE,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "tau": TAU,
        "ent_coef": ENT_COEF,
        "net_arch": NET_ARCH,
    },
    "timestamp": datetime.now().isoformat(),
}

result_path = SAVE_DIR / TASK / "result.json"
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"结果已保存到: {result_path}")

train_env.close()
eval_env.close()
test_env.close()
PYEOF

echo ""
echo "======================================================================"
echo "训练完成!"
echo "======================================================================"

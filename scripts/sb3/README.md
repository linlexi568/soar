# SB3 集成模块

本目录提供 Stable-Baselines3 (SB3) 与 Soar 的完整集成。

## 文件结构

```
scripts/sb3/
├── __init__.py          # 模块导出
├── quadrotor_env.py     # SB3 标准 Gymnasium 环境
├── train_sb3.py         # SB3 训练脚本 (PPO/SAC/TD3)
├── soar_policy.py   # Soar 程序适配器
├── compare_eval.py      # 对比评估脚本
├── isaac_gym_wrapper.py # Isaac Gym VecEnv 封装
└── README.md            # 本文件
```

## 设计理念

1. **训练**: Isaac Gym (GPU 加速) + Soar DSL 程序搜索
2. **Reward**: 对齐 SB3 标准设计 (`sb3_standard` profile)
3. **测试**: 在 SB3 Gymnasium 环境中统一评估

## 快速开始

### 1. 安装依赖

```bash
pip install stable-baselines3[extra]
pip install gymnasium
```

### 2. 使用 SB3 训练 baseline

```bash
# 训练 PPO
python scripts/sb3/train_sb3.py --algo ppo --trajectory figure8 --timesteps 100000

# 训练 SAC
python scripts/sb3/train_sb3.py --algo sac --trajectory hover --timesteps 50000
```

### 3. 使用 Soar 训练（SB3 对齐 reward）

```bash
# 修改 run.sh 中的 REWARD_PROFILE
REWARD_PROFILE="sb3_standard"  # 或 "sb3_tracking"

# 运行训练
./run.sh
```

### 4. 对比评估

```bash
python scripts/sb3/compare_eval.py \
    --sb3-model results/sb3/ppo_figure8_final.zip \
    --soar-program results/figure8-sb3_standard.json \
    --trajectory figure8 \
    --episodes 20
```

## Reward 设计

### SB3 标准 (`sb3_standard`)

```
reward = alive_bonus - pos_cost - ctrl_cost

pos_cost = pos_cost_weight * ||pos_error||
ctrl_cost = ctrl_cost_weight * ||action||^2

默认权重:
  - pos_cost_weight = 1.0
  - ctrl_cost_weight = 0.001
  - alive_bonus = 0.1
```

### SB3 轨迹跟踪 (`sb3_tracking`)

在标准基础上增加速度和姿态误差：

```
reward = alive_bonus - pos_cost - vel_cost - ctrl_cost - orient_cost

额外权重:
  - vel_cost_weight = 0.1
  - orient_cost_weight = 0.1
```

## API 参考

### QuadrotorTrackingEnv

```python
from scripts.sb3.quadrotor_env import QuadrotorTrackingEnv

env = QuadrotorTrackingEnv(
    trajectory='figure8',      # 'hover', 'figure8', 'circle', 'setpoint'
    trajectory_params={        # 轨迹参数
        'scale': 0.5,
        'period': 5.0,
        'plane': 'xz',
    },
    duration=5.0,              # Episode 时长（秒）
    control_freq=50,           # 控制频率 (Hz)
    reward_weights={           # 自定义 reward 权重
        'pos_cost_weight': 1.0,
        'ctrl_cost_weight': 0.001,
    },
)
```

**观测空间** (18D):
- `pos_error`: [3] 位置误差
- `velocity`: [3] 线速度
- `euler`: [3] 欧拉角
- `omega`: [3] 角速度
- `target_pos`: [3] 目标位置
- `target_vel`: [3] 目标速度

**动作空间** (4D):
- `[fz, tx, ty, tz]`: 归一化推力和力矩 [-1, 1]

### SoarPolicy

```python
from scripts.sb3.soar_policy import SoarPolicy

# 从 JSON 文件加载
policy = SoarPolicy('results/best_program.json')

# 预测动作
obs = env.reset()[0]
action = policy.predict(obs)
```

### 评估函数

```python
from scripts.sb3.soar_policy import evaluate_soar_in_sb3

results = evaluate_soar_in_sb3(
    program_path='results/figure8-sb3_standard.json',
    trajectory='figure8',
    n_episodes=10,
)
# 输出: mean_reward, std_reward, mean_pos_error
```

## 与 Isaac Gym 训练的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                       Soar 工作流                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   训练阶段 (Isaac Gym)                测试阶段 (SB3 Gym)         │
│   ┌─────────────────┐                ┌─────────────────┐       │
│   │ IsaacGymDroneEnv│                │QuadrotorTracking│       │
│   │   (GPU 加速)    │   ──────────>  │      Env        │       │
│   │  512+ 并行环境  │                │  (标准 Gym)     │       │
│   └─────────────────┘                └─────────────────┘       │
│           │                                   │                 │
│           ▼                                   ▼                 │
│   ┌─────────────────┐                ┌─────────────────┐       │
│   │ sb3_standard    │                │   统一评估      │       │
│   │  reward profile │                │  · Soar     │       │
│   │                 │                │  · SB3 PPO/SAC  │       │
│   └─────────────────┘                │  · PID baseline │       │
│                                      └─────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 注意事项

1. **动力学差异**: `QuadrotorTrackingEnv` 使用简化动力学，与 Isaac Gym 有差异。
   最终论文结果应在 Isaac Gym 中验证。

2. **动作缩放**: 环境内部处理 [-1, 1] 归一化动作到物理量的映射。

3. **积分状态**: `SoarPolicy` 内部维护积分状态，每个 episode 需调用 `reset()`。

4. **SB3 版本**: 测试于 stable-baselines3 >= 2.0.0

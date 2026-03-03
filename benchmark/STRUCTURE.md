# Benchmark 目录结构说明

本目录包含完整的 PPO/PID/LQR 四旋翼控制器对比实验代码。

## 核心文件

```
benchmark/
├── README.md              # 完整文档
├── QUICKSTART.md          # 快速开始指南
├── STRUCTURE.md           # 本文件
├── requirements.txt       # Python 依赖
├── setup.sh               # 环境安装脚本
├── check_env.py           # 环境验证脚本
├── run_all.sh             # 一键运行所有实验
│
├── ppo/                   # PPO 强化学习
│   ├── config.py         # PPO 配置（超参数、轨迹参数）
│   ├── train.py          # PPO 训练脚本
│   └── eval.py           # PPO 评估脚本
│
├── baselines/             # 传统控制基线
│   ├── controllers.py    # PID/LQR 控制器实现
│   ├── tune_pid.py       # PID 参数调优
│   ├── tune_lqr.py       # LQR 参数调优
│   ├── eval_pid.py       # PID 评估（待添加）
│   └── eval_lqr.py       # LQR 评估（待添加）
│
├── envs/                  # 环境封装
│   ├── isaac_gym_wrapper.py  # Isaac Gym 环境封装
│   └── reward_calculator.py  # SCG 奖励计算器
│
└── results/               # 实验结果
    ├── ppo/              # PPO 结果（模型、日志、图表）
    ├── pid/              # PID 调优结果（JSON）
    └── lqr/              # LQR 调优结果（JSON）
```

## 使用流程

### 1. 环境验证
```bash
cd benchmark
source /path/to/venv/bin/activate
python check_env.py
```

### 2. 单独运行
```bash
# 调优 PID
python baselines/tune_pid.py --task circle --trials 15

# 调优 LQR  
python baselines/tune_lqr.py --task circle --trials 20

# 训练 PPO
python ppo/train.py --task circle --max-steps 500000000

# 评估 PPO
python ppo/eval.py --task circle --use-best
```

### 3. 一键运行
```bash
./run_all.sh
```

## 支持的任务

- `hover`: 悬停
- `circle`: 圆周轨迹
- `figure8`: 8字轨迹
- `square`: 正方形轨迹
- `helix`: 螺旋上升轨迹

## 输出结果

### PID/LQR
调优结果保存为 JSON 文件：
```json
{
  "task": "circle",
  "controller": "pid",
  "best_params": {...},
  "metrics": {
    "mean_true_reward": -35.2,
    "rmse_pos": 0.45
  }
}
```

### PPO
- 模型：`results/ppo/<task>/best_model.zip`
- 归一化：`results/ppo/<task>/vec_normalize.pkl`
- 日志：`results/ppo/<task>/logs/`
- 图表：`results/ppo/<task>/eval_<task>.png`

## 修改配置

编辑 `ppo/config.py` 可修改：
- 轨迹参数（半径、周期等）
- PPO 超参数
- 训练配置（环境数、总步数等）

## 依赖项目文件

Benchmark 依赖以下项目文件（已自动复制）：
- `scripts/sb3/isaac_gym_wrapper.py` → `envs/`
- `01_soar/utils/reward_scg_exact.py` → `envs/`
- `scripts/baselines/tune_pid_lqr_isaac.py` → `baselines/`

## 论文复现

完整复现步骤：
1. 调优所有任务的 PID/LQR 参数（~1小时）
2. 训练所有任务的 PPO 模型（~12小时）
3. 评估并生成对比图表

详见 README.md

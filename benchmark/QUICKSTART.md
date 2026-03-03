# 快速使用指南

## 单独运行实验

### 1. 调优 PID
```bash
cd benchmark
python baselines/tune_pid.py --task circle --trials 15
```

### 2. 调优 LQR
```bash
python baselines/tune_lqr.py --task circle --trials 20
```

### 3. 训练 PPO
```bash
python ppo/train.py --task circle --max-steps 500000000
```

### 4. 评估 PPO
```bash
python ppo/eval.py --task circle --use-best --episodes 20
```

## 一键运行所有实验
```bash
cd benchmark
./run_all.sh
```

## 修改训练参数

编辑 `ppo/config.py` 可修改 PPO 超参数:
- `num_envs`: 并行环境数（默认 8196，需要约 24GB 显存）
- `max_steps`: 最大训练步数
- `learning_rate`: 学习率
- `net_arch`: 网络架构

## 查看训练进度

训练时会自动启动 TensorBoard:
```bash
# 或手动启动
tensorboard --logdir results/ppo/<task>/logs --port 6006
```

然后访问 http://localhost:6006

## 常见问题

### Q: 显存不足怎么办？
A: 减少 `--num-envs` 参数，例如:
```bash
python ppo/train.py --task circle --num-envs 4096
```

### Q: 训练太慢怎么办？
A: 可以减少训练步数用于测试:
```bash
python ppo/train.py --task circle --max-steps 10000000
```

### Q: 如何使用已有的 PID/LQR 参数？
A: 参数保存在 `results/pid/` 和 `results/lqr/` 目录下的 JSON 文件中，可直接使用。

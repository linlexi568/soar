# Isaac Gym 硬核场景指南

以下三类场景可以单独或组合地在 `IsaacGymDroneEnv` 中启用，用于验证控制器在极端 domain shift 下的表现。

## 1. 风场随机化 / 未见风场

| 场景名 | 描述 | 默认幅度 (可通过 `scenario_kwargs` 调整) |
| --- | --- | --- |
| `wind_random` | 每个 episode 采样不同的恒定风 + 小阵风 | 最高等效加速度 ±2.5 m/s² (XY)，垂向缩放 0.3 |
| `wind_storm` | 未在训练中见过的强侧风 + 阵风 | 默认为 `wind_test_force=[0, m·5.0, 0]`，`wind_test_gust_std=m·0.5` |

`wind_random` 会在 `reset()` 时重新采样风向/大小，`wind_storm` 则给出固定方向的强风，可叠加 `wind_test_jitter` 作为机体间微小差异。

## 2. 观测噪声 / 丢包 / 延迟

| 场景名 | 噪声 (位置/速度/姿态/角速度) | 丢包概率 | 观测延迟 |
| --- | --- | --- | --- |
| `sensor_noisy` | 0.02 / 0.05 / 0.01 / 0.02 | 5% | 0 |
| `sensor_faulty` | 0.05 / 0.12 / 0.03 / 0.05 | 15% | 2 步 |

带噪观测通过 `_apply_sensor_corruption` 实现，延迟使用环形缓冲区模拟 `TimeDelay` 传感器，丢包则回放上一帧观测。

## 3. 物理参数错配

| 场景名 | 质量缩放 | 推力缩放 | 说明 |
| --- | --- | --- | --- |
| `param_random` | `[0.85, 1.15]` | `[0.85, 1.05]` | 每次 reset 重新采样，模拟制造/挂载差异 |
| `param_shift` | `×1.4` | `×0.8` | 极限错配：重载 + 电机老化 |

真实质量通过 `gym.set_actor_rigid_body_properties` 动态更新，推力缩放体现在 `_rpm_to_forces()` 的 `thrust_multipliers` 中。

## 使用方式

```python
env = IsaacSCGVecEnv(
    num_envs=4096,
    device="cuda:0",
    task="figure8",
    duration=5.0,
    scenario="wind_storm+sensor_faulty+param_shift",
    scenario_kwargs={
        "wind_test_force": [0.0, 0.18, 0.0],
        "sensor_test_delay_steps": 3,
    },
)
```

- 多个场景用 `+` 连接，未指定的类别（wind/sensor/param）保持 `none`。
- `scenario_kwargs` 允许精细化调参；所有可调字段在 `01_soar/envs/isaac_gym_drone_env.py` 冒头注释中列出。
- 运行批量泛化测试时，可在 `run_generalization_test.sh` 中设置 `TRAIN_SCENARIO` / `TEST_SCENARIO` 以及对应的 `_KWARGS` 字符串，脚本会自动通过 `GENERALIZATION_*` 环境变量传递给 `test_generalization.py`，无须 CLI 传参。

## 建议测试矩阵

1. **风场 generalization**：`TRAIN_ENV_SCENARIO='default'`，`TEST_ENV_SCENARIO='wind_storm'`，观察 RL vs Soar 的鲁棒性差异。
2. **传感器退化**：`TRAIN_ENV_SCENARIO='sensor_noisy'`，`TEST_ENV_SCENARIO='sensor_faulty'`，重点关注丢包+延迟下的稳定性。
3. **参数错配**：`TRAIN_ENV_SCENARIO='param_random'`，`TEST_ENV_SCENARIO='param_shift'`，检验高度鲁棒控制器 vs 保守程序的表现。

> 以上场景默认全部关闭，不会影响现有 SAC/TD3 训练流程；只需在 `run_generalization_test.sh` 头部切换变量即可完成硬场景评测。

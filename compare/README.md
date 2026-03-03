# Compare Dashboard

该目录用于统一记录并对比 **PID 基线、PPO 基线以及 Soar 程序合成** 三套方案的表现。目标是在同一套评测配置下，收集可复现的运行指标与飞行质量指标。

## 指标维度

| 维度 | 说明 | 参考采样方式 |
| ---- | ---- | ------------- |
| Runtime (s) | 单次评估/训练耗时（墙钟时间） | `time`/脚本计时 |
| Mean Reward | 统一奖励定义下的平均值 | `utilities/verify_program.py`/PPO评估脚本 |
| Tracking RMSE | 位置或姿态的均方根误差 | 控制器/仿真日志 |
| Crash Rate | n 次试验中的失控或坠机比例 | 仿真返回码或日志统计 |
| Control Energy | 控制量累计范数/油门能耗 | 仿真日志或 post-process |
| Notes | 任何特殊现象或日志位置 | 手工填写 |

> 可以根据任务拓展更多指标（例如 jerk、饱和率等），只需在 CSV 中增列即可。

## 目录内容

- `metrics_template.csv`：统一的记录模板，请复制后填入真实数据。
- `logs/`：`compare_run.sh` 自动生成的原始日志与命令记录。
- `run_history.csv`：每次调用 `compare_run.sh` 自动追加的一行摘要。

## 建议流程

1. **配置命令**：编辑根目录的 `compare_run.sh`，在配置区填好 PID / PPO / PROGRAM 的具体命令（无需终端传参）。
2. **运行对比**：执行 `./compare_run.sh`，脚本会顺序运行开启的项目并在 `compare/logs/` 保存输出，同时在 `run_history.csv` 记录耗时。
3. **自动提取指标**：运行 `python compare/extract_metrics.py` 自动从日志解析关键数据并生成 `metrics_YYYYMMDD.csv`。
4. **手动补充**：打开生成的 CSV，补充自动提取无法获取的指标（如 tracking_rmse、crash_rate 等）。
5. **撰写分析**：将定量指标与飞行轨迹截图/视频对应，形成最终的对比报告。

### 快速开始示例

```bash
# 1. 启用要对比的方法（编辑 compare_run.sh，设置 RUN_PID=1 等）
vim compare_run.sh

# 2. 运行对比（会自动记录到 compare/logs/ 和 run_history.csv）
./compare_run.sh

# 3. 自动提取指标
python compare/extract_metrics.py

# 4. 查看结果
cat compare/metrics_*.csv
```

## 需要的数据来源示例

- **PID 基线**：使用 `utilities/verify_program.py` 对标准增益或手动调参的程序进行评估。
- **PPO/SAC 基线**：使用 `scripts/baselines/eval_safecontrol_baselines.py` 调用
	safe-control-gym 风格的模型，在 Isaac Gym 中评估并输出统一 JSON 指标。
- **Program (Soar)**：使用 `01_soar/train_online.py` 生成的最优程序，再用 `utilities/verify_program.py` 做统一评估。

如需新增任务或指标，只需在此目录添加相应脚本/文档并在 README 中补充说明即可。

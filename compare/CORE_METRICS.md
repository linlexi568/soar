# Soar 核心评测指标（5个必选）

## 指标选择理由

根据会议投稿需求和实际可行性，我们选择以下 5 个核心指标：

| 类别 | 指标 | 理由 |
|------|------|------|
| **效率** | 推理时间 | 边缘部署的关键约束，体现实时性 |
| **效率** | 内存占用 | 嵌入式部署的硬件约束 |
| **性能** | Position RMSE | 控制精度的黄金标准，最直观 |
| **鲁棒性** | Crash Rate | 安全性的底线指标，一票否决 |
| **鲁棒性** | Disturbance Rejection | 真实环境适应能力的核心体现 |

---

## 1. 推理时间 (Inference Time) ⚡

### 定义
单步控制决策的计算时间（微秒）

### 理论依据
- **实时控制的硬约束**：控制频率 50 Hz → 必须 < 20 ms
- **边缘部署可行性**：低算力平台（树莓派、嵌入式 MCU）的性能瓶颈
- **文献支撑**：[Liu et al., 2019] 指出推理时间是 NN 控制器部署的首要限制

### 测量方法

```python
import time
import numpy as np

def measure_inference_time(controller, state, n_iterations=1000):
    """
    测量控制器的平均推理时间
    
    Args:
        controller: 控制器对象（PID/PPO/Program）
        state: 测试状态 (dict with 'obs', 'ref_traj' etc.)
        n_iterations: 重复测试次数
    
    Returns:
        mean_time: 平均推理时间（微秒）
        std_time: 标准差（微秒）
        p95_time: 95% 分位数（微秒）
    """
    times = []
    
    for _ in range(n_iterations):
        start = time.perf_counter()
        action = controller.compute_action(state)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # 转换为微秒
    
    times = np.array(times)
    return {
        'mean_us': np.mean(times),
        'std_us': np.std(times),
        'p95_us': np.percentile(times, 95),
        'p99_us': np.percentile(times, 99)
    }
```

### 典型值（单核 CPU，无 GPU 加速）

| 方法 | 平均推理时间 | 备注 |
|------|--------------|------|
| PID | < 10 μs | 仅数学运算，极快 |
| 符号程序 | 10-100 μs | 依赖规则数量和条件复杂度 |
| 小型 NN (< 10K 参数) | 100-1000 μs | 前向传播开销 |
| 大型 NN (> 100K 参数) | 1-10 ms | 可能无法满足实时性 |

### 实时性约束

| 控制频率 | 最大推理时间 | 适用场景 |
|----------|--------------|----------|
| 50 Hz | < 20 ms | 位置控制外环 |
| 200 Hz | < 5 ms | 姿态控制内环 |
| 1000 Hz | < 1 ms | 高性能竞速无人机 |

### 论文表述示例

> "Inference time is measured as the average computation time for a single control decision over 1000 iterations on an Intel Core i7-10700K CPU (single-threaded). Our symbolic program controller achieves **μ = 45 ± 12 μs**, significantly faster than PPO's **μ = 850 ± 230 μs**, while PID baseline requires only **μ = 8 ± 2 μs**. All methods satisfy the real-time constraint of 50 Hz control (< 20 ms)."

---

## 2. 内存占用 (Memory Footprint) 💾

### 定义
控制器模型占用的存储空间（KB）

### 理论依据
- **嵌入式部署的硬约束**：
  - STM32F4（典型飞控芯片）：512 KB Flash
  - 树莓派 Zero：512 MB RAM
- **模型压缩的评价指标**：量化、剪枝、知识蒸馏的效果
- **文献支撑**：[Han et al., 2016] 的 Deep Compression，[Iandola et al., 2016] 的 SqueezeNet

### 测量方法

```python
import sys
import pickle

def measure_memory_footprint(controller):
    """
    测量控制器的内存占用
    
    Returns:
        dict: 包含参数量、存储大小、序列化大小
    """
    result = {}
    
    if hasattr(controller, 'model'):  # PyTorch NN
        import torch
        model = controller.model
        
        # 参数数量
        n_params = sum(p.numel() for p in model.parameters())
        result['n_parameters'] = n_params
        
        # 理论存储大小（FP32）
        size_fp32_kb = n_params * 4 / 1024  # 每个参数 4 字节
        result['size_fp32_kb'] = size_fp32_kb
        
        # 实际序列化大小
        temp_path = '/tmp/model_temp.pth'
        torch.save(model.state_dict(), temp_path)
        import os
        size_saved_kb = os.path.getsize(temp_path) / 1024
        result['size_saved_kb'] = size_saved_kb
        os.remove(temp_path)
        
    else:  # 符号程序或 PID
        if hasattr(controller, 'program'):
            program = controller.program
            size_pickle = len(pickle.dumps(program)) / 1024
            result['size_pickle_kb'] = size_pickle
            result['n_rules'] = len(program) if isinstance(program, list) else 1
        else:  # PID
            result['size_pickle_kb'] = 0.1  # 可忽略（几个浮点数）
            result['type'] = 'PID (negligible)'
    
    return result
```

### 典型值

| 方法 | 参数量 | 存储大小 (FP32) | 备注 |
|------|--------|-----------------|------|
| PID | 12 | < 0.1 KB | 4 轴 × 3 增益参数 |
| 符号程序 (5 规则) | ~50 | 0.5-2 KB | 取决于 AST 深度 |
| 小型 MLP (2 层 × 64) | ~5K | 20 KB | 可部署到微控制器 |
| PPO (默认网络) | ~50K | 200 KB | 需要较高算力平台 |
| 大型 NN (ResNet) | > 1M | > 4 MB | 需要 GPU 或高性能 CPU |

### 部署约束对比

| 平台 | RAM/Flash 可用 | 可部署方法 |
|------|----------------|------------|
| STM32F4 | 512 KB | PID, 符号程序 |
| ESP32 | 4 MB | PID, 符号程序, 小型 MLP |
| 树莓派 Zero | 512 MB | 所有方法 |
| Jetson Nano | 4 GB | 所有方法 |

### 论文表述示例

> "Memory footprint is critical for embedded deployment. Our symbolic program requires only **1.2 KB** storage (5 rules), compared to PPO's **215 KB** (54K parameters). This enables deployment on resource-constrained platforms such as STM32 microcontrollers (512 KB Flash), where large neural networks are infeasible."

---

## 3. Position RMSE (位置跟踪精度) 🎯

### 定义
位置误差的均方根（米）

```
RMSE_pos = sqrt( mean( (p_actual - p_target)^2 ) )
```

### 理论依据
- **控制理论的黄金标准**：稳态精度的经典指标 [Åström & Murray, 2021]
- **直观可比**：不同方法间最容易对比的性能指标
- **领域共识**：机器人/无人机领域的通用评价标准

### 计算方法

```python
def compute_position_rmse(actual_trajectory, target_trajectory):
    """
    计算整条轨迹的位置 RMSE
    
    Args:
        actual_trajectory: (T, 3) 实际位置 [x, y, z]
        target_trajectory: (T, 3) 目标位置 [x, y, z]
    
    Returns:
        rmse: 标量 RMSE 值（米）
        rmse_xyz: 各轴分解的 RMSE (dict)
    """
    errors = actual_trajectory - target_trajectory  # (T, 3)
    
    # 总体 RMSE
    rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    # 分轴 RMSE（用于诊断）
    rmse_xyz = {
        'x': np.sqrt(np.mean(errors[:, 0]**2)),
        'y': np.sqrt(np.mean(errors[:, 1]**2)),
        'z': np.sqrt(np.mean(errors[:, 2]**2))
    }
    
    return rmse, rmse_xyz
```

### 典型值（Crazyflie 2.X 规模无人机）

| 性能等级 | RMSE | 应用场景 |
|----------|------|----------|
| 优秀 | < 0.05 m | 精密操作、室内巡检 |
| 良好 | 0.05-0.15 m | 一般轨迹跟踪 |
| 可接受 | 0.15-0.30 m | 粗略导航 |
| 较差 | > 0.30 m | 不满足实用要求 |

### 不同轨迹的难度差异

| 轨迹类型 | 难度 | 期望 RMSE | 特点 |
|----------|------|-----------|------|
| Hover | 简单 | < 0.02 m | 无动态响应要求 |
| Circle (慢速) | 中等 | 0.05-0.10 m | 恒定曲率 |
| Figure-8 | 困难 | 0.10-0.20 m | 曲率变化 |
| Zigzag | 非常困难 | 0.15-0.30 m | 高频转向 |

### 统计分析

```python
def analyze_rmse_statistics(rmse_trials):
    """
    对多次试验的 RMSE 进行统计分析
    
    Args:
        rmse_trials: (n_trials,) 各次试验的 RMSE
    
    Returns:
        统计报告 (dict)
    """
    return {
        'mean': np.mean(rmse_trials),
        'std': np.std(rmse_trials),
        'median': np.median(rmse_trials),
        'min': np.min(rmse_trials),
        'max': np.max(rmse_trials),
        'q25': np.percentile(rmse_trials, 25),
        'q75': np.percentile(rmse_trials, 75),
        'iqr': np.percentile(rmse_trials, 75) - np.percentile(rmse_trials, 25)
    }
```

### 论文表述示例

> "We evaluate tracking accuracy using position RMSE across three trajectories (circle, figure-8, zigzag) with n=30 trials each. On the challenging figure-8 trajectory, our method achieves **0.087 ± 0.023 m**, significantly outperforming PID baseline's **0.142 ± 0.041 m** (p < 0.001, paired t-test) and comparable to PPO's **0.091 ± 0.029 m** (p = 0.32)."

---

## 4. Crash Rate (坠机率) 🚨

### 定义
在 n 次独立试验中发生失控/坠机的比例（百分比）

### 理论依据
- **安全性的底线指标**：一次坠机可能导致设备损坏、人员伤亡
- **实用化的关键门槛**：工业界不接受高坠机率的方法
- **鲁棒性的直接体现**：能否处理边界情况和扰动

### 计算方法

```python
def compute_crash_rate(simulation_results):
    """
    计算坠机率
    
    Args:
        simulation_results: list of dict, 每个元素包含:
            - 'crashed': bool, 是否坠机
            - 'crash_reason': str, 坠机原因（可选）
            - 'crash_time': float, 坠机时刻（可选）
    
    Returns:
        crash_rate: 坠机率（百分比）
        crash_analysis: 详细分析 (dict)
    """
    n_trials = len(simulation_results)
    n_crashes = sum(1 for r in simulation_results if r['crashed'])
    crash_rate = n_crashes / n_trials * 100
    
    # 统计坠机原因
    crash_reasons = {}
    crash_times = []
    for r in simulation_results:
        if r['crashed']:
            reason = r.get('crash_reason', 'unknown')
            crash_reasons[reason] = crash_reasons.get(reason, 0) + 1
            if 'crash_time' in r:
                crash_times.append(r['crash_time'])
    
    return {
        'crash_rate_pct': crash_rate,
        'n_crashes': n_crashes,
        'n_trials': n_trials,
        'crash_reasons': crash_reasons,
        'mean_crash_time': np.mean(crash_times) if crash_times else None
    }
```

### 坠机判定标准

在仿真环境中，满足以下任一条件视为坠机：

1. **位置越界**：`z < 0.05 m`（触地）或 `x,y,z 超出空间边界`
2. **姿态失稳**：`|roll| > 60°` 或 `|pitch| > 60°`
3. **速度过大**：`|v| > 5 m/s`（失控加速）
4. **控制饱和**：连续 50 步执行器饱和（卡死）
5. **NaN 检测**：状态或动作出现 NaN 值

```python
def check_crash(state, action, env_limits):
    """
    检查是否坠机
    """
    pos = state['position']  # (3,) [x, y, z]
    rpy = state['rpy']       # (3,) [roll, pitch, yaw]
    vel = state['velocity']  # (3,)
    
    # 检查 NaN
    if np.isnan(pos).any() or np.isnan(action).any():
        return True, 'NaN detected'
    
    # 检查位置
    if pos[2] < 0.05:
        return True, 'Hit ground'
    if not env_limits['x_min'] <= pos[0] <= env_limits['x_max']:
        return True, 'Out of bounds (x)'
    if not env_limits['y_min'] <= pos[1] <= env_limits['y_max']:
        return True, 'Out of bounds (y)'
    if not env_limits['z_min'] <= pos[2] <= env_limits['z_max']:
        return True, 'Out of bounds (z)'
    
    # 检查姿态
    if np.abs(rpy[0]) > np.radians(60) or np.abs(rpy[1]) > np.radians(60):
        return True, 'Attitude unstable'
    
    # 检查速度
    if np.linalg.norm(vel) > 5.0:
        return True, 'Velocity too high'
    
    return False, None
```

### 典型值

| 性能等级 | 坠机率 | 可接受性 |
|----------|--------|----------|
| 优秀 | 0% | 工业级 |
| 良好 | < 5% | 可商用（需故障处理） |
| 可接受 | 5-10% | 研究原型 |
| 较差 | 10-20% | 需要改进 |
| 不可接受 | > 20% | 不可实用 |

### 置信区间计算

```python
from scipy import stats

def compute_crash_rate_confidence_interval(n_crashes, n_trials, confidence=0.95):
    """
    计算坠机率的置信区间（Wilson score interval）
    """
    p = n_crashes / n_trials
    z = stats.norm.ppf((1 + confidence) / 2)  # 95% CI: z ≈ 1.96
    
    denominator = 1 + z**2 / n_trials
    center = (p + z**2 / (2 * n_trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n_trials + z**2 / (4 * n_trials**2)) / denominator
    
    return {
        'lower': max(0, center - margin) * 100,
        'upper': min(1, center + margin) * 100,
        'point_estimate': p * 100
    }
```

### 论文表述示例

> "Crash rate is evaluated over n=50 trials per trajectory. Our method achieves **2.0% crash rate (1/50)** on the aggressive zigzag trajectory, significantly lower than PID baseline's **18.0% (9/50, p < 0.01, Fisher's exact test)** and PPO's **8.0% (4/50, p = 0.08)**. The 95% confidence interval for our method is [0.1%, 10.4%], demonstrating robust performance."

---

## 5. Disturbance Rejection Ratio (抗扰动能力) 💨

### 定义
扰动下的性能衰减程度（百分比）

```
DRR = (RMSE_disturbed - RMSE_nominal) / RMSE_nominal × 100%
```

**解释**：
- DRR = 0%：扰动完全不影响性能（理想情况）
- DRR = 50%：扰动导致误差增加 50%
- DRR = 200%：扰动导致误差增加 2 倍

### 理论依据
- **鲁棒控制的核心目标**：在不确定性下保持性能 [Skogestad & Postlethwaite, 2005]
- **真实环境的必要条件**：风、质量变化、传感器噪声无法避免
- **自适应能力的体现**：学习到的策略能否应对训练集外的情况

### 测试场景设计

#### 场景 1: 持续风扰动
```python
def apply_constant_wind(env, wind_force):
    """
    在整个仿真过程中施加恒定风力
    
    Args:
        wind_force: (3,) [fx, fy, fz] in Newtons
    """
    env.disturbances['wind'] = {
        'type': 'constant',
        'force': wind_force,
        'start_time': 0.0,
        'end_time': float('inf')
    }
```

**典型风力等级**：
- 轻微：0.01 N（相当于 1-2 级风）
- 中等：0.05 N（相当于 3-4 级风）
- 强烈：0.10 N（相当于 5-6 级风，接近飞行极限）

#### 场景 2: 脉冲扰动
```python
def apply_impulse_disturbance(env, impulse_force, duration):
    """
    在特定时刻施加短时脉冲
    
    Args:
        impulse_force: (3,) 脉冲力（N）
        duration: 脉冲持续时间（秒）
    """
    env.disturbances['impulse'] = {
        'type': 'impulse',
        'force': impulse_force,
        'start_time': 5.0,  # 轨迹中段施加
        'duration': duration
    }
```

**典型脉冲**：
- 轻度：0.05 N × 0.1 s（模拟轻微碰撞）
- 中度：0.10 N × 0.5 s（模拟侧风突变）
- 重度：0.20 N × 1.0 s（模拟强阵风）

#### 场景 3: 周期性阵风
```python
def apply_periodic_wind(env, amplitude, frequency):
    """
    周期性变化的风力（模拟真实风场）
    
    Args:
        amplitude: 风力幅值（N）
        frequency: 频率（Hz）
    """
    env.disturbances['gust'] = {
        'type': 'periodic',
        'amplitude': amplitude,
        'frequency': frequency,
        'phase': np.random.uniform(0, 2*np.pi)  # 随机相位
    }
```

### 计算方法

```python
def compute_disturbance_rejection_ratio(controller, trajectory, disturbances):
    """
    计算抗扰动能力
    
    Args:
        controller: 控制器对象
        trajectory: 轨迹定义
        disturbances: 扰动配置列表
    
    Returns:
        drr_results: dict, 各扰动场景的 DRR
    """
    # 1. 无扰动基线
    rmse_nominal = evaluate_controller(controller, trajectory, disturbance=None)
    
    results = {'nominal_rmse': rmse_nominal}
    
    # 2. 各扰动场景
    for dist_name, dist_config in disturbances.items():
        rmse_dist = evaluate_controller(controller, trajectory, disturbance=dist_config)
        drr = (rmse_dist - rmse_nominal) / rmse_nominal * 100
        
        results[dist_name] = {
            'rmse_disturbed': rmse_dist,
            'drr_pct': drr,
            'rmse_increase': rmse_dist - rmse_nominal
        }
    
    # 3. 综合评分（平均 DRR）
    results['mean_drr'] = np.mean([r['drr_pct'] for r in results.values() if isinstance(r, dict)])
    
    return results
```

### 典型值

| 性能等级 | 平均 DRR | 鲁棒性评价 |
|----------|----------|------------|
| 优秀 | < 30% | 扰动影响很小 |
| 良好 | 30-60% | 可接受的性能衰减 |
| 中等 | 60-100% | 明显衰减但仍可用 |
| 较差 | 100-200% | 扰动导致性能严重下降 |
| 不可接受 | > 200% | 完全失效或坠机 |

### 恢复时间（补充指标）

除了 RMSE 衰减，还可测量恢复时间：

```python
def compute_recovery_time(trajectory, disturbance_time, threshold=0.1):
    """
    计算扰动后恢复到稳态的时间
    
    Args:
        trajectory: 完整轨迹
        disturbance_time: 扰动施加时刻（秒）
        threshold: 误差阈值（米）
    
    Returns:
        recovery_time: 恢复时间（秒），若未恢复返回 inf
    """
    errors = compute_tracking_errors(trajectory)
    dt = trajectory['dt']
    
    # 扰动后的误差序列
    start_idx = int(disturbance_time / dt)
    post_dist_errors = errors[start_idx:]
    
    # 找到首次持续低于阈值的时刻
    settled = post_dist_errors < threshold
    if not settled.any():
        return float('inf')
    
    # 要求至少持续 1 秒稳定
    window = int(1.0 / dt)
    for i in range(len(settled) - window):
        if settled[i:i+window].all():
            return i * dt
    
    return float('inf')
```

### 论文表述示例

> "We evaluate robustness under three disturbance scenarios: constant wind (0.05 N), impulse (0.1 N × 0.5 s), and periodic gust (0.03 N @ 2 Hz). Under constant wind, our method shows **DRR = 34.2%** (RMSE: 0.087 m → 0.117 m), significantly better than PID's **DRR = 78.5%** (0.142 m → 0.253 m, p < 0.01) and comparable to PPO's **DRR = 41.1%** (0.091 m → 0.128 m, p = 0.23). Average recovery time after impulse disturbance is **1.8 ± 0.4 s** for our method vs. **3.2 ± 1.1 s** for PID."

---

## 实现指南：一键评测脚本

### 自动化评测工具

创建 `compare/evaluate_core_metrics.py`：

```python
#!/usr/bin/env python3
"""
自动评测 5 个核心指标
"""

import numpy as np
import json
from pathlib import Path

def evaluate_all_methods(methods_config, test_config):
    """
    对所有方法评测 5 个核心指标
    
    Args:
        methods_config: dict, 方法配置
            {
                'PID': {'type': 'pid', 'gains': ...},
                'PPO': {'type': 'ppo', 'model_path': ...},
                'Program': {'type': 'program', 'program_path': ...}
            }
        test_config: dict, 测试配置
            {
                'trajectories': ['circle', 'figure8', 'zigzag'],
                'n_trials': 30,
                'disturbances': {...}
            }
    
    Returns:
        results: dict, 所有方法的所有指标
    """
    results = {}
    
    for method_name, method_cfg in methods_config.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {method_name}...")
        print(f"{'='*60}")
        
        controller = load_controller(method_cfg)
        
        # 1. 推理时间
        print("1/5 Measuring inference time...")
        inference_time = measure_inference_time(controller)
        
        # 2. 内存占用
        print("2/5 Measuring memory footprint...")
        memory = measure_memory_footprint(controller)
        
        # 3. Position RMSE
        print("3/5 Evaluating tracking accuracy (RMSE)...")
        rmse_results = evaluate_rmse(controller, test_config)
        
        # 4. Crash Rate
        print("4/5 Evaluating crash rate...")
        crash_results = evaluate_crash_rate(controller, test_config)
        
        # 5. Disturbance Rejection
        print("5/5 Evaluating disturbance rejection...")
        drr_results = evaluate_disturbance_rejection(controller, test_config)
        
        results[method_name] = {
            'inference_time_us': inference_time,
            'memory_footprint_kb': memory,
            'position_rmse_m': rmse_results,
            'crash_rate_pct': crash_results,
            'disturbance_rejection_pct': drr_results
        }
    
    return results


def generate_report(results, output_dir='compare/results'):
    """
    生成评测报告
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. JSON 原始数据
    with open(f'{output_dir}/core_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 2. CSV 表格
    import pandas as pd
    df = flatten_results_to_dataframe(results)
    df.to_csv(f'{output_dir}/core_metrics.csv', index=False)
    
    # 3. Markdown 报告
    with open(f'{output_dir}/REPORT.md', 'w') as f:
        f.write(generate_markdown_report(results))
    
    # 4. 对比图表
    plot_comparison_charts(results, output_dir)
    
    print(f"\n✅ 报告已生成到: {output_dir}/")


if __name__ == '__main__':
    # 配置
    methods = {
        'PID': {'type': 'pid'},
        'PPO': {'type': 'ppo', 'model': '02_PPO/checkpoints/best_model.zip'},
        'Soar': {'type': 'program', 'program': '01_soar/results/longrun_1000iters_20251114_001449.json'}
    }
    
    test_config = {
        'trajectories': ['circle', 'figure8', 'zigzag'],
        'n_trials': 30,
        'disturbances': {
            'constant_wind': {'force': [0.05, 0, 0]},
            'impulse': {'force': [0.1, 0, 0], 'duration': 0.5},
            'periodic': {'amplitude': 0.03, 'frequency': 2.0}
        }
    }
    
    # 运行评测
    results = evaluate_all_methods(methods, test_config)
    
    # 生成报告
    generate_report(results)
```

---

## 总结

### 5 个核心指标的互补性

| 维度 | 指标 | 侧重点 |
|------|------|--------|
| **部署可行性** | 推理时间 | 实时性约束 |
| **部署可行性** | 内存占用 | 硬件约束 |
| **功能性** | Position RMSE | 基本控制精度 |
| **安全性** | Crash Rate | 可靠性底线 |
| **真实环境** | Disturbance Rejection | 实用化能力 |

### 论文中的综合表述

> "We evaluate our approach using five core metrics across multiple dimensions:
> 
> **Deployment Feasibility**: Inference time (45 μs) and memory footprint (1.2 KB) enable embedded deployment on resource-constrained platforms.
> 
> **Control Performance**: Position RMSE (0.087 m on figure-8) demonstrates accurate trajectory tracking comparable to state-of-the-art PPO baseline.
> 
> **Safety**: Crash rate (2.0% on aggressive maneuvers) significantly outperforms PID baseline (18.0%, p < 0.01).
> 
> **Robustness**: Disturbance rejection ratio (34.2% under constant wind) shows superior adaptability to environmental uncertainties.
> 
> Statistical significance is verified using paired t-tests with n=30 trials per condition (α = 0.05)."

### 与 METRICS_DESIGN.md 的关系

- **METRICS_DESIGN.md**：完整的指标体系（20+ 指标），供深入分析和补充实验使用
- **CORE_METRICS.md**（本文档）：5 个必选指标，论文主体实验的最小集合

**建议**：
1. 论文正文使用这 5 个核心指标
2. 附录或补充材料可展示更多指标（Settling Time、Jerk、Rule Complexity 等）
3. 评审响应时可根据评审意见从 METRICS_DESIGN.md 中补充额外实验

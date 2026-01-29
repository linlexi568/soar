# Reward Profile 配置指南

Soar 项目提供了多套奖励权重配置（Reward Profiles），用于在不同应用场景下引导 MCTS 搜索合成不同风格的控制程序。

## 📋 可用 Profiles 列表

### 🔥 论文实验专用（推荐）

#### 1. `safety_first` - 安全优先型
**设计意图**：保守、平滑、节能  
**适用场景**：
- 安全关键应用（载人飞行、室内演示）
- 需要极低振动和平滑控制的场景
- 作为 baseline 对比，展示"最安全但可能牺牲跟踪精度"的策略

**权重特点**：
- ✅ 极高权重：`smoothness_jerk` (1.30), `saturation` (1.50), `high_freq` (1.20), `control_effort` (0.85)
- ⚠️ 中等偏低：`position_rmse` (0.70) - 允许适度误差换取安全
- 🎯 核心目标：不炸机、不饱和、不抖动

**预期行为**：
- 控制信号平滑，几乎无高频振荡
- 动作幅度保守，很少触及饱和限制
- 轨迹跟踪误差可能略高于 PID，但整体更稳定

---

#### 2. `tracking_first` - 跟踪优先型
**设计意图**：激进跟踪、允许大动作  
**适用场景**：
- 性能优先、竞速场景
- 与 PID/PPO 对比时的"上限"展示
- 需要快速响应和极低误差的应用

**权重特点**：
- ✅ 极高权重：`position_rmse` (1.50), `peak_error` (1.40), `settling_time` (1.20)
- ⚠️ 极低惩罚：`control_effort` (0.20), `smoothness_jerk` (0.15), `saturation` (0.30), `high_freq` (0.25)
- 🎯 核心目标：精确跟踪，快速响应

**预期行为**：
- 轨迹误差最小，转弯和加减速最快
- 控制信号可能频繁打满、存在抖动
- 能耗较高，但性能最优

---

#### 3. `balanced` - 平衡型
**设计意图**：折中方案，综合最优  
**适用场景**：
- 作为主实验结果展示
- 需要在多目标之间取得平衡的实际应用
- 与 PID 和 PPO 对比的"标准配置"

**权重特点**：
- ✅ 标准权重：`position_rmse` (1.00), `saturation` (1.00), `peak_error` (1.00)
- 🎯 中等平衡：所有指标权重居中，无明显偏向
- 🎯 核心目标：综合性能最优

**预期行为**：
- 误差、平滑性、能耗三者平衡
- 既不过分保守也不过分激进
- 适合大多数实际应用

---

---

#### 4. `robustness_stability` - 鲁棒性+稳定性优先型 🔥
**设计意图**：抗扰动、增益稳定、泛化能力强  
**适用场景**：
- 控制律发现研究（不过拟合单一轨迹）
- 需要强鲁棒性的实际应用
- 参数不确定性高的场景

**权重特点**：
- ✅ 极高权重：`gain_stability` (1.25), `saturation` (1.30), `peak_error` (1.15), `settling_time` (1.00)
- ⚠️ 低权重：`position_rmse` (0.60) - 允许误差，避免过拟合
- ⚠️ 零权重：`smoothness_jerk` (0.0) - 不约束平滑性，让策略自由探索
- 🎯 核心目标：快速恢复扰动、不振荡、不饱和

**预期行为**：
- 对扰动响应快、恢复能力强
- 增益参数稳定，不易振荡
- 轨迹跟踪精度中等，但泛化能力强
- 控制风格可能略显"生硬"（无平滑性约束）

**这是你之前实验用的版本！**  
（`control_law_discovery` 是它的别名，为了向后兼容）

---

### 🧪 其他 Profiles（保留兼容）

#### 5. `control_law_discovery` 
**别名**：等同于 `robustness_stability`，保留用于向后兼容旧脚本和实验结果。

#### 6. `smooth_control` - 平滑控制优先
强调 jerk 和高频能量抑制，适合需要生成人类可接受、物理可实现的控制策略。

#### 7. `balanced_smooth` - 平衡平滑型
在 `smooth_control` 基础上略微提升跟踪要求。

#### 8. `default` / `pilight_boost` / `pilight_freq_boost`
早期实验版本，保留用于向后兼容。

---

## 🚀 如何使用

### 训练时切换 Profile

编辑 `run.sh`：

```bash
# 修改此行
REWARD_PROFILE="safety_first"  # 改为 tracking_first, balanced 等
```

然后运行：
```bash
./run.sh
```

### 评估/对比时切换 Profile

编辑 `compare/compare_pid_vs_learned_v2.py`：

```python
# 修改此行
REWARD_PROFILE = "tracking_first"  # 改为 safety_first, balanced 等
```

然后运行：
```bash
cd compare
python compare_pid_vs_learned_v2.py
```

---

## 📊 论文实验建议

### 实验设计

**Experiment 1: 固定轨迹对比（figure8）**
- 比较对象：PID, PPO, Soar (safety_first), Soar (tracking_first), Soar (balanced)
- 指标：position_rmse, control_effort, peak_error, saturation_rate, high_freq_energy

**Experiment 2: Reward Sensitivity 分析**
- 展示 3 个 profile 下学到的控制程序差异
- 用表格/雷达图对比各项指标
- 讨论：reward shaping 如何系统性地改变学习策略风格

**Experiment 3: 多轨迹泛化（可选）**
- 在 hover, circle, figure8 上分别训练和测试
- 展示 Soar 在不同任务下的适应性

### 结果解读

| Profile | 跟踪精度 | 控制平滑 | 安全性 | 能耗 | 适用场景 |
|---------|---------|---------|--------|------|---------|
| `tracking_first` | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 高 | 竞速、性能优先 |
| `balanced` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 | 通用应用、主实验 |
| `safety_first` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 低 | 安全关键、演示 |

---

## 🔧 高级：自定义 Profile

如需创建新 profile，编辑 `utilities/reward_profiles.py`：

```python
_my_custom_weights: Weights = {
    "position_rmse": 1.0,
    "settling_time": 0.8,
    # ... 其他权重
}

_my_custom_ks: Coeffs = {
    "k_position": 1.1,
    # ... 其他系数
}

# 注册到 PROFILES 字典
PROFILES["my_custom"] = (_my_custom_weights, _my_custom_ks)
```

---

## 📚 权重说明

### Weights（权重）
各奖励分量的相对重要性（乘法因子）：

- `position_rmse`: 位置跟踪误差（RMSE）
- `settling_time`: 扰动恢复时间（鲁棒性）
- `control_effort`: 控制代价（输入幅度）
- `smoothness_jerk`: 平滑性（加加速度）
- `gain_stability`: 增益稳定性（避免振荡）
- `saturation`: 饱和惩罚（频繁打满）
- `peak_error`: 峰值误差（瞬态抑制）
- `high_freq`: 高频能量（避免不可实现的指令）

### Coefficients（k_* 系数）
指数/对数 shaping 函数的敏感度参数：

- 越大 = 对该指标的微小变化越敏感
- 越小 = 更宽容，允许一定程度的偏差

---

## ⚠️ 注意事项

1. **训练时间**：不同 profile 可能导致收敛速度差异
   - `tracking_first`：搜索空间更大，可能需要更多迭代
   - `safety_first`：搜索空间受限，收敛可能更快但性能上限较低

2. **轨迹依赖**：某些 profile 可能对特定轨迹更敏感
   - figure8（高速转弯）：`tracking_first` 优势明显
   - hover（静止悬停）：`safety_first` 和 `balanced` 差异不大

3. **硬件约束**：实际部署时需考虑物理限制
   - `tracking_first` 学到的策略可能超出真实硬件能力
   - 建议先用 `safety_first` 验证可行性

---

## 🎯 快速选择指南

- **不知道选哪个？** → 用 `balanced`
- **需要最安全的策略？** → 用 `safety_first`
- **需要最高性能？** → 用 `tracking_first`
- **复现之前实验？** → 用 `control_law_discovery`
- **需要演示给观众看？** → 用 `safety_first`（视觉效果最平滑）
- **需要与 RL 对比？** → 三个都跑，展示 reward shaping 的影响

---

**最后更新**：2025年11月19日  
**版本**：v1.0  
**维护者**：Soar Team

# SCG 奖励计算说明（当前版本）

本文档说明当前 **Soar 训练脚本 + SCG 评测** 中奖励是如何计算和对齐的，重点是 `safe_control_tracking` 这个 reward profile。

## 1. 训练阶段的奖励（Isaac → SCG 对齐）

训练仍然在 Isaac 环境中进行，但奖励完全对齐到 SCG 定义：

- 训练入口：`01_soar/train_online.py`
- 启动脚本：`run.sh`
- 关键参数：
  - 轨迹：`TRAJ=figure8`
  - 时长：`DURATION=5`
  - 奖励配置：`REWARD_PROFILE=safe_control_tracking`
  - 聚合方式：`REWARD_REDUCTION=sum`

在 `BatchEvaluator` / `SCGExactRewardCalculator` 中，每一步会产生两个奖励通道：

- **`reward_true`**：严格的 SCG tracking cost，公式为 `-(state_cost + action_cost)` 并在时间维度上做 `sum` 聚合；用于记录、排名、保存最优程序；
- **`reward_train`**：以 `reward_true` 为基底，再叠加复杂度、结构/稳定性先验、零动作惩罚等 shaping，只用于 MCTS/NN 训练信号。

只要 `use_scg_exact_reward=True` 且 `reward_reduction=sum`（训练脚本默认配置），`reward_true` 就完全等价于文档第二部分写的 SCG 代价。

当前 profile 下使用的分量（字段名与程序 JSON 的 `meta.reward_components` 对齐）：

- `position_rmse`：位置均方根误差（越小越好 → 奖励为负惩罚项）
- `settling_time`：误差进入容差带并保持的时间（越短越好）
- `control_effort`：控制能量 / 力矩平方和（越小越好）
- `smoothness_jerk`：控制信号的 jerk 平滑度（越平滑越好）
- `gain_stability`：程序结构与增益的稳定性先验（越稳定越好）
- `saturation`：是否频繁触发输出饱和（越少越好）
- `peak_error`：轨迹跟踪中的最大误差（越小越好）
- `high_freq`：高频震荡惩罚
- `finalize_bonus`：episode 成功完成时的终止奖励（通常是一个负偏移 + 成功加成）
- `zero_action_penalty`：长时间输出近零动作时的惩罚
- `structure_prior`：结构先验奖励（鼓励更简洁、更物理合理的结构）
- `stability_prior`：稳定性先验奖励（例如对极点/增益约束满足的鼓励）

> ⚠️ 这些组件**只**影响 `reward_train`。`reward_true` 在任何时候都只包含 SCG 的 `state_cost` 与 `action_cost`，详见本文后半部分的“严格遵循 SCG 论文”章节。
# SCG 奖励计算说明（严格遵循 SCG 论文）

> 本文只描述 **safe-control-gym 论文中明确定义的 tracking cost / reward**，
> 之前在代码/JSON 里出现的 `jerk`、`structure_prior`、`stability_prior` 等项目 **全部视为本项目的扩展项，不再称为“SCG 对齐”，也不在本文中归入 SCG 论文范畴**。

下面的描述以 SCG 官方实现（safe-control-gym 仓库）和其论文中给出的公式为准，只保留：

- 轨迹跟踪误差项（state tracking cost）
- 控制输入代价项（control effort cost）
- 时间维度上的积分 / 累加

## 1. SCG 论文中的标准 tracking cost

对于给定的期望轨迹 $x_t^{\star}$ 和控制输入 $u_t$，SCG 论文采用的典型 tracking cost 形式为：

$$
J = \int_0^T (x_t - x_t^{\star})^T Q (x_t - x_t^{\star}) + u_t^T R u_t\, dt,
$$

在离散时间下，近似为：

$$
J \approx \sum_{t=0}^{T-1} (x_t - x_t^{\star})^T Q (x_t - x_t^{\star}) + u_t^T R u_t.
$$

其中：

- $x_t$：当前系统状态（例如 quadrotor 的位置、速度、姿态角等组合）；
- $x_t^{\star}$：同一时刻的参考轨迹状态；
- $u_t$：控制输入（例如推力、力矩或等效 motor 命令）；
- $Q$：状态误差的权重矩阵；
- $R$：控制输入的权重矩阵。

SCG 论文中会给出 $Q$ 和 $R$ 的具体选择或推荐范围，用于平衡“跟踪精度”和“控制能量”。

在 safe-control-gym 的实现里，这个 cost 通常以“负号”形式转成 reward：

$$
r_t = - \bigl[(x_t - x_t^{\star})^T Q (x_t - x_t^{\star}) + u_t^T R u_t\bigr],
$$

从而 episode 的总 reward 为：

$$
R_{\text{episode}} = \sum_{t=0}^{T-1} r_t = -J.
$$

> **重要：** 本文所谓“严格对齐 SCG 论文的奖励”，指的就是只采用上述状态 tracking cost 和控制 effort cost 的组合，
> 不加入任何额外的先验项或正则项。

## 2. 本项目中严格按 SCG cost 计算的部分

在本仓库中，训练仍在 Isaac 环境进行，但“真实 reward”的设计必须满足：

1. 只依赖于与 SCG 论文一致的 **状态误差** 和 **控制输入**；
2. 按照二次型 $ (x - x^{\star})^T Q (x - x^{\star}) $ 和 $ u^T R u $ 形式组合；
3. 在时间上做简单求和（或与 SCG 代码等价的时间加权），不引入其他启发式项。

也就是说，当前和后续的奖励实现，应当可以用下面这种结构概括：

$$
r_t^{\text{SCG}} = - \Bigl[ (x_t - x_t^{\star})^T Q (x_t - x_t^{\star}) + u_t^T R u_t \Bigr],
$$

$$
R_{\text{episode}}^{\text{SCG}} = \sum_t r_t^{\text{SCG}}.
$$

在代码层面，这通常体现在：

- 从仿真环境中读取与 SCG 定义一致的状态向量（位置、速度、姿态等）；
- 从控制器侧获得实际施加的控制输入向量 $u_t$；
- 使用与 SCG 论文或 safe-control-gym 默认配置一致的 $Q, R$；
- 构造上述 cost，再取负作为 reward。


## 3. 代码实现映射（Implementation Mapping）

| 部分 | 代码位置 | 说明 |
| --- | --- | --- |
| SCG 奖励计算器 | `01_soar/utils/reward_scg_exact.py` (`SCGExactRewardCalculator`) | 精确实现 `r_t = -(x_err^T Q x_err + u^T R u)`，提供 `state_cost` / `action_cost` 分量 |
| 训练评估 | `01_soar/utils/batch_evaluation.py` (`BatchEvaluator`) | 配置 `use_scg_exact_reward=True` 时，`reward_true = sum_t r_t`，`reward_train = reward_true + shaping` |
| 测试评估 | `utilities/isaac_tester.py` (`SimulationTester`) | 运行时同样使用 `SCGExactRewardCalculator` 并做 `sum` 聚合，保证与训练日志可比 |

### 3.1 状态向量与 Q 矩阵

`SCGExactRewardCalculator` 按 safe-control-gym 的 `quadrotor_3d_track` 顺序构造 12 维状态误差：

| Index | 误差符号 | 物理量 | 源数据 |
| --- | --- | --- | --- |
| 0 | $x - x^\star$ | x 轴位置误差 | Isaac `pos[:,0] - target_pos[:,0]` |
| 1 | $\dot{x} - \dot{x}^\star$ | x 轴速度误差 | Isaac `vel[:,0] - target_vel[:,0]`（默认 0） |
| 2 | $y - y^\star$ | y 轴位置误差 | `pos[:,1] - target_pos[:,1]` |
| 3 | $\dot{y} - \dot{y}^\star$ | y 轴速度误差 | `vel[:,1]` |
| 4 | $z - z^\star$ | z 轴位置误差 | `pos[:,2] - target_pos[:,2]` |
| 5 | $\dot{z} - \dot{z}^\star$ | z 轴速度误差 | `vel[:,2]` |
| 6 | $\phi - \phi^\star$ | 翻滚角误差 | `quat_to_euler(quat)[:,0]`（目标姿态为 0） |
| 7 | $\theta - \theta^\star$ | 俯仰角误差 | `quat_to_euler(quat)[:,1]` |
| 8 | $\psi - \psi^\star$ | 偏航角误差 | `quat_to_euler(quat)[:,2]` |
| 9 | $\omega_x$ | x 轴角速度 | Isaac `omega[:,0]` |
| 10 | $\omega_y$ | y 轴角速度 | `omega[:,1]` |
| 11 | $\omega_z$ | z 轴角速度 | `omega[:,2]` |

对应的对角权重矩阵 `Q = diag(SCG_STATE_WEIGHTS)` 在代码里显式写出：

$$
Q = \mathrm{diag}(1,\ 0.01,\ 1,\ 0.01,\ 1,\ 0.01,\ 0.5,\ 0.5,\ 0.5,\ 0.01,\ 0.01,\ 0.01).
$$

### 3.2 控制输入与 R 矩阵

- 控制向量 $u_t$ 取自 Isaac 动作张量的前 4 维（对应四个推力/力矩通道）。
- `SCG_ACTION_WEIGHT = 1e-4`，因此 $R = 10^{-4} I_4$，精确匹配 safe-control-gym 默认值。

### 3.3 奖励输出字段

- `state_cost`: $x_{err}^T Q x_{err}$ 的时间累加值，由 `SCGExactRewardCalculator.get_components()` 返回。
- `action_cost`: $u^T R u$ 的时间累加值。
- `reward_true`: `-(state_cost + action_cost)`，并沿时间做 `sum` 聚合；该值用于模型选择、日志和对外报告。
- `reward_train`: `reward_true` 加上复杂度、先验、零动作惩罚等 shaping；只影响搜索/训练过程。
- 任何对外比较（基准、论文数据）均应引用 `reward_true` 以保持与 SCG 论文一致。
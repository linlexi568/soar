# Soar 项目架构文档

## 1. 项目概览

**Soar** 是一个基于 **程序合成 (Program Synthesis)** 和 **强化学习 (Reinforcement Learning)** 的无人机飞行控制律自动生成系统。

不同于传统的端到端强化学习（如 PPO 直接输出电机指令），Soar 致力于生成 **可解释、可部署、符合物理约束** 的符号化控制程序（如 PID 变体、自适应控制律）。

### 核心特性

- **混合优化架构**：结合 **MCTS**（离散结构搜索）与 **贝叶斯优化**（连续参数调优）。
- **大规模并行仿真**：利用 **Isaac Gym** 实现 4096+ 环境的 GPU 并行评估。
- **工程安全保障**：引入 **Safety Shell Prior**，强制约束输出幅值与变化率，确保物理可行性。
- **灵活的奖励配置**：支持多种飞行风格（激进跟踪、平滑节能、鲁棒稳定）的自动切换。

---

## 2. 系统架构

系统采用 **双层优化 (Bi-level Optimization)** 架构，将控制律的生成分解为"结构"与"参数"两个维度。

```mermaid
graph TD
    A[MCTS Agent] -->|选择结构动作| B(生成候选程序 AST)
    B -->|结构哈希| C{GNN 缓存?}
    C -- Yes --> D[复用结构先验]
    C -- No --> E[GNN 推理 (可选)]
    E --> D
    D --> F[贝叶斯优化器 (BO)]
    F -->|参数调优| G[Isaac Gym 并行仿真]
    G -->|Safety Shell| H[物理约束过滤]
    H -->|计算奖励| I[返回 Reward]
    I -->|反向传播| A
```

### 2.1 外层循环：结构搜索 (MCTS)
- **算法**：Monte Carlo Tree Search (AlphaZero 变体)
- **职责**：探索 DSL（领域特定语言）定义的程序空间。
- **动作**：添加规则、修改算子、增加条件分支、引入时序原语（Delay, Rate, EMA）。
- **特点**：只关注程序长什么样（Structure），不关注具体参数是多少。

### 2.2 内层循环：参数调优 (Bayesian Optimization)
- **算法**：Gaussian Process Regression + UCB
- **职责**：针对 MCTS 生成的每一个特定结构，寻找最优的常数参数组合（如 $K_p, K_i, \text{Threshold}$）。
- **优势**：利用 Isaac Gym 的并行能力，一次性评估数十组参数，快速收敛。

---

## 3. 关键组件详解

### 3.1 批量评估器 (Batch Evaluator)
位于 `01_soar/utils/batch_evaluation.py`。
这是系统的核心引擎，负责连接上层算法与底层物理仿真。

- **并行加速**：支持 4096+ 并行环境，利用 `replicas_per_program` 降低评估方差。
- **JIT 编译**：对常量程序使用 Numba JIT 编译，实现微秒级执行。
- **确定性重置**：保证评估的可复现性。

### 3.2 安全外壳先验 (Safety Shell Prior)
为了弥补纯 AI 算法在工程安全性上的缺失，我们引入了 **MAD (Magnitude-Angle-Delta)** 约束机制。

- **幅值约束 (Magnitude)**：限制推力 $f_z \in [0, 7.5]N$，力矩 $\tau \in [-0.12, 0.12]Nm$。
- **姿态约束 (Angle)**：防止无人机进入不可恢复的翻滚状态。
- **变化率约束 (Delta)**：限制控制量的变化率 $\Delta u$，防止高频震荡损坏电机。
- **硬约束过滤**：直接拒绝产生 `NaN` 或除零错误的程序。

### 3.3 GNN 策略网络 (可选组件)
位于 `01_soar/models/`。
一个基于 PyTorch Geometric 的图神经网络，用于编码程序 AST。

- **作用**：学习结构先验 $P(action|program)$，引导 MCTS 搜索方向。
- **状态**：**可选 (Optional)**。
    - 在算力受限或初期实验中，可完全禁用（使用均匀先验）。
    - 启用时，通过 **结构哈希 (Structure Hash)** 忽略常数差异，最大化缓存命中率。

### 3.4 贝叶斯调参器 (Bayesian Tuner)
位于 `01_soar/utils/bayesian_tuner.py`。

- **并行 BO**：针对无人机控制参数敏感的特点，利用 GPU 并行性，一次迭代评估多组参数。
- **自适应范围**：自动提取程序中的 `TerminalNode(float)` 并设定合理的搜索边界。

---

## 4. 奖励系统与多目标测试

系统内置了多种 **Reward Profiles**，以适应不同的控制需求。通过 `run.sh` 中的 `REWARD_PROFILE` 切换。

| Profile 名称 | 侧重点 | 适用场景 | 关键指标 |
| :--- | :--- | :--- | :--- |
| **robustness_stability** | **鲁棒性 + 稳定性** | 默认推荐，抗扰动能力强 | 姿态误差、角速度震荡 |
| **tracking_first** | **激进跟踪** | 竞速、特技飞行 | 位置误差 (RMSE)、响应速度 |
| **safety_first** | **安全节能** | 巡检、长续航 | 控制量平滑度、能量消耗 |
| **balanced** | **综合平衡** | 通用场景 | 综合加权 |

### 多奖励测试机制
在训练过程中，系统会实时监控多个分项指标：
- `pos_rmse`: 位置跟踪误差
- `stable_bonus`: 飞行稳定性
- `act_smooth`: 动作平滑度
- `crash_rate`: 炸机率

---

## 5. 快速开始

### 5.1 启动训练

使用 `run.sh` 一键启动。关键配置如下：

```bash
# 1. 启用/关闭 贝叶斯优化 (建议开启，batch_size=10)
ENABLE_BAYESIAN_TUNING=true
BO_BATCH_SIZE=10

# 2. 选择奖励风格
REWARD_PROFILE="robustness_stability"

# 3. GNN 开关 (可选，默认开启，可手动禁用以加速)
# (在代码中通过 --disable-gnn 或移除相关参数实现)
```

### 5.2 监控与日志

- 训练日志保存在 `logs/` 目录下。
- 最优程序实时保存在 `results/online_best_program.json`。
- 终端输出已精简，仅显示迭代进度、最佳奖励和 BO 状态。

---

## 6. 总结：为什么选择 Soar？

1.  **比 PPO 更安全**：Safety Shell 保证了输出永远在物理安全边界内。
2.  **比传统 PID 更智能**：能自动发现适应复杂气动特性的非线性控制律。
3.  **比遗传算法更高效**：MCTS + BO 的双层架构大幅减少了无效搜索。
4.  **工程友好**：生成的最终产物是 **代码**（Python/C++），而不是黑盒神经网络权重，可直接部署到嵌入式飞控。

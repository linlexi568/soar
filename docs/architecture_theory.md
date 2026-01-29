# Soar 控制架构与理论说明（路线 A + C · v0.2）

> 版本说明：v0.2 修复了公式渲染（统一采用 `$...$` 与 `$$...$$`），并将理论叙述与当前代码模块、配置参数做了逐段对照，便于论文撰写与代码评审。

---

## 0. 系统概览：从 DSL 程序到安全动作

| 层级 | 关键模块 / 文件 | 作用 | 与理论的对应 |
| --- | --- | --- | --- |
| **程序生成** | `core/dsl.py`, `mcts_training/*` | 定义 DSL 语法、MCTS 拓展算子 | 限定程序空间 $\mathcal{P}$，并在搜索阶段保持结构约束 |
| **硬过滤** | `utils/program_constraints.py`, `BatchEvaluator.validate_program` | 通道变量白名单、规则深度限制、非法程序大惩罚 | 定义安全程序空间 $\mathcal{P}_\text{safe}$（路线 A） |
| **仿真评估** | `utils/batch_evaluation.py` + Isaac Gym | 在 2k~4k 并行环境内评估程序性能 | 提供黑箱函数 $J(p)$ 与动作统计，保证搜索过程“可控” |
| **输出安全壳** | `BatchEvaluator._apply_output_mad` | MAD（Magnitude-Angle-Delta）裁剪与速率限制 | 将所有动作投影到 $\mathcal{U}_\text{safe}$，实现 Safety by Construction |
| **表示学习** | `models/gnn_policy_nn_v2.py` | GNN 双流编码 + `get_embedding` | 为 MCTS 提供先验 policy/value 与 Ranking embedding |
| **偏序学习** | `models/ranking_value_net.py` | Pairwise Ranking + 动作特征 | 将动作统计 $[fz, tx]$ 的 mean/std/max 纳入先验（路线 C） |
| **训练入口** | `run.sh` | 写死所有超参，包含 `RANKING_GNN_CHUNK=4` | 复现实验时的外部约束，保证“不可走捷径” |

> **概念流程**：DSL 程序 → `validate_program` 筛选 → MCTS/MAD 下发到 Isaac Gym → 收集 $R(p)$、动作统计、复杂度指标 → GNN/RBVN 更新 → 形成结构化偏好，再喂回 MCTS。

---

## 1. 安全程序空间 $\mathcal{P}_\text{safe}$ 与动作安全集合 $\mathcal{U}_\text{safe}$

### 1.1 DSL 结构约束如何落地

- **程序集合**：DSL 可表达的全部程序记为 $\mathcal{P}$，每个程序 $p$ 对应控制律 $u_p: \mathcal{X} \to \mathbb{R}^4$。
- **结构谓词**：`program_constraints.validate_program` 实现的约束可抽象为
  $$C_{\text{struct}}(p) \le 0 \iff p \text{ 的所有 AST 节点都遵守通道变量白名单、算子集合与深度上界。}$$
- **安全候选集**：
  $$\mathcal{P}_{\text{struct}} = \{ p \in \mathcal{P} \mid C_{\text{struct}}(p) \le 0 \}.$$
  当前实现中，所有能进入仿真的程序都必须在此集合内。
- **通道白名单实例**：
  ```text
  u_fz : {pos_err_z, vel_z, err_i_z, err_d_z, thrust_bias, ...}
  u_tx/u_ty : {pos_err_{x,y}, err_p_{roll,pitch}, ang_vel_{x,y}, ...}
  u_tz : {pos_err_yaw, err_p_yaw, ang_vel_z, ...}
  ```
  这些集合在 `CHANNEL_VARIABLE_WHITELISTS` 中写死，确保“传感器-执行器”映射合理。

### 1.2 MAD 输出安全壳

`BatchEvaluator._apply_output_mad` 定义了**幅值 + 变化率**双重限制：

- 幅值：
  $$
  u_{fz} \in [f_{z,\min}, f_{z,\max}], \quad \lVert (u_{tx}, u_{ty}) \rVert_2 \le T_{xy,\max}, \quad |u_{tz}| \le T_{z,\max}.
  $$
- 变化率：
  $$
  |\Delta u_{fz}| \le d_{fz,\max}, \quad |\Delta u_{tx,ty,tz}| \le d_{torque,\max}.
  $$

把 MAD 壳视作投影算子 $\Pi_{\text{safe}}$：
$$
\Pi_{\text{safe}} : \mathbb{R}^4 \rightarrow \mathcal{U}_{\text{safe}}, \qquad \tilde u_p(x) = \Pi_{\text{safe}}(u_p(x)).
$$

---

## 2. 三轴控制律的显式形式（X/Y/Z）

在当前实现中，四旋翼的控制输入向量写作：

$$
u = \begin{bmatrix} u_{fz} \\ u_{tx} \\ u_{ty} \\ u_{tz} \end{bmatrix},
$$

其中：

- $u_{fz}$：沿机体 $z$ 轴的总推力（通过姿态投影到世界坐标，主导**高度 Z 轴**控制）；
- $u_{tx}$：绕机体 $x$ 轴的力矩（主导世界坐标中**Y 向位移 / Roll 相关**控制）；
- $u_{ty}$：绕机体 $y$ 轴的力矩（主导世界坐标中**X 向位移 / Pitch 相关**控制）；
- $u_{tz}$：绕机体 $z$ 轴的力矩（主导**Yaw 方向角度**控制）。

下面给出一组与 `SAFETY_ENVELOPE_TREE_EXAMPLE.md` 中 AST 一致、又符合控制工程直觉的“目标控制律”形式，用于论文描述。

### 2.1 Z 轴（高度）控制律

DSL 中合法的一叶改动示例对应的 $u_{fz}$ 可以抽象为：

$$
u_{fz}
= k_{z,p} e_z
 + k_{z,d} \dot e_z
 + k_{r} e_{\text{roll}}
 + k_{p} e_{\text{pitch}},
$$

其中：

- $e_z = \text{pos\_err\_z} = z_{\text{ref}} - z$：高度误差；
- $\dot e_z \approx \text{vel\_z}$：高度误差的一阶导数（用垂直速度近似，方向通过符号约定）；
- $e_{\text{roll}} = \text{err\_p\_roll}$：Roll 姿态误差；
- $e_{\text{pitch}} = \text{err\_p\_pitch}$：Pitch 姿态误差；
- $k_{z,p}, k_{z,d}, k_r, k_p$ 为 DSL 中的常数节点（例如 `k_z_p, k_z_d, k_roll, k_pitch_p`）。

这条控制律实现了：

1. 经典 $PD$ 高度控制：$k_{z,p} e_z + k_{z,d} \dot e_z$；
2. 利用 Roll/Pitch 姿态误差作为前馈/补偿项，保证在大姿态机动时高度仍能稳定。

在代码层面，这一类控制律通过 DSL 表达式树自动生成，并在 `CHANNEL_ALLOWED_INPUTS['u_fz']` 白名单约束下保证**不会错误依赖 yaw 相关状态**。

### 2.2 X/Y 轴（水平位置）控制律

在世界坐标下，X/Y 方向的平移主要通过 $u_{tx}, u_{ty}$ 间接控制：

- 向前/向后运动：通过 Pitch（$u_{ty}$）改变推力在 X 方向的投影；
- 向左/向右运动：通过 Roll（$u_{tx}$）改变推力在 Y 方向的投影。

典型的目标控制律可以写为（与 DSL 变量命名对齐）：

$$
u_{tx}
= k_{x,p} e_x
 + k_{x,d} \dot e_x
 + k_{\phi,p} e_{\text{roll}}
 + k_{\phi,d} \dot e_{\text{roll}},
$$

$$
u_{ty}
= k_{y,p} e_y
 + k_{y,d} \dot e_y
 + k_{\theta,p} e_{\text{pitch}}
 + k_{\theta,d} \dot e_{\text{pitch}},
$$

其中：

- $e_x = \text{pos\_err\_x} = x_{\text{ref}} - x$；
- $e_y = \text{pos\_err\_y} = y_{\text{ref}} - y$；
- $\dot e_x, \dot e_y$ 由 `vel_x, vel_y` 近似；
- $e_{\text{roll}} = \text{err\_p\_roll}$，$\dot e_{\text{roll}} \approx \text{err\_d\_roll}$；
- $e_{\text{pitch}} = \text{err\_p\_pitch}$，$\dot e_{\text{pitch}} \approx \text{err\_d\_pitch}$。

**白名单约束对应关系：**

- 在 `CHANNEL_ALLOWED_INPUTS['u_tx']` 中，仅允许引用与 X/Y 相关的位置误差、速度、Roll 状态；
- 在 `CHANNEL_ALLOWED_INPUTS['u_ty']` 中，仅允许引用与 X/Y 相关的位置误差、速度、Pitch 状态；
- 都不会错误地看到与高度 Z 或 yaw 纯相关的状态，从而避免病态耦合（例如用 $e_z$ 直接参与 $u_{tx}$）。

在 DSL 程序空间视角下，X/Y 轴控制律本质是以 `pos_err_x/pos_err_y` 为主的 P/PD 控制，再配合姿态误差与角速度误差项，自动搜索出合适的系数组合与算子嵌套形式。

### 2.3 Yaw 轴控制律

Yaw 轴控制力矩 $u_{tz}$ 在 DSL 与白名单中被严格限制为只依赖 yaw 相关状态与少量平面速度，用于阻尼：

$$
u_{tz}
= k_{\psi,p} e_{\text{yaw}}
 + k_{\psi,d} \dot e_{\text{yaw}}
 + k_{\psi,i} \int e_{\text{yaw}} \; dt
 + k_{v,x} v_x + k_{v,y} v_y,
$$

其中：

- $e_{\text{yaw}} = \text{err\_p\_yaw}$；
- $\dot e_{\text{yaw}} \approx \text{err\_d\_yaw}$ 或 `ang_vel_z`；
- 积分项对应 `err_i_yaw`；
- $v_x, v_y$ 分别对应 `vel_x, vel_y`，以提供横向运动时的附加阻尼。

**通道约束保证：**

- `CHANNEL_ALLOWED_INPUTS['u_tz']` 中**不包括** `pos_err_z, vel_z` 等高度状态，避免通过 yaw 力矩干扰 Z 轴控制；
- 允许少量 `pos_err_xy, vel_err` 等合成量，用于实现文献中常见的 yaw-velocity damping 策略。

### 2.4 三轴联合视角

综合上述三个子控制律，可以将系统控制写成：

$$
\begin{aligned}
u_{fz} &= f_z(e_z, \dot e_z, e_{\text{roll}}, e_{\text{pitch}}; \theta_z), \\
u_{tx} &= f_x(e_x, \dot e_x, e_{\text{roll}}, \dot e_{\text{roll}}; \theta_x), \\
u_{ty} &= f_y(e_y, \dot e_y, e_{\text{pitch}}, \dot e_{\text{pitch}}; \theta_y), \\
u_{tz} &= f_{\psi}(e_{\text{yaw}}, \dot e_{\text{yaw}}, \int e_{\text{yaw}}, v_x, v_y; \theta_{\psi}),
\end{aligned}
$$

其中 $f_\bullet$ 是由 DSL 程序搜索得到的符号控制律，$\theta_\bullet$ 是由 BO/学习得到的常数参数集合。安全壳通过：

1. **输入侧**：约束每个 $f_\bullet$ 允许访问的状态集合（白名单）；
2. **算子侧**：约束调节算子的参数范围与数值幅度；
3. **输出侧**：通过 MAD 壳约束 $u$ 的幅值与变化率；

从而在 X/Y/Z/Yaw 四个通道上共同定义出一个既物理合理又控制理论可接受的 $
\mathcal{U}_{\text{safe}}$，并在搜索阶段仅在该集合内部寻找最优控制律。
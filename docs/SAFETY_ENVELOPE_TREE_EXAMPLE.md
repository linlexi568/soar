# 安全壳约束：一叶改动示例

本文档展示了通过**只改动一个状态叶节点**，如何将违反安全壳的程序修改为合法程序。

---

## 1. 违规版本（Violation）

### 程序表达式

```text
set(
  u_fz,
  k_z_p   * pos_err_z   +
  k_z_d   * vel_z       +
  k_roll  * err_p_roll  +
  k_yaw_p * err_p_yaw     # ❌ 违规：err_p_yaw 不在 u_fz 的白名单中
)
```

### 程序树结构

```text
                set
             /        \
        u_fz           (+)
                   /         \
                (+)           (*)
              /     \        /   \
           (+)       (*)  Const  err_p_yaw  ← ❌ 违规状态叶子
         /    \     /  \
       (*)    (*) Const err_p_roll
      /  \   /  \
Const pos   Const vel
 k_z_p err_z k_z_d  _z
```

### 节点统计

- **状态叶子（4个）**：`pos_err_z`, `vel_z`, `err_p_roll`, `err_p_yaw`
- **参数叶子（4个）**：`k_z_p`, `k_z_d`, `k_roll`, `k_yaw_p`
- **违规原因**：`err_p_yaw` 不在 `CHANNEL_ALLOWED_INPUTS['u_fz']` 白名单中

### 安全壳检测结果

```python
validate_action_channel(action_node)
# 返回：(False, "u_fz references disallowed inputs: ['err_p_yaw']")
# 整个程序被标记为 HARD_CONSTRAINT_PENALTY = -1e6
```

---

## 2. 合法版本（Valid）

### 程序表达式

```text
set(
  u_fz,
  k_z_p     * pos_err_z   +
  k_z_d     * vel_z       +
  k_roll    * err_p_roll  +
  k_pitch_p * err_p_pitch   # ✅ 合法：err_p_pitch 在 u_fz 的白名单中
)
```

### 最终控制律（紧凑写法）

用数学形式写，就是一条标准的带姿态前馈项的高度控制律：

$$
u_{fz} 
= k_{z,p} \cdot e_z 
 + k_{z,d} \cdot \dot e_z
 + k_{r} \cdot e_{\text{roll}}
 + k_{p} \cdot e_{\text{pitch}}
$$

其中：
- $e_z = \text{pos\_err\_z}$：高度误差
- $\dot e_z \approx \text{vel\_z}$：高度误差的导数（用垂直速度近似）
- $e_{\text{roll}} = \text{err\_p\_roll}$：Roll 姿态误差
- $e_{\text{pitch}} = \text{err\_p\_pitch}$：Pitch 姿态误差
- $k_{z,p}, k_{z,d}, k_{r}, k_{p}$ 分别对应 DSL 中的 `k_z_p, k_z_d, k_roll, k_pitch_p`

### 程序树结构

```text
                set
             /        \
        u_fz           (+)
                   /         \
                (+)           (*)
              /     \        /   \
           (+)       (*)  Const  err_p_pitch  ← ✅ 合法状态叶子
         /    \     /  \
       (*)    (*) Const err_p_roll
      /  \   /  \
Const pos   Const vel
 k_z_p err_z k_z_d  _z
```

### 节点统计

- **状态叶子（4个）**：`pos_err_z`, `vel_z`, `err_p_roll`, `err_p_pitch`
- **参数叶子（4个）**：`k_z_p`, `k_z_d`, `k_roll`, `k_pitch_p`
- **修改内容**：仅将右下角的状态叶子从 `err_p_yaw` 改为 `err_p_pitch`

### 安全壳检测结果

```python
validate_action_channel(action_node)
# 返回：(True, "")
# 程序通过验证，可以进入仿真评估
```

---

## 3. 对比分析

| 维度 | 违规版本 | 合法版本 | 说明 |
|------|---------|---------|------|
| **状态叶子数** | 4 | 4 | 相同 |
| **参数叶子数** | 4 | 4 | 相同 |
| **树结构深度** | 5 | 5 | 完全相同 |
| **算子类型** | `+`, `*`, `set` | `+`, `*`, `set` | 完全相同 |
| **唯一修改** | `err_p_yaw` | `err_p_pitch` | **仅1个状态叶子** |
| **安全壳检测** | ❌ 失败 | ✅ 通过 | - |
| **评估结果** | `-1e6` 惩罚 | 正常评估 | - |

---

## 4. 控制理论解释

### 为什么 `u_fz` 不允许使用 `err_p_yaw`？

**推力通道 `u_fz` 的物理意义**：
- 控制无人机的**垂直运动**（z 轴）
- 需要配合 **roll/pitch 姿态**产生水平分力

**Yaw（偏航角）的特性**：
- 只影响机体绕 z 轴的旋转
- **不直接参与推力分配**
- 应由 yaw 力矩 `u_tz` 控制

**耦合风险**：
- 如果 `u_fz` 依赖 `err_p_yaw`，会导致：
  - ❌ yaw 误差大时，总推力异常增大/减小
  - ❌ 高度控制与方向控制相互干扰
  - ❌ 违背"分层控制"的经典四旋翼架构

### 正确做法

**分层控制结构**：
```text
高度控制：u_fz  ← pos_err_z, vel_z, err_p_roll, err_p_pitch
Roll 控制：u_tx  ← err_p_roll, err_d_roll, pos_err_x, vel_x
Pitch控制：u_ty  ← err_p_pitch, err_d_pitch, pos_err_y, vel_y
Yaw  控制：u_tz  ← err_p_yaw, err_d_yaw, ang_vel_z
```

每个通道只看"控制理论上应该看的状态"，避免病态耦合。

---

## 5. 安全壳的白名单配置

### `u_fz` 允许的状态变量（来自 `program_constraints.py`）

```python
CHANNEL_ALLOWED_INPUTS['u_fz'] = {
    # 位置误差
    'pos_err_x', 'pos_err_y', 'pos_err_z',
    'pos_err_xy', 'pos_err_z_abs',
    
    # 速度
    'vel_x', 'vel_y', 'vel_z', 'vel_err',
    
    # 姿态误差（仅 roll/pitch，不含 yaw）
    'err_p_roll', 'err_p_pitch',
    'err_d_roll', 'err_d_pitch',
    
    # 积分项（仅 z 方向）
    'err_i_z',
}
```

**注意**：白名单中**不包含**任何 yaw 相关变量：
- ❌ `err_p_yaw`
- ❌ `err_d_yaw`
- ❌ `err_i_yaw`
- ❌ `ang_vel_z`

---

## 6. 可视化说明

### 树中的节点类型标注

| 符号 | 含义 | 示例 |
|------|------|------|
| `Const` | 参数叶子（可由 BO 优化） | `k_z_p`, `k_z_d` |
| `pos_err_z`, `vel_z` | 状态叶子（观测量） | 状态变量 |
| `(+)`, `(*)` | 二元算子节点 | 加法、乘法 |
| `set` | 赋值操作（根节点） | `set(u_fz, expr)` |

### 修改前后对比（图示）

```text
违规树（右下角）              合法树（右下角）
      ...                         ...
       |                           |
      (*)                         (*)
     /   \                       /   \
  Const  err_p_yaw  ❌    →    Const  err_p_pitch  ✅
k_yaw_p                      k_pitch_p
```

**关键点**：
- 只改了**1个状态叶子**的名称
- 参数结构、树深度、算子类型**完全不变**
- 从"物理上不合理" → "控制理论上正确"

---

## 7. 扩展：其他通道的一叶改动示例

### 示例：`u_tz` 通道（Yaw 力矩）

**违规版**：
```text
set(u_tz, k_z * pos_err_z + k_yaw * err_p_yaw)
          ❌ pos_err_z 不应出现在 u_tz
```

**合法版**：
```text
set(u_tz, k_xy * pos_err_xy + k_yaw * err_p_yaw)
          ✅ pos_err_xy 在 u_tz 白名单中
```

**解释**：
- `u_tz` 控制 yaw，可以轻微耦合 xy 平面误差（用于阻尼）
- 但**不应该看 z 方向误差**（高度与偏航无关）

---

## 8. 总结

### 安全壳的作用

安全壳通过**通道-状态白名单**约束，实现：

1. **物理正确性**：每个通道只能访问控制理论上合理的状态
2. **搜索效率**：提前过滤掉物理上不可行的程序（避免浪费仿真资源）
3. **可解释性**：生成的控制律符合分层控制架构，便于工程师理解

### 一叶改动的意义

- **最小修改**：只需调整 1 个状态叶子，无需重构整棵树
- **强对比性**：适合论文/报告中展示"安全壳的精确作用范围"
- **参数无关**：无论参数怎么调，违规的状态访问都会被拦截

---

## 参考代码位置

- 安全壳白名单定义：`01_soar/utils/program_constraints.py`
- DSL 节点定义：`01_soar/core/dsl.py`
- 验证函数：`validate_action_channel()`, `validate_program()`

---

**文档版本**：v1.0  
**更新日期**：2025年11月24日

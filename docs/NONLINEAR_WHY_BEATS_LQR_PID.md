# 为什么 `smooth(tanh)` / `sign` 在 figure8、square 上能打赢 LQR / PID（非线性视角）

本文只用你当前工程里**一致的 SCG 二次代价**口径来解释（`01_soar/utils/reward_scg_exact.py`）：

\[
J = \sum_t \big(x_\text{err}^T Q x_\text{err} + u^T R u\big),\quad
Q=\mathrm{diag}([1,0.01,1,0.01,1,0.01,0.5,0.5,0.5,0.01,0.01,0.01]),\ R=10^{-4}I
\]

要点：**state cost 权重大、action cost 很小**，因此“更快更稳地压低状态误差/姿态误差”往往比“省控制量”更重要。

---

## 1) baseline LQR/PID 在这个环境里“天然吃亏”的地方

仓库里的 baseline（`scripts/baselines/tune_pid_lqr_isaac.py`）本质是：

- 外环：位置/速度误差线性组合得到期望加速度 `acc_des`（线性）
- 中间：通过 `att_scale` 把横向加速度映射成期望 roll/pitch（小角度近似 + 人为缩放）
- 内环：姿态 PD（线性）
- 多处 clip：推力/力矩裁剪、积分限幅

这套结构在**小扰动 + 平滑参考**时非常合理；但在 figure8 / square 的两个关键场景会被放大弱点：

1. **工作点变化大**：速度、角速度、倾角在一个 episode 内变化显著，固定线性增益只能在某一段“刚好”。
2. **方轨迹的角点本质是高频/不连续参考**：需要很高带宽才能“转角不拉胯”；提高线性增益会引入振荡 + clip，导致姿态/角速度项（Q 里 roll/pitch 0.5 权重）变大。
3. **外环→内环的间接链路**：baseline 先算 `acc_des` 再算 `roll_des/pitch_des`，等效上引入额外相位滞后/动态耦合；而你发现的控制律直接用 `pos_err_y, vel_y, ang_vel_x` 去驱动 `u_tx`，相当于把“等效动态补偿”揉进了一条律里。

---

## 2) `smooth=tanh`（figure8）为什么更强：可解释为“幅值相关增益/绝对稳定友好”

工程里 `smooth(v,s)` 定义为（见 `01_soar/core/dsl.py`）：
\[
\phi(e)=s\tanh(e/s)
\]
它有三个非常关键、且能直接解释性能的性质：

### 2.1 小误差区：像高增益线性 P（压 RMSE）
当 $|e|\ll s$，有 $\tanh(e/s)\approx e/s$，所以
\[
\phi(e)\approx e\quad\Rightarrow\quad -k_p\phi(e)\approx -k_p e
\]
这等价于**在误差小的时候“线性高带宽”**，能把 figure8 周期轨道附近的跟踪误差压得很低。

### 2.2 大误差区：自动降增益/软饱和（减少姿态代价与振荡）
当 $|e|\gg s$，$\phi(e)\to \pm s$，所以比例项幅值被限制：
\[
|-k_p\phi(e)|\le k_p s
\]
这相当于**自带 gain scheduling + 输出限幅**：误差很大时不会继续把力矩推到极端，从而减少：

- 角速度变大（$\omega_x^2$）
- 大倾角（$\phi^2,\theta^2$）
- clip 后的相位滞后导致的振荡/超调

在 SCG 的 Q 权重下，这类“姿态激进/振荡”很容易把 state cost 拉爆。

### 2.3 经典非线性判稳接口：扇区有界（Lur’e / circle/popov 可用）
\[
\phi'(e)=\mathrm{sech}^2(e/s)\in(0,1]\quad\Rightarrow\quad \phi(e)\ \text{单调、奇函数、斜率受限}
\]
因此 $\phi$ 属于扇区 $[0,1]$（常见饱和/死区同类）。把“线性部分（无人机等效通道） + 扇区非线性”写成 Lur’e 形式后，可以用 **Circle criterion / Popov** 这类绝对稳定工具解释“为什么它在大范围工作点更稳”。

直觉总结：`smooth(tanh)` 让系统在不同幅值上呈现**不同等效闭环增益**，而 baseline 的线性增益是固定的。

---

## 3) `sign`（square）为什么更强：可解释为“近似滑模/时间最优 + 对角点鲁棒”

`sign(e)` 是不连续非线性。经典解释路径有两条：

### 3.1 滑模/Filippov：角点时“到达+保持”比精细线性更重要
square 的角点要求快速改变横向加速度方向。`u\propto \mathrm{sign}(e)` 等价于**变结构控制**：

- 误差在哪边，就立刻给出固定方向的最大纠偏（bang-bang）
- 配合速度项（你律里的 `+ k_d vel_y`）形成边界层，抑制抖振

在许多二阶/近二阶通道（位置-速度-加速度）上，bang-bang 控制与时间最优控制密切相关：**转角时能更快把状态拉回“可跟踪的走廊”**，直接降低位置误差的积分。

### 3.2 描述函数：等效增益随幅值变大而变小（避免过度激进）
对继电型非线性 $M\,\mathrm{sign}(\cdot)$，描述函数为：
\[
N(A)=\frac{4M}{\pi A}
\]
含义：当误差振荡幅值 $A$ 变大时，等效增益下降；当 $A$ 变小时，等效增益上升。

这正好对应 square 的需求：
- 大偏差（角点刚过去）需要“强拉回”但不希望无限放大
- 小偏差需要更强的等效增益把误差压到很小

而纯线性 PID/LQR 要同时满足这两点只能靠“固定高增益”，更容易出现振荡/超调 + clip。

---

## 4) 你这两条律里共同的“赢点”

从 `manual.md` 看，两条律都包含：

- **位置误差的非线性整形**（`smooth` 或 `sign`）→ 幅值相关增益
- **速度项**（`vel_y`）→ 提供阻尼，避免纯开关或纯比例带来的振荡
- **角速度项**（`ang_vel_x`）→ 直接压制 $\omega_x$，而 $\omega$ 虽然权重 0.01，但它会通过姿态/位置耦合放大位置误差

等效上，你把“外环误差→内环力矩”的通道做成了一个**非线性增益调度 + 阻尼注入 + 内环速率阻尼**的组合，这通常比固定结构的线性两环更贴合复杂机动。

---

## 5) 经典非线性分析怎么“证明/解释”我们的占优

下面列的是控制里最经典、也最容易跟你这两条律对上号的方法。每个方法我都写清：它能给什么“理论结论”，以及为什么会让我们在 figure8 / square 上更占优。

### 5.1 Lyapunov / ISS：解释“更大吸引域 + 抗扰更强”

- **能给的结论**：构造 $V(e,\dot e,\omega)$，证明 $\dot V\le -\alpha\|x\|^2 + \gamma\|d\|^2$（ISS 形式）或用 LaSalle 证明收敛。
- **为什么有利于我们**：`smooth(tanh)` 斜率受限、输出有界，更容易得到“负定项 + 有界扰动项”；`sign`+速度项可形成边界层，给出到达/收敛的可解释条件。

### 5.2 绝对稳定（Lur’e）/ Circle / Popov：解释“跨工作点不失稳”

- **能给的结论**：若非线性属于扇区 $[\alpha,\beta]$ 且线性部分满足 Circle/Popov 判据，则闭环对该类非线性**绝对稳定**。
- **为什么有利于我们**：`smooth=tanh` 属于典型扇区有界非线性（常用近似 $[0,1]$），这些判据就是为“饱和/斜率受限”准备的；而 baseline 的多处 clip 更像“事后截断”，理论接口更弱。

### 5.3 描述函数 / 谐波平衡：解释角点/开关导致的振荡与等效增益

- **能给的结论**：对 `sign`（继电）有 $N(A)=\frac{4M}{\pi A}$，配合线性部分 Nyquist 可预测极限环；对 `tanh` 可得到“随幅值下降”的等效增益。
- **为什么有利于我们**：square 的角点问题本质落在“开关非线性 + 线性动力学”的极限环/抖振机制上；figure8 的周期跟踪则受益于 `tanh` 的幅值相关增益（只在需要时高带宽）。

### 5.4 滑模 / Filippov：解释 `sign` 的“到达+保持”与角点鲁棒

- **能给的结论**：在 Filippov 意义下定义不连续闭环，证明切换面 $s(x)=0$ 的到达条件 $\dot s\,\mathrm{sign}(s) < -\eta$，并给出到达时间界与对匹配扰动的鲁棒性。
- **为什么有利于我们**：square 参考在角点瞬时换向，线性控制要靠固定高带宽“硬跟”→ 易振荡/clip；滑模的变结构更直接保证“误差往回拉”。

### 5.5 Poincaré / Floquet：解释 figure8 的“周期轨道稳定性”

- **能给的结论**：把周期 $T$ 的误差动力学写成 Poincaré 映射，检查 Floquet 乘子是否在单位圆内 → 周期轨道稳定性。
- **为什么有利于我们**：`tanh` 的幅值调度常让“周期内不同阶段”的等效闭环增益更合适，从而更容易把乘子压进单位圆。

---

## 6) 本仓库里可复现的“证据”入口

- toy 量化对比（不依赖 Isaac）：`results/toy_nonlinear/summary.json`
-	- 结论摘要（同一类二次代价口径的 toy 对比）：
	- `sine`：`smooth` $J\approx 4.08$ < `PID` $J\approx 4.45$ < `sign` $J\approx 4.33$ \(\ll `LQR` $J\approx 64.17$\)
	- `square`：`sign` $J\approx 9.45$ < `smooth` $J\approx 10.66$ < `PID` $J\approx 14.72$ \(\ll `LQR` $J\approx 121.96$\)
- Isaac 小信号采集（用于 FRF/Popov/描述函数落地）：
	- 数据：`results/id/u_tx_hover_sine.json`
	- 采集脚本：`scripts/nonlinear_id_collect.py`

基于当前这份 hover 小信号数据（1.5Hz 正弦激励 `u_tx`，幅值 0.12），用单频解调做的粗 FRF 估计为：

- $|G_{u\to\omega_x}(j\omega)|\approx 13.29$，相位 $\approx -1.40$ rad
- $|G_{u\to\phi}(j\omega)|\approx 1.79$，相位 $\approx -2.96$ rad
- $|G_{u\to y}(j\omega)|\approx 10.18$，相位 $\approx 3.13$ rad

这不是“完整辨识”，但足够支撑后续 Popov/Circle/描述函数的定量落地。

## 7) 可验证的预言（用于把解释落地成实验）

1. 把 figure8 的 `smooth` 令 $s\to\infty$（变得近似线性），性能应逐步靠近线性 PID/LQR。
2. 把 figure8 的 $s$ 减小：大误差时更“软饱和”，超调/姿态代价下降，但可能追踪变慢（RMSE 上升）。
3. square 的 `sign` 系数增大：转角更硬、更快，但更容易出现离散抖振/高频角速度（state cost 中的姿态项上升）。
4. 去掉 `vel_y` 项：`sign` 更容易抖、`smooth` 更容易振荡；去掉 `ang_vel_x` 项：姿态/角速度代价会上升并反过来恶化位置误差。

---

## 8) 论文式推导：用经典非线性工具说明“为何占优”

下面给出两段“可以直接放论文/报告正文”的推导：

- square：用 **Filippov/滑模** 解释角点鲁棒纠偏，用 **描述函数** 解释等效增益随幅值变化与振荡机制。
- figure8：用 **扇区/绝对稳定（Popov/Circle 入口）** 解释跨工作点稳定，用 **Floquet** 解释周期轨道收敛。

### 8.1 square：`sign` + 阻尼项为何比线性 PID/LQR 更适配角点

#### (A) 误差通道的二阶近似
对横向通道做最小近似（足以解释角点本质）：
\[
\dot e = \dot e,\quad \ddot e = -u + d(t)
\]
其中 $e=y_\text{ref}-y$，$u$ 是等效横向“加速度/力矩通道控制量”，$d(t)$ 表示耦合、建模误差与离散实现引入的匹配扰动（假设有界 $|d(t)|\le \bar d$）。

我们的 square 律在该通道上可抽象为（保留关键结构）：
\[
u = k_p\,\mathrm{sign}(e) + k_d\,\dot e
\]

#### (B) Lyapunov/ISS：强纠偏 + 有界扰动下仍稳定
取经典 Lyapunov 候选：
\[
V = \tfrac12\dot e^2 + k_p|e|\ge 0
\]
则在 $e\ne 0$ 的 Filippov 意义下：
\[
\dot V = \dot e\ddot e + k_p\,\mathrm{sign}(e)\,\dot e
	= \dot e(-k_p\,\mathrm{sign}(e) - k_d\dot e + d) + k_p\,\mathrm{sign}(e)\,\dot e
	= -k_d\dot e^2 + d\dot e
\]
从而
\[
\dot V \le -(k_d-\varepsilon)\dot e^2 + \tfrac{\bar d^2}{4\varepsilon},\quad \forall\varepsilon\in(0,k_d)
\]
这是标准 ISS/有界收敛结构：扰动有界时 $\dot e$ 收敛到与 $\bar d/k_d$ 成比例的小邻域，进而 $e$ 不会在角点“被甩飞”。

**为什么这比线性 PID/LQR 更占优**：角点会让系统短时进入大误差/大速度区。线性控制想兼顾“快转角+不振荡”通常只能提高固定增益，结果是激发高频姿态/角速度并触发 clip（在 SCG 中 roll/pitch 权重 0.5，代价会被放大）。`sign` 的强纠偏是结构性的，不依赖无限增益。

#### (C) 描述函数：等效增益随幅值自适应
将 $k_p\,\mathrm{sign}(e)$ 视为继电非线性，输入近似 $e(t)=A\sin\omega t$ 时描述函数为：
\[
N(A)=\frac{4k_p}{\pi A}
\]
即等效增益随幅值 $A$ 增大而下降。

- 偏差大（角点刚过）：不会无限增益，把系统推向更严重的振荡/clip。
- 偏差小（回到线段跟踪）：等效增益变大，把误差压得更低。

这就是 square 的理想“角点强纠偏 + 线段精跟踪”形态，而固定线性增益无法同时兼顾。

### 8.2 figure8：`smooth=tanh` 为什么更“跨工作点稳、周期轨道更收敛”

#### (A) 扇区/斜率受限：绝对稳定入口（Popov/Circle）
定义非线性：
\[
\phi(e)=s\tanh(e/s)
\]
对任意 $e\ne 0$：
\[
0 < \frac{\phi(e)}{e} \le 1,\quad \phi'(e)=\mathrm{sech}^2(e/s)\in(0,1]
\]
因此它属于典型扇区 $[0,1]$ 且斜率受限。把闭环写成 Lur’e 互联（线性块 $G$ 与静态非线性 $\phi$ 的反馈互联），即可使用 Circle/Popov 判据的逻辑：

- 误差大时 $\phi'(e)$ 变小，等效闭环增益自动下降，避免相位裕度不足导致的振荡。
- 误差小时 $\phi'(e)\approx 1$，等效增益恢复，高带宽压 RMSE。

这相当于“增益随幅值调度”，更适配 figure8 周期内不断变化的工作点。

#### (B) Floquet 视角：周期参考下为什么更容易收敛
figure8 是周期参考 $r(t+T)=r(t)$。考虑误差系统围绕周期轨道的变分方程：
\[
\delta\dot x = A(t)\,\delta x,\quad A(t+T)=A(t)
\]
单周期状态转移 $\Phi(T)$ 的特征值 $\mu_i$ 为 Floquet 乘子。若 $|\mu_i|<1$ 则周期轨道局部稳定。

`tanh` 使得“等效比例增益” $k_p\phi'(e(t))$ 满足
\[
0 < k_p\phi'(e(t)) \le k_p
\]
从而 $A(t)$ 的变化更受控，不会在周期某些阶段突然变得过“硬”（导致 $|\mu_i|$ 外逸）。直觉上这解释了：`smooth` 往往能在 figure8 的全周期同时做到“压误差 + 不爆姿态”。

如果你愿意，我可以基于你当前的 figure8 / square 环境，把“等效通道（u_tx→y）”做一个小信号辨识（FRF 或线性化），然后把 Circle/Popov 或描述函数的判据真正算出来，变成可复现实验而不是口头解释。

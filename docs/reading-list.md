# Soar 文献清单（可解释、无分支控制律）

> 详注：见 docs/reading-notes.md（第一批 8 篇深读）
> https://github.com/linlexi568/soar/blob/main/docs/reading-notes.md

围绕本仓库的目标：合成“无分支、连续、非线性状态反馈”控制律 u = f(x)，并兼顾可解释性与安全稳定性。下面为按主题精选的论文清单（持续扩充）。

> 标签说明：
> - 符号/程序策略 = 可解释/程序化策略、DSL、树/程序结构
> - 状态反馈 = 非线性/状态反馈控制律
> - 稳定性/安全 = 李雅普诺夫、收缩度量、CBF、形式化保证
> - 四旋翼/无人机 = UAV/Quadrotor 相关
> - 合成/抽象 = 反应式合成、符号抽象、SyGuS

---

## 符号/程序化策略（Programmatic/Symbolic Policies）
- Programmatically Interpretable Reinforcement Learning (PIRL) — ICML 2018  [arXiv:1804.02477](https://arxiv.org/abs/1804.02477)  ｜ 符号/程序策略
- Learning to Synthesize Programs as Interpretable and Generalizable Policies (LEAPS) — NeurIPS 2021  [arXiv:2108.13643](https://arxiv.org/abs/2108.13643)  ｜ 符号/程序策略
- Interpretable and Editable Programmatic Tree Policies for Reinforcement Learning — 2024  [arXiv:2405.14956](https://arxiv.org/abs/2405.14956)  ｜ 符号/程序策略
- Reinforcement Learning with Physics-Informed Symbolic Program Priors — 2025  [arXiv:2506.22365](https://arxiv.org/abs/2506.22365)  ｜ 符号/程序策略
- BASIL: Best-Action Symbolic Interpretable Learning — 2025  [arXiv:2506.00328](https://arxiv.org/abs/2506.00328)  ｜ 符号/程序策略

## 非线性状态反馈与稳定性保证（Lyapunov/Contraction/CBF）
- Lyapunov Neural ODE State-Feedback Control Policies — 2025/2024  [arXiv:2409.00393](https://arxiv.org/abs/2409.00393)  ｜ 状态反馈, 稳定性/安全
- Neural Contraction Metrics with Formal Guarantees for Discrete-Time Nonlinear Systems — 2025  [arXiv:2504.17102](https://arxiv.org/abs/2504.17102)  ｜ 稳定性/安全
- MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained RL — 2025  [arXiv:2504.02565](https://arxiv.org/abs/2504.02565)  ｜ 稳定性/安全
- DiffOP: RL of Optimization-Based Control Policies via Implicit Policy Gradients — 2025/2024  [arXiv:2411.07484](https://arxiv.org/abs/2411.07484)  ｜ 稳定性/安全, 可解释
- SafEDMD: Koopman-based data-driven controller design for nonlinear systems — Automatica 2025  [arXiv:2402.03145](https://arxiv.org/abs/2402.03145)  ｜ 状态反馈, 可解释
- Off Policy Lyapunov Stability in Reinforcement Learning — CoRL 2025  [arXiv:2509.09863](https://arxiv.org/abs/2509.09863)  ｜ 稳定性/安全
- Preventing Inactive CBF Safety Filters Caused by Invalid Relative Degree Assumptions — 2024/2025  [arXiv:2409.11171](https://arxiv.org/abs/2409.11171)  ｜ 稳定性/安全, CBF

## 四旋翼/无人机与对比研究（UAV/Quadrotor）
- Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking — 2025  [arXiv:2506.17832](https://arxiv.org/abs/2506.17832)  ｜ 四旋翼/无人机, 基线对比
- RAPTOR: A Foundation Policy for Quadrotor Control — 2025  [arXiv:2509.11481](https://arxiv.org/abs/2509.11481)  ｜ 四旋翼/无人机, 基础策略
- Dynamics-Invariant Quadrotor Control using Scale-Aware Deep RL — 2025  [arXiv:2503.09622](https://arxiv.org/abs/2503.09622)  ｜ 四旋翼/无人机

## 合成/抽象/形式化（Reactive Synthesis, Symbolic Abstraction, SyGuS）
- CESAR: Control Envelope Synthesis via Angelic Refinements — TACAS 2024  [arXiv:2311.02833](https://arxiv.org/abs/2311.02833)  ｜ 合成/抽象, 稳定性/安全
- Data-Driven Dynamic Controller Synthesis for Discrete-Time General Nonlinear Systems — 2025  [arXiv:2503.08060](https://arxiv.org/abs/2503.08060)  ｜ 合成/抽象, 稳定性/安全
- Data-Driven Synthesis of Symbolic Abstractions with Guaranteed Confidence — L-CSS 2022  [arXiv:2206.09397](https://arxiv.org/abs/2206.09397)  ｜ 合成/抽象
- Zonotope-based Symbolic Controller Synthesis for LTL — 2024  [arXiv:2405.00924](https://arxiv.org/abs/2405.00924)  ｜ 合成/抽象, 形式化

---

## 复用检索链接（一键打开）
- Programmatic policy reinforcement learning：
  https://arxiv.org/search/?query=programmatic+policy+reinforcement+learning&searchtype=all&abstracts=show&order=-announced_date_first&size=50
- Symbolic policy reinforcement learning：
  https://arxiv.org/search/?query=symbolic+policy+reinforcement+learning&searchtype=all&abstracts=show&order=-announced_date_first&size=50
- Quadrotor reinforcement learning control：
  https://arxiv.org/search/?query=quadrotor+reinforcement+learning+control&searchtype=all&abstracts=show&order=-announced_date_first&size=50
- Learning nonlinear state feedback control：
  https://arxiv.org/search/?query=learning+nonlinear+state+feedback+control&searchtype=all&abstracts=show&order=-announced_date_first&size=50
- Symbolic controller synthesis：
  https://arxiv.org/search/?query=symbolic+controller+synthesis&searchtype=all&abstracts=show&order=-announced_date_first&size=50
- Syntax-guided synthesis + control（SyGuS/合成）：
  https://arxiv.org/search/?query=syntax-guided+synthesis+control&searchtype=all&abstracts=show&order=-announced_date_first&size=50

---

## 使用建议（与 Soar 对齐）
- 与本仓库“无分支、连续、状态反馈”原则最贴近的优先读：PIRL、LEAPS、Programmatic Tree Policies、BASIL、Physics-Informed Symbolic Priors、Lyapunov N-ODE、Neural Contraction Metrics、MAD、SafEDMD。
- 四旋翼落地：优先看对比工作与基础策略（Leveling the Playing Field、RAPTOR），利于设定合理基线与指标。
- 安全/稳定：CBF/李雅普诺夫/收缩度量条目可为搜索空间加入软/硬约束提供思路。

> 如需我每周自动刷新清单，可追加 arXiv API/RSS 方案（可选）。

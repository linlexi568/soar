# Soar 控制规则与轨迹规范

## 轨迹起点规范

| 轨迹类型 | 起点坐标 (t=0) | 说明 | 状态 |
|---------|----------------|------|------|
| Square | [0, 0, 1] | 位于中心，先向 +y 移动 | ✅ 已修正 |
| Circle | [R, 0, 1] | 位于圆周右侧 (R=0.9时为 [0.9, 0, 1]) | ✅ 一致 |
| Figure8 | [0, 0, 1] | 位于中心 | ✅ 一致 |

## 最优控制规则

### Figure8 (Smooth 非线性项)
```
u_tx = ((-0.489 * smooth(pos_err_y, s=1.285)) + (1.062 * vel_y)) - (0.731 * ang_vel_x)
u_ty = (( 0.489 * smooth(pos_err_x, s=1.285)) - (1.062 * vel_x)) - (0.731 * ang_vel_y)
```
**参数**: k_p=0.489, k_s=1.285, k_d=1.062, k_w=0.731  
**Cost**: 81.78  
**非线性算子**: `smooth(e, s) = s * tanh(e/s)` (平滑饱和函数)

### Square (Smooth 近似 Sign)
```
u_tx = ((-1.643 * smooth(pos_err_y, s=0.786)) + (1.633 * vel_y)) - (0.629 * ang_vel_x)
u_ty = (( 1.643 * smooth(pos_err_x, s=0.786)) - (1.633 * vel_x)) - (0.629 * ang_vel_y)
```
**参数**: k_p=1.643, k_s=0.786, k_d=1.633, k_w=0.629 (BO调优结果)  
**Cost**: **44.38** (优于原 sign 版本 52.99)  
**非线性算子**: `smooth(e, s) = s * tanh(e/s)` 其中 s=0.786 提供中等饱和 (Lipschitz 连续)
**改动**: 从 sign(e) 改为 smooth(e, s)，确保 Lipschitz 连续性，性能提升 16.2%

### Circle (Smooth 非线性项)
```
u_tx = ((-2.104 * smooth(pos_err_y, s=0.296)) + (1.111 * vel_y)) - (0.727 * ang_vel_x)
u_ty = (( 2.104 * smooth(pos_err_x, s=0.296)) - (1.111 * vel_x)) - (0.727 * ang_vel_y)
```
**参数**: k_p=2.104, k_s=0.296, k_d=1.111, k_w=0.727  
**Cost**: 144.21  
**非线性算子**: `smooth(e, s) = s * tanh(e/s)` (平滑饱和函数)

---

## 技术说明

### Smooth 近似 Sign (Lipschitz 连续性)
在 Square 轨迹中，我们使用 **smooth 算子的极限形式**来近似 sign 函数：

```python
sign(e) = lim_{s→0} smooth(e, s) = lim_{s→0} s * tanh(e/s)
```

**为什么不直接用 sign？**
- `sign(e)` 在 e=0 处**不可微**，不是 Lipschitz 连续
- 导致数值优化和梯度计算困难
- 在实际控制中可能产生抖振 (chattering)

**Smooth 近似的优势：**
- `smooth(e, s)` 处处可微，Lipschitz 连续
- 当 s 很小（如 s=0.1）时，行为接近 sign
- 在 |e| >> s 时：`smooth(e, s) ≈ ±s ≈ sign(e) * s`
- 在 |e| << s 时：`smooth(e, s) ≈ e` (线性)

**参数选择：**
- **Figure8**: s=1.285 (较大，平滑过渡)
- **Square**: s=0.1 (很小，近似 bang-bang)
- **Circle**: s=0.296 (中等，平衡响应)

### 复现方式
1. **01_soar 环境**: 使用 DSL AST 构建控制律，通过 `UnaryOpNode('smooth', ...)` 调用
2. **Benchmark 环境**: 使用 `benchmark/baselines/soar_controller.py` 中的 `SoarController` 类
3. **Square 轨迹调优**: 使用 `01_soar/tune_square_smooth_bo.py` 进行 BO 调参
   ```bash
   cd 01_soar
   python tune_square_smooth_bo.py  # 默认 50 trials
   ```
4. **评估命令**:
   ```bash
   cd benchmark
   python baselines/eval_soar.py --task square --episodes 10
   python baselines/eval_soar.py --task all  # 评估所有轨迹
   ```

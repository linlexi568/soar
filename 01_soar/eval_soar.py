#!/usr/bin/env python3
"""
Soar 程序评估脚本 - 使用 SCG 精确奖励（与训练/基线完全对齐）
所有参数写在脚本顶部，直接修改即可

奖励函数: r_t = -(x_err^T Q x_err + u^T R u)
  Q = diag([1, 0.01, 1, 0.01, 1, 0.01, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
  R = 0.0001
"""

import sys
from pathlib import Path
import os

# ============================================================================
# 路径设置 (Isaac Gym 必须在 torch 之前导入)
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / '01_soar') not in sys.path:
    sys.path.insert(0, str(ROOT / '01_soar'))

_ISAAC_GYM_PY = ROOT / 'isaacgym' / 'python'
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

# ============================================================================
#                    ★★★ 评估参数配置 (修改这里) ★★★
# ============================================================================

# --------------------- 任务配置 ---------------------
TASK = "square"          # 选择: circle, square, helix, figure8, hover
DURATION = 5.0            # 每个episode时长(秒)

# --------------------- 评估配置 ---------------------
NUM_ENVS = 8196           # 并行环境数 (与训练对齐 - 训练时也是 8196)
REPLICAS = 1              # 重复评估次数
DEVICE = "cuda:0"         # 设备

# --------------------- 程序路径 ---------------------
# 修改为你要评估的程序路径
PROGRAM_PATH = ROOT / "results" / "soar_train" / f"{TASK}_safe_control_tracking_best.json"

# --------------------- 评估设置 ---------------------
STRICT_NO_PRIOR = True    # 严格无先验模式 (直接 u_* 输出，与训练对齐)
ENABLE_MAD = False        # MAD 安全壳 (训练时关闭，必须与训练一致)
REWARD_REDUCTION = "sum"  # 奖励归约方式 (训练时用 sum)
ZERO_ACTION_PENALTY = 0.0 # 零动作惩罚 (测试时设为 0)

# --------------------- 输出配置 ---------------------
OUTPUT_PATH = ROOT / "results" / "soar_test" / f"eval_{TASK}.json"

# ============================================================================
#                         评估代码 (不需要修改)
# ============================================================================

def main():
    import json
    from datetime import datetime
    
    print("=" * 70)
    print(f"Soar 程序评估 - 任务: {TASK}")
    print("=" * 70)
    print(f"  程序路径: {PROGRAM_PATH}")
    print(f"  并行环境: {NUM_ENVS}")
    print(f"  重复次数: {REPLICAS}")
    print(f"  时长: {DURATION}s")
    print(f"  MAD 安全壳: {'开启' if ENABLE_MAD else '关闭'}")
    print()
    
    # 检查程序文件
    if not PROGRAM_PATH.exists():
        print(f"❌ 错误: 程序文件不存在: {PROGRAM_PATH}")
        return
    
    # 加载程序（使用 deserialize_program 将 JSON 转换为 AST 对象）
    # 这确保时间算子（ema/delay/diff/rate）有正确的状态缓冲区
    from core.serialization import deserialize_program
    
    with open(PROGRAM_PATH, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'rules' in data:
        program = deserialize_program(data)  # 转换为 AST 对象
        meta = data.get('meta', {})
    elif isinstance(data, list):
        # 如果是旧格式（直接是 rules 列表），包装后反序列化
        program = deserialize_program({'rules': data})
        meta = {}
    else:
        print(f"❌ 错误: 无效的程序格式")
        return
    
    print(f"✓ 程序加载成功: {len(program)} 条规则")
    if meta:
        print(f"  训练时奖励: {meta.get('reward', 'N/A')}")
        print(f"  训练迭代: {meta.get('iteration', 'N/A')}")
    print()
    
    # 导入评估器
    from utils.batch_evaluation import BatchEvaluator
    from utilities.trajectory_presets import get_scg_trajectory_config
    
    # 构建轨迹配置
    traj_cfg = get_scg_trajectory_config(TASK)
    trajectory_config = {
        'type': traj_cfg.task,
        'params': dict(traj_cfg.params),
        'initial_xyz': list(traj_cfg.center)
    }
    
    # 创建评估器 (完全对齐训练配置)
    evaluator = BatchEvaluator(
        isaac_num_envs=NUM_ENVS,
        reward_profile='safe_control_tracking',
        trajectory_config=trajectory_config,
        duration=int(DURATION),
        device=DEVICE,
        use_fast_path=True,
        strict_no_prior=STRICT_NO_PRIOR,
        reward_reduction=REWARD_REDUCTION,
        zero_action_penalty=ZERO_ACTION_PENALTY,
        replicas_per_program=REPLICAS,
        enable_output_mad=ENABLE_MAD,
    )
    
    # 评估
    print("开始评估...")
    rewards_train, rewards_true, metrics_list = evaluator.evaluate_batch_with_metrics(
        programs=[program]
    )
    
    reward_train = rewards_train[0]
    reward_true = rewards_true[0]
    metrics = metrics_list[0] if metrics_list else {}
    
    print()
    print("=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"  训练奖励 (含惩罚): {reward_train:.4f}")
    print(f"  真实奖励 (不含惩罚): {reward_true:.4f}")
    print()
    if metrics:
        print("  代价分解:")
        print(f"    状态代价 (state_cost): {metrics.get('state_cost', 0):.6f}")
        print(f"    动作代价 (action_cost): {metrics.get('action_cost', 0):.6f}")
    print()
    
    if meta and 'reward' in meta:
        train_meta_reward = meta['reward']
        diff = abs(train_meta_reward - reward_train)
        print(f"  训练时记录奖励: {train_meta_reward:.4f}")
        print(f"  重新评估奖励: {reward_train:.4f}")
        print(f"  差异: {diff:.4f} ({diff/abs(train_meta_reward)*100:.1f}%)")
        print()
    
    # 保存结果
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "algorithm": "Soar",
        "task": TASK,
        "duration_sec": DURATION,
        "num_envs": NUM_ENVS,
        "replicas": REPLICAS,
        "program_path": str(PROGRAM_PATH),
        "program_rules": len(program),
        "reward_train": float(reward_train),
        "reward_true": float(reward_true),
        "state_cost": float(metrics.get('state_cost', 0)),
        "action_cost": float(metrics.get('action_cost', 0)),
        "reward_type": "scg_exact",
        "reward_reduction": REWARD_REDUCTION,
        "strict_no_prior": STRICT_NO_PRIOR,
        "enable_mad": ENABLE_MAD,
        "training_meta_reward": meta.get('reward'),
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# ==============================================================================
# Soar 训练启动脚本（仅 Train 模式）
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================
# 环境检测 & 显存优化参数
# ==============================
# Ranking训练显存控制：每批最多处理的图数量（降低此值可减少OOM风险）
# 推荐值：4（默认），高显存卡可提升至8或16；6GB以下显卡建议2或1
export RANKING_GNN_CHUNK=4

# PyTorch 内存碎片控制：避免 reserved >> allocated 导致的 OOM（可按需覆盖）
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

# 降低 BatchEvaluator/BO 的终端输出噪声（INFO→WARN）：
export BATCH_EVAL_QUIET=1
export BATCH_EVAL_LOG_LEVEL=WARN

# ==============================
# 并行度配置（可按需覆盖）
# ==============================
# CPU 线程数（缺省=物理核心数）；可通过 export CPU_THREADS=16 覆盖
CPU_THREADS_DEFAULT=64
export CPU_THREADS="${CPU_THREADS:-$CPU_THREADS_DEFAULT}"

# 绑定常见数值库/并行运行时的线程数
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS}"
export PYTORCH_NUM_THREADS="${PYTORCH_NUM_THREADS:-$CPU_THREADS}"

# Isaac Gym 库路径：PhysX/USD/rlgpu 等 native 依赖
ISAAC_BINDINGS="${REPO_ROOT}/isaacgym/python/isaacgym/_bindings/linux-x86_64"
if [[ -d "$ISAAC_BINDINGS" ]]; then
  export LD_LIBRARY_PATH="${ISAAC_BINDINGS}:${LD_LIBRARY_PATH:-}"
fi

# 🔥 关键修复：避免 Isaac Gym PhysX 初始化死锁
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
# 完全禁用 OpenBLAS 避免与 PhysX 冲突
unset OPENBLAS_CORETYPE
export OPENBLAS_NUM_THREADS=1
# 强制使用MKL或fallback到单线程
export MKL_THREADING_LAYER="GNU"
# 🔧 PhysX GPU 优化：禁用多线程调度避免死锁
export PHYSX_GPU_MAX_THREADS=1
# 🔧 确保 Python stdout 不缓冲
export PYTHONUNBUFFERED=1

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

# ==============================
# 训练参数集中配置（在这里改）
# ==============================
TRAIN_TOTAL_ITERS=500
TRAIN_MCTS_SIMS=100
TRAIN_UPDATE_FREQ=999999
TRAIN_STEPS_PER_UPDATE=10
TRAIN_ISAAC_ENVS=8192  # 单轴MCTS每批最多评估~30个程序，128环境足够
TRAIN_MCTS_LEAF_BATCH=8192  # 与环境数匹配
# 完全放开 Progressive Widening（默认关闭；开启后节点一次性扩展全部可变异，树宽可能爆炸）
DISABLE_PROGRESSIVE_WIDENING=False
ASYNC_TRAINING=false
ASYNC_UPDATE_INTERVAL=0.15
ASYNC_MAX_STEPS_PER_ITER=1

TRAJ="figure8"
DURATION=5
REWARD_PROFILE="safe_control_tracking"
REWARD_REDUCTION="sum"
EVAL_REPLICAS_PER_PROGRAM=1
MIN_STEPS_FRAC=0.0

# 放宽零动作惩罚，避免搜索早期过度压制复杂控制
ZERO_ACTION_PENALTY=1.0
ZERO_ACTION_PENALTY_DECAY=0.95
ZERO_ACTION_PENALTY_MIN=0.2
ACTION_SCALE_MULTIPLIER=1.0

# 🚨 MAD 安全壳：默认禁用（因为悬停推力约束已经在 DSL 搜索中强制 u_fz >= hover_thrust）
# 若需启用 MAD 作为双重保险，设置为 true
ENABLE_OUTPUT_MAD=false

# MAD 安全壳参数（仅在 ENABLE_OUTPUT_MAD=true 时生效）
# 强制最小悬停推力: Crazyflie mass=0.027kg, g=9.81 → mg ≈ 0.265N
MAD_MIN_FZ=0.265
MAD_MAX_FZ=7.5
MAD_MAX_XY=0.12
MAD_MAX_YAW=0.04
MAD_MAX_DELTA_FZ=1.5
MAD_MAX_DELTA_XY=0.03
MAD_MAX_DELTA_YAW=0.02

ROOT_DIRICHLET_EPS_INIT=0.25
ROOT_DIRICHLET_EPS_FINAL=0.10
ROOT_DIRICHLET_ALPHA_INIT=0.30
ROOT_DIRICHLET_ALPHA_FINAL=0.20
HEURISTIC_DECAY_WINDOW=350
POLICY_TEMPERATURE=1.0
# 限制MCTS搜索深度为3（类似PID结构复杂度）
EXPLORATION_WEIGHT=2.5
PUCT_C=1.5
MAX_DEPTH=3

USE_GPU_EXPRESSION=true
USE_META_RL=false

GNN_STRUCTURE_HIDDEN=192
GNN_STRUCTURE_LAYERS=4
GNN_STRUCTURE_HEADS=6
GNN_FEATURE_LAYERS=4
GNN_FEATURE_HEADS=8
GNN_DROPOUT=0.2

NN_BATCH_SIZE=128
NN_LEARNING_RATE=0.001
NN_REPLAY_CAPACITY=50000

USE_RANKING=false
ENABLE_RANKING_MCTS_BIAS=false
ENABLE_VALUE_HEAD=false
ENABLE_RANKING_REWEIGHT=false
RANKING_BIAS_BETA=0.3
RANKING_REWEIGHT_BETA=0.2
RANKING_BLEND_INIT=0.3
RANKING_BLEND_MAX=0.8
RANKING_BLEND_WARMUP=100

PRIOR_PROFILE="structure_stability"   # 档位键可忽略，具体权重按下方最大档覆盖
STRUCTURE_PRIOR_WEIGHT=0              # 结构先验权重(最大)
STABILITY_PRIOR_WEIGHT=0             # 稳定性先验权重(强)
PRIOR_LEVEL=2                          # 约束等级

ENABLE_BAYESIAN_TUNING=true
BO_BATCH_SIZE=8
BO_ITERATIONS=2

AST_PIPELINE=false
DEBUG_PROGRAMS=false
DEBUG_PROGRAMS_LIMIT=20
DEBUG_REWARDS=false

RESULT_DIR="${REPO_ROOT}/results"
RUN_TAG=$(date '+%Y%m%d_%H%M%S')
SCG_RESULT_DIR="${RESULT_DIR}/soar_train"
mkdir -p "$SCG_RESULT_DIR"
TRAIN_SAVE_PATH="${SCG_RESULT_DIR}/${TRAJ}_${REWARD_PROFILE}_best.json"
CHECKPOINT_FREQ=50
WARM_START=""
ELITE_ARCHIVE_SIZE=50
CURRICULUM_MODE="none"
PROGRAM_HISTORY_PATH="01_soar/results/program_history.jsonl"

TRAIN_ENTRY="${REPO_ROOT}/01_soar/train_online.py"

if [[ ! -f "$TRAIN_ENTRY" ]]; then
  echo "[run.sh] 训练入口脚本缺失: $TRAIN_ENTRY" >&2
  exit 1
fi

echo "============================================================"
echo "Soar 训练启动 (SCG 直连模式)"
echo "============================================================"
echo "总轮数: ${TRAIN_TOTAL_ITERS}"
echo "MCTS模拟: ${TRAIN_MCTS_SIMS} 次/轮"
echo "并行环境: ${TRAIN_ISAAC_ENVS}"
echo "并行度(线程): CPU_THREADS=${CPU_THREADS} | OMP=${OMP_NUM_THREADS} | MKL=${MKL_NUM_THREADS} | NUMEXPR=${NUMEXPR_NUM_THREADS} | PYTORCH=${PYTORCH_NUM_THREADS}"
echo "轨迹: ${TRAJ} | 时长: ${DURATION}s | 奖励: ${REWARD_PROFILE}(${REWARD_REDUCTION})"
echo "输出路径: $TRAIN_SAVE_PATH"
echo "============================================================"

cmd=(
  "$PYTHON_BIN" "$TRAIN_ENTRY"
  --total-iters "$TRAIN_TOTAL_ITERS"
  --mcts-simulations "$TRAIN_MCTS_SIMS"
  --update-freq "$TRAIN_UPDATE_FREQ"
  --train-steps-per-update "$TRAIN_STEPS_PER_UPDATE"
  --batch-size "$NN_BATCH_SIZE"
  --replay-capacity "$NN_REPLAY_CAPACITY"
  --learning-rate "$NN_LEARNING_RATE"
  --gnn-structure-hidden "$GNN_STRUCTURE_HIDDEN"
  --gnn-structure-layers "$GNN_STRUCTURE_LAYERS"
  --gnn-structure-heads "$GNN_STRUCTURE_HEADS"
  --gnn-feature-layers "$GNN_FEATURE_LAYERS"
  --gnn-feature-heads "$GNN_FEATURE_HEADS"
  --gnn-dropout "$GNN_DROPOUT"
  --exploration-weight "$EXPLORATION_WEIGHT"
  --puct-c "$PUCT_C"
  --max-depth "$MAX_DEPTH"
  --mcts-leaf-batch-size "$TRAIN_MCTS_LEAF_BATCH"
  --traj "$TRAJ"
  --duration "$DURATION"
  --isaac-num-envs "$TRAIN_ISAAC_ENVS"
  --eval-replicas-per-program "$EVAL_REPLICAS_PER_PROGRAM"
  --min-steps-frac "$MIN_STEPS_FRAC"
  --reward-reduction "$REWARD_REDUCTION"
  --reward-profile "$REWARD_PROFILE"
  --prior-profile "$PRIOR_PROFILE"
  --structure-prior-weight "$STRUCTURE_PRIOR_WEIGHT"
  --stability-prior-weight "$STABILITY_PRIOR_WEIGHT"
  --zero-action-penalty "$ZERO_ACTION_PENALTY"
  --zero-action-penalty-decay "$ZERO_ACTION_PENALTY_DECAY"
  --zero-action-penalty-min "$ZERO_ACTION_PENALTY_MIN"
  --action-scale-multiplier "$ACTION_SCALE_MULTIPLIER"
  --mad-min-fz "$MAD_MIN_FZ"
  --mad-max-fz "$MAD_MAX_FZ"
  --mad-max-xy "$MAD_MAX_XY"
  --mad-max-yaw "$MAD_MAX_YAW"
  --mad-max-delta-fz "$MAD_MAX_DELTA_FZ"
  --mad-max-delta-xy "$MAD_MAX_DELTA_XY"
  --mad-max-delta-yaw "$MAD_MAX_DELTA_YAW"
  --policy-temperature "$POLICY_TEMPERATURE"
  --prior-level "$PRIOR_LEVEL"
  --save-path "$TRAIN_SAVE_PATH"
  --checkpoint-freq "$CHECKPOINT_FREQ"
  --elite-archive-size "$ELITE_ARCHIVE_SIZE"
  --program-history-path "$PROGRAM_HISTORY_PATH"
)

if [[ "$DISABLE_PROGRESSIVE_WIDENING" == "true" ]]; then
  cmd+=(--disable-progressive-widening)
fi

if [[ "$AST_PIPELINE" == "true" ]]; then
  cmd+=(--ast-pipeline)
fi

if [[ "$DEBUG_PROGRAMS" == "true" ]]; then
  cmd+=(--debug-programs --debug-programs-limit "$DEBUG_PROGRAMS_LIMIT")
fi

if [[ "$DEBUG_REWARDS" == "true" ]]; then
  cmd+=(--debug-rewards)
fi

if [[ -n "$WARM_START" ]]; then
  cmd+=(--warm-start "$WARM_START")
fi

if [[ "$ENABLE_OUTPUT_MAD" == "false" ]]; then
  cmd+=(--disable-output-mad)
else
  cmd+=(--enable-output-mad)
fi

if [[ "$USE_GPU_EXPRESSION" == "false" ]]; then
  cmd+=(--disable-gpu-expression)
else
  cmd+=(--use-fast-path)
fi

if [[ "$USE_RANKING" == "false" ]]; then
  cmd+=(--use-ranking false)
else
  cmd+=(
    --use-ranking true
    --ranking-lr 0.001
    --ranking-blend-init "$RANKING_BLEND_INIT"
    --ranking-blend-max "$RANKING_BLEND_MAX"
    --ranking-blend-warmup "$RANKING_BLEND_WARMUP"
  )
fi

if [[ "$ENABLE_RANKING_MCTS_BIAS" == "true" ]]; then
  cmd+=(--enable-ranking-mcts-bias --ranking-bias-beta "$RANKING_BIAS_BETA")
fi

if [[ "$ENABLE_VALUE_HEAD" == "true" ]]; then
  cmd+=(--enable-value-head)
fi

if [[ "$ENABLE_RANKING_REWEIGHT" == "true" ]]; then
  cmd+=(--enable-ranking-reweight --ranking-reweight-beta "$RANKING_REWEIGHT_BETA")
fi

if [[ "$USE_META_RL" == "true" ]]; then
  cmd+=(--use-meta-rl --meta-rl-checkpoint "$META_CKPT")
else
  cmd+=(
    --root-dirichlet-eps-init "$ROOT_DIRICHLET_EPS_INIT"
    --root-dirichlet-eps-final "$ROOT_DIRICHLET_EPS_FINAL"
    --root-dirichlet-alpha-init "$ROOT_DIRICHLET_ALPHA_INIT"
    --root-dirichlet-alpha-final "$ROOT_DIRICHLET_ALPHA_FINAL"
    --heuristic-decay-window "$HEURISTIC_DECAY_WINDOW"
  )
fi

if [[ "$ASYNC_TRAINING" == "true" ]]; then
  cmd+=(--async-training --async-update-interval "$ASYNC_UPDATE_INTERVAL")
  if [[ -n "$ASYNC_MAX_STEPS_PER_ITER" ]]; then
    cmd+=(--async-max-steps-per-iter "$ASYNC_MAX_STEPS_PER_ITER")
  fi
fi

if [[ "$ENABLE_BAYESIAN_TUNING" == "true" ]]; then
  cmd+=(--enable-bayesian-tuning --bo-batch-size "$BO_BATCH_SIZE" --bo-iterations "$BO_ITERATIONS")
fi

LOG_FILE="${REPO_ROOT}/logs/train_${RUN_TAG}.log"
mkdir -p "${REPO_ROOT}/logs"

echo "日志将保存到: $LOG_FILE"
echo ""

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo "✅ 训练成功完成"
  echo "============================================================"
else
  echo ""
  echo "============================================================"
  echo "❌ 训练异常退出 (退出码: $TRAIN_EXIT_CODE)"
  echo "完整日志已保存: $LOG_FILE"
  echo "============================================================"
fi

exit $TRAIN_EXIT_CODE

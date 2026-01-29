#!/usr/bin/env bash
# ==============================================================================
# PID / LQR Baseline 调参启动脚本
# 用法: ./scripts/tune_baselines.sh [--isaac] [轨道类型] [时长] [迭代次数]
# 示例:
#   ./scripts/tune_baselines.sh figure8 5 20
#   ./scripts/tune_baselines.sh hover 10 30
#   ./scripts/tune_baselines.sh circle 8
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ==============================================================================
# 参数配置（可通过命令行覆盖）
# ==============================================================================
DEFAULT_TASK="square"            # 轨迹类型: hover / figure8 / circle / square / helix
DEFAULT_DURATION="5"             # 仿真时长（秒）
DEFAULT_PID_REFINE_ITERS="150"   # PID 局部精调迭代次数（旧脚本用）
DEFAULT_LQR_REFINE_ITERS="150"   # LQR 局部精调迭代次数（旧脚本用）
DEFAULT_TRIALS="200"             # Isaac 局部搜索试次数（新脚本用）

# ==============================================================================
# Isaac 基线配置：按需修改此处即可（无需通过命令行参数）
# 字段: algo | pid_mode | lqr_mode | task | duration | episodes_per_eval | trials | label
# pid_mode / lqr_mode 用 "-" 表示未使用该字段
# ==============================================================================
ISAAC_RUNS=(
    "pid|normal|-|square|5|3|200|pid_normal"
    "pid|cascade|-|square|5|3|200|pid_cascade"
    "lqr|-|pure|square|5|3|200|lqr_pure"
)

USE_ISAAC="false"
if [[ "${1:-}" == "--isaac" ]]; then
    USE_ISAAC="true"
    shift
fi

TASK="${1:-$DEFAULT_TASK}"
DURATION="${2:-$DEFAULT_DURATION}"
USER_REFINE_ITERS="${3:-}"
TRIALS="$DEFAULT_TRIALS"

PID_REFINE_ITERS="$DEFAULT_PID_REFINE_ITERS"
LQR_REFINE_ITERS="$DEFAULT_LQR_REFINE_ITERS"
if [[ -n "$USER_REFINE_ITERS" ]]; then
    PID_REFINE_ITERS="$USER_REFINE_ITERS"
    LQR_REFINE_ITERS="$USER_REFINE_ITERS"
    TRIALS="$USER_REFINE_ITERS"  # 将第三参数复用为 trials，保持简单一致
fi

VALID_TASKS=(hover figure8 circle square helix)
if [[ "$USE_ISAAC" != "true" ]]; then
    if [[ ! " ${VALID_TASKS[*]} " =~ " ${TASK} " ]]; then
        echo "[ERROR] 不支持的轨迹 '${TASK}'。可选: ${VALID_TASKS[*]}" >&2
        exit 1
    fi
fi

# ==============================================================================

# 输出文件名
if [[ "$USE_ISAAC" == "true" ]]; then
    OUTPUT_DIR="results/baseline_isaac"
    OUTPUT_FILE_NOTE="多个结果 JSON（目录: ${OUTPUT_DIR}）"
else
    OUTPUT_DIR="results/baseline"
    OUTPUT_FILE="${OUTPUT_DIR}/${TASK}_baseline.json"
    OUTPUT_FILE_NOTE="$OUTPUT_FILE"
fi

echo "============================================================"
echo "PID / LQR Baseline 调参"
echo "============================================================"
if [[ "$USE_ISAAC" == "true" ]]; then
    echo "使用 Isaac Gym + SCG 精确奖励"
    echo "运行配置组数: ${#ISAAC_RUNS[@]}"
else
    echo "轨道类型: $TASK"
    echo "仿真时长: ${DURATION}s"
    echo "PID 精调: ${PID_REFINE_ITERS} 次"
    echo "LQR 精调: ${LQR_REFINE_ITERS} 次"
fi
echo "输出文件: $OUTPUT_FILE_NOTE"
echo "============================================================"

# 激活虚拟环境
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

if [[ "$USE_ISAAC" == "true" ]]; then
    if [[ ${#ISAAC_RUNS[@]} -eq 0 ]]; then
        echo "[ERROR] ISAAC_RUNS 为空，请先在脚本顶部配置需要运行的基线" >&2
        exit 1
    fi

    ISAAC_OUTPUTS=()
    for run in "${ISAAC_RUNS[@]}"; do
        IFS="|" read -r algo pid_mode lqr_mode task duration episodes trials label <<< "$run"

        if [[ -z "$algo" ]]; then
            echo "[WARNING] 跳过无效配置: '$run'" >&2
            continue
        fi

        output_path="${OUTPUT_DIR}/${task}_${label}_scg_exact.json"
        echo ""
        echo "[RUN] ${label} | algo=${algo} pid_mode=${pid_mode} lqr_mode=${lqr_mode} task=${task} duration=${duration}s episodes=${episodes} trials=${trials}"

        cmd=(
            python3 scripts/baselines/tune_pid_lqr_isaac.py
            --algo "$algo"
            --task "$task"
            --duration "$duration"
            --episodes-per-eval "$episodes"
            --trials "$trials"
            --output "$output_path"
        )

        if [[ "$pid_mode" != "-" ]]; then
            cmd+=(--pid-mode "$pid_mode")
        fi
        if [[ "$lqr_mode" != "-" ]]; then
            cmd+=(--lqr-mode "$lqr_mode")
        fi

        "${cmd[@]}"
        ISAAC_OUTPUTS+=("$output_path")
    done

    OUTPUT_FILE_NOTE=$(printf "%s" "${ISAAC_OUTPUTS[*]}")
else
    # 使用旧的简化仿真调参脚本
    python3 scripts/baselines/tune_pid_lqr.py \
        --task "$TASK" \
        --duration "$DURATION" \
        --pid-refine-iters "$PID_REFINE_ITERS" \
        --lqr-refine-iters "$LQR_REFINE_ITERS" \
        --output "$OUTPUT_FILE"
fi

echo ""
echo "============================================================"
echo "✅ 调参完成，结果已保存到: $OUTPUT_FILE_NOTE"
echo "============================================================"

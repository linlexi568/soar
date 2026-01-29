#!/usr/bin/env python3
"""Generate summary CSV for meta-RL pretraining using real Isaac Gym evaluations."""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import statistics
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Sequence

REWARD_PATTERN = re.compile(r"奖励:\s*(-?\d+(?:\.\d+)?)")

# 全局变量用于多进程传递参数
_GLOBAL_ARGS = None
_GLOBAL_REPO_ROOT = None
_GLOBAL_SAVE_DIR = None
_GLOBAL_LOG_DIR = None


def _init_worker_globals(args, repo_root, save_dir, log_dir):
    """初始化全局变量（在每个进程中调用）"""
    global _GLOBAL_ARGS, _GLOBAL_REPO_ROOT, _GLOBAL_SAVE_DIR, _GLOBAL_LOG_DIR
    _GLOBAL_ARGS = args
    _GLOBAL_REPO_ROOT = repo_root
    _GLOBAL_SAVE_DIR = save_dir
    _GLOBAL_LOG_DIR = log_dir


def _parallel_worker(cfg: Dict[str, float]) -> List[Dict[str, float]]:
    """并行工作函数（必须在模块级别以便 pickle）"""
    return run_config(cfg, _GLOBAL_ARGS, _GLOBAL_REPO_ROOT, _GLOBAL_SAVE_DIR, _GLOBAL_LOG_DIR)


def default_grid() -> List[Dict[str, float]]:
    # ⭐⭐⭐ 平衡网格：5×5×5×4 = 500 个配置
    # 适合资源受限场景，每个超参数有 4-5 个值，足够学习趋势
    # 预计时间：~17 小时（单进程）或 ~4 小时（4 进程并行）
    eps = [0.05, 0.15, 0.25, 0.35, 0.40]                          # 5 个值
    alpha = [0.1, 0.3, 0.5, 0.7, 0.8]                             # 5 个值
    zero_penalty = [0.0, 0.1, 0.2, 0.3, 0.5]                      # 5 个值
    replicas = [2, 3, 4, 6]                                        # 4 个值
    combos = []
    for idx, (e, a, zp, rp) in enumerate(itertools.product(eps, alpha, zero_penalty, replicas), start=1):
        combos.append({
            "run_id": f"cfg_{idx:03d}",  # 支持 500 个配置
            "root_dirichlet_eps": e,
            "root_dirichlet_alpha": a,
            "zero_action_penalty": zp,
            "eval_replicas_per_program": rp,
        })
    return combos


def parse_rewards(stdout: str) -> List[float]:
    return [float(m.group(1)) for m in REWARD_PATTERN.finditer(stdout)]


def parse_iteration_rewards(stdout: str) -> Dict[int, List[float]]:
    """解析每一轮的奖励，返回 {iter_idx: [rewards]} 字典"""
    iter_rewards = {}
    current_iter = 0
    
    # 匹配 [Iter N] ... 奖励: X.XX 的模式
    for line in stdout.split('\n'):
        # 检测迭代行
        iter_match = re.search(r'\[Iter\s+(\d+)\]', line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            if current_iter not in iter_rewards:
                iter_rewards[current_iter] = []
        
        # 检测奖励
        reward_match = REWARD_PATTERN.search(line)
        if reward_match and current_iter > 0:
            iter_rewards[current_iter].append(float(reward_match.group(1)))
    
    return iter_rewards


def summarize_rewards(rewards: Sequence[float]) -> Dict[str, float]:
    if not rewards:
        return {"reward_mean": -2.0, "reward_std": 0.0}
    if len(rewards) == 1:
        return {"reward_mean": rewards[0], "reward_std": 0.0}
    return {
        "reward_mean": statistics.mean(rewards),
        "reward_std": statistics.pstdev(rewards),
    }


def derived_metrics(cfg: Dict[str, float], reward_mean: float) -> Dict[str, float]:
    zero_action_frac = max(0.0, min(1.0, 0.5 - cfg["zero_action_penalty"] * 0.5))
    entropy = max(0.2, min(2.5, 2.5 - cfg["root_dirichlet_eps"] * 2.0))
    success_rate = max(0.0, min(1.0, (reward_mean + 2.0) / 2.0))
    return {
        "zero_action_frac": zero_action_frac,
        "entropy": entropy,
        "success_rate": success_rate,
        "ranking_blend": 0.3,
        "crash_ratio": 0.0,
    }


def run_config(cfg: Dict[str, float], args: argparse.Namespace, repo_root: Path, save_dir: Path, log_dir: Path) -> List[Dict[str, float]]:
    """运行单个配置，返回时序轨迹的多行数据"""
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg['run_id']}_best.json"
    history_path = save_dir / f"{cfg['run_id']}_history.jsonl"
    log_path = log_dir / f"{cfg['run_id']}.log"

    cmd = [
        args.python,
        str(repo_root / "01_soar" / "train_online.py"),
        "--total-iters",
        str(args.total_iters),
        "--mcts-simulations",
        str(args.mcts_sims),
        "--root-dirichlet-eps",
        str(cfg["root_dirichlet_eps"]),
        "--root-dirichlet-alpha",
        str(cfg["root_dirichlet_alpha"]),
        "--zero-action-penalty",
        str(cfg["zero_action_penalty"]),
        "--eval-replicas-per-program",
        str(cfg["eval_replicas_per_program"]),
        "--save-path",
        str(save_path),
        "--program-history-path",
        str(history_path),
    ]

    if args.dry_run:
        # 生成模拟的时序数据
        stdout = ""
        for i in range(1, args.total_iters + 1):
            fake_reward = -2.0 + i * 0.003  # 模拟逐渐提升
            stdout += f"[Iter {i}] 完成 | 奖励: {fake_reward:.3f}\n"
    else:
        # 运行真实训练，将详细日志写入独立文件，终端只显示汇总
        print(f"[collect] ▶ 开始训练 {cfg['run_id']}...")
        env = os.environ.copy()
        env['TRAIN_VERBOSE_INTERVAL'] = '20'  # 统一降低详细打印频率

        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            stdout_lines = []
            for line in process.stdout:
                stdout_lines.append(line)
                log_file.write(line)
                log_file.flush()
                clean_line = line.replace('\r', '').strip()
                if '[进度' in clean_line:
                    progress_part = clean_line.split('[PW-DEBUG')[0].strip()
                    if progress_part:
                        print(f"[collect][{cfg['run_id']}] {progress_part}")

            process.wait()
            stdout = ''.join(stdout_lines)

            if process.returncode != 0 and not stdout.strip():
                raise RuntimeError(f"Config {cfg['run_id']} failed with no output; see {log_path}")

    # 解析每一轮的奖励
    iter_rewards = parse_iteration_rewards(stdout)
    
    # 每隔 sample_interval 轮记录一次（避免数据过多）
    sample_interval = args.sample_interval
    rows = []
    
    for iter_idx in sorted(iter_rewards.keys()):
        if iter_idx % sample_interval != 0 and iter_idx != args.total_iters:
            continue  # 只在采样点和最后一轮记录
        
        rewards = iter_rewards[iter_idx]
        if not rewards:
            continue
            
        stats = summarize_rewards(rewards)
        derived = derived_metrics(cfg, stats["reward_mean"])
        row = {
            "run_id": cfg["run_id"],
            "iter_idx": iter_idx,
            **stats,
            **derived,
            "root_dirichlet_eps": cfg["root_dirichlet_eps"],
            "root_dirichlet_alpha": cfg["root_dirichlet_alpha"],
            "zero_action_penalty": cfg["zero_action_penalty"],
            "eval_replicas_per_program": cfg["eval_replicas_per_program"],
        }
        rows.append(row)
    
    return rows


def write_csv(rows: List[Dict[str, float]], output: Path, append: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "iter_idx",
        "reward_mean",
        "reward_std",
        "success_rate",
        "zero_action_frac",
        "entropy",
        "ranking_blend",
        "crash_ratio",
        "root_dirichlet_eps",
        "root_dirichlet_alpha",
        "zero_action_penalty",
        "eval_replicas_per_program",
    ]
    
    mode = "a" if append and output.exists() else "w"
    write_header = not (append and output.exists())
    
    with output.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def get_next_run_prefix(output: Path) -> int:
    """读取已有 CSV，找到最大的 run_id 数字后缀"""
    if not output.exists():
        return 1
    
    max_num = 0
    with output.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id", "")
            # 提取 cfg_XX 中的数字
            match = re.match(r"cfg_(\d+)", run_id)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
    
    return max_num + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sweep data for meta-RL training")
    
    # ⚙️  可在脚本内修改的参数（所有参数都有默认值）
    parser.add_argument("--output", default="results/mcts_tune/summary.csv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--total-iters", type=int, default=100)        # ⭐ 默认 100 轮
    parser.add_argument("--mcts-sims", type=int, default=200)          # ⭐ 默认 200 模拟
    parser.add_argument("--sample-interval", type=int, default=10)     # ⭐ 默认每 10 轮采样
    parser.add_argument("--save-dir", default="results/mcts_tune/runs")
    parser.add_argument("--log-dir", default="logs/meta_rl_collect")
    parser.add_argument("--dry-run", action="store_true", help="Skip training and emit synthetic data")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    
    # 🔧 并行/分片参数（用于手动并行）
    parser.add_argument("--parallel", type=int, default=1, 
                        help="Number of parallel processes (DISABLED: use --start/--end for safe parallelism)")
    parser.add_argument("--start", type=int, default=None,
                        help="Start config index (1-based, for manual parallelism)")
    parser.add_argument("--end", type=int, default=None,
                        help="End config index (1-based, inclusive, for manual parallelism)")
    
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    output_path = Path(args.output)

    # 确定并行进程数
    num_parallel = args.parallel
    if num_parallel == 0:
        num_parallel = max(1, cpu_count() - 1)  # 留一个核心给系统
    print(f"[collect] 使用 {num_parallel} 个并行进程")

    # 如果是追加模式，调整 run_id 前缀避免冲突
    run_offset = 0
    if args.append:
        run_offset = get_next_run_prefix(output_path) - 1
        print(f"[collect] 追加模式：从 cfg_{run_offset + 1:03d} 开始编号")

    # 准备所有配置
    all_configs = list(default_grid())
    for idx, cfg in enumerate(all_configs, start=1):
        new_run_id = f"cfg_{idx + run_offset:03d}"
        cfg["run_id"] = new_run_id
    
    # 🔧 如果指定了 --start/--end，只处理该范围
    if args.start is not None or args.end is not None:
        start_idx = (args.start or 1) - 1  # 转为 0-based
        end_idx = (args.end or len(all_configs))  # 包含 end
        all_configs = all_configs[start_idx:end_idx]
        print(f"[collect] 分片模式：处理配置 {args.start or 1} 到 {args.end or len(default_grid())} (共 {len(all_configs)} 个)")
    else:
        print(f"[collect] 全量模式：总共 {len(all_configs)} 个配置")

    total_configs = len(all_configs)
    total_rows_written = 0

    def handle_result(idx: int, cfg: Dict[str, float], rows: List[Dict[str, float]]) -> None:
        nonlocal total_rows_written
        total_rows_written += len(rows)
        is_first_write = (idx == 1) and not args.append
        write_csv(rows, output_path, append=(not is_first_write))
        status = (
            f"[collect] 完成 {cfg['run_id']} ({idx}/{total_configs})"
            f" | 新增 {len(rows)} 行 | 累计 {total_rows_written} 行"
        )
        print(status, flush=True)

    if num_parallel > 1:
        print(f"[collect] 并行模式：{num_parallel} 个进程")
        with Pool(processes=num_parallel,
                  initializer=_init_worker_globals,
                  initargs=(args, repo_root, save_dir, log_dir)) as pool:
            for idx, rows in enumerate(pool.imap(_parallel_worker, all_configs), start=1):
                cfg = all_configs[idx - 1]
                handle_result(idx, cfg, rows)
    else:
        for idx, cfg in enumerate(all_configs, start=1):
            print(f"[collect] 正在运行 {cfg['run_id']} ({idx}/{total_configs})...", flush=True)
            rows = run_config(cfg, args, repo_root, save_dir, log_dir)
            handle_result(idx, cfg, rows)

    mode_str = "追加" if args.append else "写入"
    print(f"[collect] ✓ 完成！共 {mode_str} {total_rows_written} 行到 {args.output}")


if __name__ == "__main__":
    main()

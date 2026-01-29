#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse training/evaluation logs under logs/*.log and output a consolidated CSV.
- Extract per-iteration rewards, iteration time
- Extract batch evaluation throughput (programs, replicas, total seconds, ms/program)
- Extract root stats (children, visits, entropy) when available

Output: results/metrics/metrics.csv
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "results" / "metrics"
OUT_FILE = OUT_DIR / "metrics.csv"

# Regex patterns
re_iter_start = re.compile(r"^\[Iter\s+(\d+)/(\d+)\]")
re_reward = re.compile(r"^\[Iter\s+(\d+)\]\s+奖励:\s+真实=([-\d\.]+),\s+训练=([-\d\.]+)")
re_done = re.compile(r"^\[Iter\s+(\d+)\].*?耗时:\s*([\d\.]+)s")
re_batch_eval = re.compile(r"^\[BatchEvaluator\]\s+✅ 评估完成:\s+(\d+)\s+程序\s+\(×(\d+)\s+replicas\),\s*([\d\.]+)秒\s*\(([\d\.]+)ms/程序\)")
re_root = re.compile(r"^\s*\[根统计\].*?子节点数=(\d+),\s*总访问=(\d+),\s*熵=([\d\.]+)")

# Some logs may contain ASCII only; make patterns robust by ignoring locale

cols = [
    "log_file",
    "iter",
    "iters_total",
    "real_reward",
    "train_reward",
    "iter_time_s",
    "mcts_programs",
    "replicas",
    "batch_seconds",
    "ms_per_program",
    "root_children",
    "root_visits",
    "root_entropy",
]


def parse_log_file(path: Path):
    rows = []
    current = None  # dict for current iter
    iters_total = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m0 = re_iter_start.match(line)
            if m0:
                it = int(m0.group(1))
                iters_total = int(m0.group(2))
                # flush previous
                if current is not None:
                    rows.append(current)
                current = {c: None for c in cols}
                current["log_file"] = path.name
                current["iter"] = it
                current["iters_total"] = iters_total
                continue
            m1 = re_reward.match(line)
            if m1:
                it = int(m1.group(1))
                # initialize if missing (some logs print reward before [Iter n/m] 完成)
                if current is None or current.get("iter") != it:
                    if current is not None:
                        rows.append(current)
                    current = {c: None for c in cols}
                    current["log_file"] = path.name
                    current["iter"] = it
                    current["iters_total"] = iters_total
                current["real_reward"] = float(m1.group(2))
                current["train_reward"] = float(m1.group(3))
                continue
            m2 = re_done.match(line)
            if m2 and current is not None:
                it = int(m2.group(1))
                if current.get("iter") == it:
                    current["iter_time_s"] = float(m2.group(2))
                continue
            m3 = re_batch_eval.match(line)
            if m3 and current is not None:
                programs = int(m3.group(1))
                replicas = int(m3.group(2))
                sec = float(m3.group(3))
                mspp = float(m3.group(4))
                # Heuristic: the large batch (>= 50 programs) corresponds to MCTS candidate set throughput
                if programs >= 50:
                    current["mcts_programs"] = programs
                    current["replicas"] = replicas
                    current["batch_seconds"] = sec
                    current["ms_per_program"] = mspp
                continue
            m4 = re_root.match(line)
            if m4 and current is not None:
                current["root_children"] = int(m4.group(1))
                current["root_visits"] = int(m4.group(2))
                current["root_entropy"] = float(m4.group(3))
                continue
    # flush last
    if current is not None:
        rows.append(current)
    return rows


def main():
    if not LOG_DIR.exists():
        print(f"Log dir not found: {LOG_DIR}", file=sys.stderr)
        sys.exit(1)
    log_files = sorted(LOG_DIR.glob("*.log"))
    if not log_files:
        print("No log files found.")
        sys.exit(1)

    all_rows = []
    for p in log_files:
        try:
            rows = parse_log_file(p)
            all_rows.extend(rows)
            print(f"Parsed {p.name}: {len(rows)} rows")
        except Exception as e:
            print(f"Failed to parse {p}: {e}", file=sys.stderr)

    # write CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"Wrote {OUT_FILE} with {len(all_rows)} rows")


if __name__ == "__main__":
    main()

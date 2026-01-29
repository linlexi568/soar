#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce a small LaTeX table summarizing key metrics for the latest train_* log.
Output: results/metrics/summary.tex
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "metrics" / "metrics.csv"
OUT = ROOT / "results" / "metrics" / "summary.tex"


def main():
    df = pd.read_csv(CSV)
    cand = [n for n in df["log_file"].unique() if n.startswith("train_")]
    if not cand:
        print("No train_* logs found.")
        return
    log = sorted(cand)[-1]
    d = df[df["log_file"] == log].copy()
    # learning curve stats
    best_reward = d["real_reward"].max(skipna=True)
    last_reward = d.sort_values("iter")["real_reward"].dropna().tail(1).values
    last_reward = float(last_reward[0]) if len(last_reward) else float("nan")
    # throughput stats (only MCTS batch entries)
    dt = d["ms_per_program"].dropna()
    if len(dt) > 0:
        p50 = dt.quantile(0.5)
        p90 = dt.quantile(0.9)
        p95 = dt.quantile(0.95)
        mean = dt.mean()
    else:
        p50 = p90 = p95 = mean = float("nan")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated from results/metrics/metrics.csv\n")
        f.write("\\begin{table}[h]\\centering\n")
        f.write("\\caption{Summary of latest training run (" + log.replace("_", "\\_") + ")}.\\label{tab:summary}\n")
        f.write("\\begin{tabular}{l r}\\toprule\n")
        f.write(f"Best real reward & {best_reward:.4f} \\ \\n")
        f.write(f"Last real reward & {last_reward:.4f} \\ \\n")
        f.write(f"Throughput mean (ms/program) & {mean:.1f} \\ \\n")
        f.write(f"Throughput p50 / p90 / p95 & {p50:.1f} / {p90:.1f} / {p95:.1f} \\ \\n")
        f.write("\\bottomrule\\end{tabular}\n\\end{table}\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

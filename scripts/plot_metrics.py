#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot key figures from results/metrics/metrics.csv
- Learning curve: real reward vs iteration (per log)
- Throughput curve: ms per program vs iteration (for MCTS batch)
- Optionally histogram of ms/program

Saved to results/plots/
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "metrics" / "metrics.csv"
PLOT_DIR = ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def plot_learning_curve(df: pd.DataFrame, log_name: str):
    d = df[df["log_file"] == log_name].copy()
    d = d.sort_values("iter")
    if d.empty:
        print(f"[skip] no data for {log_name}")
        return
    plt.figure(figsize=(6, 3.5))
    plt.plot(d["iter"], d["real_reward"], label="Real reward", lw=1.8)
    if d["train_reward"].notna().any():
        plt.plot(d["iter"], d["train_reward"], label="Train reward", lw=1.2, alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Reward (higher is better)")
    plt.title(f"Learning curve: {log_name}")
    plt.legend()
    plt.grid(True, ls=":", alpha=0.4)
    out = PLOT_DIR / f"{log_name}_learning_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def plot_throughput(df: pd.DataFrame, log_name: str):
    d = df[(df["log_file"] == log_name) & (df["ms_per_program"].notna())].copy()
    d = d.sort_values("iter")
    if d.empty:
        print(f"[skip] no throughput for {log_name}")
        return
    plt.figure(figsize=(6, 3.5))
    plt.plot(d["iter"], d["ms_per_program"], label="ms per program", lw=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("ms / program (MCTS batch)")
    plt.title(f"Evaluation throughput: {log_name}")
    plt.grid(True, ls=":", alpha=0.4)
    out = PLOT_DIR / f"{log_name}_throughput.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def plot_histogram(df: pd.DataFrame, log_name: str):
    d = df[(df["log_file"] == log_name) & (df["ms_per_program"].notna())]["ms_per_program"].dropna()
    if d.empty:
        print(f"[skip] no histogram for {log_name}")
        return
    plt.figure(figsize=(5.2, 3.6))
    plt.hist(d, bins=30, color="#4472c4", edgecolor="white")
    plt.xlabel("ms / program")
    plt.ylabel("count")
    plt.title(f"Throughput distribution: {log_name}")
    plt.grid(True, ls=":", alpha=0.3)
    out = PLOT_DIR / f"{log_name}_throughput_hist.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    # choose the latest train_* log as primary curve
    cand = [n for n in df["log_file"].unique() if n.startswith("train_")]
    if not cand:
        print("No train_* logs found in CSV.")
        sys.exit(0)
    # pick the most recent by timestamp embedded in name
    log_name = sorted(cand)[-1]
    print(f"Using log: {log_name}")
    plot_learning_curve(df, log_name)
    plot_throughput(df, log_name)
    plot_histogram(df, log_name)

if __name__ == "__main__":
    main()

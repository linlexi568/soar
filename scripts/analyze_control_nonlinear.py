#!/usr/bin/env python3
"""Nonlinear control analysis helper for Soar logs.

Usage:
    python scripts/analyze_control_nonlinear.py \
        --log results/rollout.csv \
        --outdir results/plots

Expected log columns (CSV or JSONL with flat dict per line):
    time, pos_err_x, pos_err_y, pos_err_z,
    vel_x, vel_y, vel_z,
    ang_vel_x, ang_vel_y, ang_vel_z,
    u_fz, u_tx, u_ty, u_tz
Missing columns are tolerated; plots that need them will be skipped.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keep matplotlib quiet in headless runs
plt.switch_backend("Agg")


def load_log(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        rows: List[Dict] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported log format: {path.suffix}")
    return df


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def plot_phase(df: pd.DataFrame, x: str, dx: str, title: str, out: Path) -> None:
    if x not in df or dx not in df:
        return
    plt.figure(figsize=(4, 4))
    plt.plot(df[x], df[dx], linewidth=1.0, alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(dx)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, out: Path, hlines=None) -> None:
    if x not in df or y not in df:
        return
    plt.figure(figsize=(4, 3))
    plt.scatter(df[x], df[y], s=4, alpha=0.4, edgecolors="none")
    if hlines:
        for h in hlines:
            plt.axhline(h, color="r", linestyle="--", linewidth=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_time(df: pd.DataFrame, cols: List[str], title: str, out: Path) -> None:
    missing = [c for c in cols if c not in df]
    if missing:
        return
    t = df["time"] if "time" in df else np.arange(len(df))
    plt.figure(figsize=(6, 3))
    for c in cols:
        plt.plot(t, df[c], label=c, linewidth=1.0)
    plt.xlabel("time")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Nonlinear analysis plots for Soar logs")
    ap.add_argument("--log", required=True, help="CSV or JSONL log path")
    ap.add_argument("--outdir", default="results/plots", help="Output directory for figures")
    args = ap.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df = load_log(log_path)

    # Phase planes
    plot_phase(df, "pos_err_z", "vel_z", "Phase: pos_err_z vs vel_z", outdir / "phase_z.png")
    plot_phase(df, "pos_err_y", "vel_y", "Phase: pos_err_y vs vel_y", outdir / "phase_y.png")

    # Saturation / nonlinearity views
    plot_scatter(df, "pos_err_z", "u_fz", "u_fz vs pos_err_z", outdir / "ufz_vs_poserrz.png", hlines=[0.265-2, 0.265+2])
    plot_scatter(df, "ang_vel_y", "u_ty", "u_ty vs ang_vel_y (square term visible)", outdir / "uty_vs_angvely.png")
    plot_scatter(df, "vel_y", "u_tz", "u_tz vs vel_y (division sensitivity)", outdir / "utz_vs_vely.png")
    plot_scatter(df, "pos_err_y", "u_tz", "u_tz vs pos_err_y", outdir / "utz_vs_poserry.png")

    # Time series
    plot_time(df, [c for c in ["u_fz", "u_tx", "u_ty", "u_tz"] if c in df], "Controls", outdir / "controls.png")
    plot_time(df, [c for c in ["pos_err_x", "pos_err_y", "pos_err_z"] if c in df], "Position errors", outdir / "pos_err.png")
    plot_time(df, [c for c in ["vel_x", "vel_y", "vel_z"] if c in df], "Velocities", outdir / "vel.png")

    print(f"Saved plots to {outdir.resolve()}")


if __name__ == "__main__":
    main()

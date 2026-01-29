#!/usr/bin/env python3
"""
Simple GPU monitor that logs utilization, memory, temperature to a CSV file.
Usage:
  python3 scripts/gpu_monitor.py --interval 2 --output gpu_usage.csv
Stop with Ctrl+C.
"""
import argparse
import csv
import datetime as dt
import subprocess
import sys

def query_nvidia_smi():
    fmt = "csv,noheader,nounits"
    q = "index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,fan.speed,clocks.sm,power.draw"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={q}",
        f"--format={fmt}"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        rows = []
        for l in lines:
            parts = [p.strip() for p in l.split(',')]
            rows.append(parts)
        return rows
    except Exception as e:
        print(f"[gpu_monitor] Failed to run nvidia-smi: {e}")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=2.0, help="sampling interval seconds")
    ap.add_argument("--output", type=str, default="gpu_usage.csv", help="output CSV path")
    args = ap.parse_args()

    header = [
        "timestamp",
        "gpu_index",
        "name",
        "temp_c",
        "util_percent",
        "mem_used_mb",
        "mem_total_mb",
        "fan_percent",
        "sm_clock_mhz",
        "power_w"
    ]

    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header only if file is empty
        if f.tell() == 0:
            writer.writerow(header)
        try:
            while True:
                now = dt.datetime.now().isoformat(timespec='seconds')
                rows = query_nvidia_smi()
                for r in rows:
                    # Ensure length
                    r = r + [""] * (len(header)-1)
                    writer.writerow([now] + r[:len(header)-1])
                f.flush()
                if args.interval > 0:
                    import time
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[gpu_monitor] Stopped.")
            sys.exit(0)

if __name__ == "__main__":
    main()

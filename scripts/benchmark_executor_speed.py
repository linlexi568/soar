#!/usr/bin/env python3
"""Benchmark GPUProgramExecutor CPU vs GPU paths.

This measures pure expression-evaluation cost (不含 IsaacGym), 对比：
- build_state_tensors + evaluate(force_cpu=True)  作为 CPU 路径
- evaluate_from_raw_obs(force_cpu=False)          作为 GPU 控制循环路径

注意：这里只测算子执行本身，不代表端到端训练耗时；但可以直观看到控制律部分的加速比。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any, Dict, List

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "01_soar"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from compare_gpu_cpu_executor import _load_programs  # type: ignore
from utils.gpu_program_executor import GPUProgramExecutor  # type: ignore


def _rand_integral_states(batch: int, seed: int) -> List[Dict[str, float]]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    data = torch.randn(batch, 6, generator=gen)
    states: List[Dict[str, float]] = []
    for row in data:
        states.append(
            {
                "err_i_x": float(row[0]),
                "err_i_y": float(row[1]),
                "err_i_z": float(row[2]),
                "err_i_roll": float(row[3]),
                "err_i_pitch": float(row[4]),
                "err_i_yaw": float(row[5]),
            }
        )
    return states


def benchmark(
    program_paths: List[str],
    device_str: str,
    steps: int,
    batch: int,
    warmup: int,
    seed: int,
) -> None:
    device = torch.device(device_str)
    programs_all = _load_programs(program_paths)
    # 只取前 batch 个程序，不够就重复
    if len(programs_all) == 0:
        raise SystemExit("No programs loaded")
    if len(programs_all) < batch:
        factor = (batch + len(programs_all) - 1) // len(programs_all)
        programs = (programs_all * factor)[:batch]
    else:
        programs = programs_all[:batch]

    executor = GPUProgramExecutor(device=str(device))
    executor.reset_state()
    token = executor.prepare_batch(programs)

    def rand_tensor(dim: int, gen: torch.Generator) -> torch.Tensor:
        # 生成在 CPU 上，再搬到目标 device，避免 generator 设备类型冲突
        cpu = torch.randn(batch, dim, generator=gen)
        return cpu.to(device)

    gen = torch.Generator(device="cpu").manual_seed(seed)

    use_mask = torch.ones(batch, dtype=torch.bool, device=device)

    # 预先生成一批随机输入，避免计时里掺杂太多 host 侧开销
    pos_seq = []
    vel_seq = []
    omega_seq = []
    quat_seq = []
    target_seq = []
    integral_seq = []
    for _ in range(steps + warmup):
        pos = rand_tensor(3, gen)
        vel = rand_tensor(3, gen)
        omega = rand_tensor(3, gen)
        quat = torch.randn(batch, 4, generator=gen)
        quat = quat.to(device)
        quat = quat / quat.norm(dim=1, keepdim=True)
        target = torch.zeros(batch, 3, device=device)
        integrals = _rand_integral_states(batch, seed)
        pos_seq.append(pos)
        vel_seq.append(vel)
        omega_seq.append(omega)
        quat_seq.append(quat)
        target_seq.append(target)
        integral_seq.append(integrals)

    # --- CPU 路径：build_state_tensors + evaluate(force_cpu=True) ---
    def run_cpu() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for i in range(warmup, warmup + steps):
            pos = pos_seq[i]
            vel = vel_seq[i]
            omega = omega_seq[i]
            quat = quat_seq[i]
            target = target_seq[i]
            integrals = integral_seq[i]
            state_tensors, _, _ = executor.build_state_tensors(
                pos,
                vel,
                omega,
                quat,
                target,
                integrals,
            )
            _ = executor.evaluate(
                token,
                state_tensors,
                use_mask,
                active_mask=None,
                force_cpu=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        return t1 - t0

    # --- GPU 控制循环路径：evaluate_from_raw_obs(force_cpu=False) ---
    def run_gpu() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for i in range(warmup, warmup + steps):
            pos = pos_seq[i]
            vel = vel_seq[i]
            omega = omega_seq[i]
            quat = quat_seq[i]
            target = target_seq[i]
            integrals = integral_seq[i]
            _outputs, _pos_err, _rpy = executor.evaluate_from_raw_obs(
                token,
                pos,
                vel,
                omega,
                quat,
                target,
                integrals,
                use_mask,
                active_mask=None,
                force_cpu=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        return t1 - t0

    # 轻量 warmup（已经通过预生成 + steps 控制，额外 warmup 只做一次）
    _ = run_gpu()

    t_cpu = run_cpu()
    t_gpu = run_gpu()

    executor.release_batch(token)

    total_evals = steps * batch
    cpu_per = t_cpu * 1e6 / total_evals
    gpu_per = t_gpu * 1e6 / total_evals
    speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")

    print("==== GPUProgramExecutor speed benchmark ====")
    print(f"device           : {device}")
    print(f"programs(batch)  : {batch}")
    print(f"steps            : {steps}")
    print(f"total evaluations: {total_evals}")
    print("--- per-eval cost (us) ---")
    print(f"CPU path         : {cpu_per:.3f} us / eval")
    print(f"GPU ctrl path    : {gpu_per:.3f} us / eval")
    print(f"speedup (CPU/GPU): {speedup:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GPUProgramExecutor CPU vs GPU paths.")
    parser.add_argument("--program", action="append", dest="programs", required=True,
                        help="Path to a serialized DSL program JSON. Repeat for multiple programs.")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--steps", type=int, default=2000, help="Number of evaluation steps per program")
    parser.add_argument("--batch", type=int, default=64, help="Number of programs in parallel batch")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup steps to skip from timing")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed")
    args = parser.parse_args()

    benchmark(args.programs, args.device, args.steps, args.batch, args.warmup, args.seed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU/GPU executor consistency checker.

This script runs arbitrary SOAR DSL programs through the new GPUProgramExecutor
and compares the outputs with the CPU fallback evaluator under randomized states.
It helps quantify numerical drift when migrating Level 1/2/3 programs fully to GPU.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
from typing import Any, Dict, List

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "01_soar"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    from core.serialization import load_program_json  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("Unable to import core.serialization.load_program_json") from exc

try:
    from utils.gpu_program_executor import GPUProgramExecutor  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(f"GPUProgramExecutor unavailable: {exc}")


def _synthesize_state_tensors(batch: int, device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    """Generate synthetic state tensors that mimic BatchEvaluator's layout."""

    gen = torch.Generator(device="cpu").manual_seed(seed)
    pos_err = torch.randn(batch, 3, generator=gen) * 0.3
    vel = torch.randn(batch, 3, generator=gen) * 0.2
    omega = torch.randn(batch, 3, generator=gen) * 0.15
    rpy = torch.randn(batch, 3, generator=gen) * 0.1
    integral = torch.randn(batch, 6, generator=gen) * 0.05

    pos_err_xy = torch.linalg.norm(pos_err[:, :2], dim=1)
    pos_err_mag = torch.linalg.norm(pos_err, dim=1)
    vel_err = torch.linalg.norm(vel, dim=1)
    ang_vel_mag = torch.linalg.norm(omega, dim=1)
    rpy_err_mag = torch.linalg.norm(rpy, dim=1)

    state = {
        "pos_err_x": pos_err[:, 0],
        "pos_err_y": pos_err[:, 1],
        "pos_err_z": pos_err[:, 2],
        "pos_err": pos_err_mag,
        "pos_err_xy": pos_err_xy,
        "pos_err_z_abs": torch.abs(pos_err[:, 2]),
        "vel_x": vel[:, 0],
        "vel_y": vel[:, 1],
        "vel_z": vel[:, 2],
        "vel_err": vel_err,
        "err_p_roll": rpy[:, 0],
        "err_p_pitch": rpy[:, 1],
        "err_p_yaw": rpy[:, 2],
        "ang_err": rpy_err_mag,
        "rpy_err_mag": rpy_err_mag,
        "ang_vel_x": omega[:, 0],
        "ang_vel_y": omega[:, 1],
        "ang_vel_z": omega[:, 2],
        "ang_vel": ang_vel_mag,
        "ang_vel_mag": ang_vel_mag,
        "err_i_x": integral[:, 0],
        "err_i_y": integral[:, 1],
        "err_i_z": integral[:, 2],
        "err_i_roll": integral[:, 3],
        "err_i_pitch": integral[:, 4],
        "err_i_yaw": integral[:, 5],
        "err_d_x": -vel[:, 0],
        "err_d_y": -vel[:, 1],
        "err_d_z": -vel[:, 2],
        "err_d_roll": -omega[:, 0],
        "err_d_pitch": -omega[:, 1],
        "err_d_yaw": -omega[:, 2],
    }
    # Move to device lazily to avoid generator device issues on CUDA.
    return {k: v.to(device) for k, v in state.items()}


def _load_programs(paths: List[str]) -> List[List[Dict[str, Any]]]:
    programs = []
    for raw_path in paths:
        path = pathlib.Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Program JSON not found: {path}")
        programs.append(load_program_json(str(path)))
    return programs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GPU vs CPU DSL executor outputs.")
    parser.add_argument("--program", action="append", dest="programs", required=True,
                        help="Path to a serialized DSL program JSON. Repeat for multiple programs.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for GPU executor (default: cuda:0)")
    parser.add_argument("--samples", type=int, default=128, help="Number of randomized state batches to test")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Alert threshold for absolute difference (per channel)")
    args = parser.parse_args()

    device = torch.device(args.device)
    programs = _load_programs(args.programs)
    batch_size = len(programs)
    executor = GPUProgramExecutor(device=str(device))

    max_abs = torch.zeros(4)
    max_rel = torch.zeros(4)
    sum_abs = torch.zeros(4)
    sample_count = 0
    fallback_total = 0

    for idx in range(args.samples):
        executor.reset_state()
        batch_programs = [copy.deepcopy(p) for p in programs]
        token = executor.prepare_batch(batch_programs)
        batch_info = executor._batches[token]  # type: ignore[attr-defined]
        compiled_mask = torch.tensor([c is not None for c in batch_info["compiled"]], dtype=torch.bool)
        fallback_total += int((~compiled_mask).sum().item())

        state_tensors = _synthesize_state_tensors(batch_size, device, args.seed + idx)
        use_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        cpu_out = executor.evaluate(token, state_tensors, use_mask, force_cpu=True)
        gpu_out = executor.evaluate(token, state_tensors, use_mask)
        executor.release_batch(token)

        if compiled_mask.any():
            mask = compiled_mask.to(device)
            diff = (gpu_out - cpu_out)[mask]
            cpu_ref = cpu_out[mask]
            max_abs = torch.maximum(max_abs, diff.abs().max(dim=0).values.cpu())
            rel = diff.abs() / (cpu_ref.abs() + 1e-6)
            max_rel = torch.maximum(max_rel, rel.max(dim=0).values.cpu())
            sum_abs += diff.abs().sum(dim=0).cpu()
            sample_count += diff.shape[0]

    print("==== GPU vs CPU Consistency Report ====")
    print(f"Programs tested       : {batch_size}")
    print(f"Random batches        : {args.samples}")
    print(f"Total fallbacks       : {fallback_total}")
    if sample_count == 0:
        print("No programs compiled to GPU IR; nothing to compare.")
        return
    mean_abs = sum_abs / max(sample_count, 1)
    channel_names = ["u_fz", "u_tx", "u_ty", "u_tz"]
    for i, name in enumerate(channel_names):
        print(f"  {name:4s} | max_abs={max_abs[i]:.6e} | max_rel={max_rel[i]:.6e} | mean_abs={mean_abs[i]:.6e}")
        if max_abs[i] > args.tolerance:
            print(f"    ⚠️  exceeds tolerance {args.tolerance:.2e}")


if __name__ == "__main__":
    main()

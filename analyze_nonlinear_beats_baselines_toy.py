#!/usr/bin/env python3
"""Toy evidence: why nonlinear (tanh/sign) can beat LQR/PID under SCG-like cost.

We cannot run Isaac Gym here because the bundled IsaacGym bindings in this repo
only include `py36/` binaries, while the workspace Python is 3.8.

So we build a minimal lateral-channel surrogate:
    y_dot = v
    v_dot = u
tracking y_ref(t).

We compare controllers under a cost aligned with SCG reward weights:
    J = ∫ (q_pos*e^2 + q_vel*ev^2 + r*u^2) dt
with q_pos=1, q_vel=0.01, r=1e-4.

Controllers:
- LQR on (e, ev)
- PID
- Smooth: u = kp * s * tanh(e/s) + kd * ev
- Sign:   u = kp * sign(e)      + kd * ev

We include actuator saturation |u|<=u_max to emulate torque/acc limits.

Outputs:
- prints best costs for each controller per task (sine/square)
- saves plots to results/toy_nonlinear/
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Callable, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


@dataclass
class CostWeights:
    q_pos: float = 1.0
    q_vel: float = 0.01
    r_u: float = 1e-4


def smooth_tanh(e: np.ndarray, s: float) -> np.ndarray:
    return s * np.tanh(e / max(1e-9, s))


def sat(u: np.ndarray, u_max: float) -> np.ndarray:
    return np.clip(u, -u_max, u_max)


def ref_sine(t: np.ndarray, amp: float = 1.0, f_hz: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    w = 2.0 * math.pi * f_hz
    y = amp * np.sin(w * t)
    v = amp * w * np.cos(w * t)
    return y, v


def ref_square_smooth(t: np.ndarray, amp: float = 1.0, f_hz: float = 0.2, edge_s: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed square wave using tanh transitions (finite derivative).

    y_ref ≈ amp * sign(sin(w t)) but with smooth edges.
    v_ref computed analytically from tanh derivative.
    """
    w = 2.0 * math.pi * f_hz
    s = np.sin(w * t)
    # smooth sign: tanh(s/edge)
    y = amp * np.tanh(s / max(1e-9, edge_s))
    # dy/dt = amp * sech^2(s/edge) * (cos(wt)*w/edge)
    sech2 = 1.0 / np.cosh(s / max(1e-9, edge_s)) ** 2
    v = amp * sech2 * (np.cos(w * t) * w / max(1e-9, edge_s))
    return y, v


def simulate(
    controller: Callable[[float, float, float, float, float], float],
    y_ref_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    T: float,
    dt: float,
    u_max: float,
    weights: CostWeights,
) -> Dict[str, np.ndarray]:
    n = int(T / dt)
    t = np.arange(n) * dt
    y_ref, v_ref = y_ref_fn(t)

    y = 0.0
    v = 0.0
    i_e = 0.0

    ys = np.zeros(n)
    vs = np.zeros(n)
    us = np.zeros(n)
    es = np.zeros(n)
    evs = np.zeros(n)

    for k in range(n):
        e = float(y_ref[k] - y)
        ev = float(v_ref[k] - v)
        i_e += e * dt

        u = float(controller(t[k], e, ev, i_e, dt))
        u = float(np.clip(u, -u_max, u_max))

        # dynamics: y_dot=v, v_dot=u
        y = y + v * dt
        v = v + u * dt

        ys[k] = y
        vs[k] = v
        us[k] = u
        es[k] = e
        evs[k] = ev

    # cost
    J = float(np.sum(weights.q_pos * es**2 + weights.q_vel * evs**2 + weights.r_u * us**2) * dt)

    return {
        "t": t,
        "y": ys,
        "v": vs,
        "u": us,
        "y_ref": y_ref,
        "v_ref": v_ref,
        "e": es,
        "ev": evs,
        "J": np.array([J]),
    }


def make_lqr_controller(weights: CostWeights) -> Callable[[float, float, float, float, float], float]:
    # state: [e, ev], dynamics x_dot = A x + B u, with e_dot=ev, ev_dot=-u (since ev = v_ref - v)
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [-1.0]])
    Q = np.diag([weights.q_pos, weights.q_vel])
    R = np.array([[weights.r_u]])
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ P)  # 1x2

    def ctrl(_t: float, e: float, ev: float, _i_e: float, _dt: float) -> float:
        x = np.array([e, ev])
        u = -float(K @ x)
        return u

    return ctrl


def make_pid_controller(kp: float, kd: float, ki: float) -> Callable[[float, float, float, float, float], float]:
    def ctrl(_t: float, e: float, ev: float, i_e: float, _dt: float) -> float:
        return kp * e + kd * ev + ki * i_e

    return ctrl


def make_smooth_controller(kp: float, kd: float, s: float) -> Callable[[float, float, float, float, float], float]:
    def ctrl(_t: float, e: float, ev: float, _i_e: float, _dt: float) -> float:
        return kp * float(smooth_tanh(np.array([e]), s)[0]) + kd * ev

    return ctrl


def make_sign_controller(kp: float, kd: float) -> Callable[[float, float, float, float, float], float]:
    def ctrl(_t: float, e: float, ev: float, _i_e: float, _dt: float) -> float:
        return kp * float(np.sign(e)) + kd * ev

    return ctrl


def grid_search(
    kind: str,
    y_ref_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    T: float,
    dt: float,
    u_max: float,
    weights: CostWeights,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    best_J = float("inf")
    best_params: Dict[str, float] = {}
    best_trace: Dict[str, np.ndarray] = {}

    if kind == "lqr":
        ctrl = make_lqr_controller(weights)
        trace = simulate(ctrl, y_ref_fn, T, dt, u_max, weights)
        return {"K": 0.0}, trace

    if kind == "pid":
        kps = [0.5, 1.0, 2.0, 4.0, 8.0]
        kds = [0.1, 0.3, 0.6, 1.0, 2.0]
        kis = [0.0, 0.05, 0.1, 0.2]
        for kp in kps:
            for kd in kds:
                for ki in kis:
                    ctrl = make_pid_controller(kp, kd, ki)
                    tr = simulate(ctrl, y_ref_fn, T, dt, u_max, weights)
                    J = float(tr["J"][0])
                    if J < best_J:
                        best_J = J
                        best_params = {"kp": kp, "kd": kd, "ki": ki}
                        best_trace = tr
        return best_params, best_trace

    if kind == "smooth":
        kps = [0.5, 1.0, 2.0, 4.0, 8.0]
        kds = [0.1, 0.3, 0.6, 1.0, 2.0]
        ss = [0.2, 0.5, 1.0, 2.0]
        for kp in kps:
            for kd in kds:
                for s in ss:
                    ctrl = make_smooth_controller(kp, kd, s)
                    tr = simulate(ctrl, y_ref_fn, T, dt, u_max, weights)
                    J = float(tr["J"][0])
                    if J < best_J:
                        best_J = J
                        best_params = {"kp": kp, "kd": kd, "s": s}
                        best_trace = tr
        return best_params, best_trace

    if kind == "sign":
        kps = [0.2, 0.5, 1.0, 2.0, 4.0]
        kds = [0.1, 0.3, 0.6, 1.0, 2.0]
        for kp in kps:
            for kd in kds:
                ctrl = make_sign_controller(kp, kd)
                tr = simulate(ctrl, y_ref_fn, T, dt, u_max, weights)
                J = float(tr["J"][0])
                if J < best_J:
                    best_J = J
                    best_params = {"kp": kp, "kd": kd}
                    best_trace = tr
        return best_params, best_trace

    raise ValueError(f"unknown kind: {kind}")


def plot_trace(trace: Dict[str, np.ndarray], title: str, out: Path) -> None:
    t = trace["t"]
    plt.figure(figsize=(10, 7))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, trace["y_ref"], label="y_ref")
    ax1.plot(t, trace["y"], label="y")
    ax1.set_ylabel("y")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, trace["e"], label="e")
    ax2.set_ylabel("error")
    ax2.grid(True)
    ax2.legend(loc="best")

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, trace["u"], label="u")
    ax3.set_ylabel("u")
    ax3.set_xlabel("t (s)")
    ax3.grid(True)
    ax3.legend(loc="best")

    plt.suptitle(title + f"  J={float(trace['J'][0]):.3f}")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    out_dir = Path("results/toy_nonlinear")
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = CostWeights()
    dt = 0.01
    T = 20.0
    u_max = 3.0

    tasks = {
        "sine": lambda tt: ref_sine(tt, amp=1.0, f_hz=0.25),
        "square": lambda tt: ref_square_smooth(tt, amp=1.0, f_hz=0.2, edge_s=0.10),
    }
    controllers = ["lqr", "pid", "smooth", "sign"]

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for task_name, ref_fn in tasks.items():
        summary[task_name] = {}
        print("=" * 72)
        print(f"Task={task_name}")
        for kind in controllers:
            params, trace = grid_search(kind, ref_fn, T, dt, u_max, weights)
            J = float(trace["J"][0])
            summary[task_name][kind] = {**params, "J": J}
            print(f"  {kind:6s}  J={J:.3f}  params={params}")

            plot_path = out_dir / f"{task_name}_{kind}.png"
            plot_trace(trace, f"{task_name} / {kind}", plot_path)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved plots + summary to {out_dir}")


if __name__ == "__main__":
    main()

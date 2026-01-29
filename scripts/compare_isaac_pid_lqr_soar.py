#!/usr/bin/env python3
"""Compare PID / LQR / Soar controllers inside the Isaac tester with SCG rewards."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
for candidate in (ROOT, ROOT / "scripts", ROOT / "01_soar", ROOT / "utilities"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

def _evict_if_external(name: str) -> None:
    mod = sys.modules.get(name)
    if mod is None:
        return
    mod_file = getattr(mod, "__file__", "")
    if not mod_file:
        sys.modules.pop(name, None)
        return
    repo_str = str(ROOT)
    if not mod_file.startswith(repo_str):
        sys.modules.pop(name, None)


_evict_if_external("core")
_evict_if_external("utils")

from baselines.tune_pid_lqr import (  # type: ignore
    PIDParams,
    LQRParams,
    TunablePIDController,
    TunableLQRController,
)
from core.serialization import load_program_json  # type: ignore
from core.program_executor import MathProgramController  # type: ignore
from utilities.isaac_tester import SimulationTester  # type: ignore
from utilities.trajectory_presets import get_scg_trajectory_config

KF = 2.8e-08
KM = 1.1e-10
ARM_LENGTH = 0.046
YAW_COEFF = KM / KF if KM > 0 else 0.0
MAX_RPM = 25000.0

SPLIT_MATRIX = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, -1.0],
    [-1.0, 0.0, 1.0, 0.0],
    [1.0, -1.0, 1.0, -1.0],
], dtype=np.float64)


def forces_to_rpm(force_vec: np.ndarray) -> np.ndarray:
    fz, tx, ty, tz = force_vec
    
    # 限制力矩以确保所有电机推力非负
    # 最大可行力矩 = 0.5 * fz * L (当一个电机推力为0时)
    max_torque = 0.5 * max(fz, 0.01) * ARM_LENGTH  # ~0.006 N·m at hover
    tx = np.clip(tx, -max_torque, max_torque)
    ty = np.clip(ty, -max_torque, max_torque)
    tz = np.clip(tz, -max_torque * 0.5, max_torque * 0.5)  # yaw 更小
    
    rhs = np.array([
        fz,
        tx / ARM_LENGTH,
        ty / ARM_LENGTH,
        (tz * KF / max(KM, 1e-12)) if KM > 0 else 0.0,
    ], dtype=np.float64)
    try:
        thrusts = np.linalg.solve(SPLIT_MATRIX, rhs)
    except np.linalg.LinAlgError:
        thrusts = np.linalg.lstsq(SPLIT_MATRIX, rhs, rcond=None)[0]
    thrusts = np.clip(thrusts, 0.0, None)
    omega = np.sqrt(thrusts / max(KF, 1e-12))
    rpm = omega * 60.0 / (2.0 * math.pi)
    rpm = np.clip(rpm, 0.0, MAX_RPM)
    return rpm.astype(np.float32)


def build_trajectory(name: str) -> Dict[str, Any]:
    cfg = get_scg_trajectory_config(name)
    # 计算 t=0 时刻轨迹上的位置作为初始位置
    from utilities.trajectory_presets import scg_position
    initial_pos = scg_position(cfg.task, t=0.0, params=cfg.params, center=cfg.center)
    return {
        "type": cfg.task,
        "params": dict(cfg.params),
        "initial_xyz": initial_pos.tolist(),
    }


class PIDIsaacWrapper:
    def __init__(self, params: PIDParams):
        self._controller = TunablePIDController(params)

    def computeControl(self, control_timestep: float, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos):
        rpy = Rotation.from_quat(cur_quat).as_euler("XYZ", degrees=False)
        action = self._controller.compute(cur_pos, cur_vel, rpy, cur_ang_vel, target_pos)
        rpm = forces_to_rpm(action)
        pos_e = target_pos - cur_pos
        return rpm, pos_e, rpy


class LQRIsaacWrapper:
    def __init__(self, params: LQRParams):
        self._controller = TunableLQRController(params)

    def computeControl(self, control_timestep: float, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos):
        rpy = Rotation.from_quat(cur_quat).as_euler("XYZ", degrees=False)
        action = self._controller.compute(cur_pos, cur_vel, rpy, cur_ang_vel, target_pos)
        rpm = forces_to_rpm(action)
        pos_e = target_pos - cur_pos
        return rpm, pos_e, rpy


def load_params(path: Path) -> Tuple[PIDParams, LQRParams]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pid_block = data.get("pid", {})
    lqr_block = data.get("lqr", {})
    pid_params = PIDParams(**pid_block.get("params", pid_block))
    lqr_params = LQRParams(**lqr_block.get("params", lqr_block))
    return pid_params, lqr_params


def run_controller(name: str, controller, args) -> Dict[str, Any]:
    weights = {"state_cost": 1.0}
    trajectory = build_trajectory(args.trajectory)
    tester = SimulationTester(
        controller=controller,
        test_scenarios=[],
        weights=weights,
        duration_sec=args.duration,
        output_folder=args.log_dir,
        gui=args.gui,
        trajectory=trajectory,
        log_skip=max(1, args.log_skip),
        in_memory=True,
        quiet=args.quiet,
    )
    reward = tester.run()
    stats = dict(tester.last_results)
    stats["true_reward"] = reward
    stats["controller"] = name
    stats["log_path"] = tester.last_log_path
    return stats


def load_program(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Program file not found: {path}")
    return load_program_json(str(path))


class DirectForceSoarWrapper:
    """
    直接输出 6D 力/力矩的 Soar 包装器，与 BatchEvaluator 的训练方式一致。
    避免 RPM 中转导致的 KF 不一致问题。
    """
    def __init__(self, program: List[Dict[str, Any]]):
        self.program = program or []
        self.integral_pos_e = np.zeros(3, dtype=float)
        self.integral_rpy_e = np.zeros(3, dtype=float)
        # 导入 DSL 节点类型
        from core.dsl import BinaryOpNode, TerminalNode
        self.BinaryOpNode = BinaryOpNode
        self.TerminalNode = TerminalNode
        
    def _eval_ast(self, node, state: Dict[str, float]) -> float:
        try:
            v = node.evaluate(state)
            return float(v)
        except Exception:
            return 0.0
    
    def computeControl(self, control_timestep: float, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos):
        """返回 (rpm[4], pos_e[3], rpy_e[3])，与 MathProgramController 接口一致。
        但这里 rpm 实际上是 dummy，真正的力/力矩会被放进 _last_forces。
        """
        from scipy.spatial.transform import Rotation
        
        target_rpy = np.zeros(3, dtype=float)
        pos_e = np.asarray(target_pos) - np.asarray(cur_pos)
        rpy = Rotation.from_quat(cur_quat).as_euler('XYZ', degrees=False)
        rpy_e = target_rpy - rpy
        
        # 更新积分状态
        dt = max(0.0, float(control_timestep))
        self.integral_pos_e = self.integral_pos_e + pos_e * dt
        self.integral_rpy_e = self.integral_rpy_e + rpy_e * dt
        
        # 构造状态字典
        state = {
            'err_p_roll': float(rpy_e[0]), 'err_p_pitch': float(rpy_e[1]), 'err_p_yaw': float(rpy_e[2]),
            'err_d_roll': float(-cur_ang_vel[0]), 'err_d_pitch': float(-cur_ang_vel[1]), 'err_d_yaw': float(-cur_ang_vel[2]),
            'ang_vel_x': float(cur_ang_vel[0]), 'ang_vel_y': float(cur_ang_vel[1]), 'ang_vel_z': float(cur_ang_vel[2]),
            'err_i_roll': float(self.integral_rpy_e[0]), 'err_i_pitch': float(self.integral_rpy_e[1]), 'err_i_yaw': float(self.integral_rpy_e[2]),
            'pos_err_x': float(pos_e[0]), 'pos_err_y': float(pos_e[1]), 'pos_err_z': float(pos_e[2]),
            'err_i_x': float(self.integral_pos_e[0]), 'err_i_y': float(self.integral_pos_e[1]), 'err_i_z': float(self.integral_pos_e[2]),
            'pos_err_xy': float(np.linalg.norm(pos_e[:2])),
            'rpy_err_mag': float(np.linalg.norm(rpy_e)),
            'ang_vel_mag': float(np.linalg.norm(cur_ang_vel)),
            'pos_err_z_abs': float(abs(pos_e[2])),
            # 速度变量（程序可能使用这些）
            'vel_x': float(cur_vel[0]), 'vel_y': float(cur_vel[1]), 'vel_z': float(cur_vel[2]),
        }
        
        # 执行 DSL 规则
        u_out = {}
        for rule in self.program:
            for act in rule.get('action', []) or []:
                if hasattr(act, 'op') and act.op == 'set' and hasattr(act, 'left') and hasattr(act.left, 'value'):
                    key = str(getattr(act.left, 'value', ''))
                    right_node = getattr(act, 'right', None)
                    if right_node is not None and hasattr(right_node, 'evaluate'):
                        u_out[key] = self._eval_ast(right_node, state)
        
        # 提取 (fz, tx, ty, tz) 并裁剪到物理合理范围（与 BatchEvaluator 一致）
        fz = float(np.clip(u_out.get('u_fz', 0.0), -5.0, 5.0))
        tx = float(np.clip(u_out.get('u_tx', 0.0), -0.02, 0.02))
        ty = float(np.clip(u_out.get('u_ty', 0.0), -0.02, 0.02))
        tz = float(np.clip(u_out.get('u_tz', 0.0), -0.01, 0.01))
        
        # 存储力/力矩以供 SimulationTester 使用
        self._last_forces = np.array([0.0, 0.0, fz, tx, ty, tz], dtype=np.float32)
        
        # 将力/力矩转为 RPM（仅用于与现有接口兼容）
        rpm = forces_to_rpm(np.array([fz, tx, ty, tz]))
        return rpm, pos_e, rpy_e
    
    def get_last_forces(self) -> np.ndarray:
        """返回上一次 computeControl 计算的 6D 力/力矩。"""
        return getattr(self, '_last_forces', np.zeros(6, dtype=np.float32))


def build_soar_controller(program_path: Path) -> DirectForceSoarWrapper:
    program = load_program(program_path)
    return DirectForceSoarWrapper(program=program)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PID/LQR/Soar inside Isaac tester with SCG reward")
    parser.add_argument("--program", type=str, default="results/scg_aligned/figure8_safe_control_tracking_best.json")
    parser.add_argument("--params-json", type=str, default="results/tuned_baselines_final.json")
    parser.add_argument("--trajectory", type=str, choices=["hover", "figure8", "circle", "square", "helix"], default="figure8")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--log-skip", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/isaac_compare_pid_lqr_soar.json")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params_path = Path(args.params_json)
    program_path = Path(args.program)
    pid_params, lqr_params = load_params(params_path)

    pid_ctrl = PIDIsaacWrapper(pid_params)
    lqr_ctrl = LQRIsaacWrapper(lqr_params)
    soar_ctrl = build_soar_controller(program_path)

    results = []
    results.append(run_controller("PID", pid_ctrl, args))
    results.append(run_controller("LQR", lqr_ctrl, args))
    results.append(run_controller("Soar", soar_ctrl, args))

    output = {entry["controller"]: entry for entry in results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Controller        Reward        StateCost    ActionCost    LogFile")
    print("-" * 80)
    for entry in results:
        print(
            f"{entry['controller']:<15} {entry['true_reward']:<12.3f} {entry['state_cost']:<12.3f} "
            f"{entry['action_cost']:<12.6f} {entry.get('log_path','-')}"
        )
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

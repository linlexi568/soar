from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation
from .dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode

class MathProgramController:
    """
    使用“数学原语 DSL”直接输出控制量的控制器。
    - 规则列表 program: [{'condition': AST, 'action': [BinaryOpNode('set', Terminal('key'), expr_ast), ...]}]
    - 支持输出键：
      1) 机体力/力矩（LOCAL）：'u_fz','u_tx','u_ty','u_tz'
      2) 电机推力（牛顿）：'m0','m1','m2','m3'
    - 返回四电机 RPM，沿用现有 tester 流水线。
    """
    def __init__(self,
                 program: Optional[List[Dict[str, Any]]] = None,
                 suppress_init_print: bool = True):
        self.program = program or []
        self._suppress = suppress_init_print
        # 简单积分器（与 SimplePIDControl 分离，避免引入 PID 语义）
        self.integral_pos_e = np.zeros(3, dtype=float)
        self.integral_rpy_e = np.zeros(3, dtype=float)
        # 物理与混配常量（Crazyflie 近似）
        self.KF = 3.16e-10  # N/(rad/s)^2
        self.KM = 7.94e-12  # N*m/(rad/s)^2
        self.L = 0.046      # m（臂长一半）
        self.MAX_RPM = 20000.0

    # --- 小工具 ---
    def _eval_ast(self, node: ProgramNode, state: Dict[str, float]) -> float:
        try:
            # Disable conditional branches: treat IfNode as its then-branch only
            if isinstance(node, IfNode):
                return self._eval_ast(node.then_branch, state)
            v = node.evaluate(state)
            return float(v)
        except Exception:
            return 0.0

    def _state_from_inputs(self,
                           control_timestep: float,
                           cur_pos: np.ndarray,
                           cur_quat: np.ndarray,
                           cur_vel: np.ndarray,
                           cur_ang_vel: np.ndarray,
                           target_pos: np.ndarray,
                           target_rpy: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        if target_rpy is None:
            target_rpy = np.zeros(3, dtype=float)
        pos_e = target_pos - cur_pos
        rpy = Rotation.from_quat(cur_quat).as_euler('XYZ', degrees=False)
        rpy_e = target_rpy - rpy
        # 积分更新（原始误差）
        self.integral_pos_e = self.integral_pos_e + pos_e * float(max(0.0, control_timestep))
        self.integral_rpy_e = self.integral_rpy_e + rpy_e * float(max(0.0, control_timestep))
        pos_err_xy = float(np.linalg.norm(pos_e[:2]))
        rpy_err_mag = float(np.linalg.norm(rpy_e))
        ang_vel_mag = float(np.linalg.norm(cur_ang_vel))
        pos_err_z_abs = float(abs(pos_e[2]))
        st = {
            'err_p_roll': float(rpy_e[0]), 'err_p_pitch': float(rpy_e[1]), 'err_p_yaw': float(rpy_e[2]),
            'err_d_roll': float(-cur_ang_vel[0]), 'err_d_pitch': float(-cur_ang_vel[1]), 'err_d_yaw': float(-cur_ang_vel[2]),
            'ang_vel_x': float(cur_ang_vel[0]), 'ang_vel_y': float(cur_ang_vel[1]), 'ang_vel_z': float(cur_ang_vel[2]),
            'err_i_roll': float(self.integral_rpy_e[0]), 'err_i_pitch': float(self.integral_rpy_e[1]), 'err_i_yaw': float(self.integral_rpy_e[2]),
            'pos_err_x': float(pos_e[0]), 'pos_err_y': float(pos_e[1]), 'pos_err_z': float(pos_e[2]),
            'err_i_x': float(self.integral_pos_e[0]), 'err_i_y': float(self.integral_pos_e[1]), 'err_i_z': float(self.integral_pos_e[2]),
            'pos_err_xy': pos_err_xy, 'rpy_err_mag': rpy_err_mag, 'ang_vel_mag': ang_vel_mag, 'pos_err_z_abs': pos_err_z_abs,
        }
        return st, pos_e, rpy_e

    def _mix_to_motors(self, u: Dict[str, float]) -> np.ndarray:
        # 支持 m0..m3 直接指定（牛顿）
        if all(k in u for k in ('m0','m1','m2','m3')):
            T = np.array([max(0.0, float(u['m0'])),
                          max(0.0, float(u['m1'])),
                          max(0.0, float(u['m2'])),
                          max(0.0, float(u['m3']))], dtype=float)
        else:
            # 使用合力/力矩混配（忽略 yaw 扭矩）
            Fz = float(u.get('u_fz', 0.0))
            tx = float(u.get('u_tx', 0.0))
            ty = float(u.get('u_ty', 0.0))
            # yaw 扭矩暂不混配（KM 值很小，可忽略）
            # 解算：
            #   T1 = 0.25*Fz - 0.5/L * ty
            #   T2 = 0.25*Fz + 0.5/L * tx
            #   T3 = 0.25*Fz + 0.5/L * ty
            #   T4 = 0.25*Fz - 0.5/L * tx
            invL2 = 0.5 / max(1e-6, self.L)
            T1 = 0.25 * Fz - invL2 * ty
            T2 = 0.25 * Fz + invL2 * tx
            T3 = 0.25 * Fz + invL2 * ty
            T4 = 0.25 * Fz - invL2 * tx
            T = np.array([T1, T2, T3, T4], dtype=float)
            T = np.maximum(0.0, T)
        # 推力 -> RPM
        # T = KF * omega^2, omega = RPM * 2π/60
        # RPM = sqrt(T/KF) * 60/(2π)
        rpm = np.sqrt(np.maximum(0.0, T) / max(1e-12, self.KF)) * (60.0 / (2.0 * np.pi))
        rpm = np.clip(rpm, 0.0, self.MAX_RPM)
        return rpm

    def computeControl(self,
                       control_timestep: float,
                       cur_pos: np.ndarray,
                       cur_quat: np.ndarray,
                       cur_vel: np.ndarray,
                       cur_ang_vel: np.ndarray,
                       target_pos: np.ndarray,
                       target_rpy: np.ndarray = np.zeros(3),
                       target_vel: np.ndarray = np.zeros(3),
                       target_rpy_rates: np.ndarray = np.zeros(3)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, pos_e, rpy_e = self._state_from_inputs(control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy)
        # 输出累加（直接忽略条件，所有 action 按顺序应用，后写覆盖前写）
        u_out: Dict[str, float] = {}
        for rule in (self.program or []):
            acts = rule.get('action', []) or []
            for act in acts:
                if isinstance(act, BinaryOpNode) and act.op == 'set' and isinstance(act.left, TerminalNode):
                    try:
                        k = str(act.left.value)
                        v = self._eval_ast(act.right, state) if hasattr(act, 'right') else 0.0
                        u_out[k] = float(v)
                    except Exception:
                        continue
        rpm = self._mix_to_motors(u_out)
        return rpm, pos_e, rpy_e

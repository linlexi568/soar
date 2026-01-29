#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe-Control-Gym × Soar Adapter (skeleton)

Purpose
- Wrap Soar controller (DSL or baseline PD) into a callable policy for safe-control-gym
- Provide action-space adapters for either [thrust, torques] or motor commands
- Keep strong comments so mapping can be finalized quickly once env keys are known

Quick use
- As a library: from adapters.safecontrol_soar_adapter import SoarPolicyAdapter
- In eval script: adapter = SoarPolicyAdapter(mode='pid', action_space='thrust_torque')

Notes
- This file avoids hard dependency on Soar internals at import time; actual import is lazy.
- When mode='soar', it will attempt to load a JSON program via compare_gpu_cpu_executor._load_programs.
"""
from __future__ import annotations
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ------------------------------
# Action mixer: thrust/torques ↔ motors
# ------------------------------
@dataclass
class MixerCoeffs:
    kf: float = 2.8e-08  # thrust coefficient (from repo's batch_evaluation)
    km: float = 1.1e-10  # yaw moment coefficient
    L: float = 0.046     # arm length (m)
    min_rpm: float = 0.0
    max_rpm: float = 30000.0

class QuadXActionMixer:
    """Simple quad-X mixer consistent with our repo coefficients.

    Conventions follow batch_evaluation._rpm_to_forces_local for sign/order as closely as possible.
    T_i = kf * omega_i^2 (omega in rad/s). We convert desired [fz, tx, ty, tz] to individual thrusts,
    then to omega^2, then to RPM (and clip). Ordering assumed [m0, m1, m2, m3] in X configuration.
    """
    def __init__(self, coeffs: MixerCoeffs = MixerCoeffs()) -> None:
        self.c = coeffs

    def thrust_torque_to_motors(self, fz: float, tx: float, ty: float, tz: float) -> np.ndarray:
        c = self.c
        # Solve for per-motor thrusts T = [T0, T1, T2, T3]
        # Equations (match _rpm_to_forces_local):
        # fz = sum(T)
        # tx =  L * (T1 - T3)
        # ty =  L * (T2 - T0)
        # tz =  KM * ((w0^2 - w1^2 + w2^2 - w3^2)) ≈ KM/KF * (T0 - T1 + T2 - T3)
        # Approximate tz using thrusts via ratio km/kf to keep it linear in T.
        r = c.km / c.kf if c.kf > 0 else 0.0
        A = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0,  c.L, 0.0, -c.L],
            [-c.L, 0.0,  c.L, 0.0],
            [ r,  -r,   r,   -r ],
        ], dtype=np.float64)
        b = np.array([fz, tx, ty, tz], dtype=np.float64)
        try:
            T = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            T = np.linalg.lstsq(A, b, rcond=None)[0]
        T = np.clip(T, 0.0, None)  # thrust cannot be negative
        # Convert thrust to omega^2 then RPM
        omega2 = T / max(c.kf, 1e-12)
        omega = np.sqrt(np.maximum(omega2, 0.0))
        rpm = omega * 60.0 / (2.0 * math.pi)
        rpm = np.clip(rpm, c.min_rpm, c.max_rpm)
        return rpm.astype(np.float32)

# ------------------------------
# Adapter
# ------------------------------
class SoarPolicyAdapter:
    """
    A callable policy that maps safe-control-gym observations to actions using:
    - mode='soar': Soar DSL program via GPUProgramExecutor (if available)
    - mode='pid': baseline PID controller (no repo dependency), useful for quick validation
    - mode='lqr': axis-decoupled LQR using a double-integrator model solved via SciPy C.A.R.E.

    action_space:
    - 'thrust_torque' -> returns [u_fz, u_tx, u_ty, u_tz]
    - 'motors' -> returns 4x motor RPM (or normalized 0..1 if normalize_motors=True)
    """
    def __init__(
        self,
        mode: str = 'pid',
        action_space: str = 'thrust_torque',
        program_json: Optional[str] = None,
        device: str = 'cpu',
        normalize_motors: bool = False,
        mixer: Optional[QuadXActionMixer] = None,
    ) -> None:
        assert action_space in ('thrust_torque', 'motors')
        assert mode in ('pid', 'lqr', 'soar')
        self.mode = mode
        self.action_space = action_space
        self.normalize_motors = normalize_motors
        self.mixer = mixer or QuadXActionMixer()

        self._soar = None
        self._token = None
        self._device = device
        self._program_json = program_json

        # PD gains (baseline, conservative; tune as needed)
        self.k_z_p = 9.0
        self.k_z_d = 3.0
        self.k_x_p = 4.0
        self.k_x_d = 2.0
        self.k_y_p = 4.0
        self.k_y_d = 2.0
        self.k_roll_p = 2.5
        self.k_roll_d = 0.8
        self.k_pitch_p = 2.5
        self.k_pitch_d = 0.8
        self.k_yaw_p = 1.5
        self.k_yaw_d = 0.4

        # Pre-compute LQR gains if requested
        if self.mode == 'lqr':
            self._init_lqr()

        if self.mode == 'soar':
            self._lazy_init_soar()

    # --------- LQR helpers ---------
    def _init_lqr(self) -> None:
        from scipy.linalg import solve_continuous_are

        def _gain(q_pos: float, q_vel: float, r: float) -> np.ndarray:
            A = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
            B = np.array([[0.0], [1.0]], dtype=np.float64)
            Q = np.diag([q_pos, q_vel])
            R = np.array([[max(r, 1e-6)]], dtype=np.float64)
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.solve(R, B.T @ P)
            return K.reshape(2)

        self._K_altitude = _gain(4.0, 1.5, 0.2)
        self._K_xy = _gain(6.0, 2.5, 0.3)
        self._K_attitude = _gain(3.0, 1.0, 0.1)
        self._K_yaw = _gain(2.0, 0.8, 0.05)

    # --------- Soar lazy init ---------
    def _lazy_init_soar(self) -> None:
        # Inject repo paths
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sys.path.insert(0, os.path.join(repo_root, '01_soar'))
        sys.path.insert(0, os.path.join(repo_root, 'scripts'))
        try:
            from utils.gpu_program_executor import GPUProgramExecutor  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import GPUProgramExecutor: {e}")
        self._executor_cls = GPUProgramExecutor
        # Load program
        programs = None
        if self._program_json:
            try:
                from compare_gpu_cpu_executor import _load_programs  # type: ignore
                programs = _load_programs([self._program_json])
            except Exception:
                programs = None
        if not programs:
            # Fallback: will use PD
            self.mode = 'pid'
            return
        self._soar = self._executor_cls(device=self._device)
        self._soar.reset_state()
        self._token = self._soar.prepare_batch(programs)

    # --------- Observation mapping ---------
    def _parse_obs(self, obs: Any) -> Dict[str, float]:
        """Map safe-control-gym obs -> state components needed by controller.

        This is a placeholder that supports both dict and flat arrays. Please adjust
        indices/keys to match your exact safe-control-gym env version.
        """
        if isinstance(obs, dict):
            p = np.asarray(obs.get('pos', obs.get('p', [0, 0, 0])), dtype=np.float64)
            v = np.asarray(obs.get('vel', obs.get('v', [0, 0, 0])), dtype=np.float64)
            rpy = np.asarray(obs.get('rpy', [0, 0, 0]), dtype=np.float64)
            omega = np.asarray(obs.get('omega', obs.get('ang_vel', [0, 0, 0])), dtype=np.float64)
            rpy_err = np.asarray(obs.get('rpy_err', [0, 0, 0]), dtype=np.float64)
            pos_err = np.asarray(obs.get('pos_err', [0, 0, 0]), dtype=np.float64)
        else:
            arr = np.asarray(obs, dtype=np.float64).reshape(-1)
            # Heuristic fallback indices (customize): [px,py,pz,vx,vy,vz,roll,pitch,yaw,wx,wy,wz]
            p = arr[0:3] if arr.size >= 3 else np.zeros(3)
            v = arr[3:6] if arr.size >= 6 else np.zeros(3)
            rpy = arr[6:9] if arr.size >= 9 else np.zeros(3)
            omega = arr[9:12] if arr.size >= 12 else np.zeros(3)
            # Without reference deltas, set errors as negatives of state (hover at origin)
            pos_err = -p
            rpy_err = -rpy
        return {
            'pos_err_x': float(pos_err[0]),
            'pos_err_y': float(pos_err[1]),
            'pos_err_z': float(pos_err[2]),
            'vel_x': float(v[0]),
            'vel_y': float(v[1]),
            'vel_z': float(v[2]),
            'err_p_roll': float(rpy_err[0]),
            'err_p_pitch': float(rpy_err[1]),
            'err_p_yaw': float(rpy_err[2]),
            'ang_vel_x': float(omega[0]),
            'ang_vel_y': float(omega[1]),
            'ang_vel_z': float(omega[2]),
        }

    # --------- PD baseline ---------
    def _act_pid(self, st: Dict[str, float]) -> Tuple[float, float, float, float]:
        ez = st['pos_err_z']
        evz = st['vel_z']
        ex = st['pos_err_x']
        evx = st['vel_x']
        ey = st['pos_err_y']
        evy = st['vel_y']
        e_roll = st['err_p_roll']
        e_pitch = st['err_p_pitch']
        e_yaw = st['err_p_yaw']
        w_x = st['ang_vel_x']
        w_y = st['ang_vel_y']
        w_z = st['ang_vel_z']
        # Z (height): PD with roll/pitch assist
        u_fz = (
            self.k_z_p * ez +
            self.k_z_d * evz +
            0.3 * self.k_roll_p * e_roll +
            0.3 * self.k_pitch_p * e_pitch
        )
        # X/Y via roll/pitch (indirect position control)
        u_tx = (
            self.k_x_p * ex +
            self.k_x_d * evx +
            self.k_roll_p * e_roll +
            self.k_roll_d * w_x
        )
        u_ty = (
            self.k_y_p * ey +
            self.k_y_d * evy +
            self.k_pitch_p * e_pitch +
            self.k_pitch_d * w_y
        )
        u_tz = (
            self.k_yaw_p * e_yaw +
            self.k_yaw_d * w_z
        )
        return float(u_fz), float(u_tx), float(u_ty), float(u_tz)

    # --------- LQR baseline ---------
    def _act_lqr(self, st: Dict[str, float]) -> Tuple[float, float, float, float]:
        if not hasattr(self, '_K_altitude'):
            self._init_lqr()

        def _dot(K: np.ndarray, state: Tuple[float, float]) -> float:
            vec = np.array([state[0], state[1]], dtype=np.float64)
            return -float(np.dot(K, vec))

        u_fz = _dot(self._K_altitude, (st['pos_err_z'], st['vel_z']))
        # Outer-loop translation → desired attitude, inner-loop stabilizes attitude directly
        pitch_outer = _dot(self._K_xy, (st['pos_err_x'], st['vel_x']))
        roll_outer = _dot(self._K_xy, (st['pos_err_y'], st['vel_y']))
        pitch_inner = _dot(self._K_attitude, (st['err_p_pitch'], st['ang_vel_y']))
        roll_inner = _dot(self._K_attitude, (st['err_p_roll'], st['ang_vel_x']))
        u_ty = pitch_outer + pitch_inner
        u_tx = roll_outer + roll_inner
        u_tz = _dot(self._K_yaw, (st['err_p_yaw'], st['ang_vel_z']))
        return u_fz, u_tx, u_ty, u_tz

    # --------- Soar DSL path ---------
    def _act_soar(self, st: Dict[str, float]) -> Tuple[float, float, float, float]:
        if not self._soar or not self._token:
            # Fallback to PD if executor not ready
            return self._act_pid(st)
        import torch
        device = torch.device(self._device)
        batch = 1
        pos = torch.tensor([[ -st['pos_err_x'], -st['pos_err_y'], -st['pos_err_z'] ]], device=device, dtype=torch.float32)
        vel = torch.tensor([[ st['vel_x'], st['vel_y'], st['vel_z'] ]], device=device, dtype=torch.float32)
        omega = torch.tensor([[ st['ang_vel_x'], st['ang_vel_y'], st['ang_vel_z'] ]], device=device, dtype=torch.float32)
        quat = torch.tensor([[0,0,0,1]], device=device, dtype=torch.float32)  # assume small angles
        target = torch.zeros((batch,3), device=device, dtype=torch.float32)
        integral_states = [{
            'err_i_x': 0.0, 'err_i_y': 0.0, 'err_i_z': 0.0,
            'err_i_roll': 0.0, 'err_i_pitch': 0.0, 'err_i_yaw': 0.0,
        }]
        use_mask = torch.ones((batch,), dtype=torch.bool, device=device)
        (outputs, _, _) = self._soar.evaluate_from_raw_obs(
            self._token, pos, vel, omega, quat, target, integral_states,
            use_mask, active_mask=None, force_cpu=(self._device=='cpu')
        )
        u = outputs[0].detach().cpu().numpy().tolist()  # [u_fz, u_tx, u_ty, u_tz]
        return float(u[0]), float(u[1]), float(u[2]), float(u[3])

    # --------- public API ---------
    def act(self, obs: Any) -> np.ndarray:
        st = self._parse_obs(obs)
        if self.mode == 'soar':
            u_fz, u_tx, u_ty, u_tz = self._act_soar(st)
        elif self.mode == 'lqr':
            u_fz, u_tx, u_ty, u_tz = self._act_lqr(st)
        else:
            u_fz, u_tx, u_ty, u_tz = self._act_pid(st)
        if self.action_space == 'thrust_torque':
            return np.array([u_fz, u_tx, u_ty, u_tz], dtype=np.float32)
        rpm = self.mixer.thrust_torque_to_motors(u_fz, u_tx, u_ty, u_tz)
        if self.normalize_motors:
            rpm_norm = np.clip(rpm / max(self.mixer.c.max_rpm, 1e-6), 0.0, 1.0)
            return rpm_norm
        return rpm

    def close(self) -> None:
        try:
            if self._soar and self._token is not None:
                self._soar.release_batch(self._token)
        except Exception:
            pass
        self._soar = None
        self._token = None

if __name__ == '__main__':
    adapter = SoarPolicyAdapter(mode='pid', action_space='thrust_torque')
    dummy_obs = {
        'pos': [0.2, -0.1, -0.05],
        'vel': [0.0, 0.0, 0.0],
        'rpy': [0.05, -0.04, 0.02],
        'omega': [0.0, 0.0, 0.0],
        'pos_err': [-0.2, 0.1, 0.05],
        'rpy_err': [-0.05, 0.04, -0.02]
    }
    a = adapter.act(dummy_obs)
    print('action:', a)

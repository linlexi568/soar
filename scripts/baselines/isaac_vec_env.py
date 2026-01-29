from __future__ import annotations

"""Stable-Baselines compatible Isaac Gym vectorized environment.

This wrapper exposes the native :class:`IsaacGymDroneEnv` using the same
observation/action conventions required by the safe-control-gym baselines so we
can evaluate PPO/SAC policies under the Soar reward infrastructure.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "01_soar") not in sys.path:
    sys.path.insert(0, str(_ROOT / "01_soar"))

from envs.isaac_gym_drone_env import IsaacGymDroneEnv
from utils.reward_stepwise import StepwiseRewardCalculator
from utilities.reward_profiles import get_reward_profile
from utilities.trajectory_presets import get_scg_trajectory_config


@dataclass(frozen=True)
class TrajectorySpec:
    task: str
    params: Dict[str, float]
    initial_xyz: Sequence[float]


def default_trajectory_params() -> Dict[str, Dict[str, float]]:
    catalog: Dict[str, Dict[str, float]] = {}
    for task in ("hover", "figure8", "circle", "square", "helix"):
        cfg = get_scg_trajectory_config(task)
        params = dict(cfg.params)
        params["center"] = cfg.center
        catalog[task] = params
    return catalog


INITIAL_XYZ_DEFAULT: Sequence[float] = (0.0, 0.0, 1.0)


class IsaacGymVecEnv(VecEnv):
    """Light-weight VecEnv wrapper around :class:`IsaacGymDroneEnv`."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str,
        task: str,
        duration: float,
        reward_profile: str = 'safe_control_tracking',
        traj_params: Optional[Dict[str, float]] = None,
        initial_xyz: Optional[Iterable[float]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.task = task
        self.duration = float(duration)
        cfg = get_scg_trajectory_config(task)
        resolved_params = dict(cfg.params)
        if traj_params:
            # 忽略用户传入的 center，center 通过 initial_xyz 控制
            resolved_params.update({k: v for k, v in traj_params.items() if k != "center"})
        self.target_params = resolved_params
        init_xyz = initial_xyz if initial_xyz is not None else cfg.center
        self.initial_xyz = torch.as_tensor(init_xyz, device=self.device, dtype=torch.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        super().__init__(self.num_envs, self.observation_space, self.action_space)

        if reward_profile != 'safe_control_tracking':
            raise ValueError(
                "IsaacGymVecEnv only supports the 'safe_control_tracking' reward profile."
            )

        self.env_pool = IsaacGymDroneEnv(
            num_envs=self.num_envs,
            device=device,
            headless=True,
            duration_sec=self.duration,
        )

        self.control_freq = getattr(self.env_pool, "control_freq", 48.0)
        self.dt = 1.0 / float(self.control_freq)

        weights, coeffs = get_reward_profile('safe_control_tracking')
        self.reward_calc = StepwiseRewardCalculator(
            weights,
            coeffs,
            dt=self.dt,
            num_envs=self.num_envs,
            device=device,
        )

        self.env_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.actions_buf: Optional[torch.Tensor] = None

        self.env_pool.reset()

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:  # type: ignore[override]
        if seed is None:
            return
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> np.ndarray:  # type: ignore[override]
        self.env_pool.reset()
        self.env_steps.zero_()
        weights = self.reward_calc.w
        coeffs = self.reward_calc.k
        self.reward_calc = StepwiseRewardCalculator(
            dict(weights),
            dict(coeffs),
            dt=self.dt,
            num_envs=self.num_envs,
            device=str(self.device),
        )
        obs_dict = self.env_pool.get_obs()
        t = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        obs_tensor = self._format_obs(obs_dict, t)
        return obs_tensor.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions_buf = torch.as_tensor(actions, device=self.device, dtype=torch.float32)

    def step_wait(self):  # type: ignore[override]
        if self.actions_buf is None:
            raise RuntimeError("step_wait called before step_async")

        u_fz = self.actions_buf[:, 0] * 20.0
        u_tx = self.actions_buf[:, 1] * 10.0
        u_ty = self.actions_buf[:, 2] * 10.0
        u_tz = self.actions_buf[:, 3] * 10.0

        zeros = torch.zeros(self.num_envs, device=self.device)
        forces_6d = torch.stack(
            [
                zeros,
                zeros,
                u_fz,
                u_tx,
                u_ty,
                u_tz,
            ],
            dim=1,
        )

        obs_terminal_dict, _, dones, _ = self.env_pool.step(forces_6d)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.bool)

        t = self.env_steps.float() * self.dt
        targets = self._compute_targets(t)

        pos = torch.as_tensor(obs_terminal_dict["position"], device=self.device)
        vel = torch.as_tensor(obs_terminal_dict["velocity"], device=self.device)
        quat = torch.as_tensor(obs_terminal_dict["orientation"], device=self.device)
        omega = torch.as_tensor(obs_terminal_dict["angular_velocity"], device=self.device)

        rewards = self.reward_calc.compute_step(
            pos=pos,
            target=targets,
            vel=vel,
            omega=omega,
            actions=forces_6d,
            done_mask=dones,
            quat=quat,
        )

        self.env_steps += 1

        final_obs_dict = obs_terminal_dict
        if dones.any():
            reset_ids = torch.nonzero(dones).squeeze(-1)
            if reset_ids.numel() > 0:
                self.env_steps[reset_ids] = 0
            final_obs_dict = self.env_pool.get_obs()

        current_t = self.env_steps.float() * self.dt
        final_obs_tensor = self._format_obs(final_obs_dict, current_t)

        infos = [{} for _ in range(self.num_envs)]
        if dones.any():
            terminal_obs_tensor = self._format_obs(obs_terminal_dict, t)
            done_indices = torch.nonzero(dones).squeeze(-1).cpu().numpy()
            for idx in done_indices:
                infos[idx]["terminal_observation"] = terminal_obs_tensor[idx].cpu().numpy()

        return (
            final_obs_tensor.cpu().numpy(),
            rewards.cpu().numpy(),
            dones.cpu().numpy(),
            infos,
        )

    def close(self) -> None:  # type: ignore[override]
        if hasattr(self, "env_pool"):
            self.env_pool.close()

    # ------------------------------------------------------------------
    # VecEnv protocol helpers
    # ------------------------------------------------------------------
    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.num_envs if indices is None else len(indices)
        return [False] * n

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        value = getattr(self, attr_name, None)
        n = self.num_envs if indices is None else len(indices)
        return [value] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_targets(self, t_tensor: torch.Tensor) -> torch.Tensor:
        params = self.target_params
        base = self.initial_xyz.unsqueeze(0).expand(self.num_envs, -1)

        if self.task == "hover":
            return base

        if self.task == "figure8":
            scale = float(params.get("scale", params.get("A", 0.8)))
            A = float(params.get("A", scale))
            B = float(params.get("B", scale))
            period = float(params.get("period", 12.0))
            plane = str(params.get("plane", "xy")).lower()
            idx_a, idx_b = self._plane_indices(plane)
            w = 2.0 * np.pi / period
            x = A * torch.sin(w * t_tensor)
            y = B * torch.sin(w * t_tensor) * torch.cos(w * t_tensor)
            delta = torch.zeros(self.num_envs, 3, device=self.device)
            delta[:, idx_a] = x
            delta[:, idx_b] = y
            return base + delta

        if self.task == "circle":
            R = float(params.get("R", 0.9))
            period = float(params.get("period", 10.0))
            plane = str(params.get("plane", "xy")).lower()
            idx_a, idx_b = self._plane_indices(plane)
            w = 2.0 * np.pi / period
            x = R * torch.cos(w * t_tensor)
            y = R * torch.sin(w * t_tensor)
            delta = torch.zeros(self.num_envs, 3, device=self.device)
            delta[:, idx_a] = x
            delta[:, idx_b] = y
            return base + delta

        if self.task == "helix":
            R = float(params.get("R", 0.7))
            period = float(params.get("period", 10.0))
            vz = float(params.get("v_z", 0.15))
            plane = str(params.get("plane", "xy")).lower()
            idx_a, idx_b = self._plane_indices(plane)
            w = 2.0 * np.pi / period
            x = R * torch.cos(w * t_tensor)
            y = R * torch.sin(w * t_tensor)
            z = vz * t_tensor
            delta = torch.zeros(self.num_envs, 3, device=self.device)
            delta[:, idx_a] = x
            delta[:, idx_b] = y
            delta[:, 2] += z
            return base + delta

        if self.task == "square":
            scale = float(params.get("scale", params.get("side", 0.8)))
            period = float(params.get("period", 8.0))
            plane = str(params.get("plane", "xy")).lower()
            idx_a, idx_b = self._plane_indices(plane)
            seg_period = period / 4.0
            traverse_speed = scale / max(seg_period, 1e-6)
            t_mod = torch.remainder(t_tensor, period)
            seg_float = torch.clamp(torch.floor(t_mod / seg_period), max=3)
            seg_time = t_mod - seg_float * seg_period
            seg_pos = traverse_speed * seg_time
            coord_a = torch.zeros_like(t_tensor)
            coord_b = torch.zeros_like(t_tensor)
            mask0 = seg_float == 0
            mask1 = seg_float == 1
            mask2 = seg_float == 2
            mask3 = seg_float == 3
            coord_a[mask1] = -seg_pos[mask1]
            coord_a[mask2] = -scale
            coord_a[mask3] = -scale + seg_pos[mask3]
            coord_b[mask0] = seg_pos[mask0]
            coord_b[mask1] = scale
            coord_b[mask2] = scale - seg_pos[mask2]
            delta = torch.zeros(self.num_envs, 3, device=self.device)
            delta[:, idx_a] = coord_a
            delta[:, idx_b] = coord_b
            return base + delta

        return base

    @staticmethod
    def _plane_indices(plane: str) -> tuple[int, int]:
        axis = {"x": 0, "y": 1, "z": 2}
        if not isinstance(plane, str):
            return 0, 1
        plane = plane.lower()
        if len(plane) != 2 or plane[0] == plane[1]:
            return 0, 1
        idx_a = axis.get(plane[0], 0)
        idx_b = axis.get(plane[1], 1)
        return idx_a, idx_b

    def _format_obs(self, obs_dict, t_tensor: torch.Tensor) -> torch.Tensor:
        pos = torch.as_tensor(obs_dict["position"], device=self.device)
        vel = torch.as_tensor(obs_dict["velocity"], device=self.device)
        quat = torch.as_tensor(obs_dict["orientation"], device=self.device)
        omega = torch.as_tensor(obs_dict["angular_velocity"], device=self.device)

        target_pos = self._compute_targets(t_tensor)
        pos_err = pos - target_pos
        rpy = self._quat_to_rpy(quat)
        obs = torch.cat([pos_err, vel, omega, rpy], dim=1)
        return obs

    def _quat_to_rpy(self, q: torch.Tensor) -> torch.Tensor:
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (np.pi / 2), torch.asin(sinp))

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=1)


__all__ = ["IsaacGymVecEnv", "TrajectorySpec", "default_trajectory_params", "INITIAL_XYZ_DEFAULT"]

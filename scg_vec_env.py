#!/usr/bin/env python3
"""Stable-Baselines friendly Isaac Gym vectorized env with SCG reward.

This module exposes :class:`IsaacSCGVecEnv`, a thin wrapper around
``scripts.sb3.isaac_gym_wrapper.IsaacGymSB3VecEnv`` so training scripts can
instantiate a ready-to-use VecEnv without wiring CLI arguments. All reward
computations go through ``SCGExactRewardCalculator`` to stay bitwise-aligned
with Soar's evaluator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import sys
import os

# ⚠️ CRITICAL: Setup Isaac Gym paths BEFORE any torch import
_REPO_ROOT = Path(__file__).resolve().parent
_ISAAC_GYM_PY = _REPO_ROOT / 'isaacgym' / 'python'
if _ISAAC_GYM_PY.exists() and str(_ISAAC_GYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAAC_GYM_PY))

_ISAAC_BINDINGS = _ISAAC_GYM_PY / 'isaacgym' / '_bindings' / 'linux-x86_64'
if _ISAAC_BINDINGS.exists():
    os.environ['LD_LIBRARY_PATH'] = str(_ISAAC_BINDINGS) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

# Import Isaac Gym BEFORE torch
try:
    from isaacgym import gymapi  # type: ignore
except Exception:
    pass

# Now safe to import numpy and torch-dependent modules
import numpy as np

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

# Make sure repo modules are importable
for extra in (_REPO_ROOT, _REPO_ROOT / 'scripts', _REPO_ROOT / '01_soar'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts.sb3.isaac_gym_wrapper import IsaacGymSB3VecEnv  # type: ignore  # noqa: E402


class IsaacSCGVecEnv(VecEnv):
    """Expose the SCG-aligned Isaac Gym batch env as an SB3 VecEnv."""

    def __init__(
        self,
        num_envs: int = 256,
        task: str = 'square',
        duration: float = 5.0,
        device: str = 'cuda:0',
        trajectory_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._inner = IsaacGymSB3VecEnv(
            num_envs=num_envs,
            trajectory_type=task,
            duration=duration,
            reward_type='scg_exact',
            trajectory_params=trajectory_params,
            device=device,
        )
        super().__init__(
            num_envs=self._inner.num_envs,
            observation_space=self._inner.observation_space,
            action_space=self._inner.action_space,
        )
        self._pending_actions: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        return self._inner.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        if self._pending_actions is None:
            raise RuntimeError('step_wait() called before step_async()')
        obs, rewards, dones, infos = self._inner.step(self._pending_actions)
        self._pending_actions = None
        # SB3 expects np.ndarray outputs with shape (n_env,)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        # Underlying env already resets finished replicas; ensure infos list len.
        if len(infos) != self.num_envs:
            infos = list(infos) + [{} for _ in range(self.num_envs - len(infos))]
        return obs, rewards, dones, infos

    def close(self) -> None:
        self._inner.close()

    def render(self, mode: str = 'human'):
        return None

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environments are wrapped with a given wrapper."""
        return [False] * self.num_envs

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        return [None] * self.num_envs

    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from vectorized environments."""
        return [getattr(self._inner, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name: str, value, indices=None):
        """Set attribute inside vectorized environments."""
        setattr(self._inner, attr_name, value)

    @property
    def unwrapped(self) -> 'IsaacSCGVecEnv':
        return self


__all__ = ['IsaacSCGVecEnv']

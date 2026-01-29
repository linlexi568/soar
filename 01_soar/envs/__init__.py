"""
01_soar/envs - 仿真环境适配层

当前实现：
- Isaac Gym (GPU): 100-500× 加速
"""

try:
    from .isaac_gym_drone_env import IsaacGymDroneEnv, ISAAC_GYM_AVAILABLE
    __all__ = ['IsaacGymDroneEnv', 'ISAAC_GYM_AVAILABLE']
except ImportError:
    ISAAC_GYM_AVAILABLE = False
    __all__ = ['ISAAC_GYM_AVAILABLE']

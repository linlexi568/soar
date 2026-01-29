import time
import os, importlib.util
import sys
import numpy as np
import pandas as pd
import torch
import types
from typing import Optional, List, Dict, Any
from scipy.spatial.transform import Rotation

from utilities.trajectory_presets import get_scg_trajectory_config, scg_position_velocity

# 动态加载 Isaac Gym 环境类，避免包名以数字开头导致的导入问题
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(_ROOT, '01_soar', 'envs', 'isaac_gym_drone_env.py')
_ENV_MOD_NAME = 'soar_env.isaac_gym_drone_env'
spec = importlib.util.spec_from_file_location(_ENV_MOD_NAME, _ENV_FILE)
if spec and spec.loader:
    _mod = importlib.util.module_from_spec(spec)
    sys.modules[_ENV_MOD_NAME] = _mod
    spec.loader.exec_module(_mod)  # type: ignore
    IsaacGymDroneEnv = getattr(_mod, 'IsaacGymDroneEnv')
else:
    raise ImportError('Failed to load IsaacGymDroneEnv')

_pkg_name = 'soar_env'
if _pkg_name not in sys.modules:
    pkg = types.ModuleType(_pkg_name)
    pkg.__path__ = [os.path.join(_ROOT, '01_soar', 'utils')]
    sys.modules[_pkg_name] = pkg
else:
    pkg = sys.modules[_pkg_name]

_REWARD_DIR = os.path.join(_ROOT, '01_soar', 'utils')
if isinstance(getattr(pkg, '__path__', None), list):
    pkg.__path__ = list(dict.fromkeys(list(pkg.__path__) + [_REWARD_DIR]))

_REWARD_SC_FILE = os.path.join(_REWARD_DIR, 'reward_scg_exact.py')
_REWARD_SC_MOD_NAME = f'{_pkg_name}.reward_scg_exact'
_reward_sc_spec = importlib.util.spec_from_file_location(_REWARD_SC_MOD_NAME, _REWARD_SC_FILE)
if _reward_sc_spec and _reward_sc_spec.loader:
    _reward_sc_mod = importlib.util.module_from_spec(_reward_sc_spec)
    sys.modules[_REWARD_SC_MOD_NAME] = _reward_sc_mod  # type: ignore
    _reward_sc_spec.loader.exec_module(_reward_sc_mod)  # type: ignore
else:
    raise ImportError('Failed to load reward_scg_exact')

_REWARD_FILE = os.path.join(_REWARD_DIR, 'reward_stepwise.py')
_REWARD_MOD_NAME = f'{_pkg_name}.reward_stepwise'
_reward_spec = importlib.util.spec_from_file_location(_REWARD_MOD_NAME, _REWARD_FILE)
if _reward_spec and _reward_spec.loader:
    _reward_mod = importlib.util.module_from_spec(_reward_spec)
    sys.modules[_REWARD_MOD_NAME] = _reward_mod  # type: ignore
    _reward_spec.loader.exec_module(_reward_mod)  # type: ignore
    RewardCalculator = getattr(_reward_mod, 'StepwiseRewardCalculator')
else:  # pragma: no cover - hard failure without SCG reward implementation
    raise ImportError('Failed to load StepwiseRewardCalculator')

# 轻量 CPU 后备环境（当 Isaac Gym 不可用时用于冒烟测试）
class SimpleCPUHoverEnv:
    def __init__(self, num_envs: int = 1, control_freq_hz: int = 48, duration_sec: int = 4):
        import torch
        self.num_envs = num_envs
        self.device = torch.device('cpu')
        self.control_freq_hz = control_freq_hz
        self.dt = 1.0 / float(control_freq_hz)
        self.duration_sec = duration_sec
        # 状态张量，形状尽量贴近 Isaac 接口
        self.pos = torch.zeros((num_envs, 3), dtype=torch.float32)
        self.pos[:, 2] = 1.0  # 初始高度
        self.quat = torch.zeros((num_envs, 4), dtype=torch.float32)
        self.quat[:, 3] = 1.0  # 单位四元数
        self.lin_vel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self.ang_vel = torch.zeros((num_envs, 3), dtype=torch.float32)
        # 简化动力学参数（与 Isaac Gym 保持一致）
        self.mass = 0.027
        self.g = 9.81
        self.KF = 2.8e-08  # 与 Isaac Gym 一致

    def get_states_batch(self) -> Dict[str, Any]:
        return {
            'pos': self.pos.clone(),
            'vel': self.lin_vel.clone(),
            'quat': self.quat.clone(),
            'omega': self.ang_vel.clone(),
        }

    def reset(self):
        import torch
        self.pos[:] = 0.0
        self.pos[:, 2] = 1.0
        self.lin_vel[:] = 0.0
        self.ang_vel[:] = 0.0
        self.quat[:] = 0.0
        self.quat[:, 3] = 1.0  # 单位四元数 [0,0,0,1]
        return self.get_states_batch()

    def step(self, actions):
        # actions: [N,4] RPM 或 [N,6] 力/力矩 [fx, fy, fz, tx, ty, tz]
        import torch
        
        if actions.shape[-1] == 6:
            # 直接力/力矩模式 [fx, fy, fz, tx, ty, tz]
            Fz = actions[:, 2].float()
            tau_x = actions[:, 3].float()
            tau_y = actions[:, 4].float()
            tau_z = actions[:, 5].float()
        elif actions.shape[-1] == 4:
            # RPM 模式
            omega = actions.float() * (2.0 * 3.1415926535 / 60.0)
            T = self.KF * (omega ** 2)  # [N,4]
            Fz = torch.sum(T, dim=1)
            KM = 7.94e-12
            L = 0.046
            tau_x = L * (T[:, 1] - T[:, 3])
            tau_y = L * (T[:, 2] - T[:, 0])
            tau_z = KM * (omega[:, 0]**2 - omega[:, 1]**2 + omega[:, 2]**2 - omega[:, 3]**2)
        else:
            raise ValueError(f'SimpleCPUHoverEnv 支持 [N,4] RPM 或 [N,6] 力/力矩, 收到 {actions.shape}')
        
        # 从四元数提取当前姿态角
        qx, qy, qz, qw = self.quat[:, 0], self.quat[:, 1], self.quat[:, 2], self.quat[:, 3]
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        
        # 推力在世界坐标系的分解
        cos_roll = torch.cos(roll)
        cos_pitch = torch.cos(pitch)
        sin_roll = torch.sin(roll)
        sin_pitch = torch.sin(pitch)
        
        thrust_world_x = Fz * sin_pitch * cos_roll / self.mass
        thrust_world_y = -Fz * sin_roll / self.mass
        thrust_world_z = Fz * cos_pitch * cos_roll / self.mass
        
        # 加速度（世界坐标系）
        acc_x = thrust_world_x
        acc_y = thrust_world_y
        acc_z = thrust_world_z - self.g
        
        # 速度和位置积分
        self.lin_vel[:, 0] += acc_x * self.dt
        self.lin_vel[:, 1] += acc_y * self.dt
        self.lin_vel[:, 2] += acc_z * self.dt
        self.pos[:, 0] += self.lin_vel[:, 0] * self.dt
        self.pos[:, 1] += self.lin_vel[:, 1] * self.dt
        self.pos[:, 2] += self.lin_vel[:, 2] * self.dt
        
        # 姿态动力学（简化：假设对角惯性矩阵）
        inertia = torch.tensor([1.4e-5, 1.4e-5, 2.17e-5], dtype=torch.float32)
        alpha_x = tau_x / inertia[0]
        alpha_y = tau_y / inertia[1]
        alpha_z = tau_z / inertia[2]
        self.ang_vel[:, 0] += alpha_x * self.dt
        self.ang_vel[:, 1] += alpha_y * self.dt
        self.ang_vel[:, 2] += alpha_z * self.dt
        self.ang_vel = torch.clamp(self.ang_vel, -10.0, 10.0)
        
        # 更新四元数（小角度近似）
        droll = self.ang_vel[:, 0] * self.dt
        dpitch = self.ang_vel[:, 1] * self.dt
        dyaw = self.ang_vel[:, 2] * self.dt
        new_roll = torch.clamp(roll + droll, -0.5, 0.5)
        new_pitch = torch.clamp(pitch + dpitch, -0.5, 0.5)
        new_yaw = dyaw  # yaw 累积
        # 重建四元数
        cy = torch.cos(new_yaw * 0.5)
        sy = torch.sin(new_yaw * 0.5)
        cp = torch.cos(new_pitch * 0.5)
        sp = torch.sin(new_pitch * 0.5)
        cr = torch.cos(new_roll * 0.5)
        sr = torch.sin(new_roll * 0.5)
        self.quat[:, 3] = cr * cp * cy + sr * sp * sy  # qw
        self.quat[:, 0] = sr * cp * cy - cr * sp * sy  # qx
        self.quat[:, 1] = cr * sp * cy + sr * cp * sy  # qy
        self.quat[:, 2] = cr * cp * sy - sr * sp * cy  # qz
        
        # 地面碰撞近似
        below = self.pos[:, 2] < 0.0
        if torch.any(below):
            self.pos[below, 2] = 0.0
            self.lin_vel[below, :] = 0.0
        # 其他返回值按 Isaac Gym 接口占位
        obs = {
            'position': self.pos.numpy(),
            'velocity': self.lin_vel.numpy(),
            'orientation': self.quat.numpy(),
            'angular_velocity': self.ang_vel.numpy(),
        }
        rewards = torch.zeros((self.num_envs,), dtype=torch.float32)
        dones = torch.zeros((self.num_envs,), dtype=torch.bool)
        return obs, rewards, dones, {}

# 单例环境，避免反复创建/销毁导致底层 Foundation 冲突
_ENV_SINGLETON = None  # type: ignore

class SimulationTester:
    """
    使用 Isaac Gym 环境的单环境测试器，提供与原 test.py 近似的接口，
    以便 verify_program/main 等脚本在单一后端下运行。
    """
    def __init__(self, controller, test_scenarios: list, weights: dict, duration_sec: int = 20,
                 output_folder: str = 'results', gui: bool = False, trajectory: Optional[dict] = None,
                 log_skip: int = 1, in_memory: bool = True, early_stop_rmse: Optional[float] = None,
                 early_min_seconds: float = 4.0, quiet: bool = True):
        self.controller = controller
        self.test_scenarios = test_scenarios
        self.weights = weights
        self.duration_sec = duration_sec
        self.output_folder = output_folder
        self.gui = gui
        self.trajectory = trajectory
        self.CONTROL_FREQ_HZ = 48
        self.CONTROL_TIMESTEP = 1.0 / self.CONTROL_FREQ_HZ
        self.log_skip = max(1, int(log_skip))
        self.in_memory = in_memory
        self.early_stop_rmse = early_stop_rmse
        self.early_min_seconds = max(0.0, early_min_seconds)
        self.quiet = quiet
        self.INITIAL_XYZ = np.array([[0, 0, 1.0]]) if not (trajectory and 'initial_xyz' in trajectory) else np.array([trajectory['initial_xyz']])
        self.last_log_df: Optional[pd.DataFrame] = None
        self.last_log_path: Optional[str] = None
        self.last_results: Dict[str, Any] = {}

    @staticmethod
    def _rpm_to_forces_cpu(rpm: np.ndarray) -> np.ndarray:
        KF = 2.8e-08
        KM = 1.1e-10
        L = 0.046
        omega = np.asarray(rpm, dtype=np.float64) * (2.0 * np.pi / 60.0)
        thrust = KF * (omega ** 2)
        fz = np.sum(thrust)
        tx = L * (thrust[1] - thrust[3])
        ty = L * (thrust[2] - thrust[0])
        tz = KM * (omega[0] ** 2 - omega[1] ** 2 + omega[2] ** 2 - omega[3] ** 2)
        return np.array([[0.0, 0.0, fz, tx, ty, tz]], dtype=np.float32)

    def _generate_trajectory_dataframe(self) -> Optional[pd.DataFrame]:
        if self.trajectory is None:
            return None
        num_steps = int(self.duration_sec * self.CONTROL_FREQ_HZ)
        timestamps = np.linspace(0, self.duration_sec, num=num_steps, endpoint=False)
        traj_type = self.trajectory.get('type', 'hover')
        params_override = dict(self.trajectory.get('params', {}))
        cfg = get_scg_trajectory_config(traj_type, overrides=params_override)
        params = cfg.params
        initial_xyz = self.INITIAL_XYZ[0]
        positions = np.zeros((num_steps, 3))
        for i, t in enumerate(timestamps):
            pos, _ = scg_position_velocity(traj_type, float(t), params=params, center=initial_xyz)
            positions[i] = pos
        traj_df = pd.DataFrame(positions, columns=['target_x','target_y','target_z'])
        traj_df['timestamp'] = timestamps
        return traj_df.set_index('timestamp')

    def run(self) -> float:
        traj_df = self._generate_trajectory_dataframe()
        effective_dt = self.CONTROL_TIMESTEP * self.log_skip
        # StepwiseRewardCalculator 初始化参数: weights, ks, dt, num_envs, device
        default_ks = {'K_VZ': 0.2, 'K_VAR': 0.1}  # 默认控制系数
        reward_calculator = RewardCalculator(weights=self.weights, ks=default_ks, dt=effective_dt, num_envs=1, device='cpu')
        reward_calculator.reset(num_envs=1)
        # 每次 run() 都创建新环境，确保控制器之间独立
        try:
            env = IsaacGymDroneEnv(num_envs=1, control_freq_hz=self.CONTROL_FREQ_HZ, duration_sec=self.duration_sec, headless=not self.gui)
        except ImportError:
            # Isaac Gym 不可用时，使用简化 CPU 后备环境进行冒烟测试
            env = SimpleCPUHoverEnv(num_envs=1, control_freq_hz=self.CONTROL_FREQ_HZ, duration_sec=self.duration_sec)
        env.reset()
        # 日志缓冲
        timestamps = []
        xs, ys, zs = [], [], []
        rs, ps, ys_ang = [], [], []
        vxs, vys, vzs = [], [], []
        wxs, wys, wzs = [], [], []
        rpm0, rpm1, rpm2, rpm3 = [], [], [], []
        scg_rewards, scg_state_costs, scg_action_costs = [], [], []

        start = time.time()
        running_sq_error_sum = 0.0; running_count = 0
        disturbance_times: List[float] = [s['time'] for s in self.test_scenarios if s.get('type','PULSE') in ('PULSE','MASS_CHANGE')]
        prev_state_cost = 0.0
        prev_action_cost = 0.0

        for i in range(int(self.duration_sec * self.CONTROL_FREQ_HZ)):
            t = i * self.CONTROL_TIMESTEP
            # 状态
            states = env.get_states_batch()
            cur_pos = states['pos'][0].cpu().numpy()
            cur_quat = states['quat'][0].cpu().numpy()
            cur_vel = states['vel'][0].cpu().numpy()
            cur_omega = states['omega'][0].cpu().numpy()

            # 目标
            if traj_df is not None:
                target_pos = traj_df.asof(t)[['target_x','target_y','target_z']].values
            else:
                target_pos = self.INITIAL_XYZ[0]

            rpm, pos_e, rpy_e = self.controller.computeControl(
                control_timestep=self.CONTROL_TIMESTEP,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_omega,
                target_pos=target_pos
            )
            
            # 检测控制器是否提供直接力/力矩输出（与 BatchEvaluator 训练方式一致）
            if hasattr(self.controller, 'get_last_forces'):
                forces_6d = self.controller.get_last_forces()
                actions = torch.from_numpy(forces_6d.reshape(1, 6)).to(env.device).float()
                forces_local = actions.clone()
            else:
                # 施加动作（RPM）
                actions = torch.from_numpy(rpm.reshape(1,4)).to(env.device).float()
                if hasattr(env, '_rpm_to_forces'):
                    forces_local = env._rpm_to_forces(actions).detach()
                else:
                    forces_np = self._rpm_to_forces_cpu(rpm)
                    forces_local = torch.from_numpy(forces_np).to(dtype=torch.float32)
            _, _, _, _ = env.step(actions)

            # 计算 SCG 奖励（使用 CPU 张量以复用 StepwiseRewardCalculator）
            pos_t = torch.from_numpy(cur_pos.reshape(1, 3)).to(dtype=torch.float32)
            vel_t = torch.from_numpy(cur_vel.reshape(1, 3)).to(dtype=torch.float32)
            quat_t = torch.from_numpy(cur_quat.reshape(1, 4)).to(dtype=torch.float32)
            omega_t = torch.from_numpy(cur_omega.reshape(1, 3)).to(dtype=torch.float32)
            target_t = torch.from_numpy(target_pos.reshape(1, 3)).to(dtype=torch.float32)
            reward_t = reward_calculator.compute_step(
                pos=pos_t,
                target=target_t,
                vel=vel_t,
                omega=omega_t,
                actions=forces_local.to('cpu'),
                quat=quat_t,
            )
            comps = reward_calculator.get_component_totals()
            total_state_cost = float(comps['state_cost'][0].item())
            total_action_cost = float(comps['action_cost'][0].item())
            step_state_cost = total_state_cost - prev_state_cost
            step_action_cost = total_action_cost - prev_action_cost
            prev_state_cost = total_state_cost
            prev_action_cost = total_action_cost

            if (i % self.log_skip) == 0:
                # 记录日志
                euler = Rotation.from_quat(cur_quat).as_euler('XYZ', degrees=False)
                timestamps.append(t)
                xs.append(cur_pos[0]); ys.append(cur_pos[1]); zs.append(cur_pos[2])
                rs.append(euler[0]); ps.append(euler[1]); ys_ang.append(euler[2])
                vxs.append(cur_vel[0]); vys.append(cur_vel[1]); vzs.append(cur_vel[2])
                wxs.append(cur_omega[0]); wys.append(cur_omega[1]); wzs.append(cur_omega[2])
                rpm0.append(rpm[0]); rpm1.append(rpm[1]); rpm2.append(rpm[2]); rpm3.append(rpm[3])
                scg_rewards.append(float(reward_t[0].item()))
                scg_state_costs.append(step_state_cost)
                scg_action_costs.append(step_action_cost)
                # 早停判断
                se = float(np.dot(pos_e, pos_e)); running_sq_error_sum += se; running_count += 1
                if (self.early_stop_rmse is not None) and (t >= self.early_min_seconds) and (running_count>0):
                    rmse = float(np.sqrt(running_sq_error_sum / running_count))
                    if rmse > self.early_stop_rmse:
                        break
            if self.gui:
                time.sleep(max(0.0, t - (time.time() - start)))

        # 组装 DataFrame
        data = np.column_stack([
            np.array(timestamps), np.array(xs), np.array(ys), np.array(zs),
            np.array(rs), np.array(ps), np.array(ys_ang),
            np.array(vxs), np.array(vys), np.array(vzs),
            np.array(wxs), np.array(wys), np.array(wzs),
            np.array(rpm0), np.array(rpm1), np.array(rpm2), np.array(rpm3),
            np.array(scg_rewards), np.array(scg_state_costs), np.array(scg_action_costs)
        ])
        cols = ['timestamp','x','y','z','r','p','y_angle','vx','vy','vz','wx','wy','wz','rpm0','rpm1','rpm2','rpm3',
                'reward_scg','state_cost','action_cost']
        log_df = pd.DataFrame(data, columns=cols)
        log_df['total_cost'] = log_df['state_cost'] + log_df['action_cost']
        
        # 简化的烟测评分：计算位置RMSE（用于调试输出）
        target_pos = self.INITIAL_XYZ[0]
        pos_errors = np.sqrt((log_df['x'] - target_pos[0])**2 + 
                            (log_df['y'] - target_pos[1])**2 + 
                            (log_df['z'] - target_pos[2])**2)
        scg_totals = reward_calculator.get_component_totals()
        total_state_cost = float(scg_totals['state_cost'][0].item())
        total_action_cost = float(scg_totals['action_cost'][0].item())
        
        # 🔧 使用累积总和奖励（与训练 BatchEvaluator reward_reduction='sum' 对齐）
        final_score = -(total_state_cost + total_action_cost)
        num_steps = len(timestamps)
        
        log_path = None
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            traj_name = self.trajectory['type'] if self.trajectory and 'type' in self.trajectory else 'custom'
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            log_filename = f"isaac_{traj_name}_{timestamp}.csv"
            log_path = os.path.join(self.output_folder, log_filename)
            log_df.to_csv(log_path, index=False)
        if self.in_memory:
            self.last_log_df = log_df
        self.last_log_path = log_path
        self.last_results = {
            'scg_reward': final_score,
            'state_cost': total_state_cost,
            'action_cost': total_action_cost,
            'mean_pos_rmse': float(np.mean(pos_errors)),
            'max_pos_error': float(np.max(pos_errors)),
            'log_path': log_path,
        }
        
        if not self.quiet:
            print(f"烟测完成 ({num_steps} 步):")
            print(f"  平均位置误差: {np.mean(pos_errors):.4f}m")
            print(f"  最大位置误差: {np.max(pos_errors):.4f}m")
            print(f"  SCG state cost (累积): {total_state_cost:.4f}")
            print(f"  SCG action cost (累积): {total_action_cost:.4f}")
            print(f"  SCG 奖励 (每步平均): {final_score:.4f}")
        
        # 单例环境不在此关闭，进程结束由外部回收
        return float(final_score)

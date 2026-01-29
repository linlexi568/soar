"""批量程序评估模块 - Isaac Gym GPU并行加速

仅支持Isaac Gym批量并行仿真（512+ 环境）
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import hashlib
import time

# Isaac Gym检测（尝试从本仓库的 vendor 目录加载）
# ⚠️ CRITICAL: Isaac Gym必须在torch导入前初始化
import sys, pathlib, os
ISAAC_GYM_AVAILABLE = False
try:
    # 优先直接导入
    from isaacgym import gymapi  # type: ignore
    ISAAC_GYM_AVAILABLE = True
except Exception:
    # 尝试将 repo 内置路径加入 sys.path
    try:
        _HERE = pathlib.Path(__file__).resolve()
        _PKG_ROOT = _HERE.parent  # 01_soar
        _REPO_ROOT = _PKG_ROOT.parent  # repo root
        _GYM_PY = _REPO_ROOT / 'isaacgym' / 'python'
        if _GYM_PY.exists() and str(_GYM_PY) not in sys.path:
            sys.path.insert(0, str(_GYM_PY))
        from isaacgym import gymapi  # type: ignore
        ISAAC_GYM_AVAILABLE = True
        # 配置必要的环境变量以定位插件信息
        try:
            os.environ.setdefault('GYM_USD_PLUG_INFO_PATH', str(_GYM_PY / 'isaacgym' / '_bindings' / 'linux-x86_64' / 'usd' / 'plugInfo.json'))
        except Exception:
            pass
    except Exception:
        ISAAC_GYM_AVAILABLE = False

# ⚠️ CRITICAL: torch必须在Isaac Gym之后导入
import torch

# Stepwise 奖励计算器与权重
try:
    from utils.reward_stepwise import StepwiseRewardCalculator  # type: ignore
except Exception:
    try:
        # 添加路径以支持直接运行
        import sys, pathlib
        _parent = pathlib.Path(__file__).resolve().parent.parent
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        from utils.reward_stepwise import StepwiseRewardCalculator  # type: ignore
    except Exception:
        StepwiseRewardCalculator = None  # type: ignore

# SCG 精确 reward 计算器
try:
    from utils.reward_scg_exact import SCGExactRewardCalculator  # type: ignore
except Exception:
    try:
        from reward_scg_exact import SCGExactRewardCalculator  # type: ignore
    except Exception:
        SCGExactRewardCalculator = None  # type: ignore
try:
    from utils.gpu_program_executor import GPUProgramExecutor  # type: ignore
except Exception:
    try:
        from gpu_program_executor import GPUProgramExecutor  # type: ignore
    except Exception:
        GPUProgramExecutor = None  # type: ignore
try:
    from utilities.reward_profiles import get_reward_profile  # type: ignore
except Exception:
    get_reward_profile = None  # type: ignore
try:
    from utilities.trajectory_presets import scg_position
except Exception:
    scg_position = None  # type: ignore
try:
    from utils.prior_scoring import compute_prior_scores  # type: ignore
except Exception:
    try:
        import sys, pathlib
        _parent = pathlib.Path(__file__).resolve().parent.parent
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        from utils.prior_scoring import compute_prior_scores  # type: ignore
    except Exception:
        compute_prior_scores = None  # type: ignore

try:
    # 用于结构化序列化程序，生成稳定哈希
    from core.serialization import to_serializable_dict as _to_serializable_dict  # type: ignore
except Exception:
    _to_serializable_dict = None  # type: ignore

try:
    from core.serialization import serialize_program as _serialize_program  # type: ignore
except Exception:
    _serialize_program = None  # type: ignore

# 重置 AST 节点状态（确保每次评估的确定性）
try:
    from core.dsl import reset_program_state  # type: ignore
except Exception:
    try:
        from dsl import reset_program_state  # type: ignore
    except Exception:
        reset_program_state = None  # type: ignore


@dataclass
class ProgramParamCandidate:
    """轻量级 BO 候选，延迟注入参数 & 延迟构造 DSL AST。"""

    base_program: List[Dict[str, Any]]
    param_paths: Tuple[str, ...]
    param_values: Tuple[float, ...]
    cache_key: Optional[str] = None
    allow_cache: bool = False
    _materialized: Optional[List[Dict[str, Any]]] = None

    def materialize(self) -> List[Dict[str, Any]]:
        if self._materialized is None:
            import copy
            try:
                from utils.bayesian_tuner import inject_tuned_params  # type: ignore
            except ImportError:
                from .bayesian_tuner import inject_tuned_params  # type: ignore
            prog_copy = copy.deepcopy(self.base_program)
            tuned_values = {path: self.param_values[idx] for idx, path in enumerate(self.param_paths)}
            inject_tuned_params(prog_copy, tuned_values)
            self._materialized = prog_copy
        return self._materialized


def _normalize_program_structure_for_cache(obj: Any):
    """递归去除程序内的常数值，仅保留结构信息用于缓存键。

    - 所有 dict 中 key 为 'value' 的数值会被占位符替换；
    - 其他任意 int/float 也统一替换，确保结构相同即命中缓存；
    - 其余类型保持不变。
    """
    if isinstance(obj, dict):
        normalized = {}
        for k, v in obj.items():
            if k == 'value' and isinstance(v, (int, float)):
                normalized[k] = '<CONST>'
            else:
                normalized[k] = _normalize_program_structure_for_cache(v)
        return normalized
    if isinstance(obj, list):
        return [_normalize_program_structure_for_cache(item) for item in obj]
    if isinstance(obj, (int, float)):
        return '<CONST>'
    return obj


try:
    from utils.program_constraints import validate_program, HARD_CONSTRAINT_PENALTY
except Exception:
    try:
        from program_constraints import validate_program, HARD_CONSTRAINT_PENALTY  # type: ignore
    except Exception:
        def validate_program(_program):  # type: ignore
            return True, ""
        HARD_CONSTRAINT_PENALTY = -1e6  # type: ignore

class BatchEvaluator:
    """批量程序评估器（仅支持Isaac Gym）"""

    def __init__(self, 
                 trajectory_config: Dict[str, Any],
                 duration: int = 20,
                 isaac_num_envs: int = 96,
                 device: str = 'cuda:0',
                 replicas_per_program: int = 5,
                 min_steps_frac: float = 0.0,
                 reward_reduction: str = 'mean',
                 reward_profile: str = 'control_law_discovery',
                 strict_no_prior: bool = True,
                 zero_action_penalty: float = 5.0,
                 use_fast_path: bool = True,
                 use_gpu_expression_executor: bool = True,
                 complexity_bonus: float = 0.1,
                 action_scale_multiplier: float = 1.0,
                 structure_prior_weight: float = 0.0,
                 stability_prior_weight: float = 0.0,
                 enable_output_mad: bool = True,
                 mad_min_fz: float = 0.0,
                 mad_max_fz: float = 7.5,
                 mad_max_xy: float = 1.0,           # 扩大范围以允许有效控制增益
                 mad_max_yaw: float = 0.5,          # 扩大范围
                 mad_max_delta_fz: float = 1.5,
                 mad_max_delta_xy: float = 0.5,     # 扩大变化率限制
                 mad_max_delta_yaw: float = 0.2,    # 扩大变化率限制
                 enable_bayesian_tuning: bool = False,
                 bo_batch_size: int = 50,
                 bo_iterations: int = 3,
                 bo_param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 gpu_control_loop: Optional[bool] = None,
                 use_scg_exact_reward: bool = False):
        """
        Args:
            trajectory_config: 轨迹配置 {'type': 'figure8', 'params': {...}}
            duration: 仿真时长（秒）
            isaac_num_envs: Isaac Gym并行环境数 (优化后默认96)
            device: GPU设备
            replicas_per_program: 每个程序评估N次取平均，减少方差 (优化后默认5)
            min_steps_frac: 每次评估至少执行的步数比例（0-1），避免过早 done 提前退出
            reward_reduction: 奖励归约方式：'sum'（步次求和）或 'mean'（步次平均，抵消存活时长偏差）
            reward_profile: 奖励配置文件名称
            zero_action_penalty: 零动作惩罚 (优化后默认5.0)
            complexity_bonus: 复杂度奖励系数 (每个唯一变量+0.1, 每条规则+0.05*bonus)
            structure_prior_weight: 结构先验加成权重（0=关闭）
            stability_prior_weight: 稳定性先验加成权重（0=关闭）
            enable_bayesian_tuning: 是否启用贝叶斯优化对程序常数进行调参
            bo_batch_size: BO每次并行评估的参数组数
            bo_iterations: BO迭代次数
            bo_param_ranges: 参数范围字典 {'const': (min, max), ...}
        """
        # 保险起见：运行期再尝试一次导入
        global ISAAC_GYM_AVAILABLE
        if not ISAAC_GYM_AVAILABLE:
            try:
                from isaacgym import gymapi  # type: ignore
                ISAAC_GYM_AVAILABLE = True
            except Exception:
                # 再尝试 vendor 路径
                try:
                    _HERE = pathlib.Path(__file__).resolve()
                    _PKG_ROOT = _HERE.parent
                    _REPO_ROOT = _PKG_ROOT.parent
                    _GYM_PY = _REPO_ROOT / 'isaacgym' / 'python'
                    if _GYM_PY.exists() and str(_GYM_PY) not in sys.path:
                        sys.path.insert(0, str(_GYM_PY))
                    from isaacgym import gymapi  # type: ignore
                    os.environ.setdefault('GYM_USD_PLUG_INFO_PATH', str(_GYM_PY / 'isaacgym' / '_bindings' / 'linux-x86_64' / 'usd' / 'plugInfo.json'))
                    ISAAC_GYM_AVAILABLE = True
                except Exception:
                    ISAAC_GYM_AVAILABLE = False
        # 不在此处硬性失败；在真正创建环境时再进行检测并报错
        
        self.trajectory_config = trajectory_config
        self.duration = duration
        self.isaac_num_envs = isaac_num_envs
        self.device = device
        self.replicas_per_program = max(1, int(replicas_per_program))
        self.min_steps_frac = float(min_steps_frac) if 0.0 <= float(min_steps_frac) <= 1.0 else 0.0
        self.reward_reduction = reward_reduction if reward_reduction in ('sum', 'mean') else 'sum'
        self.reward_profile = reward_profile
        # 严格无先验（默认开启）：强制使用直接 u_* 动作路径，完全不依赖内置 PID 框架
        self.strict_no_prior = bool(strict_no_prior)
        # 对整集始终为“零动作”的程序加罚，避免搜索停留在空程序
        try:
            self.zero_action_penalty = float(zero_action_penalty)
        except Exception:
            self.zero_action_penalty = 0.0  # AlphaZero: 让NN自己学习
        
        # 复杂度奖励系数（鼓励使用多变量和多规则）
        try:
            self.complexity_bonus = float(complexity_bonus)
        except Exception:
            self.complexity_bonus = 0.0  # AlphaZero: 让NN自己学习复杂度权衡
        
        # 动作全局缩放系数（诊断用）
        try:
            self.action_scale_multiplier = float(action_scale_multiplier)
        except Exception:
            self.action_scale_multiplier = 1.0

        self.structure_prior_weight = float(structure_prior_weight)
        self.stability_prior_weight = float(stability_prior_weight)
        self.metric_export_keys: Tuple[str, ...] = (
            'position_rmse',
            'control_effort',
        )

        # MAD（Magnitude-Angle-Delta）安全壳参数
        self.enable_output_mad = bool(enable_output_mad)
        self.mad_min_fz = float(mad_min_fz)
        self.mad_max_fz = float(mad_max_fz)
        self.mad_max_xy = float(abs(mad_max_xy))
        self.mad_max_yaw = float(abs(mad_max_yaw))
        self.mad_max_delta_fz = float(abs(mad_max_delta_fz))
        self.mad_max_delta_xy = float(abs(mad_max_delta_xy))
        self.mad_max_delta_yaw = float(abs(mad_max_delta_yaw))
        self._mad_eps = 1e-6
        
        # 🎯 选择 reward 计算器
        self.use_scg_exact_reward = bool(use_scg_exact_reward)
        if self.reward_profile == 'safe_control_tracking':
            # Force SCG exact reward path so we faithfully mirror the benchmark.
            self.use_scg_exact_reward = True
            self.metric_export_keys = ('state_cost', 'action_cost')
        self._step_reward_calc = None
        self._scg_reward_calc = None
        
        if self.use_scg_exact_reward and SCGExactRewardCalculator is not None:
            # 使用精确 SCG reward 计算器
            try:
                self._scg_reward_calc = SCGExactRewardCalculator(
                    num_envs=self.isaac_num_envs,
                    device=self.device
                )
                print(f"[BatchEvaluator] ✅ 使用精确 SCG reward 计算器")
            except Exception as e:
                print(f"[BatchEvaluator] ⚠️ SCG reward 初始化失败: {e}，回退 Stepwise")
                self.use_scg_exact_reward = False
        
        if not self.use_scg_exact_reward:
            # 初始化 Stepwise 奖励计算器
            try:
                weights, ks = get_reward_profile(self.reward_profile)
                # 估计 dt: Isaac 默认物理频率 240 Hz，控制频率 48 Hz -> dt ≈ 1/48
                self._step_dt = 1.0 / 48.0
                self._step_reward_calc = StepwiseRewardCalculator(weights, ks, dt=self._step_dt, num_envs=self.isaac_num_envs, device=self.device)
            except Exception:
                self._step_reward_calc = None

        # 记录最近一次安全裁剪后的 [fz, tx, ty, tz]
        self._last_safe_actions = torch.zeros((self.isaac_num_envs, 4), device=self.device)

        # Isaac Gym环境池（延迟初始化）
        self._isaac_env_pool = None
        self._envs_ready = False  # 环境池持久化标记
        self._last_reset_size = 0  # 上次reset的环境数
        
        # 🚀 快速路径优化
        self.use_fast_path = use_fast_path
        self._program_cache = {}  # 预编译缓存: {prog_hash: (fz,tx,ty,tz)}
        disable_gpu_env = os.getenv('DISABLE_GPU_EXPRESSION', '').lower()
        if disable_gpu_env in ('1', 'true', 'yes'):
            use_gpu_expression_executor = False
        self.use_gpu_expression_executor = bool(use_gpu_expression_executor)
        self._gpu_executor = None
        if self.use_gpu_expression_executor and GPUProgramExecutor is not None:
            try:
                self._gpu_executor = GPUProgramExecutor(device=self.device)
                print("[BatchEvaluator] ✅ GPU表达式执行器已启用")
            except Exception as gpu_exc:
                self._gpu_executor = None
                self.use_gpu_expression_executor = False
                print(f"[BatchEvaluator] ⚠️ GPU表达式执行器初始化失败，回退CPU: {gpu_exc}")
        elif self.use_gpu_expression_executor:
            print("[BatchEvaluator] ⚠️ GPUProgramExecutor 不可用，回退CPU")
            self.use_gpu_expression_executor = False

        env_gpu_loop = os.getenv('ENABLE_GPU_CONTROL_LOOP', '0').lower() in ('1', 'true', 'yes')
        if gpu_control_loop is None:
            self._use_gpu_control_loop = bool(env_gpu_loop)
        else:
            self._use_gpu_control_loop = bool(gpu_control_loop)
        if self._use_gpu_control_loop and (self._gpu_executor is None or not self.use_gpu_expression_executor):
            self._use_gpu_control_loop = False
        if self._use_gpu_control_loop:
            print("[BatchEvaluator] 🚀 控制循环全GPU路径已启用")
        
        # 🚀🚀 超高性能执行器 (完全向量化 + JIT)
        if use_fast_path:
            try:
                from .ultra_fast_executor import UltraFastExecutor
                self._ultra_executor = UltraFastExecutor()
            except Exception as e:
                try:
                    from ultra_fast_executor import UltraFastExecutor
                    self._ultra_executor = UltraFastExecutor()
                except Exception:
                    print(f"[BatchEvaluator] ⚠️ 超高性能执行器加载失败: {e}")
                    self._ultra_executor = None
        else:
            self._ultra_executor = None
            # 清理可能残留的编译缓存
            if hasattr(self, '_compiled_forces'):
                delattr(self, '_compiled_forces')
        
        # 🔥 贝叶斯优化调参模块
        self.enable_bayesian_tuning = bool(enable_bayesian_tuning)
        self.bo_batch_size = int(bo_batch_size)
        self.bo_iterations = int(bo_iterations)
        self.bo_param_ranges = bo_param_ranges or {'default': (-3.0, 3.0)}
        self._bo_tuner = None  # 延迟创建（因为依赖程序实际参数）
        # 程序评估结果缓存：避免对完全相同的程序重复仿真
        self._eval_cache: Dict[str, float] = {}
        self._eval_cache_limit: int = 5000
        
        print(f"[BatchEvaluator] 初始化完成")
        print(f"  - Isaac Gym: {'✅ 启用' if ISAAC_GYM_AVAILABLE else '❌ 未启用'}")
        print(f"  - 并行环境数: {self.isaac_num_envs}")
        print(f"  - GPU设备: {self.device}")
        print(f"  - 单程序副本数: {self.replicas_per_program}")
        if self.enable_bayesian_tuning:
            print(f"  - 贝叶斯调参: ✅ 启用 (batch={self.bo_batch_size}, iters={self.bo_iterations})")
        print(f"  - 最小步数比例: {self.min_steps_frac}")
        print(f"  - 奖励归约: {self.reward_reduction}")
        print(f"  - 严格无先验(u_*直接控制): {'✅ 是' if self.strict_no_prior else '❌ 否'}")
        if self.strict_no_prior:
            print(f"  - 零动作惩罚: {self.zero_action_penalty}")

    # ---------------------- 程序评估缓存辅助 ----------------------
    def _program_eval_key(self, program: List[Dict[str, Any]]) -> str:
        """生成稳定的程序键，用于评估缓存。

        使用 core.serialization.to_serializable_dict 的 JSON 表示，再做 blake2s 哈希；
        若不可用则退化为 str(program)。
        """
        if isinstance(program, ProgramParamCandidate):
            if not program.allow_cache:
                return None
            if program.cache_key:
                return program.cache_key

        try:
            import json
            if isinstance(program, ProgramParamCandidate):
                base_prog = program.base_program
                if _serialize_program is not None:
                    serial_source = _serialize_program(base_prog)  # type: ignore
                elif _to_serializable_dict is not None:
                    serial_source = _to_serializable_dict(base_prog)
                else:
                    serial_source = base_prog
            elif _serialize_program is not None:
                serial_source = _serialize_program(program)  # type: ignore
            elif _to_serializable_dict is not None:
                serial_source = _to_serializable_dict(program)
            else:
                serial_source = program
            serial = _normalize_program_structure_for_cache(serial_source)
            s = json.dumps(serial, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            digest = hashlib.blake2s(s.encode("utf-8")).hexdigest()
            if isinstance(program, ProgramParamCandidate):
                program.cache_key = digest
            return digest
        except Exception:
            try:
                return str(program)
            except Exception:
                return str(id(program))

    
    def _init_isaac_gym_pool(self):
        """延迟初始化Isaac Gym环境池"""
        if self._isaac_env_pool is not None:
            return
        
        print(f"[BatchEvaluator] 初始化Isaac Gym环境池...")
        
        # 导入Isaac Gym环境
        try:
            from envs.isaac_gym_drone_env import IsaacGymDroneEnv
        except ImportError:
            try:
                # 添加路径以支持直接运行
                import sys, pathlib
                _parent = pathlib.Path(__file__).resolve().parent.parent
                if str(_parent) not in sys.path:
                    sys.path.insert(0, str(_parent))
                from envs.isaac_gym_drone_env import IsaacGymDroneEnv
            except ImportError:
                raise ImportError("无法导入IsaacGymDroneEnv，请检查envs目录")
        
        # 创建环境池
        self._isaac_env_pool = IsaacGymDroneEnv(
            num_envs=self.isaac_num_envs,
            device=self.device,
            headless=True,
            duration_sec=self.duration
        )
        # 保存控制周期
        try:
            self._control_freq = int(self._isaac_env_pool.control_freq)
        except Exception:
            self._control_freq = 48
        self._control_dt = 1.0 / float(self._control_freq)
        
        print(f"[BatchEvaluator] ✅ Isaac Gym环境池就绪（{self.isaac_num_envs} 环境）")

    # ---------------------- 贝叶斯优化调参模块 ----------------------
    def _batch_tune_programs_with_bo(self, programs: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """🚀 真正的批量贝叶斯优化：对多个程序同时进行 GP-UCB 迭代优化
        
        核心改进（相比之前的Sobol采样）：
        1. 使用 Gaussian Process 建模参数空间
        2. 每轮迭代根据 UCB 采集函数智能选择下一批候选
        3. 所有程序的候选仍然批量并行评估（利用Isaac Gym）
        
        工作流程：
        - Iteration 1: 初始化采样（Sobol）→ 批量评估 → 更新 GP
        - Iteration 2+: UCB 选择候选 → 批量评估 → 更新 GP
        - 最终：每个程序返回最佳参数
        
        Args:
            programs: 待调优的程序列表
            
        Returns:
            tuned_programs: 调优后的程序列表
        """
        try:
            from utils.bayesian_tuner import (
                BayesianTuner, ParameterSpec, 
                extract_tunable_params, inject_tuned_params
            )
        except ImportError:
            print("[BatchEvaluator] Warning: BayesianTuner not available, skipping BO")
            return programs
        
        # 禁用递归BO
        old_bo_flag = self.enable_bayesian_tuning
        self.enable_bayesian_tuning = False
        
        try:
            # 🔧 第一步：为每个程序初始化独立的 BayesianTuner
            program_tuners = []  # [(prog_idx, tuner, params), ...]
            param_paths_map: Dict[int, Tuple[str, ...]] = {}
            cache_key_map: Dict[int, Optional[str]] = {}
            
            for prog_idx, program in enumerate(programs):
                params = extract_tunable_params(program)
                if not params:
                    # 无参数，跳过BO
                    program_tuners.append((prog_idx, None, None))
                    param_paths_map[prog_idx] = tuple()
                    cache_key_map[prog_idx] = self._program_eval_key(program)
                    continue
                
                # 定义参数空间
                param_specs = []
                for path, init_value in params:
                    if 'default' in self.bo_param_ranges:
                        low, high = self.bo_param_ranges['default']
                    else:
                        low = init_value - 2.0
                        high = init_value + 2.0
                    param_specs.append(ParameterSpec(name=path, low=low, high=high, log_scale=False))
                
                param_paths = tuple(path for path, _ in params)
                param_paths_map[prog_idx] = param_paths
                cache_key_map[prog_idx] = self._program_eval_key(program)
                
                # 创建 BayesianTuner 实例
                tuner = BayesianTuner(
                    param_specs=param_specs,
                    batch_size=self.bo_batch_size,
                    n_iterations=self.bo_iterations,
                    ucb_kappa=2.0,
                    random_seed=hash(str(program)) % (2**31)
                )
                program_tuners.append((prog_idx, tuner, params))
            
            # 🔧 第二步：迭代式批量BO（真正的 Bayesian Optimization）
            import time as time_module
            bo_start_time = time_module.time()
            print(f"[BatchEvaluator] 🧠 真实BO: {len([t for t in program_tuners if t[1] is not None])} 个程序, "
                  f"{self.bo_iterations} 轮迭代, {self.bo_batch_size} 个候选/轮")
            
            for iter_idx in range(self.bo_iterations):
                iter_start_time = time_module.time()
                # 2.1 收集本轮所有程序的候选参数
                all_candidates = []  # [(prog_idx, candidate_program), ...]
                candidate_metadata = []  # [(prog_idx, X_raw_row), ...] 用于更新GP
                gen_start_time = time_module.time()
                
                for prog_idx, tuner, params in program_tuners:
                    if tuner is None:
                        # 无参数程序，只在第一轮添加一次
                        if iter_idx == 0:
                            all_candidates.append((prog_idx, programs[prog_idx]))
                            candidate_metadata.append((prog_idx, None))
                        continue
                    
                    # 生成候选：第一轮用Sobol，后续用UCB
                    if iter_idx == 0:
                        X_norm = tuner._sobol_sample(tuner.batch_size)
                    else:
                        X_norm = tuner._select_next_batch()
                    
                    X_raw = tuner._denormalize(X_norm)
                    
                    # 为每组参数创建程序副本
                    param_paths = param_paths_map.get(prog_idx, tuple())
                    for i in range(len(X_raw)):
                        param_values = tuple(float(X_raw[i, j]) for j in range(len(param_paths)))
                        candidate = ProgramParamCandidate(
                            base_program=programs[prog_idx],
                            param_paths=param_paths,
                            param_values=param_values,
                        )
                        all_candidates.append((prog_idx, candidate))
                        candidate_metadata.append((prog_idx, X_raw[i]))
                
                gen_time = time_module.time() - gen_start_time
                print(f"[BO] 第{iter_idx+1}轮候选生成完成: {len(all_candidates)}个程序 | 耗时{gen_time:.1f}秒 (含deepcopy)")

                # 📊 统计当前轮候选中独特的结构模板数量（忽略常数）
                if all_candidates:
                    structure_keys = set()
                    for _, cand_prog in all_candidates:
                        base_prog = cand_prog.base_program if isinstance(cand_prog, ProgramParamCandidate) else cand_prog
                        try:
                            key = self._program_eval_key(base_prog)
                        except Exception:
                            key = None
                        if key is not None:
                            structure_keys.add(key)
                    unique_structures = len(structure_keys) if structure_keys else len(all_candidates)
                    print(f"[BO] 第{iter_idx+1}轮结构覆盖: {unique_structures}/{len(all_candidates)} unique templates")
                
                # 2.2 批量评估所有候选
                if not all_candidates:
                    break
                    
                all_candidate_programs = [prog for _, prog in all_candidates]
                eval_start_time = time_module.time()
                
                # 🔥 BO 内层评估时，重置 SCG reward calculator 以匹配新的批量大小
                bo_batch_size = len(all_candidate_programs)
                if self.use_scg_exact_reward and self._scg_reward_calc is not None:
                    from .reward_scg_exact import SCGExactRewardCalculator
                    self._scg_reward_calc = SCGExactRewardCalculator(
                        num_envs=bo_batch_size,
                        device=self.device,
                        state_weights=self._scg_reward_calc.Q,
                        action_weight=self._scg_reward_calc.R,
                    )
                
                all_rewards = self.evaluate_batch(all_candidate_programs)
                eval_time = time_module.time() - eval_start_time
                print(f"[BO] 第{iter_idx+1}轮评估完成: {len(all_candidate_programs)}个候选 | 耗时{eval_time:.1f}秒")
                
                # 2.3 更新每个程序的 GP 模型
                for idx, ((prog_idx, _), reward) in enumerate(zip(all_candidates, all_rewards)):
                    _, tuner, _ = program_tuners[prog_idx]
                    if tuner is None:
                        continue
                    
                    # 获取对应的参数值
                    X_raw_row = candidate_metadata[idx][1]
                    if X_raw_row is not None:
                        X_norm_row = tuner._normalize(X_raw_row.reshape(1, -1))
                        tuner.X_history.append(X_norm_row)
                        tuner.y_history.append(np.array([reward]))
                
                # 2.4 拟合 GP（为下一轮做准备）
                if iter_idx < self.bo_iterations - 1:  # 最后一轮不需要拟合
                    gp_start_time = time_module.time()
                    for prog_idx, tuner, _ in program_tuners:
                        if tuner is not None and tuner.X_history:
                            X_all = np.vstack(tuner.X_history)
                            y_all = np.concatenate(tuner.y_history)
                            tuner.gp.fit(X_all, y_all)
                    gp_time = time_module.time() - gp_start_time
                    print(f"[BO] GP模型拟合完成: {len([t for t in program_tuners if t[1] is not None])}个模型 | 耗时{gp_time:.2f}秒")
                
                iter_time = time_module.time() - iter_start_time
                print(f"[BO] 第{iter_idx+1}轮完成 | 总耗时{iter_time:.1f}秒")
            
            # 🔧 第三步：为每个程序选择最佳参数
            tuned_programs = []
            for prog_idx, tuner, params in program_tuners:
                if tuner is None or not tuner.y_history:
                    # 无参数或BO失败，保留原程序
                    tuned_programs.append(programs[prog_idx])
                    continue
                
                # 找到最佳参数
                y_all = np.concatenate(tuner.y_history)
                best_idx = np.argmax(y_all)
                X_all = np.vstack(tuner.X_history)
                best_X_norm = X_all[best_idx]
                best_X_raw = tuner._denormalize(best_X_norm.reshape(1, -1))[0]
                
                # 注入最佳参数
                import copy
                tuned_prog = copy.deepcopy(programs[prog_idx])
                param_dict = {params[j][0]: best_X_raw[j] for j in range(len(params))}
                inject_tuned_params(tuned_prog, param_dict)
                tuned_programs.append(tuned_prog)
            
            bo_total_time = time_module.time() - bo_start_time
            print(f"[BatchEvaluator] ✅ 真实BO完成: {len(tuned_programs)} 个程序已通过GP-UCB优化 | 总耗时{bo_total_time:.1f}秒")
            
            # 🔥 BO 完成后，恢复原始批量大小的 SCG calculator
            if self.use_scg_exact_reward and self._scg_reward_calc is not None:
                from .reward_scg_exact import SCGExactRewardCalculator
                self._scg_reward_calc = SCGExactRewardCalculator(
                    num_envs=self.isaac_num_envs,
                    device=self.device,
                    state_weights=self._scg_reward_calc.Q,
                    action_weight=self._scg_reward_calc.R,
                )
            
            return tuned_programs
            
        finally:
            self.enable_bayesian_tuning = old_bo_flag
    
    def _tune_program_with_bo(self, program: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        """对单个程序使用贝叶斯优化调整常数参数
        
        Args:
            program: 原始程序（包含初始参数）
            
        Returns:
            tuned_program: 调优后的程序
            best_reward: 对应的最佳奖励
        """
        try:
            from utils.bayesian_tuner import (
                BayesianTuner, ParameterSpec, 
                extract_tunable_params, inject_tuned_params
            )
        except ImportError:
            print("[BatchEvaluator] Warning: BayesianTuner not available, skipping BO")
            return program, float('-inf')
        
        # 1. 提取可调参数
        params = extract_tunable_params(program)
        if not params:
            # 没有常数参数，无需调优
            return program, float('-inf')
        
        # 2. 定义参数空间
        param_specs = []
        for path, init_value in params:
            # 根据初始值或全局配置确定范围
            if 'default' in self.bo_param_ranges:
                low, high = self.bo_param_ranges['default']
            else:
                # 自适应：以初始值为中心，±2倍范围
                low = init_value - 2.0
                high = init_value + 2.0
            
            param_specs.append(ParameterSpec(
                name=path,
                low=low,
                high=high,
                log_scale=False
            ))
        
        # 3. 定义评估函数（🚀 批量并行优化：一次评估所有候选参数）
        def eval_fn(X_batch):
            """X_batch: [bo_batch_size, n_params]"""
            import copy
            batch_size = len(X_batch)
            
            # 🚀 关键优化：批量构造所有候选程序（避免串行循环）
            all_programs = []
            for i in range(batch_size):
                prog_copy = copy.deepcopy(program)
                param_dict = {params[j][0]: X_batch[i, j] for j in range(len(params))}
                inject_tuned_params(prog_copy, param_dict)
                all_programs.append(prog_copy)
            
            # 🚀 一次性评估所有程序（利用 Isaac Gym 4096 并行环境）
            # 禁用递归 BO 避免无限循环
            old_bo_flag = self.enable_bayesian_tuning
            self.enable_bayesian_tuning = False
            try:
                rewards = self.evaluate_batch(all_programs)  # ✅ 批量并行评估
            finally:
                self.enable_bayesian_tuning = old_bo_flag
            
            return np.array(rewards)
        
        # 4. 运行 BO
        tuner = BayesianTuner(
            param_specs=param_specs,
            batch_size=min(self.bo_batch_size, self.isaac_num_envs),
            n_iterations=self.bo_iterations,
            ucb_kappa=2.0,
            random_seed=hash(str(program)) % 2**31
        )
        
        best_params, best_reward = tuner.optimize(eval_fn, verbose=False)
        
        # 5. 注入最佳参数
        import copy
        tuned_program = copy.deepcopy(program)
        param_dict = {params[j][0]: best_params[j] for j in range(len(params))}
        inject_tuned_params(tuned_program, param_dict)
        
        return tuned_program, best_reward

    # ---------------------- DSL 辅助：AST 求值与动作解析 ----------------------
    def _ast_eval(self, node, state: Dict[str, float]) -> float:
        """最小求值器：支持 MCTS 生成的算子集（数值表达式）。"""
        try:
            # 延迟导入 DSL 结点类型
            try:
                from core.dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore
            except Exception:
                # 添加路径以支持直接运行
                import sys, pathlib
                _parent = pathlib.Path(__file__).resolve().parent.parent
                if str(_parent) not in sys.path:
                    sys.path.insert(0, str(_parent))
                from core.dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore

            # 递归求值
            if isinstance(node, (int, float)):
                return float(node)
            # 终端：变量名或常数
            if hasattr(node, 'value') and not hasattr(node, 'op'):
                v = getattr(node, 'value', 0.0)
                if isinstance(v, str):
                    return float(state.get(v, 0.0))
                return float(v)
            # 一元
            if hasattr(node, 'op') and hasattr(node, 'child'):
                x = float(self._ast_eval(node.child, state))
                op = str(getattr(node, 'op', ''))
                if op == 'abs':
                    return abs(x)
                if op == 'sin':
                    import math
                    return float(math.sin(x))
                if op == 'cos':
                    import math
                    return float(math.cos(x))
                if op == 'tan':
                    import math
                    return float(max(-10.0, min(10.0, math.tan(x))))
                if op == 'log1p':
                    import math
                    return float(math.log1p(abs(x)))
                if op == 'sqrt':
                    import math
                    return float(math.sqrt(abs(x)))
                if op == 'sign':
                    return float(1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
                return float(x)
            # 二元
            if hasattr(node, 'op') and hasattr(node, 'left') and hasattr(node, 'right'):
                op = str(getattr(node, 'op', ''))
                if op in ('+', '-', '*', '/', 'max', 'min'):
                    a = float(self._ast_eval(node.left, state))
                    b = float(self._ast_eval(node.right, state))
                    if op == '+':
                        return a + b
                    if op == '-':
                        return a - b
                    if op == '*':
                        return a * b
                    if op == '/':
                        return a / b if abs(b) > 1e-9 else (a * 1.0)
                    if op == 'max':
                        return a if a >= b else b
                    if op == 'min':
                        return a if a <= b else b
                elif op in ('<', '>', '==', '!='):
                    a = float(self._ast_eval(node.left, state))
                    b = float(self._ast_eval(node.right, state))
                    if op == '<':
                        return 1.0 if a < b else 0.0
                    if op == '>':
                        return 1.0 if a > b else 0.0
                    if op == '==':
                        return 1.0 if abs(a - b) < 1e-9 else 0.0
                    if op == '!=':
                        return 1.0 if abs(a - b) >= 1e-9 else 0.0
            # IfNode
            if hasattr(node, 'condition') and hasattr(node, 'then_branch') and hasattr(node, 'else_branch'):
                c = float(self._ast_eval(node.condition, state))
                return float(self._ast_eval(node.then_branch if c > 0 else node.else_branch, state))
        except Exception:
            pass
        return 0.0

    def _program_uses_u(self, program: List[Dict[str, Any]]) -> bool:
        """检测动作是否使用了 u_fz/u_tx/u_ty/u_tz 键。"""
        try:
            for rule in program or []:
                acts = rule.get('action', []) or []
                for a in acts:
                    try:
                        # a 为 BinaryOpNode('set', TerminalNode(key), expr)
                        if hasattr(a, 'op') and a.op == 'set' and hasattr(a, 'left') and hasattr(a.left, 'value'):
                            key = str(getattr(a.left, 'value', ''))
                            if key in ('u_fz', 'u_tx', 'u_ty', 'u_tz'):
                                return True
                    except Exception:
                        continue
        except Exception:
            return False
        return False

    def _mirror_expand_single_axis_program(self, program: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """若程序仅输出 u_tx，则镜像生成 u_ty，并附带固定 yaw/thrust 稳定器。

        目的：当搜索在单轴空间内进行时，仍能得到四通道可执行的完整控制律。
        """
        import copy
        try:
            from core.dsl import TerminalNode, ConstantNode, UnaryOpNode, BinaryOpNode  # type: ignore
        except Exception:
            # 无法导入 DSL 节点时直接返回原程序，避免中断训练
            return copy.deepcopy(program)

        var_mapping = {
            'err_p_roll': 'err_p_pitch',
            'err_d_roll': 'err_d_pitch',
            'err_i_roll': 'err_i_pitch',
            'ang_vel_x': 'ang_vel_y',
            'pos_err_y': 'pos_err_x',  # u_tx→u_ty: y轴位置误差映射为x轴
            'vel_y': 'vel_x',          # u_tx→u_ty: y轴速度映射为x轴
        }

        def map_expr(expr):
            """递归映射表达式中的轴向变量。"""
            if isinstance(expr, TerminalNode):
                val = getattr(expr, 'value', None)
                if isinstance(val, str) and val in var_mapping:
                    return TerminalNode(var_mapping[val])
                return copy.deepcopy(expr)
            if isinstance(expr, ConstantNode):
                return copy.deepcopy(expr)
            if isinstance(expr, UnaryOpNode):
                return UnaryOpNode(expr.op, map_expr(expr.child), params=copy.deepcopy(getattr(expr, 'params', {})))
            if isinstance(expr, BinaryOpNode):
                return BinaryOpNode(expr.op, map_expr(expr.left), map_expr(expr.right))
            if isinstance(expr, dict):
                etype = expr.get('type')
                if etype in ('TerminalNode', 'Terminal'):
                    val = expr.get('value')
                    if isinstance(val, str) and val in var_mapping:
                        new_expr = copy.deepcopy(expr)
                        new_expr['value'] = var_mapping[val]
                        return new_expr
                    return copy.deepcopy(expr)
                if etype in ('ConstantNode', 'Constant'):
                    return copy.deepcopy(expr)
                if etype in ('UnaryOpNode', 'Unary'):
                    return {
                        'type': 'UnaryOpNode',
                        'op': expr.get('op'),
                        'child': map_expr(expr.get('child')),
                        'params': copy.deepcopy(expr.get('params')) if expr.get('params') is not None else None,
                    }
                if etype in ('BinaryOpNode', 'Binary'):
                    return {
                        'type': 'BinaryOpNode',
                        'op': expr.get('op'),
                        'left': map_expr(expr.get('left')),
                        'right': map_expr(expr.get('right')),
                    }
            return copy.deepcopy(expr)

        def extract_target(action) -> Optional[str]:
            if isinstance(action, BinaryOpNode) and getattr(action, 'op', None) == 'set':
                left = getattr(action, 'left', None)
                if isinstance(left, TerminalNode) and isinstance(getattr(left, 'value', None), str):
                    return str(left.value)
            if isinstance(action, dict) and action.get('op') == 'set':
                left = action.get('left', {})
                if isinstance(left, dict):
                    val = left.get('value')
                    if isinstance(val, str):
                        return val
            return None

        # 汇总当前程序的输出通道
        targets = []
        for rule in program or []:
            for act in rule.get('action', []) or []:
                tgt = extract_target(act)
                if tgt:
                    targets.append(tgt)
        unique_targets = set(targets)
        if len(unique_targets) != 1 or 'u_tx' not in unique_targets:
            return copy.deepcopy(program)

        # 找到首个 u_tx 规则，克隆并镜像到 u_ty
        base_rule = None
        base_action = None
        for rule in program or []:
            for act in rule.get('action', []) or []:
                if extract_target(act) == 'u_tx':
                    base_rule = rule
                    base_action = act
                    break
            if base_action is not None:
                break
        if base_rule is None or base_action is None:
            return copy.deepcopy(program)

        # 构造修正后的 u_tx 规则（智能取反位置误差和速度项）
        # 🔧 物理映射分析:
        #   +tx → +roll → -Y 位移
        #   +ty → +pitch → +X 位移
        # 正确控制律:
        #   要追踪 +Y (pos_err_y > 0): 需要 -tx → u_tx = -Kp*err_y + Kd*vel_y - Kd_omega*ang_vel
        #   要追踪 +X (pos_err_x > 0): 需要 +ty → u_ty = +Kp*err_x - Kd*vel_x - Kd_omega*ang_vel
        # 
        # 假设原始单轴程序为:
        #   u_tx = Kp*pos_err_y - Kd*vel_y - Kd_omega*ang_vel_x  (原始符号)
        # 则需要取反 pos_err 和 vel 项系数，保持 ang_vel 项:
        #   u_tx = -Kp*pos_err_y + Kd*vel_y - Kd_omega*ang_vel_x (修正后)
        
        # 🔧 智能取反：取反 pos_err 和 vel 相关项的系数，保持 ang_vel 阻尼项不变
        def negate_pos_vel_coefficients(expr):
            """取反 pos_err 和 vel 相关项的系数，保持 ang_vel 阻尼项不变。"""
            if isinstance(expr, TerminalNode):
                return copy.deepcopy(expr)
            if isinstance(expr, ConstantNode):
                return copy.deepcopy(expr)
            if isinstance(expr, BinaryOpNode):
                op = expr.op
                # 处理乘法：检查是否涉及 pos_err 或 vel 变量
                if op == '*':
                    left = expr.left
                    right = expr.right
                    involves_target = False
                    
                    # 检查 left
                    if isinstance(left, TerminalNode):
                        val = str(getattr(left, 'value', ''))
                        if 'pos_err' in val or (val.startswith('vel_') and 'ang_vel' not in val):
                            involves_target = True
                    # 检查 right
                    if isinstance(right, TerminalNode):
                        val = str(getattr(right, 'value', ''))
                        if 'pos_err' in val or (val.startswith('vel_') and 'ang_vel' not in val):
                            involves_target = True
                    
                    if involves_target:
                        # 找到常数项并取反
                        if isinstance(left, ConstantNode):
                            new_left = ConstantNode(-left.value, name=getattr(left, 'name', None))
                            return BinaryOpNode('*', new_left, copy.deepcopy(right))
                        elif isinstance(right, ConstantNode):
                            new_right = ConstantNode(-right.value, name=getattr(right, 'name', None))
                            return BinaryOpNode('*', copy.deepcopy(left), new_right)
                        else:
                            # 没有常数项，用 -1 * 包裹
                            return BinaryOpNode('*', ConstantNode(-1.0), copy.deepcopy(expr))
                    else:
                        # 不涉及目标变量，保持原样但递归处理子表达式
                        return BinaryOpNode(op, negate_pos_vel_coefficients(left), negate_pos_vel_coefficients(right))
                else:
                    # +, -, 等运算：递归处理两边
                    return BinaryOpNode(op, negate_pos_vel_coefficients(expr.left), negate_pos_vel_coefficients(expr.right))
            if isinstance(expr, UnaryOpNode):
                return UnaryOpNode(expr.op, negate_pos_vel_coefficients(expr.child), params=copy.deepcopy(getattr(expr, 'params', {})))
            return copy.deepcopy(expr)
        
        # 🔧 简化：直接使用原始 u_tx 规则，不做取反
        # MCTS 搜索应该自己找到正确符号的控制律
        # 🔧 但需要添加姿态阻尼项来稳定系统！
        
        # 添加 roll 阻尼到 u_tx：u_tx_final = u_tx_search - Kd_att * ang_vel_x
        att_damp_x = ConstantNode(0.15, name='c_att_damp_x')  # 姿态阻尼
        if isinstance(base_action, BinaryOpNode) and getattr(base_action, 'op', None) == 'set':
            search_expr = copy.deepcopy(getattr(base_action, 'right', None))
            # u_tx = search_expr - 0.15 * ang_vel_x
            damped_tx_expr = BinaryOpNode('-', search_expr, BinaryOpNode('*', att_damp_x, TerminalNode('ang_vel_x')))
            corrected_tx_action = BinaryOpNode('set', TerminalNode('u_tx'), damped_tx_expr)
            corrected_tx_rule = {
                'condition': copy.deepcopy(base_rule.get('condition')),
                'action': [corrected_tx_action]
            }
        else:
            corrected_tx_rule = copy.deepcopy(base_rule)
        
        # 构造 u_ty 规则（变量映射 y→x，添加 pitch 阻尼）
        att_damp_y = ConstantNode(0.15, name='c_att_damp_y')  # 姿态阻尼
        mirrored_rule = {
            'condition': copy.deepcopy(base_rule.get('condition')),
            'action': []
        }
        if isinstance(base_action, BinaryOpNode):
            # 变量映射 + 添加 pitch 阻尼
            mapped_expr = map_expr(copy.deepcopy(getattr(base_action, 'right', None)))
            # u_ty = mapped_expr - 0.15 * ang_vel_y
            damped_ty_expr = BinaryOpNode('-', mapped_expr, BinaryOpNode('*', att_damp_y, TerminalNode('ang_vel_y')))
            mirrored_action = BinaryOpNode('set', TerminalNode('u_ty'), damped_ty_expr)
        else:
            mirrored_action = copy.deepcopy(base_action)
            if isinstance(mirrored_action, dict):
                if isinstance(mirrored_action.get('left'), dict):
                    mirrored_action['left']['value'] = 'u_ty'
                mirrored_action['right'] = map_expr(mirrored_action.get('right'))
        mirrored_rule['action'] = [mirrored_action]

        # 固定 yaw 通道 PID
        yaw_p = ConstantNode(4.0, name='c_yaw_p', min_val=4.0, max_val=4.0)
        yaw_d = ConstantNode(0.8, name='c_yaw_d', min_val=0.8, max_val=0.8)
        yaw_expr = BinaryOpNode('-', BinaryOpNode('*', yaw_p, TerminalNode('err_p_yaw')), BinaryOpNode('*', yaw_d, TerminalNode('ang_vel_z')))
        yaw_rule = {
            'condition': None,
            'action': [BinaryOpNode('set', TerminalNode('u_tz'), yaw_expr)]
        }

        # 固定 thrust 通道为简单高度PD控制（避免坠落或飞走）
        # 🔧 Isaac Gym 需要更低的增益，因为已经有 FZ_SCALE 缩放
        thrust_p = ConstantNode(0.5, name='c_thrust_p', min_val=0.5, max_val=0.5)
        thrust_d = ConstantNode(0.2, name='c_thrust_d', min_val=0.2, max_val=0.2)
        thrust_ff = ConstantNode(0.65, name='c_thrust_ff', min_val=0.65, max_val=0.65)
        thrust_expr = BinaryOpNode('+',
            BinaryOpNode('-', BinaryOpNode('*', thrust_p, TerminalNode('pos_err_z')), BinaryOpNode('*', thrust_d, TerminalNode('vel_z'))),
            thrust_ff
        )
        thrust_rule = {
            'condition': None,
            'action': [BinaryOpNode('set', TerminalNode('u_fz'), thrust_expr)]
        }

        # 构造新程序：用添加阻尼的 u_tx 规则替换原始规则，添加 u_ty/u_tz/u_fz
        new_program = []
        for rule in program or []:
            has_u_tx = False
            for act in rule.get('action', []) or []:
                if extract_target(act) == 'u_tx':
                    has_u_tx = True
                    break
            if has_u_tx:
                new_program.append(corrected_tx_rule)
            else:
                new_program.append(copy.deepcopy(rule))
        new_program.append(mirrored_rule)
        new_program.append(yaw_rule)
        new_program.append(thrust_rule)
        return new_program

    def _compute_prior_bonus(self, programs: List[List[Dict[str, Any]]]):
        if compute_prior_scores is None:
            return None
        if (abs(self.structure_prior_weight) < 1e-9 and
                abs(self.stability_prior_weight) < 1e-9):
            return None
        batch_size = len(programs)
        if batch_size == 0:
            return None
        structure_tensor = torch.zeros(batch_size, device=self.device)
        stability_tensor = torch.zeros(batch_size, device=self.device)
        for idx, prog in enumerate(programs):
            try:
                scores = compute_prior_scores(prog)
                structure_tensor[idx] = float(scores.get('structure', 0.0))
                stability_tensor[idx] = float(scores.get('stability', 0.0))
            except Exception:
                continue
        struct_component = self.structure_prior_weight * structure_tensor
        stab_component = self.stability_prior_weight * stability_tensor
        total = struct_component + stab_component
        return total, struct_component, stab_component

    def _reset_action_history(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if self._last_safe_actions is None:
            return
        if env_ids is None:
            self._last_safe_actions.zero_()
        else:
            self._last_safe_actions[env_ids.long().to(self.device)] = 0.0

    def _partition_programs_by_constraints(self, programs: List[List[Dict[str, Any]]]) -> Tuple[List[List[Dict[str, Any]]], List[int], Dict[int, str]]:
        valid_programs: List[List[Dict[str, Any]]] = []
        valid_indices: List[int] = []
        invalid_info: Dict[int, str] = {}
        for idx, program in enumerate(programs):
            if isinstance(program, ProgramParamCandidate):
                valid_programs.append(program)
                valid_indices.append(idx)
                continue
            ok, reason = validate_program(program)
            if ok:
                valid_programs.append(program)
                valid_indices.append(idx)
            else:
                invalid_info[idx] = reason or "violates hard constraints"
        return valid_programs, valid_indices, invalid_info

    def _log_invalid_programs(self, invalid_info: Dict[int, str]) -> None:
        if not invalid_info:
            return
        for idx, reason in invalid_info.items():
            print(f"[HardConstraint] Skip program #{idx}: {reason}")

    def _merge_rewards_with_invalid(self,
                                    valid_indices: List[int],
                                    valid_rewards: List[float],
                                    invalid_info: Dict[int, str],
                                    total_count: int) -> List[float]:
        merged = [float(HARD_CONSTRAINT_PENALTY)] * total_count
        reward_iter = iter(valid_rewards)
        for idx in valid_indices:
            merged[idx] = float(next(reward_iter))
        self._log_invalid_programs(invalid_info)
        return merged

    def _metric_template(self) -> Dict[str, float]:
        # 仅保留与 SCG 论文一致的两项：状态代价和控制代价
        return {
            'state_cost': 0.0,
            'action_cost': 0.0,
            'hard_constraint_violation': 0.0,
        }

    def _merge_metrics_with_invalid(self,
                                    valid_indices: List[int],
                                    rewards_train: List[float],
                                    rewards_true: List[float],
                                    metrics: List[Dict[str, float]],
                                    invalid_info: Dict[int, str],
                                    total_count: int) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
        final_train = [float(HARD_CONSTRAINT_PENALTY)] * total_count
        final_true = [float(HARD_CONSTRAINT_PENALTY)] * total_count
        final_metrics = [self._metric_template() for _ in range(total_count)]
        train_iter = iter(rewards_train)
        true_iter = iter(rewards_true)
        metric_iter = iter(metrics)
        for idx in valid_indices:
            final_train[idx] = float(next(train_iter))
            final_true[idx] = float(next(true_iter))
            merged_metric = self._metric_template()
            merged_metric.update(next(metric_iter))
            merged_metric['hard_constraint_violation'] = 0.0
            final_metrics[idx] = merged_metric
        for idx in invalid_info:
            final_metrics[idx]['hard_constraint_violation'] = 1.0
        self._log_invalid_programs(invalid_info)
        return final_train, final_true, final_metrics

    def _apply_output_mad(self,
                          actions: torch.Tensor,
                          use_u_flags: List[bool],
                          batch_size: int) -> torch.Tensor:
        if actions is None or actions.shape[0] == 0:
            return actions
        # 全局动作缩放（诊断用途）
        if abs(self.action_scale_multiplier - 1.0) > 1e-6:
            actions[:batch_size, 2:6] *= self.action_scale_multiplier

        if not self.enable_output_mad:
            return actions

        if not use_u_flags:
            return actions

        mask = torch.tensor(use_u_flags, device=self.device, dtype=torch.bool)
        if not mask.any():
            return actions

        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return actions

        action_slice = actions[:batch_size, 2:6]
        current = action_slice[idx].clone()
        prev = self._last_safe_actions[idx]

        # Magnitude clamp（力/力矩幅值）
        current[:, 0] = current[:, 0].clamp(self.mad_min_fz, self.mad_max_fz)
        lateral = current[:, 1:3]
        lat_norm = torch.linalg.norm(lateral, dim=1, keepdim=True)
        lat_scale = torch.clamp(self.mad_max_xy / (lat_norm + self._mad_eps), max=1.0)
        current[:, 1:3] = lateral * lat_scale
        current[:, 3] = current[:, 3].clamp(-self.mad_max_yaw, self.mad_max_yaw)

        # Delta clamp（相邻步变化率）
        delta = current - prev
        delta[:, 0] = delta[:, 0].clamp(-self.mad_max_delta_fz, self.mad_max_delta_fz)
        delta[:, 1] = delta[:, 1].clamp(-self.mad_max_delta_xy, self.mad_max_delta_xy)
        delta[:, 2] = delta[:, 2].clamp(-self.mad_max_delta_xy, self.mad_max_delta_xy)
        delta[:, 3] = delta[:, 3].clamp(-self.mad_max_delta_yaw, self.mad_max_delta_yaw)
        safe = prev + delta

        action_slice[idx] = safe
        self._last_safe_actions.index_copy_(0, idx, safe)
        return actions

    def _compile_program_fast(self, program: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        """
        🚀 快速路径: 预编译常量程序 (u_fz/u_tx/u_ty/u_tz = const)
        
        对于简单的常量控制程序,直接提取常量值,避免重复AST求值
        """
        fz = tx = ty = tz = 0.0
        for rule in program or []:
            if rule.get('op') == 'set':
                var = rule.get('var', '')
                expr = rule.get('expr', {})
                if expr.get('type') == 'const':
                    val = float(expr.get('value', 0.0))
                    if var == 'u_fz':
                        fz = val
                    elif var == 'u_tx':
                        tx = val
                    elif var == 'u_ty':
                        ty = val
                    elif var == 'u_tz':
                        tz = val
        # 裁剪
        fz = float(max(-5.0, min(5.0, fz)))
        tx = float(max(-0.02, min(0.02, tx)))
        ty = float(max(-0.02, min(0.02, ty)))
        tz = float(max(-0.01, min(0.01, tz)))
        return fz, tx, ty, tz
    
    def _extract_variables_from_node(self, node) -> set:
        """递归提取节点中的所有变量名"""
        variables = set()
        if node is None:
            return variables
        return variables

    # ----- 执行路径判定：仅当程序为“无条件常量 set u_*”时才允许 UltraFast -----
    def _is_const_program(self, program) -> bool:
        """判断程序是否为仅包含无条件常量 set u_* 的形式。

        满足条件：
        - 每条规则为 dict 且 op == 'set'
        - 不包含 condition 或 condition 为 None/False
        - expr 为 {'type': 'const', 'value': ...}
        只要出现任意复杂表达式/条件/非常量，就返回 False。
        """
        try:
            for rule in program or []:
                if not isinstance(rule, dict):
                    return False
                if rule.get('op') != 'set':
                    return False
                if rule.get('condition') not in (None, False):
                    return False
                expr = rule.get('expr', None)
                if not isinstance(expr, dict) or expr.get('type') != 'const':
                    return False
                # 变量名必须在允许集合内（u_fz/u_tx/u_ty/u_tz），否则忽略但视为非常量程序
                var = str(rule.get('var', ''))
                if var not in ('u_fz','u_tx','u_ty','u_tz'):
                    return False
            return True
        except Exception:
            return False

    def _all_programs_const(self, programs) -> bool:
        try:
            return all(self._is_const_program(p) for p in (programs or []))
        except Exception:
            return False
        
        # 检查节点类型
        node_type = type(node).__name__
        
        # TerminalNode: 检查是否是变量（字符串）
        if node_type == 'TerminalNode':
            if hasattr(node, 'value') and isinstance(node.value, str):
                variables.add(node.value)
        
        # UnaryOpNode: 递归检查子节点
        elif node_type == 'UnaryOpNode':
            if hasattr(node, 'child'):
                variables.update(self._extract_variables_from_node(node.child))
        
        # BinaryOpNode: 递归检查左右子节点
        elif node_type == 'BinaryOpNode':
            if hasattr(node, 'left'):
                variables.update(self._extract_variables_from_node(node.left))
            if hasattr(node, 'right'):
                variables.update(self._extract_variables_from_node(node.right))
        
        return variables
    
    def _eval_program_forces(self, program: List[Dict[str, Any]], state: Dict[str, float]) -> Tuple[float, float, float, float]:
        """在给定数值 state 下，求解程序产生的 (fz, tx, ty, tz)。

        当前版本将 DSL 输出视为 *残差控制* u_residual，最终控制律为

            u_total = u_base(state) + u_residual(program, state)

        其中 u_base 由底层 Isaac 控制器/segmented PID 提供，本函数仅负责计算
        u_residual 部分（并做适度裁剪），理论分析上可以将其视为有界扰动项。

        策略：聚合所有满足条件的规则，将 set 的值累加（可适度裁剪）。
        注意：仅当程序为“无条件常量 set u_*”形式时，才启用字典制式的快速路径缓存；
        对于 AST 形式（rule={'condition':..., 'action':[BinaryOpNode('set',...)]}），必须走 AST 求值，否则会被错误地当作零动作缓存。
        """
        # 🚀 快速路径: 仅在“无条件常量 set u_*”程序时启用
        if self.use_fast_path and self._is_const_program(program):
            try:
                # 使用稳定的键，仅针对常量 set 规则
                prog_key = str([(r.get('op'), r.get('var'), r.get('expr')) for r in program])
                if prog_key in self._program_cache:
                    return self._program_cache[prog_key]
                # 常量编译
                result = self._compile_program_fast(program)
                self._program_cache[prog_key] = result
                return result
            except Exception:
                # 回退到 AST 求值
                pass
        
        # 慢速路径: 完整AST求值（AST-first 程序或包含条件/非常量表达式）
        # 使用节点的 evaluate() 方法来支持时间算子（ema, rate, delay 等）
        fz = tx = ty = tz = 0.0
        try:
            for rule in program or []:
                # 求值条件（使用 evaluate 而不是 _ast_eval）
                cond_node = rule.get('condition')
                if cond_node is not None and hasattr(cond_node, 'evaluate'):
                    cond = float(cond_node.evaluate(state))
                else:
                    cond = 1.0  # 无条件默认为真
                    
                if cond > 0.0:
                    for a in rule.get('action', []) or []:
                        try:
                            if hasattr(a, 'op') and a.op == 'set' and hasattr(a, 'left') and hasattr(a.left, 'value'):
                                key = str(getattr(a.left, 'value', ''))
                                right_node = getattr(a, 'right', None)
                                # 使用 evaluate() 方法来支持时间算子
                                if right_node is not None and hasattr(right_node, 'evaluate'):
                                    val = float(right_node.evaluate(state))
                                else:
                                    val = 0.0
                                    
                                if key == 'u_fz':
                                    fz += val
                                elif key == 'u_tx':
                                    tx += val
                                elif key == 'u_ty':
                                    ty += val
                                elif key == 'u_tz':
                                    tz += val
                        except Exception:
                            continue
        except Exception:
            pass
        # 适度裁剪（物理合理范围，经验值）
        fz = float(max(-5.0, min(5.0, fz)))     # N（向上为正）
        tx = float(max(-0.02, min(0.02, tx)))   # N*m
        ty = float(max(-0.02, min(0.02, ty)))   # N*m
        tz = float(max(-0.01, min(0.01, tz)))   # N*m（气动力矩较小）
        # 应用全局动作缩放系数（诊断专用）
        scale = float(self.action_scale_multiplier)
        return fz * scale, tx * scale, ty * scale, tz * scale

    def _ensure_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if hasattr(value, 'to'):  # numpy array
            return torch.as_tensor(value, device=self.device)
        return torch.tensor(value, device=self.device)

    def _prepare_gpu_state_tensors(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        omega: torch.Tensor,
        quat: torch.Tensor,
        tgt: torch.Tensor,
        integral_states: List[Dict[str, float]],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        batch_size = pos.shape[0]
        tgt_view = tgt.view(1, 3)
        pos_err = tgt_view - pos
        pos_err_xy = torch.linalg.norm(pos_err[:, :2], dim=1)
        pos_err_mag = torch.linalg.norm(pos_err, dim=1)
        vel_err = torch.linalg.norm(vel, dim=1)
        ang_vel_mag = torch.linalg.norm(omega, dim=1)
        if self._gpu_executor is not None:
            rpy = self._gpu_executor.quat_to_rpy_gpu(quat)
        else:
            rpy = torch.zeros_like(pos)
        rpy_err_mag = torch.linalg.norm(rpy, dim=1)

        integral_tensor = torch.zeros((batch_size, 6), device=self.device)
        for idx in range(batch_size):
            buf = integral_states[idx]
            integral_tensor[idx, 0] = float(buf.get('err_i_x', 0.0))
            integral_tensor[idx, 1] = float(buf.get('err_i_y', 0.0))
            integral_tensor[idx, 2] = float(buf.get('err_i_z', 0.0))
            integral_tensor[idx, 3] = float(buf.get('err_i_roll', 0.0))
            integral_tensor[idx, 4] = float(buf.get('err_i_pitch', 0.0))
            integral_tensor[idx, 5] = float(buf.get('err_i_yaw', 0.0))

        state_tensors = {
            'pos_err_x': pos_err[:, 0],
            'pos_err_y': pos_err[:, 1],
            'pos_err_z': pos_err[:, 2],
            'pos_err': pos_err_mag,
            'pos_err_xy': pos_err_xy,
            'pos_err_z_abs': torch.abs(pos_err[:, 2]),
            'vel_x': vel[:, 0],
            'vel_y': vel[:, 1],
            'vel_z': vel[:, 2],
            'vel_err': vel_err,
            'err_p_roll': rpy[:, 0],
            'err_p_pitch': rpy[:, 1],
            'err_p_yaw': rpy[:, 2],
            'ang_err': rpy_err_mag,
            'rpy_err_mag': rpy_err_mag,
            'ang_vel_x': omega[:, 0],
            'ang_vel_y': omega[:, 1],
            'ang_vel_z': omega[:, 2],
            'ang_vel': ang_vel_mag,
            'ang_vel_mag': ang_vel_mag,
            'err_i_x': integral_tensor[:, 0],
            'err_i_y': integral_tensor[:, 1],
            'err_i_z': integral_tensor[:, 2],
            'err_i_roll': integral_tensor[:, 3],
            'err_i_pitch': integral_tensor[:, 4],
            'err_i_yaw': integral_tensor[:, 5],
            'err_d_x': -vel[:, 0],
            'err_d_y': -vel[:, 1],
            'err_d_z': -vel[:, 2],
            'err_d_roll': -omega[:, 0],
            'err_d_pitch': -omega[:, 1],
            'err_d_yaw': -omega[:, 2],
        }
        return state_tensors, pos_err, rpy

    def _update_integral_states(
        self,
        integral_states: List[Dict[str, float]],
        pos_err: torch.Tensor,
        rpy: torch.Tensor,
        done_mask: torch.Tensor,
        dt: float,
    ) -> None:
        pos_err_det = pos_err.detach()
        rpy_det = rpy.detach()
        done = done_mask.detach().bool()
        for idx, buf in enumerate(integral_states):
            if done[idx]:
                continue
            buf['err_i_x'] = float(buf.get('err_i_x', 0.0) + pos_err_det[idx, 0].item() * dt)
            buf['err_i_y'] = float(buf.get('err_i_y', 0.0) + pos_err_det[idx, 1].item() * dt)
            buf['err_i_z'] = float(buf.get('err_i_z', 0.0) + pos_err_det[idx, 2].item() * dt)
            buf['err_i_roll'] = float(buf.get('err_i_roll', 0.0) + rpy_det[idx, 0].item() * dt)
            buf['err_i_pitch'] = float(buf.get('err_i_pitch', 0.0) + rpy_det[idx, 1].item() * dt)
            buf['err_i_yaw'] = float(buf.get('err_i_yaw', 0.0) + rpy_det[idx, 2].item() * dt)

    def _apply_pid_controllers(
        self,
        controllers: List[Any],
        use_u_flags: List[bool],
        actions: torch.Tensor,
        step: int,
        pos,
        quat,
        vel,
        omega,
        tgt_np,
        integral_states: List[Dict[str, float]],
        ever_nonzero: torch.Tensor,
        debug_enabled: bool,
    ) -> None:
        if not controllers:
            return
        dt = float(getattr(self, '_control_dt', 1.0 / 48.0))
        import numpy as _np

        for i, ctrl in enumerate(controllers):
            if use_u_flags[i] or ctrl is None:
                continue
            try:
                pos_i = pos[i]
                quat_i = quat[i]
                vel_i = vel[i]
                omega_i = omega[i]
                if isinstance(pos_i, torch.Tensor):
                    pos_i = pos_i.detach().cpu().numpy()
                if isinstance(quat_i, torch.Tensor):
                    quat_i = quat_i.detach().cpu().numpy()
                if isinstance(vel_i, torch.Tensor):
                    vel_i = vel_i.detach().cpu().numpy()
                if isinstance(omega_i, torch.Tensor):
                    omega_i = omega_i.detach().cpu().numpy()
                ctrl_actions = ctrl.step(
                    time_step=step,
                    pos_x=float(pos_i[0]),
                    pos_y=float(pos_i[1]),
                    pos_z=float(pos_i[2]),
                    target_x=float(tgt_np[0]),
                    target_y=float(tgt_np[1]),
                    target_z=float(tgt_np[2]),
                )
                actions[i, 0] = float(ctrl_actions.get('fx', 0.0))
                actions[i, 1] = float(ctrl_actions.get('fy', 0.0))
                actions[i, 2] = float(ctrl_actions.get('fz', 0.0))
                actions[i, 3] = float(ctrl_actions.get('tx', 0.0))
                actions[i, 4] = float(ctrl_actions.get('ty', 0.0))
                actions[i, 5] = float(ctrl_actions.get('tz', 0.0))
                if self.strict_no_prior:
                    nz = (abs(actions[i, 2]) > 1e-6) or (abs(actions[i, 3]) > 1e-8) or \
                         (abs(actions[i, 4]) > 1e-8) or (abs(actions[i, 5]) > 1e-8)
                    if nz:
                        ever_nonzero[i] = True
                pe = _np.asarray(tgt_np, dtype=_np.float32) - _np.asarray(pos_i, dtype=_np.float32)
                integral_states[i]['err_i_x'] += float(pe[0]) * dt
                integral_states[i]['err_i_y'] += float(pe[1]) * dt
                integral_states[i]['err_i_z'] += float(pe[2]) * dt
            except Exception as exc:
                if debug_enabled:
                    print(f"[DebugReward] Controller step failed for env {i}: {exc}")
                continue

    # ---------------------- 资源清理 ----------------------
    def close(self):
        """关闭底层环境池，释放GPU/PhysX资源（供基准或多次初始化场景使用）。"""
        try:
            if self._isaac_env_pool is not None:
                try:
                    self._isaac_env_pool.close()
                except Exception:
                    pass
                self._isaac_env_pool = None
        except Exception:
            pass

    def _rpm_to_forces_local(self, rpm: np.ndarray) -> Tuple[float, float, float, float]:
        """将 4 电机 RPM 转换为 (fz, tx, ty, tz)，系数需与环境一致。"""
        KF = 2.8e-08
        KM = 1.1e-10
        L = 0.046
        omega = np.asarray(rpm, dtype=np.float64) * (2.0 * np.pi / 60.0)
        T = KF * (omega ** 2)
        fz = float(np.sum(T))
        tx = float(L * (T[1] - T[3]))
        ty = float(L * (T[2] - T[0]))
        tz = float(KM * (omega[0] ** 2 - omega[1] ** 2 + omega[2] ** 2 - omega[3] ** 2))
        return fz, tx, ty, tz

    def _target_pos(self, t: float) -> np.ndarray:
        """根据 trajectory_config 计算期望位置 [x,y,z]"""
        cfg = self.trajectory_config or {}
        tp = cfg.get('type', 'figure8')
        params = cfg.get('params', {})
        # 支持 initial_xyz / center 两种键名
        init = np.array(cfg.get('initial_xyz', params.get('center', [0.0, 0.0, 1.0])), dtype=np.float32)
        if tp == 'hover':
            # 悬停模式：目标点固定不动
            return init
        elif tp == 'circle':
            # 支持 R / radius 两种键名
            R = float(params.get('R', params.get('radius', 0.9))); period = float(params.get('period', 10.0))
            w = 2.0 * np.pi / max(1e-6, period)
            x = R * np.cos(w * t); y = R * np.sin(w * t); z = 0.0
            return init + np.array([x, y, z], dtype=np.float32)
        elif tp == 'helix':
            R = float(params.get('R', 0.7)); period = float(params.get('period', 10.0)); vz = float(params.get('v_z', 0.15))
            w = 2.0 * np.pi / max(1e-6, period)
            x = R * np.cos(w * t); y = R * np.sin(w * t); z = vz * t
            return init + np.array([x, y, z], dtype=np.float32)
        elif tp == 'square':
            scale = float(params.get('scale', params.get('side', 0.8)))
            period = float(params.get('period', 8.0))
            plane = str(params.get('plane', 'xy')).lower()
            axis = {'x': 0, 'y': 1, 'z': 2}
            if len(plane) == 2 and plane[0] != plane[1]:
                ia = axis.get(plane[0], 0); ib = axis.get(plane[1], 1)
            else:
                ia, ib = 0, 1
            seg_period = max(period / 4.0, 1e-6)
            traverse_speed = scale / seg_period
            cycle = 0.0
            if period > 0:
                cycle = float(np.fmod(t, period))
            seg_idx = int(cycle // seg_period) % 4
            seg_time = cycle - seg_idx * seg_period
            seg_pos = traverse_speed * seg_time
            coord_a = 0.0
            coord_b = 0.0
            if seg_idx == 0:
                coord_a = 0.0
                coord_b = seg_pos
            elif seg_idx == 1:
                coord_a = -seg_pos
                coord_b = scale
            elif seg_idx == 2:
                coord_a = -scale
                coord_b = scale - seg_pos
            else:
                coord_a = -scale + seg_pos
                coord_b = 0.0
            delta = np.zeros(3, dtype=np.float32)
            delta[ia] = coord_a
            delta[ib] = coord_b
            return init + delta
        else:  # figure8
            # 严格对齐 safe-control-gym quadrotor_3D_track: 在给定平面内画 8 字
            A = float(params.get('A', 1.0))
            B = float(params.get('B', 1.0))
            period = float(params.get('period', 5.0))
            # 🔧 默认 xy 平面：u_tx 控制 Y，u_ty 控制 X，匹配单轴搜索
            plane = str(params.get('plane', 'xy')).lower()
            w = 2.0 * np.pi / max(1e-6, period)
            a_coord = A * np.sin(w * t)
            b_coord = B * np.sin(w * t) * np.cos(w * t)

            # plane 选择哪个坐标轴承载 8 字轨迹（例如 xz）
            axis = {'x': 0, 'y': 1, 'z': 2}
            if len(plane) == 2 and plane[0] != plane[1]:
                ia = axis.get(plane[0], 0)
                ib = axis.get(plane[1], 2)
            else:
                ia, ib = 0, 1  # 回退 xy

            delta = np.zeros(3, dtype=np.float32)
            delta[ia] = a_coord
            delta[ib] = b_coord
            return init + delta
    
    def evaluate_batch(self, programs: List[List[Dict[str, Any]]]) -> List[float]:
        """
        使用Isaac Gym批量评估程序
        
        Args:
            programs: 程序列表，每个程序是规则列表
        
        Returns:
            rewards: 每个程序的奖励（负值=误差，越大越好）
        """
        total_requested = len(programs)

        # 先按硬约束过滤
        valid_programs, valid_indices, invalid_info = self._partition_programs_by_constraints(programs)
        if not valid_programs:
            self._log_invalid_programs(invalid_info)
            return [float(HARD_CONSTRAINT_PENALTY)] * total_requested
        programs = valid_programs

        # 初始化环境池（在BO之前，避免BO第1轮触发初始化开销）
        if self._isaac_env_pool is None:
            self._init_isaac_gym_pool()

        # 🔥 贝叶斯优化调参（🚀 批量并行优化：所有程序的BO候选参数一起评估）
        if self.enable_bayesian_tuning:
            programs = self._batch_tune_programs_with_bo(programs)

        # 延迟导入 torch：确保在 isaacgym 成功导入之后
        import torch  # type: ignore

        # 评估缓存：为每个有效程序生成键，拆分缓存命中与待评估子集
        cache_keys: List[Optional[str]] = []
        cached_rewards: Dict[int, float] = {}
        indices_to_eval: List[int] = []
        for idx, prog in enumerate(programs):
            try:
                key = self._program_eval_key(prog)
            except Exception:
                key = None
            cache_keys.append(key)
            if key is not None and key in self._eval_cache:
                cached_rewards[idx] = float(self._eval_cache[key])
            else:
                indices_to_eval.append(idx)

        # 👀 轻量级缓存命中率日志（主要用于观察 BO 内部复用情况）
        num_valid = len(programs)
        num_cached = len(cached_rewards)
        num_new = len(indices_to_eval)
        if num_valid > 0 and (num_cached > 0 or num_new > 0):
            hit_rate = num_cached / float(num_valid)
            print(f"[EvalCache] valid={num_valid}, cached={num_cached}, new={num_new}, hit={hit_rate:.3f}")

        if len(indices_to_eval) == 0:
            # 全部命中缓存，直接合并无效程序并返回
            cached_list = [cached_rewards[i] for i in range(len(programs))]
            return self._merge_rewards_with_invalid(valid_indices, cached_list, invalid_info, total_requested)

        # 构造待真实评估的子列表
        programs_to_eval = [programs[i] for i in indices_to_eval]

        # 对仍需真实仿真的候选，延迟构造实际 DSL 程序
        programs_to_eval = [
            prog.materialize() if isinstance(prog, ProgramParamCandidate) else prog
            for prog in programs_to_eval
        ]

        # 🔧 镜像展开：如果程序只有 u_tx，则自动生成 u_ty（取反）及 yaw/thrust 稳定器
        programs_to_eval = [
            self._mirror_expand_single_axis_program(prog) for prog in programs_to_eval
        ]

        num_programs_original = len(programs_to_eval)
        
        # 🔧 扩展replicas: 每个程序复制 replicas_per_program 次
        if self.replicas_per_program > 1:
            programs_expanded = []
            for prog in programs_to_eval:
                programs_expanded.extend([prog] * self.replicas_per_program)
            programs_to_eval = programs_expanded

        num_programs = len(programs_to_eval)
        rewards = []
        
        start_time = time.time()
        
        # 分批评估（考虑replicas: 每批最多 isaac_num_envs // replicas_per_program 个程序）
        programs_per_batch = max(1, self.isaac_num_envs // self.replicas_per_program)
        
        for batch_start in range(0, num_programs, programs_per_batch):
            batch_end = min(batch_start + programs_per_batch, num_programs)
            batch_programs = programs_to_eval[batch_start:batch_end]
            batch_size = len(batch_programs)

            
            # ✅ 确定性评估：强制每批评估前完全重置环境（确保相同程序得到相同奖励）
            # 原因：环境池复用会导致新程序从上一个程序的结束状态开始，引入不可控的随机性
            # 修复：永远执行 reset()，保证每个程序都从固定初始状态 (0,0,h) 开始评估
            
            # 轻量级确定性重置：保留环境池，只重置状态
            # 🔧 计算初始位置：等于 t=0 时的目标位置
            initial_pos_np = self._target_pos(0.0)
            initial_pos_tensor = torch.tensor(initial_pos_np, device=self.device, dtype=torch.float32)
            if not self._envs_ready:
                # 首次：完整初始化环境池
                # 扩展 initial_pos 为 [num_envs, 3]
                initial_pos_batch = initial_pos_tensor.unsqueeze(0).expand(self.isaac_num_envs, -1).clone()
                obs = self._isaac_env_pool.reset(initial_pos=initial_pos_batch)
                self._envs_ready = True
                self._last_reset_size = self.isaac_num_envs
                self._reset_action_history()
            else:
                # 后续：只重置前 batch_size 个环境到初始状态 (快速，无重建开销)
                env_ids_to_reset = torch.arange(batch_size, dtype=torch.long, device=self.device)
                initial_pos_batch = initial_pos_tensor.unsqueeze(0).expand(batch_size, -1).clone()
                obs = self._isaac_env_pool.reset(env_ids=env_ids_to_reset, initial_pos=initial_pos_batch)
                self._reset_action_history(env_ids_to_reset)
            
            # 运行仿真（环境池大小可能大于本批大小，按前 batch_size 个槽位使用）
            total_rewards = torch.zeros(self.isaac_num_envs, device=self.device)
            done_flags = torch.zeros(self.isaac_num_envs, dtype=torch.bool, device=self.device)
            # 为当前批次创建专属 done 标志和 stepwise 奖励计算器（匹配 batch_size）
            done_flags_batch = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            if self._step_reward_calc is not None:
                try:
                    weights, ks = get_reward_profile(self.reward_profile) if get_reward_profile else ({}, {})
                    self._step_reward_calc = StepwiseRewardCalculator(weights, ks, dt=self._step_dt, num_envs=batch_size, device=self.device)
                except Exception:
                    self._step_reward_calc = None
            
            # 🔥 为当前批次重建 SCG reward calculator（匹配 batch_size）
            if self.use_scg_exact_reward and self._scg_reward_calc is not None:
                from .reward_scg_exact import SCGExactRewardCalculator
                self._scg_reward_calc = SCGExactRewardCalculator(
                    num_envs=batch_size,
                    device=self.device,
                    state_weights=self._scg_reward_calc.Q,
                    action_weight=self._scg_reward_calc.R,
                )
            # 记录每个环境累计了多少个有效步（用于 mean 归约）
            steps_count = torch.zeros(self.isaac_num_envs, device=self.device)
            # 记录是否曾经产生过非零动作（仅针对前 batch_size）
            ever_nonzero = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            
            # 初始化积分状态（持久化跨步）
            integral_states = [
                {
                    'err_i_x': 0.0, 'err_i_y': 0.0, 'err_i_z': 0.0,
                    'err_i_roll': 0.0, 'err_i_pitch': 0.0, 'err_i_yaw': 0.0
                }
                for i in range(batch_size)
            ]

            # 调试开关（需尽早声明，避免未定义引用）
            debug_enabled = bool(int(os.getenv('DEBUG_STEPWISE', '0')))

            # 所有程序统一使用 u_* 直接输出路径（不再依赖 PID 封装）
            controllers = [None for _ in range(batch_size)]
            use_u_flags = [True for _ in range(batch_size)]  # 全部使用直接力/力矩输出
            gpu_batch_token = None
            
            # 🔧 重置每个程序的时间算子状态（ema/delay/diff/rate）
            # 确保每次评估从零状态开始，保证训练与测试一致性
            if reset_program_state is not None:
                for prog in batch_programs:
                    reset_program_state(prog)
            
            if debug_enabled:
                print("[DebugReward] All programs use direct u_* (force/torque) output path")

            # 🔧 修复: gpu_batch_token 初始化应该在 debug_enabled 条件外
            if (self._gpu_executor is not None and self.use_gpu_expression_executor and any(use_u_flags)):
                try:
                    gpu_batch_token = self._gpu_executor.prepare_batch(batch_programs)
                except Exception as gpu_batch_exc:
                    gpu_batch_token = None
                    if debug_enabled:
                        print(f"[GPUExecutor] ⚠️ 批次绑定失败，使用CPU路径: {gpu_batch_exc}")

            # 控制步数（以控制频率计，不再按物理频率）
            max_steps = int(self.duration * float(getattr(self, '_control_freq', 48)))
            min_steps = int(max_steps * self.min_steps_frac)
            
            # 调试辅助：记录首末位置误差（仅在开启 DEBUG_STEPWISE 时）
            first_pos_err = None
            last_pos_err = None
            
            # 统计整个 Episode 的动作幅度
            episode_stats = {
                'sum_fz': 0.0, 'max_fz': 0.0,
                'sum_tx': 0.0, 'max_tx': 0.0,
                'count': 0
            }

            # 预先分配动作张量，循环内复用以减少反复分配
            actions = torch.zeros((self.isaac_num_envs, 6), device=self.device)

            for step in range(max_steps):
                # 计算目标点（所有 env 相同目标轨迹，使用动态轨迹而不是静态 cfg.target）
                t = step * float(getattr(self, '_control_dt', 1.0/48.0))
                tgt_np = self._target_pos(t)  # numpy array [3]
                tgt_tensor = torch.tensor(tgt_np, device=self.device, dtype=torch.float32)

                # 生成动作（统一为 [fx,fy,fz,tx,ty,tz] 6 维格式，便于混用）
                actions.zero_()
                pos = obs['position'][:batch_size]
                quat = obs['orientation'][:batch_size]
                vel = obs['velocity'][:batch_size]
                omega = obs['angular_velocity'][:batch_size]
                gpu_actions_applied = False
                if gpu_batch_token is not None and (not hasattr(self, '_cuda_executor') or self._cuda_executor is None or not hasattr(self, '_compiled_forces_gpu')):
                    try:
                        pos_tensor = self._ensure_tensor(pos)
                        vel_tensor = self._ensure_tensor(vel)
                        omega_tensor = self._ensure_tensor(omega)
                        quat_tensor = self._ensure_tensor(quat)
                        gpu_use_mask = torch.tensor(use_u_flags, dtype=torch.bool, device=self.device)
                        if self._use_gpu_control_loop:
                            gpu_outputs, pos_err_tensor, rpy_tensor = self._gpu_executor.evaluate_from_raw_obs(
                                gpu_batch_token,
                                pos_tensor,
                                vel_tensor,
                                omega_tensor,
                                quat_tensor,
                                tgt_tensor,
                                integral_states,
                                gpu_use_mask,
                                active_mask=(~done_flags_batch)
                            )
                        else:
                            state_tensors, pos_err_tensor, rpy_tensor = self._prepare_gpu_state_tensors(
                                pos_tensor, vel_tensor, omega_tensor, quat_tensor, tgt_tensor, integral_states
                            )
                            gpu_outputs = self._gpu_executor.evaluate(
                                gpu_batch_token,
                                state_tensors,
                                gpu_use_mask,
                                active_mask=(~done_flags_batch)
                            )
                        actions[:batch_size, 2:6] = torch.where(
                            gpu_use_mask.unsqueeze(-1),
                            gpu_outputs,
                            actions[:batch_size, 2:6]
                        )
                        if self.strict_no_prior:
                            nz_mask = (
                                gpu_outputs[:, 0].abs() > 1e-6
                            ) | (
                                gpu_outputs[:, 1].abs() > 1e-8
                            ) | (
                                gpu_outputs[:, 2].abs() > 1e-8
                            ) | (
                                gpu_outputs[:, 3].abs() > 1e-8
                            )
                            ever_nonzero |= (gpu_use_mask & nz_mask)
                        self._update_integral_states(
                            integral_states,
                            pos_err_tensor,
                            rpy_tensor,
                            done_flags_batch,
                            float(getattr(self, '_control_dt', 1.0 / 48.0))
                        )
                        if not self.strict_no_prior:
                            self._apply_pid_controllers(
                                controllers,
                                use_u_flags,
                                actions,
                                step,
                                pos_tensor,
                                quat_tensor,
                                vel_tensor,
                                omega_tensor,
                                tgt_np,
                                integral_states,
                                ever_nonzero,
                                debug_enabled,
                            )
                        gpu_actions_applied = True
                    except Exception as gpu_step_exc:
                        gpu_actions_applied = False
                        if debug_enabled:
                            print(f"[GPUExecutor] ⚠️ step{step} 回退CPU: {gpu_step_exc}")
                
                # 🚀🚀🚀 CUDA超高性能路径: 完全GPU执行 (step 0时初始化)
                if not gpu_actions_applied and self.use_fast_path and step == 0:
                    try:
                        # 优先尝试CUDA执行器 (零CPU传输)
                        if not hasattr(self, '_cuda_executor_initialized'):
                            self._cuda_executor_initialized = True
                            try:
                                from .cuda_program_executor import CUDAProgramExecutor
                                self._cuda_executor = CUDAProgramExecutor(device=str(self.device))
                                print(f"[CUDA] 🚀 初始化CUDA执行器 (设备: {self.device})")
                            except Exception as e:
                                print(f"[CUDA] ⚠️ CUDA执行器不可用: {e}")
                                self._cuda_executor = None
                        
                        # CUDA编译
                        if self._cuda_executor is not None and not hasattr(self, '_compiled_forces_gpu'):
                            if self._all_programs_const(batch_programs):
                                t0 = time.time()
                                self._compiled_forces_gpu = self._cuda_executor.compile_constant_programs(batch_programs)
                                compile_time = (time.time() - t0) * 1000
                                
                                if self._compiled_forces_gpu is not None:
                                    print(f"[CUDA] ✅ GPU预编译{len(batch_programs)}程序 ({compile_time:.2f}ms)")
                                    print(f"[CUDA] 💾 Forces shape: {self._compiled_forces_gpu.shape}, device: {self._compiled_forces_gpu.device}")
                                else:
                                    print(f"[CUDA] ⚠️ 包含非常量程序，回退到CPU路径")
                                    self._cuda_executor = None
                            else:
                                print(f"[CUDA] ⚠️ 存在条件/表达式程序，回退到CPU路径")
                                self._cuda_executor = None
                    except Exception as e:
                        print(f"[CUDA] ❌ 编译失败: {e}, 回退到CPU路径")
                        self._cuda_executor = None
                
                # 🚀🚀 超高性能路径: 完全向量化 + JIT (CPU fallback)
                if not gpu_actions_applied and self.use_fast_path and self._ultra_executor is not None and step == 0:
                    # 只有当CUDA不可用时才使用CPU UltraFast
                    if not hasattr(self, '_cuda_executor') or self._cuda_executor is None:
                        # 首次步骤: 预编译所有程序 (只做一次)
                        try:
                            if not hasattr(self, '_compiled_forces'):
                                # 仅当所有程序皆为“无条件常量 set u_*”时，才启用 UltraFast
                                if self._all_programs_const(batch_programs):
                                    self._compiled_forces = self._ultra_executor.compile_programs(batch_programs)
                                    print(f"[UltraFast CPU] ✅ 预编译{len(batch_programs)}程序 → 缓存{len(self._ultra_executor.program_cache)}个唯一程序")
                                # 若全部常量结果几乎为零，且严格无先验，则放弃 UltraFast 以避免长期零动作退化
                                try:
                                    import numpy as _np
                                    if _np.all(_np.abs(self._compiled_forces) < 1e-8) and self.strict_no_prior:
                                        print("[UltraFast] ⚠️ 全常量为零，禁用UltraFast以避免零动作退化")
                                        self._ultra_executor = None
                                        if hasattr(self, '_compiled_forces'):
                                            delattr(self, '_compiled_forces')
                                except Exception:
                                    pass
                            else:
                                # 存在条件/非常量表达式：禁用 UltraFast，回退到逐步AST评估，确保动作依赖状态
                                self._ultra_executor = None
                        except Exception as e:
                            print(f"[UltraFast] ⚠️ 预编译失败: {e}, 回退到标准快速路径")
                            self._ultra_executor = None
                
                # 🚀🚀🚀 完全GPU路径: 零CPU传输 (CUDA加速)
                if not gpu_actions_applied and self.use_fast_path and hasattr(self, '_cuda_executor') and self._cuda_executor is not None:
                    try:
                        # 100% GPU执行: 无CPU↔GPU传输!
                        if hasattr(self, '_compiled_forces_gpu'):
                            # ✅ CUDA执行器已经返回正确大小的tensor [batch_size, 6]
                            actions[:batch_size] = self._cuda_executor.apply_constant_forces_vectorized(
                                self._compiled_forces_gpu,
                                batch_size,
                                self.isaac_num_envs
                            )
                    except Exception as e:
                        print(f"[CUDA Fast Path] ⚠️ GPU执行失败: {e}, 回退到CPU路径")
                        self._cuda_executor = None
                
                # 🚀 快速路径: 批量处理 u_* 路径 (CPU fallback)
                if not gpu_actions_applied and self.use_fast_path and (not hasattr(self, '_cuda_executor') or self._cuda_executor is None):
                    # 预先导入scipy（避免循环内重复导入）
                    try:
                        from scipy.spatial.transform import Rotation
                    except ImportError:
                        Rotation = None
                    
                    # 批量计算位置误差 [batch_size, 3]
                    # 注意: Isaac Gym的obs可能是torch tensor或numpy array
                    if isinstance(pos, torch.Tensor):
                        pos_np = pos.cpu().numpy()
                        quat_np = quat.cpu().numpy()
                        vel_np = vel.cpu().numpy()
                        omega_np = omega.cpu().numpy()
                    else:
                        pos_np = np.asarray(pos)
                        quat_np = np.asarray(quat)
                        vel_np = np.asarray(vel)
                        omega_np = np.asarray(omega)
                    
                    tgt_batch = np.tile(tgt_np, (batch_size, 1))  # [batch_size, 3]
                    pe_batch = tgt_batch - pos_np  # [batch_size, 3]
                    
                    # 批量计算RPY
                    if Rotation is not None:
                        try:
                            rpy_batch = Rotation.from_quat(quat_np).as_euler('XYZ', degrees=False)  # [batch_size, 3]
                        except Exception:
                            rpy_batch = np.zeros((batch_size, 3), dtype=np.float32)
                    else:
                        rpy_batch = np.zeros((batch_size, 3), dtype=np.float32)
                    
                    # 🚀🚀 超高性能执行: 批量应用预编译的力
                    if self._ultra_executor is not None and hasattr(self, '_compiled_forces'):
                        try:
                            # 批量执行 (消除Python循环)
                            try:
                                from .ultra_fast_executor import apply_forces_jit, update_integral_jit
                            except ImportError:
                                from ultra_fast_executor import apply_forces_jit, update_integral_jit
                            
                            use_u_array = np.array(use_u_flags, dtype=np.bool_)
                            actions_np = np.zeros((batch_size, 6), dtype=np.float32)
                            apply_forces_jit(actions_np, self._compiled_forces, use_u_array)
                            
                            # 转为tensor
                            actions[:batch_size] = torch.from_numpy(actions_np).to(self.device)
                            
                            # 更新积分项 (JIT加速)
                            if not all(done_flags[:batch_size].cpu().numpy()):
                                err_i = np.array([
                                    [s['err_i_x'], s['err_i_y'], s['err_i_z'],
                                     s['err_i_roll'], s['err_i_pitch'], s['err_i_yaw']]
                                    for s in integral_states
                                ], dtype=np.float32)
                                done_array = done_flags[:batch_size].cpu().numpy().astype(np.bool_)
                                dt = float(getattr(self, '_control_dt', 1.0/48.0))
                                update_integral_jit(err_i, pe_batch, rpy_batch, done_array, dt)
                                
                                # 写回integral_states
                                for i in range(batch_size):
                                    integral_states[i]['err_i_x'] = float(err_i[i, 0])
                                    integral_states[i]['err_i_y'] = float(err_i[i, 1])
                                    integral_states[i]['err_i_z'] = float(err_i[i, 2])
                                    integral_states[i]['err_i_roll'] = float(err_i[i, 3])
                                    integral_states[i]['err_i_pitch'] = float(err_i[i, 4])
                                    integral_states[i]['err_i_yaw'] = float(err_i[i, 5])
                            
                            # 检查ever_nonzero (向量化)
                            if self.strict_no_prior:
                                nonzero_mask = (np.abs(actions_np[:, 2]) > 1e-6) | \
                                               (np.abs(actions_np[:, 3]) > 1e-8) | \
                                               (np.abs(actions_np[:, 4]) > 1e-8) | \
                                               (np.abs(actions_np[:, 5]) > 1e-8)
                                for i in range(batch_size):
                                    if use_u_flags[i] and nonzero_mask[i]:
                                        ever_nonzero[i] = True
                            
                            # 处理非u_*路径（PID控制器）
                            for i in range(batch_size):
                                if not use_u_flags[i]:
                                    ctrl = controllers[i]
                                    try:
                                        if ctrl is not None:
                                            pe = pe_batch[i]
                                            ctrl_actions = ctrl.step(
                                                time_step=step,
                                                pos_x=float(pos[i][0]),
                                                pos_y=float(pos[i][1]),
                                                pos_z=float(pos[i][2]),
                                                target_x=float(tgt_np[0]),
                                                target_y=float(tgt_np[1]),
                                                target_z=float(tgt_np[2]),
                                            )
                                            actions[i, 0] = float(ctrl_actions.get('fx', 0.0))
                                            actions[i, 1] = float(ctrl_actions.get('fy', 0.0))
                                            actions[i, 2] = float(ctrl_actions.get('fz', 0.0))
                                            actions[i, 3] = float(ctrl_actions.get('tx', 0.0))
                                            actions[i, 4] = float(ctrl_actions.get('ty', 0.0))
                                            actions[i, 5] = float(ctrl_actions.get('tz', 0.0))
                                            if self.strict_no_prior:
                                                if (abs(actions[i, 2]) > 1e-6) or (abs(actions[i, 3]) > 1e-8) or \
                                                   (abs(actions[i, 4]) > 1e-8) or (abs(actions[i, 5]) > 1e-8):
                                                    ever_nonzero[i] = True
                                            
                                            # 更新积分项
                                            dt = float(getattr(self, '_control_dt', 1.0/48.0))
                                            integral_states[i]['err_i_x'] += pe[0] * dt
                                            integral_states[i]['err_i_y'] += pe[1] * dt
                                            integral_states[i]['err_i_z'] += pe[2] * dt
                                    except Exception as e:
                                        if debug_enabled:
                                            print(f"[DebugReward] Controller step failed for env {i}: {e}")
                                        pass
                            
                        except Exception as e:
                            if step == 0:
                                import traceback
                                print(f"[UltraFast] ⚠️ 执行失败: {e}")
                                traceback.print_exc()
                            print(f"[UltraFast] 回退到标准路径")
                            # 回退到下面的标准快速路径
                            self._ultra_executor = None
                    
                    # 标准快速路径 (如果超高性能路径未激活)
                    if self._ultra_executor is None or not hasattr(self, '_compiled_forces'):
                        # 向量化处理所有使用u_*的程序
                        for i in range(batch_size):
                            if use_u_flags[i]:
                                pe = pe_batch[i]
                                rpy = rpy_batch[i]
                                
                                state = {
                                'pos_err_x': float(pe[0]),
                                'pos_err_y': float(pe[1]),
                                'pos_err_z': float(pe[2]),
                                'pos_err': float(np.linalg.norm(pe)),
                                'pos_err_xy': float(np.linalg.norm(pe[:2])),
                                'pos_err_z_abs': float(abs(pe[2])),
                                'vel_x': float(vel_np[i][0]),
                                'vel_y': float(vel_np[i][1]),
                                'vel_z': float(vel_np[i][2]),
                                'vel_err': float(np.linalg.norm(vel_np[i])),
                                'err_p_roll': float(rpy[0]),
                                'err_p_pitch': float(rpy[1]),
                                'err_p_yaw': float(rpy[2]),
                                'ang_err': float(np.linalg.norm(rpy)),
                                'rpy_err_mag': float(np.linalg.norm(rpy)),
                                'ang_vel_x': float(omega_np[i][0]),
                                'ang_vel_y': float(omega_np[i][1]),
                                'ang_vel_z': float(omega_np[i][2]),
                                'ang_vel': float(np.linalg.norm(omega_np[i])),
                                'ang_vel_mag': float(np.linalg.norm(omega_np[i])),
                                'err_i_x': float(integral_states[i]['err_i_x']),
                                'err_i_y': float(integral_states[i]['err_i_y']),
                                'err_i_z': float(integral_states[i]['err_i_z']),
                                'err_i_roll': float(integral_states[i]['err_i_roll']),
                                'err_i_pitch': float(integral_states[i]['err_i_pitch']),
                                'err_i_yaw': float(integral_states[i]['err_i_yaw']),
                                'err_d_x': float(-vel_np[i][0]),
                                'err_d_y': float(-vel_np[i][1]),
                                'err_d_z': float(-vel_np[i][2]),
                                    'err_d_roll': float(-omega_np[i][0]),
                                    'err_d_pitch': float(-omega_np[i][1]),
                                    'err_d_yaw': float(-omega_np[i][2]),
                                }
                                fz, tx, ty, tz = self._eval_program_forces(batch_programs[i], state)
                                actions[i, 0] = 0.0
                                actions[i, 1] = 0.0
                                actions[i, 2] = float(fz)
                                actions[i, 3] = float(tx)
                                actions[i, 4] = float(ty)
                                actions[i, 5] = float(tz)
                                if self.strict_no_prior:
                                    if (abs(fz) > 1e-6) or (abs(tx) > 1e-8) or (abs(ty) > 1e-8) or (abs(tz) > 1e-8):
                                        ever_nonzero[i] = True
                                
                                # 更新积分项
                                dt = float(getattr(self, '_control_dt', 1.0/48.0))
                                integral_states[i]['err_i_x'] += pe[0] * dt
                                integral_states[i]['err_i_y'] += pe[1] * dt
                                integral_states[i]['err_i_z'] += pe[2] * dt
                                integral_states[i]['err_i_roll'] += rpy[0] * dt
                                integral_states[i]['err_i_pitch'] += rpy[1] * dt
                                integral_states[i]['err_i_yaw'] += rpy[2] * dt
                    
                    # 处理非u_*路径（PID控制器）
                    for i in range(batch_size):
                        if not use_u_flags[i]:
                            ctrl = controllers[i]
                            try:
                                if ctrl is not None:
                                    pe = pe_batch[i]
                                    ctrl_actions = ctrl.step(
                                        time_step=step,
                                        pos_x=float(pos[i][0]),
                                        pos_y=float(pos[i][1]),
                                        pos_z=float(pos[i][2]),
                                        target_x=float(tgt_np[0]),
                                        target_y=float(tgt_np[1]),
                                        target_z=float(tgt_np[2]),
                                    )
                                    actions[i, 0] = float(ctrl_actions.get('fx', 0.0))
                                    actions[i, 1] = float(ctrl_actions.get('fy', 0.0))
                                    actions[i, 2] = float(ctrl_actions.get('fz', 0.0))
                                    actions[i, 3] = float(ctrl_actions.get('tx', 0.0))
                                    actions[i, 4] = float(ctrl_actions.get('ty', 0.0))
                                    actions[i, 5] = float(ctrl_actions.get('tz', 0.0))
                                    if self.strict_no_prior:
                                        if (abs(actions[i, 2]) > 1e-6) or (abs(actions[i, 3]) > 1e-8) or \
                                           (abs(actions[i, 4]) > 1e-8) or (abs(actions[i, 5]) > 1e-8):
                                            ever_nonzero[i] = True
                                    
                                    # 更新积分项
                                    dt = float(getattr(self, '_control_dt', 1.0/48.0))
                                    integral_states[i]['err_i_x'] += pe[0] * dt
                                    integral_states[i]['err_i_y'] += pe[1] * dt
                                    integral_states[i]['err_i_z'] += pe[2] * dt
                            except Exception as e:
                                if debug_enabled:
                                    print(f"[DebugReward] Controller step failed for env {i}: {e}")
                                pass
                elif not gpu_actions_applied:
                    # 慢速路径: 原始串行处理
                    for i in range(batch_size):
                        ctrl = controllers[i]
                        try:
                            if use_u_flags[i]:
                                # 构造完整三轴 state（支持精细 PID）
                                pe = np.asarray(tgt_np, dtype=np.float32) - np.asarray(pos[i], dtype=np.float32)
                                # 获取四元数 → RPY（简化：仅用于姿态误差估算）
                                try:
                                    from scipy.spatial.transform import Rotation
                                    rpy = Rotation.from_quat(quat[i]).as_euler('XYZ', degrees=False)
                                except Exception:
                                    # 无 scipy 时退化为零
                                    rpy = np.zeros(3, dtype=np.float32)
                                
                                # TODO: 积分项需要跨步累积（当前简化为零）
                                state = {
                                # 位置误差（三轴）
                                'pos_err_x': float(pe[0]),
                                'pos_err_y': float(pe[1]),
                                'pos_err_z': float(pe[2]),
                                'pos_err': float(np.linalg.norm(pe)),
                                'pos_err_xy': float(np.linalg.norm(pe[:2])),
                                'pos_err_z_abs': float(abs(pe[2])),
                                # 速度（三轴 + 模长）
                                'vel_x': float(vel[i][0]),
                                'vel_y': float(vel[i][1]),
                                'vel_z': float(vel[i][2]),
                                'vel_err': float(np.linalg.norm(vel[i])),
                                # 姿态误差（RPY，目标假设为 0）
                                'err_p_roll': float(rpy[0]),
                                'err_p_pitch': float(rpy[1]),
                                'err_p_yaw': float(rpy[2]),
                                'ang_err': float(np.linalg.norm(rpy)),
                                'rpy_err_mag': float(np.linalg.norm(rpy)),
                                # 角速度（三轴 + 模长）
                                'ang_vel_x': float(omega[i][0]),
                                'ang_vel_y': float(omega[i][1]),
                                'ang_vel_z': float(omega[i][2]),
                                'ang_vel': float(np.linalg.norm(omega[i])),
                                'ang_vel_mag': float(np.linalg.norm(omega[i])),
                                # 积分项（累积）
                                'err_i_x': float(integral_states[i]['err_i_x']),
                                'err_i_y': float(integral_states[i]['err_i_y']),
                                'err_i_z': float(integral_states[i]['err_i_z']),
                                'err_i_roll': float(integral_states[i]['err_i_roll']),
                                'err_i_pitch': float(integral_states[i]['err_i_pitch']),
                                'err_i_yaw': float(integral_states[i]['err_i_yaw']),
                                # 微分项（近似为速度/角速度的负值）
                                'err_d_x': float(-vel[i][0]),
                                'err_d_y': float(-vel[i][1]),
                                'err_d_z': float(-vel[i][2]),
                                'err_d_roll': float(-omega[i][0]),
                                'err_d_pitch': float(-omega[i][1]),
                                'err_d_yaw': float(-omega[i][2]),
                                }
                                fz, tx, ty, tz = self._eval_program_forces(batch_programs[i], state)
                                actions[i, 0] = 0.0
                                actions[i, 1] = 0.0
                                actions[i, 2] = float(fz)
                                actions[i, 3] = float(tx)
                                actions[i, 4] = float(ty)
                                actions[i, 5] = float(tz)
                                # 记录是否产生非零动作
                                if self.strict_no_prior:
                                    if (abs(fz) > 1e-6) or (abs(tx) > 1e-8) or (abs(ty) > 1e-8) or (abs(tz) > 1e-8):
                                        ever_nonzero[i] = True
                                # 更新积分状态（仅对未完成的环境）
                                if not done_flags[i]:
                                    dt = float(self._control_dt)
                                    integral_states[i]['err_i_x'] += float(pe[0]) * dt
                                    integral_states[i]['err_i_y'] += float(pe[1]) * dt
                                    integral_states[i]['err_i_z'] += float(pe[2]) * dt
                                    integral_states[i]['err_i_roll'] += float(rpy[0]) * dt
                                    integral_states[i]['err_i_pitch'] += float(rpy[1]) * dt
                                    integral_states[i]['err_i_yaw'] += float(rpy[2]) * dt
                            else:
                                if ctrl is None:
                                    continue
                                rpm, _pos_e, _rpy_e = ctrl.computeControl(
                                    self._control_dt,
                                    cur_pos=pos[i],
                                    cur_quat=quat[i],
                                    cur_vel=vel[i],
                                    cur_ang_vel=omega[i],
                                    target_pos=tgt_np,
                                )
                                rpm = np.clip(np.asarray(rpm, dtype=np.float32), 0.0, 25000.0)
                                fz, tx, ty, tz = self._rpm_to_forces_local(rpm)
                                actions[i, 2] = float(fz)
                                actions[i, 3] = float(tx)
                                actions[i, 4] = float(ty)
                                actions[i, 5] = float(tz)
                        except Exception:
                            # 失败则保持零动作
                            pass
                
                # 输出安全壳 (MAD) + 步进仿真
                actions = self._apply_output_mad(actions, use_u_flags, batch_size)
                
                # 更新统计
                if debug_enabled or batch_start == 0:
                    try:
                        fz_vals = actions[:batch_size, 2].abs()
                        tx_vals = actions[:batch_size, 3].abs()
                        episode_stats['sum_fz'] += float(fz_vals.sum().item())
                        episode_stats['max_fz'] = max(episode_stats['max_fz'], float(fz_vals.max().item()))
                        episode_stats['sum_tx'] += float(tx_vals.sum().item())
                        episode_stats['max_tx'] = max(episode_stats['max_tx'], float(tx_vals.max().item()))
                        episode_stats['count'] += batch_size
                    except Exception:
                        pass

                obs, step_rewards_env, dones, infos = self._isaac_env_pool.step(actions)

                # 直接从 Isaac Gym 获取 GPU 张量快照，避免 CPU↔GPU 往返
                tensor_obs = self._isaac_env_pool.get_states_batch()
                pos_gpu = tensor_obs['pos']
                vel_gpu = tensor_obs['vel']
                omega_gpu = tensor_obs['omega']
                quat_gpu = tensor_obs['quat']  # 姿态四元数 [qx, qy, qz, qw]
                # 目标（悬停或轨迹）
                if self.trajectory_config.get('type') == 'hover':
                    tgt = np.array([0.0, 0.0, self.trajectory_config.get('height', 1.0)], dtype=np.float32)
                else:
                    tgt = np.array(self.trajectory_config.get('target', [0.0, 0.0, 1.0]), dtype=np.float32)
                
                # 计算 Reward
                if self.use_scg_exact_reward and self._scg_reward_calc is not None:
                    # 🎯 精确 SCG reward（二次代价，无 shaping）
                    step_reward = self._scg_reward_calc.compute_step(
                        pos_gpu[:batch_size, :],
                        vel_gpu[:batch_size, :],
                        quat_gpu[:batch_size, :],
                        omega_gpu[:batch_size, :],
                        tgt_tensor,
                        actions[:batch_size, 2:6],  # [fz, tx, ty, tz]
                        done_mask=done_flags_batch
                    )
                elif self._step_reward_calc is not None:
                    # Stepwise 奖励（带 shaping）
                    step_total = self._step_reward_calc.compute_step(
                        pos_gpu[:batch_size, :],
                        tgt_tensor,
                        vel_gpu[:batch_size, :],
                        omega_gpu[:batch_size, :],
                        actions[:batch_size, :],
                        done_flags_batch,
                        quat=quat_gpu[:batch_size, :],  # 🔧 修复: 传递 quat 参数
                    )
                    step_reward = step_total
                else:
                    # 退回旧逻辑
                    if self.trajectory_config.get('type') == 'hover':
                        w_pos, w_vel = 2.0, 0.3  # 悬停：更看重精确定点和静止
                    else:
                        w_pos, w_vel = 1.0, 0.1  # 轨迹跟踪：允许一定速度
                    pos_err = pos_gpu[:batch_size, :] - tgt_tensor
                    step_reward = - w_pos * torch.norm(pos_err, dim=1)
                    step_reward -= w_vel * torch.norm(vel_gpu[:batch_size, :], dim=1)
                    act_pen = 1e-7 * torch.sum(actions[:batch_size, :] ** 2, dim=1)
                    step_reward -= act_pen
                    crashed = pos_gpu[:batch_size, 2] < 0.1
                    step_reward[crashed] -= 5.0

                # 调试：记录首末位置误差（使用动态目标）
                if debug_enabled:
                    # 计算当前步的绝对位置误差模长
                    cur_pos_err = torch.norm(pos_gpu[:batch_size, :] - tgt_tensor.view(1, 3), dim=1)
                    if step == 0:
                        first_pos_err = cur_pos_err.detach()[:min(8, batch_size)].cpu()
                    last_pos_err = cur_pos_err.detach()[:min(8, batch_size)].cpu()
                # 累积奖励
                active_mask = (~done_flags_batch).float()
                total_rewards[:batch_size] += step_reward * active_mask
                steps_count[:batch_size] += active_mask
                # 更新批次 done 标志（仅前 batch_size 有效）
                done_flags_batch |= dones[:batch_size]
                done_flags[:batch_size] = done_flags_batch
                if step >= min_steps and done_flags_batch.all():
                    break
            # 额外的 episode 末尾奖励（仅 Stepwise 模式）
            if self._step_reward_calc is not None and not self.use_scg_exact_reward:
                bonus = self._step_reward_calc.finalize()[:batch_size]
                total_rewards[:batch_size] += bonus
            # 在严格无先验模式下：对整集始终零动作的程序施加惩罚
            if self.strict_no_prior and self.zero_action_penalty > 0:
                zero_mask = (~ever_nonzero).float()
                total_rewards[:batch_size] -= self.zero_action_penalty * zero_mask
                if debug_enabled:
                    try:
                        zero_cnt = int((~ever_nonzero).sum().item())
                        print(f"[DebugReward] zero-action programs in batch: {zero_cnt}/{batch_size}")
                    except Exception:
                        pass

            if gpu_batch_token is not None:
                self._gpu_executor.release_batch(gpu_batch_token)
            
            # 🔍 动作幅度统计（诊断用）：计算本批动作输出的平均幅度与最大值
            # 注释掉以减少日志输出噪音
            # if debug_enabled or batch_start == 0:
            #     try:
            #         count = max(1, episode_stats['count'])
            #         avg_fz = episode_stats['sum_fz'] / count
            #         max_fz = episode_stats['max_fz']
            #         avg_tx = episode_stats['sum_tx'] / count
            #         max_tx = episode_stats['max_tx']
            #         print(f"[ActionAmp] Batch{batch_start//programs_per_batch}: avg_fz={avg_fz:.4f}, max_fz={max_fz:.4f}, avg_tx={avg_tx:.6f}, max_tx={max_tx:.6f}")
            #     except Exception:
            #         pass
            
            # 复杂度激励和先验：仅影响训练奖励，不改真实环境奖励
            complexity_rewards = torch.zeros(batch_size, device=self.device)
            if self.complexity_bonus > 0:
                for i in range(batch_size):
                    prog = batch_programs[i]
                    unique_vars = set()
                    for rule in prog:
                        node = rule.get('node', None)
                        if node is not None:
                            vars_in_node = self._extract_variables_from_node(node)
                            unique_vars.update(vars_in_node)
                    num_rules = sum(1 for rule in prog if rule.get('node', None) is not None)
                    bonus = self.complexity_bonus * len(unique_vars) + 0.5 * self.complexity_bonus * num_rules
                    complexity_rewards[i] = bonus
                if debug_enabled:
                    try:
                        print(f"[DebugReward] complexity bonuses: {complexity_rewards[:min(8, batch_size)].cpu().numpy()}")
                    except Exception:
                        pass

            prior_bonus = self._compute_prior_bonus(batch_programs)
            prior_struct = torch.zeros(batch_size, device=self.device)
            prior_stab = torch.zeros(batch_size, device=self.device)
            if prior_bonus is not None:
                # prior_bonus: (total, struct, stab)
                prior_struct = prior_bonus[1]
                prior_stab = prior_bonus[2]

            # 归约
            if self.reward_reduction == 'mean':
                denom = torch.clamp(steps_count[:batch_size], min=1.0)
                batch_scores = (total_rewards[:batch_size] / denom).cpu().numpy().tolist()
            else:
                batch_scores = total_rewards[:batch_size].cpu().numpy().tolist()
            rewards.extend(batch_scores)

            # 调试输出（仅首批 & 开启时）
            if debug_enabled and batch_start == 0:
                try:
                    print("[DebugReward] batch_size={} mean_final_reward={:.4f}".format(
                        batch_size, float(np.mean(batch_scores))))
                    if first_pos_err is not None and last_pos_err is not None:
                        diff = (last_pos_err - first_pos_err).numpy()
                        print("[DebugReward] first_pos_err[:8] =", [f"{x:.3f}" for x in first_pos_err.numpy()])
                        print("[DebugReward] last_pos_err[:8]  =", [f"{x:.3f}" for x in last_pos_err.numpy()])
                        print("[DebugReward] Δpos_err[:8]      =", [f"{x:.3f}" for x in diff])
                except Exception:
                    pass
        
        elapsed = time.time() - start_time
        # 显示原始程序数(未扩展replicas前)
        display_count = num_programs_original if self.replicas_per_program > 1 else num_programs
        # 注释掉详细评估日志，减少输出噪音
        # print(f"[BatchEvaluator] ✅ 评估完成: {display_count} 程序 (×{self.replicas_per_program} replicas), {elapsed:.2f}秒 ({elapsed/display_count*1000:.1f}ms/程序)")
        
        # 先将新评估结果写入缓存（以“单程序”粒度，而非 replicas）
        # rewards 当前长度为 num_programs_original×replicas_per_program（或无replicas时为 num_programs_original）
        # 先得到每个原始程序的平均奖励，用于缓存和后续合并
        per_program_rewards: List[float]
        if self.replicas_per_program > 1:
            per_program_rewards = []
            for i in range(num_programs_original):
                start_idx = i * self.replicas_per_program
                end_idx = start_idx + self.replicas_per_program
                avg_reward = float(np.mean(rewards[start_idx:end_idx]))
                per_program_rewards.append(avg_reward)
        else:
            per_program_rewards = [float(r) for r in rewards]

        # 写入 eval cache
        for local_idx, prog_reward in zip(indices_to_eval, per_program_rewards):
            key = cache_keys[local_idx]
            if key is None:
                continue
            self._eval_cache[key] = float(prog_reward)
        if len(self._eval_cache) > self._eval_cache_limit:
            remove_n = max(1, int(self._eval_cache_limit * 0.2))
            for _ in range(remove_n):
                try:
                    self._eval_cache.pop(next(iter(self._eval_cache)))
                except Exception:
                    break

        # 将缓存命中与新评估结果组合成“仅有效程序”的完整列表
        merged_valid_rewards: List[float] = []
        eval_iter = iter(per_program_rewards)
        for idx in range(len(programs)):
            if idx in cached_rewards:
                merged_valid_rewards.append(cached_rewards[idx])
            else:
                merged_valid_rewards.append(float(next(eval_iter)))

        if len(valid_indices) == total_requested:
            return merged_valid_rewards
        return self._merge_rewards_with_invalid(valid_indices, merged_valid_rewards, invalid_info, total_requested)

    def evaluate_batch_with_metrics(self, programs: List[List[Dict[str, Any]]]) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
        """与 evaluate_batch 类似，但额外返回逐分量奖励汇总（加权后）用于分析/记录。

        Returns:
            rewards_train: 每个程序的训练奖励（含零动作惩罚，对 replicas 取平均后）
            rewards_true: 每个程序的真实奖励（不含惩罚，对 replicas 取平均后）
            metrics: 每个程序的组件字典（同样对 replicas 平均），键包含：
                     ['position_rmse','settling_time','control_effort','smoothness_jerk',
                      'gain_stability','saturation','peak_error','high_freq','finalize_bonus',
                      'zero_action_penalty','structure_prior','stability_prior']
        """
        total_requested = len(programs)
        valid_programs, valid_indices, invalid_info = self._partition_programs_by_constraints(programs)
        if not valid_programs:
            self._log_invalid_programs(invalid_info)
            penalty = [float(HARD_CONSTRAINT_PENALTY)] * total_requested
            metrics = [self._metric_template() for _ in range(total_requested)]
            for idx in invalid_info:
                metrics[idx]['hard_constraint_violation'] = 1.0
            return penalty, penalty[:], metrics
        programs = valid_programs

        # 🔧 镜像展开：如果程序只有 u_tx，则自动生成 u_ty（取反）及 yaw/thrust 稳定器
        programs = [
            self._mirror_expand_single_axis_program(prog) for prog in programs
        ]

        # 初始化环境池
        if self._isaac_env_pool is None:
            self._init_isaac_gym_pool()

        import torch  # type: ignore

        num_programs_original = len(programs)
        # 扩展 replicas
        if self.replicas_per_program > 1:
            programs_expanded = []
            for prog in programs:
                programs_expanded.extend([prog] * self.replicas_per_program)
            programs = programs_expanded

        num_programs = len(programs)
        rewards: List[float] = []  # 训练奖励（含惩罚）
        rewards_true: List[float] = []  # 真实奖励（不含惩罚）
        metrics_all: List[Dict[str, float]] = []  # 与 rewards 顺序一一对应（扩展后）

        start_time = time.time()
        programs_per_batch = max(1, self.isaac_num_envs // self.replicas_per_program)

        for batch_start in range(0, num_programs, programs_per_batch):
            batch_end = min(batch_start + programs_per_batch, num_programs)
            batch_programs = programs[batch_start:batch_end]
            batch_size = len(batch_programs)

            # 轻量级确定性重置：保留环境池，只重置状态 (fast_path版本)
            num_needed = batch_size
            
            # 🔧 计算初始位置：等于 t=0 时的目标位置
            initial_pos_np = self._target_pos(0.0)
            initial_pos_tensor = torch.tensor(initial_pos_np, device=self.device, dtype=torch.float32)
            
            if not self._envs_ready:
                initial_pos_batch = initial_pos_tensor.unsqueeze(0).expand(self.isaac_num_envs, -1).clone()
                obs = self._isaac_env_pool.reset(initial_pos=initial_pos_batch)
                self._envs_ready = True
                self._last_reset_size = self.isaac_num_envs
                self._reset_action_history()
            else:
                env_ids_to_reset = torch.arange(batch_size, dtype=torch.long, device=self.device)
                initial_pos_batch = initial_pos_tensor.unsqueeze(0).expand(batch_size, -1).clone()
                obs = self._isaac_env_pool.reset(env_ids=env_ids_to_reset, initial_pos=initial_pos_batch)
                self._reset_action_history(env_ids_to_reset)

            if self.use_scg_exact_reward and self._scg_reward_calc is not None:
                self._scg_reward_calc.reset(num_envs=batch_size)

            total_rewards = torch.zeros(self.isaac_num_envs, device=self.device)
            done_flags = torch.zeros(self.isaac_num_envs, dtype=torch.bool, device=self.device)
            done_flags_batch = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            # 初始化逐分量计算器（匹配 batch_size）
            if self._step_reward_calc is not None:
                try:
                    weights, ks = get_reward_profile(self.reward_profile) if get_reward_profile else ({}, {})
                    self._step_reward_calc = StepwiseRewardCalculator(weights, ks, dt=self._step_dt, num_envs=batch_size, device=self.device)
                except Exception:
                    self._step_reward_calc = None
            # 记录每个环境累计步数
            steps_count = torch.zeros(self.isaac_num_envs, device=self.device)
            ever_nonzero = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            integral_states = [
                {
                    'err_i_x': 0.0, 'err_i_y': 0.0, 'err_i_z': 0.0,
                    'err_i_roll': 0.0, 'err_i_pitch': 0.0, 'err_i_yaw': 0.0
                }
                for _ in range(batch_size)
            ]
            debug_enabled = bool(int(os.getenv('DEBUG_STEPWISE', '0')))

            # 所有程序统一使用 u_* 直接输出路径（不再依赖 PID 封装）
            gpu_batch_token = None
            controllers = [None for _ in range(batch_size)]
            use_u_flags = [True for _ in range(batch_size)]  # 全部使用直接力/力矩输出
            
            # 🔧 重置每个程序的时间算子状态（ema/delay/diff/rate）
            # 确保每次评估从零状态开始，保证训练与测试一致性
            if reset_program_state is not None:
                for prog in batch_programs:
                    reset_program_state(prog)
            
            # UltraFast 仅在所有程序为常量 set 情况下启用（metrics 评估同理）
            if self.use_fast_path and self._ultra_executor is not None:
                try:
                    if not self._all_programs_const(batch_programs):
                        self._ultra_executor = None
                except Exception:
                    self._ultra_executor = None

            # 🔧 修复: gpu_batch_token 初始化应该在 try/except 块外
            if (self._gpu_executor is not None and self.use_gpu_expression_executor and any(use_u_flags)):
                try:
                    gpu_batch_token = self._gpu_executor.prepare_batch(batch_programs)
                except Exception as gpu_batch_exc:
                    gpu_batch_token = None
                    if batch_start == 0:
                        print(f"[GPUExecutor] ⚠️ metrics批次绑定失败，使用CPU路径: {gpu_batch_exc}")

            max_steps = int(self.duration * float(getattr(self, '_control_freq', 48)))
            min_steps = int(max_steps * self.min_steps_frac)
            bonus_vec = None

            for step in range(max_steps):
                t = step * float(getattr(self, '_control_dt', 1.0/48.0))
                tgt_np = self._target_pos(t)
                tgt_tensor = torch.tensor(tgt_np, device=self.device, dtype=torch.float32)

                actions = torch.zeros((self.isaac_num_envs, 6), device=self.device)
                pos = obs['position'][:batch_size]
                quat = obs['orientation'][:batch_size]
                vel = obs['velocity'][:batch_size]
                omega = obs['angular_velocity'][:batch_size]
                gpu_actions_applied = False
                if gpu_batch_token is not None:
                    try:
                        pos_tensor = self._ensure_tensor(pos)
                        vel_tensor = self._ensure_tensor(vel)
                        omega_tensor = self._ensure_tensor(omega)
                        quat_tensor = self._ensure_tensor(quat)
                        gpu_use_mask = torch.tensor(use_u_flags, dtype=torch.bool, device=self.device)
                        if self._use_gpu_control_loop:
                            gpu_outputs, pos_err_tensor, rpy_tensor = self._gpu_executor.evaluate_from_raw_obs(
                                gpu_batch_token,
                                pos_tensor,
                                vel_tensor,
                                omega_tensor,
                                quat_tensor,
                                tgt_tensor,
                                integral_states,
                                gpu_use_mask,
                                active_mask=(~done_flags_batch)
                            )
                        else:
                            state_tensors, pos_err_tensor, rpy_tensor = self._prepare_gpu_state_tensors(
                                pos_tensor, vel_tensor, omega_tensor, quat_tensor, tgt_tensor, integral_states
                            )
                            gpu_outputs = self._gpu_executor.evaluate(
                                gpu_batch_token,
                                state_tensors,
                                gpu_use_mask,
                                active_mask=(~done_flags_batch)
                            )
                        actions[:batch_size, 2:6] = torch.where(
                            gpu_use_mask.unsqueeze(-1),
                            gpu_outputs,
                            actions[:batch_size, 2:6]
                        )
                        if self.strict_no_prior:
                            nz_mask = (
                                gpu_outputs[:, 0].abs() > 1e-6
                            ) | (
                                gpu_outputs[:, 1].abs() > 1e-8
                            ) | (
                                gpu_outputs[:, 2].abs() > 1e-8
                            ) | (
                                gpu_outputs[:, 3].abs() > 1e-8
                            )
                            ever_nonzero |= (gpu_use_mask & nz_mask)
                        self._update_integral_states(
                            integral_states,
                            pos_err_tensor,
                            rpy_tensor,
                            done_flags_batch,
                            float(getattr(self, '_control_dt', 1.0 / 48.0))
                        )
                        if not self.strict_no_prior:
                            self._apply_pid_controllers(
                                controllers,
                                use_u_flags,
                                actions,
                                step,
                                pos_tensor,
                                quat_tensor,
                                vel_tensor,
                                omega_tensor,
                                tgt_np,
                                integral_states,
                                ever_nonzero,
                                debug_enabled,
                            )
                        gpu_actions_applied = True
                    except Exception as gpu_metrics_exc:
                        gpu_actions_applied = False
                        if batch_start == 0:
                            print(f"[GPUExecutor] ⚠️ metrics step 回退CPU: {gpu_metrics_exc}")

                # 为简化，这里复用 evaluate_batch 的标准快速路径（不展开全部超快路径细节），
                # 但保留正确性：逐程序求值生成 u_*。
                if not gpu_actions_applied:
                    try:
                        from scipy.spatial.transform import Rotation
                    except ImportError:
                        Rotation = None
                    if isinstance(pos, torch.Tensor):
                        pos_np = pos.cpu().numpy(); quat_np = quat.cpu().numpy(); vel_np = vel.cpu().numpy(); omega_np = omega.cpu().numpy()
                    else:
                        pos_np = np.asarray(pos); quat_np = np.asarray(quat); vel_np = np.asarray(vel); omega_np = np.asarray(omega)
                    tgt_batch = np.tile(tgt_np, (batch_size, 1))
                    pe_batch = tgt_batch - pos_np
                    if Rotation is not None:
                        try:
                            rpy_batch = Rotation.from_quat(quat_np).as_euler('XYZ', degrees=False)
                        except Exception:
                            rpy_batch = np.zeros((batch_size, 3), dtype=np.float32)
                    else:
                        rpy_batch = np.zeros((batch_size, 3), dtype=np.float32)

                    for i in range(batch_size):
                        if use_u_flags[i]:
                            pe = pe_batch[i]; rpy = rpy_batch[i]
                            state = {
                                'pos_err_x': float(pe[0]), 'pos_err_y': float(pe[1]), 'pos_err_z': float(pe[2]),
                                'pos_err': float(np.linalg.norm(pe)), 'pos_err_xy': float(np.linalg.norm(pe[:2])), 'pos_err_z_abs': float(abs(pe[2])),
                                'vel_x': float(vel_np[i][0]), 'vel_y': float(vel_np[i][1]), 'vel_z': float(vel_np[i][2]), 'vel_err': float(np.linalg.norm(vel_np[i])),
                                'err_p_roll': float(rpy[0]), 'err_p_pitch': float(rpy[1]), 'err_p_yaw': float(rpy[2]), 'ang_err': float(np.linalg.norm(rpy)), 'rpy_err_mag': float(np.linalg.norm(rpy)),
                                'ang_vel_x': float(omega_np[i][0]), 'ang_vel_y': float(omega_np[i][1]), 'ang_vel_z': float(omega_np[i][2]), 'ang_vel': float(np.linalg.norm(omega_np[i])), 'ang_vel_mag': float(np.linalg.norm(omega_np[i])),
                                'err_i_x': float(integral_states[i]['err_i_x']), 'err_i_y': float(integral_states[i]['err_i_y']), 'err_i_z': float(integral_states[i]['err_i_z']),
                                'err_i_roll': float(integral_states[i]['err_i_roll']), 'err_i_pitch': float(integral_states[i]['err_i_pitch']), 'err_i_yaw': float(integral_states[i]['err_i_yaw']),
                                'err_d_x': float(-vel_np[i][0]), 'err_d_y': float(-vel_np[i][1]), 'err_d_z': float(-vel_np[i][2]), 'err_d_roll': float(-omega_np[i][0]), 'err_d_pitch': float(-omega_np[i][1]), 'err_d_yaw': float(-omega_np[i][2]),
                            }
                            fz, tx, ty, tz = self._eval_program_forces(batch_programs[i], state)
                            actions[i, 2] = float(fz); actions[i, 3] = float(tx); actions[i, 4] = float(ty); actions[i, 5] = float(tz)
                            if self.strict_no_prior:
                                if (abs(fz) > 1e-6) or (abs(tx) > 1e-8) or (abs(ty) > 1e-8) or (abs(tz) > 1e-8):
                                    ever_nonzero[i] = True
                            if not done_flags[i]:
                                dt = float(getattr(self, '_control_dt', 1.0/48.0))
                                integral_states[i]['err_i_x'] += float(pe[0]) * dt
                                integral_states[i]['err_i_y'] += float(pe[1]) * dt
                                integral_states[i]['err_i_z'] += float(pe[2]) * dt
                                integral_states[i]['err_i_roll'] += float(rpy[0]) * dt
                                integral_states[i]['err_i_pitch'] += float(rpy[1]) * dt
                                integral_states[i]['err_i_yaw'] += float(rpy[2]) * dt
                        else:
                            ctrl = controllers[i]
                            if ctrl is not None:
                                rpm, _pos_e, _rpy_e = ctrl.computeControl(
                                    self._control_dt,
                                    cur_pos=pos[i], cur_quat=quat[i], cur_vel=vel[i], cur_ang_vel=omega[i], target_pos=tgt_np,
                                )
                                rpm = np.clip(np.asarray(rpm, dtype=np.float32), 0.0, 25000.0)
                                fz, tx, ty, tz = self._rpm_to_forces_local(rpm)
                                actions[i, 2] = float(fz); actions[i, 3] = float(tx); actions[i, 4] = float(ty); actions[i, 5] = float(tz)
                            if self.strict_no_prior:
                                if (abs(actions[i, 2]) > 1e-6) or (abs(actions[i, 3]) > 1e-8) or (abs(actions[i, 4]) > 1e-8) or (abs(actions[i, 5]) > 1e-8):
                                    ever_nonzero[i] = True

                # 环境步进前应用 MAD 安全壳
                actions = self._apply_output_mad(actions, use_u_flags, batch_size)
                obs, step_rewards_env, dones, infos = self._isaac_env_pool.step(actions)

                tensor_obs = self._isaac_env_pool.get_states_batch()
                pos_t = tensor_obs['pos']
                vel_t = tensor_obs['vel']
                omega_t = tensor_obs['omega']
                quat_t = tensor_obs['quat']  # 姿态四元数
                
                # 计算 Reward
                if self.use_scg_exact_reward and self._scg_reward_calc is not None:
                    # 🎯 精确 SCG reward
                    step_reward = self._scg_reward_calc.compute_step(
                        pos_t[:batch_size, :],
                        vel_t[:batch_size, :],
                        quat_t[:batch_size, :],
                        omega_t[:batch_size, :],
                        tgt_tensor,
                        actions[:batch_size, 2:6],
                        done_mask=done_flags_batch
                    )
                elif self._step_reward_calc is not None:
                    # Stepwise 奖励
                    step_total = self._step_reward_calc.compute_step(
                        pos_t[:batch_size, :],
                        tgt_tensor,
                        vel_t[:batch_size, :],
                        omega_t[:batch_size, :],
                        actions[:batch_size, :],
                        done_flags_batch,
                        quat=quat_t[:batch_size, :],  # 🔧 修复: 传递 quat 参数
                    )
                    step_reward = step_total
                else:
                    pos_err = pos_t[:batch_size, :] - tgt_tensor
                    w_pos, w_vel = (2.0, 0.3) if self.trajectory_config.get('type') == 'hover' else (1.0, 0.1)
                    step_reward = - w_pos * torch.norm(pos_err, dim=1)
                    step_reward -= w_vel * torch.norm(vel_t[:batch_size, :], dim=1)
                    act_pen = 1e-7 * torch.sum(actions[:batch_size, :] ** 2, dim=1)
                    step_reward -= act_pen
                    crashed = pos_t[:batch_size, 2] < 0.1
                    step_reward[crashed] -= 5.0

                active_mask = (~done_flags_batch).float()
                total_rewards[:batch_size] += step_reward * active_mask
                steps_count[:batch_size] += active_mask
                done_flags_batch |= dones[:batch_size]
                done_flags[:batch_size] = done_flags_batch
                if step >= min_steps and done_flags_batch.all():
                    break

            # finalize & 额外奖惩
            if self.use_scg_exact_reward and self._scg_reward_calc is not None:
                # SCG 精确模式：无 finalize bonus
                bonus_vec = torch.zeros(batch_size, device=self.device)
                comp_totals = self._scg_reward_calc.get_components()
            elif self._step_reward_calc is not None:
                bonus = self._step_reward_calc.finalize()[:batch_size]
                total_rewards[:batch_size] += bonus
                bonus_vec = bonus
                comp_totals = self._step_reward_calc.get_component_totals()
            else:
                bonus_vec = torch.zeros(batch_size, device=self.device)
                comp_totals = {k: torch.zeros(batch_size, device=self.device) for k in [
                    'position_rmse','settling_time','control_effort','smoothness_jerk',
                    'gain_stability','saturation','peak_error','high_freq']}

            # 初始化复杂度和先验奖励（metrics模式下默认为0）
            complexity_rewards = torch.zeros(batch_size, device=self.device)
            prior_struct = torch.zeros(batch_size, device=self.device)
            prior_stab = torch.zeros(batch_size, device=self.device)

            # 🔍 分离真实奖励和训练奖励
            # reward_true: 纯环境奖励（仅 SCG 代价，不含任何 shaping）
            # reward_train: 训练信号（在真实奖励基础上叠加复杂度、先验、零动作惩罚等）
            batch_rewards_true = total_rewards[:batch_size].clone()
            batch_rewards_train = total_rewards[:batch_size].clone()
            # 复杂度和先验：只加到训练奖励，不改真实奖励
            batch_rewards_train += complexity_rewards
            batch_rewards_train += prior_struct
            batch_rewards_train += prior_stab
            
            # 零动作惩罚：仅加到训练奖励上
            zero_penalty_applied = torch.zeros(batch_size, device=self.device)
            if self.strict_no_prior and self.zero_action_penalty > 0:
                zero_mask = (~ever_nonzero).float()
                zero_penalty_applied = self.zero_action_penalty * zero_mask
                batch_rewards_train -= zero_penalty_applied

            # 归约（对两个奖励分别处理）
            if self.reward_reduction == 'mean':
                denom = torch.clamp(steps_count[:batch_size], min=1.0)
                batch_scores_true = (batch_rewards_true / denom).cpu().numpy().tolist()
                batch_scores_train = (batch_rewards_train / denom).cpu().numpy().tolist()
            else:
                batch_scores_true = batch_rewards_true.cpu().numpy().tolist()
                batch_scores_train = batch_rewards_train.cpu().numpy().tolist()
            
            # rewards列表存储训练奖励（用于NN训练）
            rewards.extend(batch_scores_train)
            # rewards_true列表存储真实奖励（用于保存、输出、对比）
            rewards_true.extend(batch_scores_true)

            # 逐环境组件字典：只导出 SCG 对齐指标
            for i in range(batch_size):
                d: Dict[str, float] = {}
                # 直接从 SCG 组件中读取 state_cost / action_cost
                state_tensor = comp_totals.get('state_cost')
                action_tensor = comp_totals.get('action_cost')
                d['state_cost'] = float(state_tensor[i].item()) if state_tensor is not None else 0.0
                d['action_cost'] = float(action_tensor[i].item()) if action_tensor is not None else 0.0
                metrics_all.append(d)

        elapsed = time.time() - start_time
        display_count = num_programs_original if self.replicas_per_program > 1 else num_programs
        # 注释掉详细评估日志，减少输出噪音
        # print(f"[BatchEvaluator] ✅ 评估完成: {display_count} 程序 (×{self.replicas_per_program} replicas), {elapsed:.2f}秒 ({elapsed/display_count*1000:.1f}ms/程序)")

        # 汇总 replicas：对每个原始程序的组件逐键取平均
        if self.replicas_per_program > 1:
            averaged_rewards: List[float] = []
            averaged_rewards_true: List[float] = []
            averaged_metrics: List[Dict[str, float]] = []
            for i in range(num_programs_original):
                start_idx = i * self.replicas_per_program
                end_idx = start_idx + self.replicas_per_program
                avg_reward = float(np.mean(rewards[start_idx:end_idx]))
                avg_reward_true = float(np.mean(rewards_true[start_idx:end_idx]))
                averaged_rewards.append(avg_reward)
                averaged_rewards_true.append(avg_reward_true)
                # 平均组件
                keys = list(metrics_all[start_idx].keys())
                avg_dict = {k: float(np.mean([metrics_all[j][k] for j in range(start_idx, end_idx)])) for k in keys}
                averaged_metrics.append(avg_dict)
            if len(valid_indices) == total_requested:
                return averaged_rewards, averaged_rewards_true, averaged_metrics
            return self._merge_metrics_with_invalid(
                valid_indices,
                averaged_rewards,
                averaged_rewards_true,
                averaged_metrics,
                invalid_info,
                total_requested,
            )

        if len(valid_indices) == total_requested:
            return rewards, rewards_true, metrics_all
        return self._merge_metrics_with_invalid(
            valid_indices,
            rewards,
            rewards_true,
            metrics_all,
            invalid_info,
            total_requested,
        )

    def evaluate_single_with_metrics(self, program: List[Dict[str, Any]]) -> Tuple[float, float, Dict[str, float]]:
        """评估单个程序（支持 replicas），返回训练奖励、真实奖励与组件字典。
        
        Returns:
            reward_train: 训练奖励（含惩罚）
            reward_true: 真实奖励（不含惩罚）
            components: 组件字典
        """
        rewards_train, rewards_true, metrics = self.evaluate_batch_with_metrics([program])
        return rewards_train[0], rewards_true[0], metrics[0]
    
    def _compute_action_from_program(self, program: List[Dict[str, Any]], 
                                      obs: np.ndarray, step: int) -> np.ndarray:
        """
        从程序计算控制输入（简化版）
        
        Args:
            program: DSL程序规则列表
            obs: 观测 [obs_dim]
            step: 当前步数
        
        Returns:
            action: [4] = [thrust, roll_rate, pitch_rate, yaw_rate]
        
        Note: 现在所有程序直接输出 u_fz/u_tx/u_ty/u_tz，不再使用 PID 封装
        """
        # 当前返回悬停控制（占位符）
        # 实际应该：
        # 1. 从obs提取状态（位置、速度等）
        # 2. 计算轨迹目标点
        # 3. 使用program规则计算PID输出
        # 4. 转换为电机指令
        
        return np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def evaluate_single(self, program: List[Dict[str, Any]]) -> float:
        """评估单个程序：可并行复制多个副本并取平均，提升GPU利用率/稳定性"""
        # evaluate_batch 会自动处理 replicas，不需要在这里复制
        rewards = self.evaluate_batch([program])
        return float(np.mean(rewards))
    
    def evaluate_batch_programs(self, programs: List[List[Dict[str, Any]]]) -> List[float]:
        """批量评估多个程序（用于 MCTS 叶节点并行化）
        
        这个方法专门为 MCTS 并行化设计，支持一次评估多个不同的程序。
        每个程序仍然使用完整的 isaac_num_envs 环境评估，但通过连续调用
        减少 Python/Isaac Gym 的开销。
        
        Args:
            programs: 程序列表，每个程序是 List[Dict[str, Any]]
            
        Returns:
            rewards_train: 训练奖励列表（含惩罚）
            
        Example:
            >>> programs = [program1, program2, program3]
            >>> rewards = evaluator.evaluate_batch_programs(programs)
            >>> # rewards = [-1.5, -2.3, -0.8]
        """
        if not programs:
            return []
        
        # 连续评估每个程序（仍使用完整的 isaac_num_envs）
        # 注意：这里不是真正的"并行"，而是减少调用开销
        rewards_train = []
        for program in programs:
            reward_train, _, _ = self.evaluate_single_with_metrics(program)
            rewards_train.append(reward_train)
        
        return rewards_train


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试Isaac Gym批量评估器")
    print("=" * 80)
    
    if not ISAAC_GYM_AVAILABLE:
        print("❌ Isaac Gym未安装，无法测试")
        exit(1)
    
    trajectory = {
        'type': 'figure8',
        'initial_xyz': [0, 0, 1.0],
        'params': {'A': 0.8, 'B': 0.5, 'period': 12}
    }
    
    evaluator = BatchEvaluator(
        trajectory_config=trajectory,
        duration=5,
        isaac_num_envs=64,
        device='cuda:0'
    )
    
    # 创建测试程序
    test_programs = [
        [{'name': 'rule1', 'condition': None, 'action': [], 'multiplier': [1, 1, 1]}]
    ] * 8
    
    print(f"\n评估 {len(test_programs)} 个程序...")
    rewards = evaluator.evaluate_batch(test_programs)
    print(f"奖励: {[f'{r:.3f}' for r in rewards]}")
    print("\n✅ 测试完成")

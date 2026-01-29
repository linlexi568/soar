"""在线训练主循环 - AlphaZero式程序合成

从零开始训练：NN随机初始化 → MCTS搜索 → 收集样本 → 更新NN → 循环
"""
from __future__ import annotations

# 【修复Python 3.13兼容性】禁用PyTorch编译功能
import os
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import argparse, time, json, random, os
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import numpy as np

# 导入现有模块 - 简化导入,只导入必需组件
import sys, pathlib
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Ensure Isaac Gym python bindings are importable (repo vendor path)
try:
    _REPO_ROOT = _PKG_ROOT.parent
    _GYM_PY = _REPO_ROOT / 'isaacgym' / 'python'
    if _GYM_PY.exists() and str(_GYM_PY) not in sys.path:
        sys.path.insert(0, str(_GYM_PY))
    # 提前导入 isaacgym，确保其先于 torch 导入
    try:
        from isaacgym import gymapi  # type: ignore
    except Exception:
        pass
except Exception:
    pass

# 直接导入必需模块（避免循环依赖）
from mcts_training.mcts import MCTS_Agent, MCTSNode
from mcts_training.policy.policy_nn import EDIT_TYPES  # PolicyValueNNLarge 已移除 (固定特征网络弃用)
from core.ast_pipeline import to_ast_program, has_u_set, to_serializable_dict  # AST-first pipeline

# GNN v2模块（分层架构）
try:
    from models.gnn_features import ast_to_pyg_graph, batch_programs_to_graphs
    from models.gnn_policy_nn_v2 import create_gnn_policy_value_net_v2
    from torch_geometric.data import Batch as PyGBatch
    GNN_V2_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] GNN v2模块不可用: {e}")

# Ranking Value Network（用于自适应奖励学习，打破平坦奖励困境）
try:
    from models.ranking_value_net import (
        RankingValueNet, PairwiseRankingBuffer,
        compute_ranking_loss, generate_program_pairs,
        setup_ranking_training, train_ranking_step
    )
    RANKING_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Ranking网络不可用: {e}")
    RANKING_AVAILABLE = False
    GNN_V2_AVAILABLE = False
    ast_to_pyg_graph = None
    batch_programs_to_graphs = None
    create_gnn_policy_value_net_v2 = None  # type: ignore
    PyGBatch = None

# 导入batch_evaluation（可能需要Isaac Gym）；确保在导入 torch 之前尝试导入 isaacgym
try:
    from utils.batch_evaluation import BatchEvaluator
    BATCH_EVAL_AVAILABLE = True
except Exception as e:
    print(f"[Warning] BatchEvaluator不可用: {e}")
    BATCH_EVAL_AVAILABLE = False
    BatchEvaluator = None  # type: ignore

try:
    from utilities.trajectory_presets import get_scg_trajectory_config
except Exception as e:
    raise ImportError(f"无法导入 safe-control-gym 轨迹助手: {e}")

try:
    from utils.prior_scoring import PRIOR_PROFILES
except Exception:
    PRIOR_PROFILES = {"none": (0.0, 0.0)}

try:
    from utils.program_constraints import validate_program, HARD_CONSTRAINT_PENALTY
except Exception:
    try:
        from program_constraints import validate_program, HARD_CONSTRAINT_PENALTY  # type: ignore
    except Exception:
        def validate_program(_program):  # type: ignore
            return True, ""
        HARD_CONSTRAINT_PENALTY = -1e6  # type: ignore

# 现在再导入 torch 及其子模块，避免破坏 isaacgym 的导入顺序要求
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 导入serialization
try:
    from core.serialization import save_program_json as _save_prog
    def save_program_json(program, path, meta=None):  # type: ignore
        _save_prog(program, path, meta=meta)
except Exception:
    def save_program_json(program, path, meta=None):  # type: ignore
        import json, os, time
        # 简化版保存（不包含节点对象）
        simplified = []
        for rule in program:
            simple_rule = {
                'name': rule.get('name', 'rule'),
                'multiplier': rule.get('multiplier', [1.0, 1.0, 1.0])
            }
            simplified.append(simple_rule)
        
        payload = {'rules': simplified, 'note': 'Simplified format'}
        if meta:
            payload['meta'] = meta
        payload.setdefault('meta', {})['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)


def _normalize_constants_for_hash(obj):
    """递归替换所有浮点常数为占位符，用于结构哈希
    
    目的：让结构相同但参数不同的程序共享同一个GNN缓存
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == 'value' and isinstance(v, (int, float)):
                # 常数值统一替换为占位符
                result[k] = '<CONST>'
            else:
                result[k] = _normalize_constants_for_hash(v)
        return result
    elif isinstance(obj, list):
        return [_normalize_constants_for_hash(item) for item in obj]
    elif isinstance(obj, (int, float)):
        return '<CONST>'
    else:
        return obj


def get_program_hash(program, ignore_constants=True):
    """生成程序的稳定哈希值用于缓存。
    
    使用AST序列化后的JSON（排序键）计算blake2s，避免内存地址导致的伪差异，
    极大提高缓存命中率。
    
    Args:
        program: 程序表示（可以是DSL程序、AST等）
        ignore_constants: 是否忽略常数值（仅保留结构），默认True
                         True时，所有常数值替换为'<CONST>'，大幅提高BO场景下的缓存命中率
        
    Returns:
        str: 程序的哈希值（16进制字符串）
    """
    try:
        import json, hashlib
        from core.serialization import to_serializable_dict
        serial = to_serializable_dict(program)  # {'rules': ...}
        
        # 🚀 关键优化：忽略常数值，只基于结构哈希
        if ignore_constants:
            serial = _normalize_constants_for_hash(serial)
        
        s = json.dumps(serial, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.blake2s(s.encode('utf-8')).hexdigest()
    except Exception:
        # 回退：使用字符串表示（尽量稳定）；失败则使用id（最差情况）
        try:
            return str(program)
        except Exception:
            return str(id(program))


class ReplayBuffer:
    """经验回放缓冲区（支持固定特征和GNN图数据）"""
    
    def __init__(self, capacity: int = 50000, use_gnn: bool = False):
        self.capacity = capacity
        self.use_gnn = use_gnn
        self.buffer = deque(maxlen=capacity)
    
    def push(self, sample: Dict[str, Any]):
        """添加样本
        
        GNN模式: sample = {'graph': PyG Data, 'policy_target': tensor}
        注意：已移除value_target，只训练policy
        """
        self.buffer.append(sample)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """随机采样"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class OnlineTrainer:
    """在线训练器 - AlphaZero范式"""
    
    def __init__(self, args):
        self.args = args
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] 使用设备: {self.device}")
        
        # 统一使用 GNN v2，若不可用直接报错终止（移除固定特征回退）
        if not GNN_V2_AVAILABLE:
            raise ImportError("GNN v2 模块不可用，已移除固定特征网络回退，请安装 torch-geometric 等依赖。")
        self.use_gnn = True
        
        # 初始化NN（GNN统一使用v2分层架构）
        gnn_structure_hidden = getattr(args, 'gnn_structure_hidden', 256)
        gnn_structure_layers = getattr(args, 'gnn_structure_layers', 5)
        gnn_structure_heads = getattr(args, 'gnn_structure_heads', 8)
        gnn_feature_layers = getattr(args, 'gnn_feature_layers', 3)
        gnn_feature_heads = getattr(args, 'gnn_feature_heads', 8)
        gnn_dropout = getattr(args, 'gnn_dropout', 0.1)
        
        print(f"[Trainer] 使用 GNN v2 (Hierarchical Dual) 分层网络")
        print(f"[Trainer] GNN架构: structure({gnn_structure_hidden}d×{gnn_structure_layers}L×{gnn_structure_heads}H), "
              f"feature({gnn_feature_layers}L×{gnn_feature_heads}H), dropout={gnn_dropout}")
        
        self.nn_model = create_gnn_policy_value_net_v2(
            node_feature_dim=24,
            policy_output_dim=len(EDIT_TYPES),
            structure_hidden=gnn_structure_hidden,
            structure_layers=gnn_structure_layers,
            structure_heads=gnn_structure_heads,
            feature_layers=gnn_feature_layers,
            feature_heads=gnn_feature_heads,
            dropout=gnn_dropout
        ).to(self.device)
        
        # 禁用torch compile避免Python 3.13兼容性问题
        try:
            import os
            os.environ['PYTORCH_JIT'] = '0'
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
        except Exception:
            pass
        
        try:
            self.optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=args.learning_rate,
                weight_decay=1e-4
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # 如果标准Adam失败，尝试手动创建
            print(f"[Warning] Adam初始化失败，使用简化版: {e}")
            self.optimizer = optim.SGD(
                self.nn_model.parameters(),
                lr=args.learning_rate,
                momentum=0.9
            )
        
        print(f"[Trainer] NN初始化完成 (参数: {sum(p.numel() for p in self.nn_model.parameters())})")
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=args.replay_capacity, use_gnn=self.use_gnn)

        # 先验配置（结构/稳定性）
        prior_profile_key = getattr(args, 'prior_profile', 'none')
        preset_structure, preset_stability = PRIOR_PROFILES.get(prior_profile_key, PRIOR_PROFILES.get('none', (0.0, 0.0)))
        structure_override = getattr(args, 'structure_prior_weight', None)
        stability_override = getattr(args, 'stability_prior_weight', None)
        self.structure_prior_weight = float(preset_structure if structure_override is None else structure_override)
        self.stability_prior_weight = float(preset_stability if stability_override is None else stability_override)
        print(f"[Trainer] 先验配置: profile={prior_profile_key} => structure={self.structure_prior_weight:.3f}, stability={self.stability_prior_weight:.3f}")
        
        # 调试：逐步奖励与零动作统计
        if getattr(args, 'debug_rewards', False):
            os.environ['DEBUG_STEPWISE'] = '1'
        if not BATCH_EVAL_AVAILABLE:
            raise RuntimeError("BatchEvaluator 不可用：请确保 Isaac Gym 及相关依赖已正确安装。项目已移除 DummyEvaluator 回退，所有训练必须使用真实奖励。")

        print("[Trainer] 使用 BatchEvaluator（真实 Isaac Gym 奖励）")
        self.evaluator = BatchEvaluator(
            trajectory_config=self._build_trajectory(),
            duration=args.duration,
            isaac_num_envs=args.isaac_num_envs,
            device=str(self.device),
            replicas_per_program=getattr(args, 'eval_replicas_per_program', 1),
            min_steps_frac=getattr(args, 'min_steps_frac', 0.0),
            reward_reduction=getattr(args, 'reward_reduction', 'sum'),
            reward_profile=getattr(args, 'reward_profile', 'safe_control_tracking'),  # SCG-only reward profile
            use_scg_exact_reward=True,
            strict_no_prior=True,  # 统一使用直接u_*控制（不依赖内置PID框架）
            zero_action_penalty=float(getattr(args, 'zero_action_penalty', 0.0)),  # 参数化零动作惩罚
            complexity_bonus=0.0,  # AlphaZero哲学：让NN自己学习复杂度权衡
            use_fast_path=getattr(args, 'use_fast_path', False),
            use_gpu_expression_executor=not getattr(args, 'disable_gpu_expression', False),
            action_scale_multiplier=float(getattr(args, 'action_scale_multiplier', 1.0)),  # 动作缩放系数
            structure_prior_weight=self.structure_prior_weight,
            stability_prior_weight=self.stability_prior_weight,
            enable_output_mad=getattr(args, 'enable_output_mad', True),
            mad_min_fz=float(getattr(args, 'mad_min_fz', 0.0)),
            mad_max_fz=float(getattr(args, 'mad_max_fz', 7.5)),
            mad_max_xy=float(getattr(args, 'mad_max_xy', 0.12)),
            mad_max_yaw=float(getattr(args, 'mad_max_yaw', 0.04)),
            mad_max_delta_fz=float(getattr(args, 'mad_max_delta_fz', 1.5)),
            mad_max_delta_xy=float(getattr(args, 'mad_max_delta_xy', 0.03)),
            mad_max_delta_yaw=float(getattr(args, 'mad_max_delta_yaw', 0.02)),
            enable_bayesian_tuning=getattr(args, 'enable_bayesian_tuning', False),
            bo_batch_size=getattr(args, 'bo_batch_size', 50),
            bo_iterations=getattr(args, 'bo_iterations', 3),
            # 参数范围兜底：如果节点自身没有 min/max，则使用此处默认值
            bo_param_ranges={'default': (-3.0, 3.0)}
        )
        
        # 统计
        self.iteration = 0
        self.best_reward = -float('inf')
        self.best_program = None
        self.best_program_copy = None  # 🔒 深拷贝保护,防止cleanup_tree清理
        self.training_stats = []
        self._mcts_stats = {}  # MCTS性能统计
        
        # 🔄 异步训练支持
        self.async_training = getattr(args, 'async_training', False)
        self.async_trainer = None  # 稍后初始化
        
        # � 三合一优化开关
        self.enable_ranking_mcts_bias = getattr(args, 'enable_ranking_mcts_bias', True)
        self.ranking_bias_beta = getattr(args, 'ranking_bias_beta', 0.3)
        self.enable_value_head = getattr(args, 'enable_value_head', True)
        self.enable_ranking_reweight = getattr(args, 'enable_ranking_reweight', True)
        self.ranking_reweight_beta = getattr(args, 'ranking_reweight_beta', 0.2)
        
        if self.enable_ranking_mcts_bias:
            print(f"[Trainer] ✅ Ranking→MCTS偏置已启用 (beta={self.ranking_bias_beta})")
        if self.enable_value_head:
            print(f"[Trainer] ✅ Value头辅助训练已启用（纯训练信号，MCTS仍用真实仿真）")
        if self.enable_ranking_reweight:
            print(f"[Trainer] ✅ Ranking→Policy重加权已启用 (beta={self.ranking_reweight_beta})")

        # Progressive Widening 开关（可选完全放开树宽）
        self.disable_progressive_widening = bool(getattr(args, 'disable_progressive_widening', False))
        if self.disable_progressive_widening:
            print("[Trainer] ⚠️ Progressive Widening 已禁用：节点将直接按全部可变异数扩展")
        
        # 🚀 悬停推力约束配置（Hover Thrust Constraint）
        # 强制 u_fz = hover_thrust + delta，确保无人机始终有最小升力
        self._enforce_hover_thrust = getattr(args, 'enforce_hover_thrust', True)
        self._hover_thrust_value = float(getattr(args, 'hover_thrust_value', 0.265))
        self._hover_thrust_min = float(getattr(args, 'hover_thrust_min', 0.20))
        self._hover_thrust_max = float(getattr(args, 'hover_thrust_max', 0.35))
        self._hover_delta_max = float(getattr(args, 'hover_delta_max', 2.0))
        if self._enforce_hover_thrust:
            print(f"[Trainer] 🚁 悬停推力约束已启用: hover={self._hover_thrust_value:.3f}N [{self._hover_thrust_min:.2f}, {self._hover_thrust_max:.2f}], delta_max={self._hover_delta_max:.1f}N")
        else:
            print(f"[Trainer] ⚠️ 悬停推力约束已禁用（允许程序输出零推力）")
        
        # 🏆 精英程序池 (Elite Archive) - 保留Top-K最优程序
        self.elite_archive = []  # [(reward, program_copy, iter_idx), ...]
        self.elite_archive_size = getattr(args, 'elite_archive_size', 100)  
        print(f"[Trainer] 🏆 精英程序池: 保留Top-{self.elite_archive_size}最优程序")

        # MCTS 搜索参数对外封闭：固定为内部常量（仅保留“模拟次数”可调）
        # 这些参数不通过 CLI 暴露，确保“只调 NN”策略
        self._exploration_weight = 2.5
        self._puct_c = 1.5
        self._max_depth = getattr(args, 'max_depth', 3)  # 从命令行读取，默认3
        # 注意：已移除value head，全部使用真实仿真
        # Dirichlet / 温度探索参数（内部固定 + 退火日程）
        # 🔥 Meta-RL 或启发式衰减参数配置
        self.use_meta_rl = getattr(args, 'use_meta_rl', False)
        self.meta_rl_controller = None
        
        if self.use_meta_rl:
            # 加载 Meta-RL RNN 控制器
            from meta_rl.controller import MetaRLController
            meta_ckpt = getattr(args, 'meta_rl_checkpoint', 'meta_rl/checkpoints/meta_policy.pt')
            print(f"[Trainer] 🧠 启用 Meta-RL 动态调参 (模型: {meta_ckpt})")
            self.meta_rl_controller = MetaRLController(checkpoint_path=meta_ckpt, device=self.device)
            # Meta-RL 模式下初始值由控制器决定
            self._root_dirichlet_eps = 0.25
            self._root_dirichlet_alpha = 0.30
        else:
            # 启发式衰减模式：支持命令行参数覆盖
            if getattr(args, 'root_dirichlet_eps_init', None) is not None:
                # 用户指定了启发式参数
                self._root_dirichlet_eps_init = float(args.root_dirichlet_eps_init)
                self._root_dirichlet_eps_final = float(getattr(args, 'root_dirichlet_eps_final', self._root_dirichlet_eps_init))
                self._root_dirichlet_alpha_init = float(getattr(args, 'root_dirichlet_alpha_init', args.root_dirichlet_alpha))
                self._root_dirichlet_alpha_final = float(getattr(args, 'root_dirichlet_alpha_final', self._root_dirichlet_alpha_init))
                self._root_dirichlet_decay_iters = int(getattr(args, 'heuristic_decay_window', 200))
                print(f"[Trainer] 📉 启发式退火: eps={self._root_dirichlet_eps_init:.2f}→{self._root_dirichlet_eps_final:.2f}, alpha={self._root_dirichlet_alpha_init:.2f}→{self._root_dirichlet_alpha_final:.2f} (窗口={self._root_dirichlet_decay_iters})")
            else:
                # 使用内部默认值
                self._root_dirichlet_eps_init = 0.60
                self._root_dirichlet_eps_final = 0.15
                self._root_dirichlet_alpha_init = 0.50
                self._root_dirichlet_alpha_final = 0.30
                self._root_dirichlet_decay_iters = 600
                print(f"[Trainer] 📉 默认退火日程: eps={self._root_dirichlet_eps_init:.2f}→{self._root_dirichlet_eps_final:.2f}, alpha={self._root_dirichlet_alpha_init:.2f}→{self._root_dirichlet_alpha_final:.2f}")
            
            self._root_dirichlet_eps = self._root_dirichlet_eps_init
            self._root_dirichlet_alpha = self._root_dirichlet_alpha_init
        # 温度退火日程：从高温（探索）逐步降至低温（利用）
        self._policy_temperature_init = 2.0  # 🔧 提高初始温度：1.5→2.0 - 更强探索
        self._policy_temperature_final = 0.8  # 🔧 提高最终温度：0.5→0.8 - 保持探索性
        self._policy_temperature_decay_iters = 500  # 500轮内完成退火
        self._policy_temperature = self._policy_temperature_init
        print(
            f"[Trainer] MCTS参数已封闭: exploration_weight=2.5, puct_c=1.5, max_depth={self._max_depth}, "
            f"root_dirichlet=(eps={self._root_dirichlet_eps}, alpha={self._root_dirichlet_alpha})；仅 --mcts-simulations 可调；全部使用真实仿真"
        )
        print(f"[Trainer] 温度退火: {self._policy_temperature_init:.2f} → {self._policy_temperature_final:.2f} (over {self._policy_temperature_decay_iters} iters)")
        # 根节点先验覆盖率阈值（用于自适应最小分支控制，避免手工指定固定分支数）
        self._root_prior_coverage_tau = 0.80
        print(f"[Trainer] Root最小分支自适应：先验累计覆盖率 τ={self._root_prior_coverage_tau:.2f}")
        # NN 参数校验：记录一次参数校验和用于后续微小变更观测（不影响训练）
        self._last_param_checksum = self._compute_param_checksum()

        # 一元原语参数网格课程：先粗后细，提升 prior 召回
        self._unary_grid_stage1_iters = 200  # 粗网格阶段
        self._unary_grid_stage2_iters = 600  # 过渡阶段

        # 精英程序根种子：适度利用历史最优，提升 prior 复用
        self._elite_seed_prob = 0.25
        self._elite_seed_topk = 5
        self._elite_seed_delay = 20  # 至少积累若干轮后再启用
        
        # 🚀 Ranking Value Network（自适应奖励学习，解决平坦奖励问题）
        self.use_ranking = getattr(args, 'use_ranking', True) and RANKING_AVAILABLE
        if self.use_ranking:
            print(f"[Trainer] 🔥 启用 Ranking Value Network (Ranking Policy Gradient)")
            self.ranking_net, self.ranking_buffer, self.ranking_optimizer = setup_ranking_training(
                gnn_model=self.nn_model,  # 传递GNN模型
                device=self.device,
                learning_rate=getattr(args, 'ranking_lr', 1e-3),
                embed_dim=gnn_structure_hidden  # 使用 GNN 的实际 hidden size
            )
            # 混合系数：逐步从MCTS value过渡到ranking value
            self.ranking_blend_factor = float(getattr(args, 'ranking_blend_init', 0.3))
            self.ranking_blend_max = float(getattr(args, 'ranking_blend_max', 0.8))
            self.ranking_blend_warmup_iters = int(getattr(args, 'ranking_blend_warmup', 100))
            print(f"[Trainer] Ranking混合: 初始={self.ranking_blend_factor:.2f} → 最大={self.ranking_blend_max:.2f} (warmup={self.ranking_blend_warmup_iters}轮)")
            # Ranking 样本质量控制：仅保留奖励差足够大的样本对，降低噪声（内部常量，不暴露CLI）
            self._ranking_min_delta = 0.05
            print(f"[Trainer] Ranking对过滤: |Δreward| ≥ {self._ranking_min_delta:.2f}")
        else:
            self.ranking_net = None
            self.ranking_buffer = None
            self.ranking_optimizer = None
            print(f"[Trainer] ⚠️ Ranking网络未启用 (use_ranking={getattr(args, 'use_ranking', True)}, available={RANKING_AVAILABLE})")
    
    def _build_trajectory(self) -> Dict[str, Any]:
        """构建与 safe-control-gym 对齐的轨迹配置。
        
        起点规范 (t=0):
        - Square:  [0, 0, 1]   (中心，先向 +y 移动)
        - Circle:  [R, 0, 1]   (圆周右侧，R=0.9时为 [0.9, 0, 1])
        - Figure8: [0, 0, 1]   (中心)
        - Hover:   center      (悬停点)
        """
        traj_cfg = get_scg_trajectory_config(self.args.traj)
        params = dict(traj_cfg.params)
        
        # 计算 t=0 时刻轨迹上的位置作为初始位置
        initial_xyz = self._compute_trajectory_start(traj_cfg)

        if traj_cfg.task == 'hover':
            import random as _r
            curriculum = getattr(self.args, 'curriculum_mode', 'none')
            stage = getattr(self, '_curriculum_stage', 1)
            if curriculum != 'none':
                if stage == 1:
                    initial_xyz = list(traj_cfg.center)
                else:
                    amp_xy = 0.2 if stage == 2 else 0.5
                    amp_z = 0.1 if stage == 2 else 0.3
                    initial_xyz = [
                        traj_cfg.center[0] + _r.uniform(-amp_xy, amp_xy),
                        traj_cfg.center[1] + _r.uniform(-amp_xy, amp_xy),
                        traj_cfg.center[2] + _r.uniform(-amp_z, amp_z),
                    ]
            else:
                initial_xyz = [
                    traj_cfg.center[0] + _r.uniform(-0.5, 0.5),
                    traj_cfg.center[1] + _r.uniform(-0.5, 0.5),
                    traj_cfg.center[2] + _r.uniform(-0.3, 0.3),
                ]

        return {
            'type': traj_cfg.task,
            'initial_xyz': initial_xyz,
            'params': params,
        }
    
    def _compute_trajectory_start(self, traj_cfg) -> list:
        """计算 t=0 时刻轨迹上的位置。
        
        这确保无人机从轨迹的起点开始，而不是轨迹的中心。
        """
        from utilities.trajectory_presets import scg_position
        
        # 使用 scg_position 计算 t=0 时刻的位置
        pos_t0 = scg_position(traj_cfg.task, t=0.0, params=traj_cfg.params, center=traj_cfg.center)
        return pos_t0.tolist()

    def _compute_param_checksum(self) -> float:
        """计算模型参数的简单校验和（L2范数求和），用于观测参数是否发生更新。"""
        with torch.no_grad():
            s = 0.0
            for p in self.nn_model.parameters():
                if p is not None and p.requires_grad:
                    try:
                        s += float(p.data.norm(2).item())
                    except Exception:
                        pass
            return float(s)
    
    def _generate_random_program(self) -> List[Dict[str, Any]]:
        """生成随机初始程序"""
        # 使用MCTS的随机生成逻辑
        mcts = MCTS_Agent(
            evaluation_function=lambda p: 0.0,  # 占位符
            dsl_variables=['pos_err', 'vel_err'],
            dsl_constants=[0.0, 1.0],
            dsl_operators=['+', '-', '*'],
            structure_prior_weight=self.structure_prior_weight,
            stability_prior_weight=self.stability_prior_weight
        )
        # 单轴搜索：默认仅 Roll 通道，避免空间爆炸
        mcts._active_channels = ['u_tx']
        return mcts._generate_random_segmented_program()
    
    def _load_program_from_json(self, path: str) -> Optional[List[Dict[str, Any]]]:
        """从 JSON 文件加载程序（用于 warm start）"""
        try:
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            
            # 尝试提取 rules 字段
            if isinstance(data, dict) and 'rules' in data:
                rules = data['rules']
            elif isinstance(data, list):
                rules = data
            else:
                print(f"[Warning] 无法解析程序文件格式: {path}")
                return None
            
            # 简单验证
            if not isinstance(rules, list) or len(rules) == 0:
                print(f"[Warning] 程序文件为空或格式错误: {path}")
                return None
            
            print(f"[Trainer] ✅ 从 {path} 加载了 {len(rules)} 条规则")
            return rules
        except Exception as e:
            print(f"[Warning] 加载程序失败: {e}")
            return None
    
    def _quick_action_features(self, program: List[Dict[str, Any]]) -> List[float]:
        """快速提取程序的动作幅度特征（用于Ranking NN）
        
        简化实现：返回程序结构统计作为代理特征
        避免导入program_executor（可能有循环依赖问题）
        
        Returns:
            [fz_mean, fz_std, fz_max, tx_mean, tx_std, tx_max]
            实际返回: [num_rules, num_vars, max_depth, 0, 0, 0]（结构代理）
        """
        try:
            # 统计程序结构特征作为动作幅度的代理
            # 假设：复杂程序 → 更多规则/变量 → 更大动作幅度
            num_rules = len([r for r in program if r.get('node') is not None])
            
            # 统计唯一变量数
            unique_vars = set()
            def extract_vars(node):
                if node is None:
                    return
                if isinstance(node, dict):
                    if node.get('type') == 'variable':
                        unique_vars.add(node.get('name', ''))
                    for child in ['left', 'right', 'condition', 'true_branch', 'false_branch']:
                        if child in node:
                            extract_vars(node[child])
            
            for rule in program:
                extract_vars(rule.get('node'))
            
            num_vars = len(unique_vars)
            
            # 计算最大深度（复杂度指标）
            def max_depth(node):
                if node is None or not isinstance(node, dict):
                    return 0
                depths = []
                for child in ['left', 'right', 'condition', 'true_branch', 'false_branch']:
                    if child in node:
                        depths.append(max_depth(node[child]))
                return 1 + max(depths) if depths else 1
            
            depths = [max_depth(r.get('node')) for r in program if r.get('node') is not None]
            max_d = max(depths) if depths else 0
            
            # 返回结构特征作为动作特征的代理
            # [规则数, 变量数, 最大深度, 0, 0, 0]
            # 网络可以学习："更复杂的程序通常有更大动作"
            return [
                float(num_rules) / 10.0,  # 归一化
                float(num_vars) / 5.0,
                float(max_d) / 5.0,
                0.0,  # 占位
                0.0,  # 占位
                0.0   # 占位
            ]
        except Exception as e:
            # 失败时返回零特征
            return [0.0] * 6
    
    def _curriculum_config(self, iter_idx: int) -> Tuple[List[str], List[str]]:
        """根据当前迭代返回课程学习限制的变量与算子集合。
        阶段划分 (basic 模式):
          Stage 1 (0%~33%): 仅位置误差 pos_err_x/y/z, 允许 '+' '*' (线性/缩放)；表达式深度由 MCTS 自控但算子少。
          Stage 2 (33%~66%): 加入速度 vel_x/y/z 与简单减法；允许 '+' '-' '*' '/'.
          Stage 3 (66%~100%): 完整 prior_level 对应变量集合与全算子。
        返回 (allowed_vars, allowed_ops)。若 curriculum_mode=none 则返回空表示不限制。
        """
        mode = getattr(self.args, 'curriculum_mode', 'none')
        if mode == 'none':
            return [], []  # 不限制
        progress = (iter_idx + 1) / float(self.args.total_iters)
        if progress <= 0.33:
            self._curriculum_stage = 1
            return ['pos_err_x', 'pos_err_y', 'pos_err_z'], ['+', '*']
        elif progress <= 0.66:
            self._curriculum_stage = 2
            return ['pos_err_x', 'pos_err_y', 'pos_err_z', 'vel_x', 'vel_y', 'vel_z'], ['+', '-', '*', '/']
        else:
            self._curriculum_stage = 3
            return [], []  # Stage 3 不做额外裁剪（使用 prior_level 完整集合）

    def _analyze_program(self, program: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取程序使用的变量与基本结构信息"""
        used = set()
        def collect(node):
            # 使用包式导入：优先尝试从 core.dsl 导入，若脚本直接运行则回退添加项目根到 sys.path
            try:
                from core.dsl import TerminalNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore
            except Exception:
                import sys, pathlib
                # 将项目根加入 sys.path，保证 `from core.dsl import ...` 可用
                _parent = pathlib.Path(__file__).resolve().parent.parent
                if str(_parent) not in sys.path:
                    sys.path.insert(0, str(_parent))
                from core.dsl import TerminalNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore

            if node is None:
                return
            if isinstance(node, TerminalNode) and isinstance(node.value, str):
                used.add(node.value)
            elif isinstance(node, UnaryOpNode):
                collect(node.child)
            elif isinstance(node, BinaryOpNode):
                collect(node.left); collect(node.right)
            elif isinstance(node, IfNode):
                collect(node.condition); collect(node.then_branch); collect(node.else_branch)
        for rule in program:
            cond = rule.get('condition')
            collect(cond)
            for act in rule.get('action', []):
                try:
                    # 'set' 二元: left 是输出键, right 是表达式
                    if hasattr(act, 'op') and act.op == 'set' and hasattr(act, 'right'):
                        collect(act.right)
                except Exception:
                    pass
        return {
            'rule_count': len(program),
            'used_variables': sorted(list(used))
        }

    def _program_to_str(self, program: List[Dict[str, Any]], max_rules: int = 3) -> str:
        """将程序转成可读字符串，便于迭代日志打印。
        仅打印前 max_rules 条规则，避免过长输出。
        """
        try:
            from core.dsl import TerminalNode, ConstantNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore
        except Exception:
            import sys, pathlib
            _parent = pathlib.Path(__file__).resolve().parent.parent
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            from core.dsl import TerminalNode, ConstantNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore

        def ast_str(node):
            if node is None:
                return "None"
            if isinstance(node, BinaryOpNode):
                return f"({ast_str(node.left)} {node.op} {ast_str(node.right)})"
            if isinstance(node, UnaryOpNode):
                return f"{node.op}({ast_str(node.child)})"
            if isinstance(node, IfNode):
                # Conditions are disabled; show then-branch only
                return ast_str(node.then_branch)
            if isinstance(node, ConstantNode):
                name = f"{node.name}=" if node.name else ""
                return f"Const({name}{node.value:.3f})"
            if isinstance(node, TerminalNode):
                return str(node.value)
            if isinstance(node, dict):
                # JSON dict fallback
                ntype = node.get('type', '')
                if ntype == 'BinaryOpNode' or ntype == 'Binary':
                    return f"({ast_str(node.get('left'))} {node.get('op')} {ast_str(node.get('right'))})"
                if ntype == 'UnaryOpNode' or ntype == 'Unary':
                    return f"{node.get('op')}({ast_str(node.get('child'))})"
                if ntype == 'ConstantNode' or ntype == 'Constant':
                    name = node.get('name')
                    name_prefix = f"{name}=" if name else ""
                    return f"Const({name_prefix}{node.get('value')})"
                if ntype == 'TerminalNode' or ntype == 'Terminal':
                    return str(node.get('value'))
            return str(node)

        parts = []
        for i, rule in enumerate(program or []):
            if i >= max_rules:
                parts.append("...")
                break
            cond = rule.get('condition')
            acts = rule.get('action', []) or []
            act_strs = []
            for a in acts:
                if hasattr(a, 'op') and a.op == 'set':
                    lhs = getattr(a.left, 'value', '?') if hasattr(a, 'left') else '?'
                    rhs = ast_str(getattr(a, 'right', None))
                    act_strs.append(f"set {lhs} = {rhs}")
                elif isinstance(a, dict) and a.get('op') == 'set':
                    lhs = a.get('left', {}).get('value', '?')
                    rhs = ast_str(a.get('right'))
                    act_strs.append(f"set {lhs} = {rhs}")
            # Conditions are disabled; only print actions
            parts.append(f"[{'; '.join(act_strs)}]")
        return " | ".join(parts) if parts else "<empty>"

    def _append_program_history(self, iter_idx: int, reward: float, program: List[Dict[str, Any]]):
        path = getattr(self.args, 'program_history_path', None)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            info = self._analyze_program(program)
            # 如果评估器支持细粒度组件奖励,一并记录
            components = None
            if hasattr(self.evaluator, 'evaluate_single_with_metrics'):
                try:
                    r_total, comp = self.evaluator.evaluate_single_with_metrics(program)
                    components = comp
                except Exception:
                    components = None
            rec = {
                'iter': iter_idx + 1,
                'reward': reward,
                **info,
                'reward_components': components
            }
            with open(path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[History] 追加程序记录失败: {e}")
    
    def _get_saved_program_reward(self, save_path: str) -> float:
        """
        读取已保存程序文件中的奖励值
        
        Returns:
            已保存程序的奖励值，如果文件不存在或读取失败则返回负无穷
        """
        if not os.path.exists(save_path):
            return float('-inf')  # 文件不存在，任何新程序都应该保存
        
        try:
            import json
            with open(save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 尝试从meta中读取reward
            if 'meta' in data and 'reward' in data['meta']:
                saved_reward = float(data['meta']['reward'])
                return saved_reward
            else:
                # 旧版本文件可能没有meta，返回负无穷以允许保存
                return float('-inf')
        except Exception as e:
            print(f"  ⚠️  读取已保存程序奖励失败: {e}，将允许保存")
            return float('-inf')  # 读取失败，允许保存以防万一

    def mcts_search(self, root_program: List[Dict[str, Any]], num_simulations: int = 800, iter_idx: int = 0) -> Tuple[List[Any], List[int]]:
        """
        执行MCTS搜索（使用当前NN引导）
        
        Returns:
            children: 所有子节点
            visit_counts: 访问次数分布
        """
        # 变量集合：按 prior_level 分级裁剪
        # level 2 (中度零先验): 保留三轴分量+姿态，去掉模长聚合/积分/微分
        # level 3 (严格零先验): 仅保留位置误差、速度、角速度三轴原始分量
        prior_level = getattr(self.args, 'prior_level', 2)
        
        if prior_level == 3:
            # 严格零先验：仅最原始信号
            dsl_variables = [
                'pos_err_x', 'pos_err_y', 'pos_err_z',
                'vel_x', 'vel_y', 'vel_z',
                'ang_vel_x', 'ang_vel_y', 'ang_vel_z'
            ]
        elif prior_level == 2:
            # 中度零先验 + PID完整支持：保留三轴+姿态+积分项+微分项（用于合成PID控制器）
            dsl_variables = [
                'pos_err_x', 'pos_err_y', 'pos_err_z',
                'vel_x', 'vel_y', 'vel_z',
                'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                'err_p_roll', 'err_p_pitch', 'err_p_yaw',
                # 积分项（PID的I）
                'err_i_x', 'err_i_y', 'err_i_z',
                # 微分项（PID的D，姿态专用）
                'err_d_roll', 'err_d_pitch', 'err_d_yaw'
            ]
        else:
            # 回退到全特征（不推荐，仅用于调试）
            dsl_variables = [
                'pos_err_x','pos_err_y','pos_err_z','pos_err_xy','pos_err_z_abs',
                'vel_x','vel_y','vel_z','vel_err',
                'ang_vel_x','ang_vel_y','ang_vel_z','ang_vel','ang_vel_mag',
                'err_i_x','err_i_y','err_i_z',
                'err_p_roll','err_p_pitch','err_p_yaw','rpy_err_mag',
                'err_d_x','err_d_y','err_d_z','err_d_roll','err_d_pitch','err_d_yaw'
            ]

        # 课程学习裁剪
        curriculum_vars, curriculum_ops = self._curriculum_config(iter_idx)
        if curriculum_vars:  # 非空表示阶段限制变量集合
            dsl_variables = [v for v in dsl_variables if v in curriculum_vars]
        # 算子裁剪：去掉除法，保留常用安全算子，降低搜索爆炸与数值不稳
        base_ops_full = ['+','-','*','max','min','abs','sqrt','log1p']
        if curriculum_ops:  # 限制算子集合
            dsl_operators = [op for op in base_ops_full if op in curriculum_ops]
        else:
            dsl_operators = list(base_ops_full)
        # 默认加入时序/稳定性一元原语（参数化为不同op名，便于MCTS选择）
        # 采用课程化的参数网格：先粗后细，提高 prior 召回
        temporal_ops: List[str] = []
        stage1 = int(getattr(self, '_unary_grid_stage1_iters', 200))
        stage2 = int(getattr(self, '_unary_grid_stage2_iters', 600))
        if iter_idx < stage1:
            ema_list = [0.2]
            k_list = [1]  # delay/diff 仅 k=1，避免长时延引入不稳
            clamp_list = [(-2.0, 2.0)]
            dz_list = [0.05]
            rate_list = [1.0]
            smooth_list = [1.0]
        elif iter_idx < stage2:
            ema_list = [0.1, 0.5]
            k_list = [1]  # 仍限制 k=1，先收敛后再放宽
            clamp_list = [(-1.0, 1.0), (-2.0, 2.0)]
            dz_list = [0.01, 0.1]
            rate_list = [0.5, 2.0]
            smooth_list = [0.5, 2.0]
        else:
            ema_list = [0.1, 0.2, 0.5]
            k_list = [1, 2]  # 后期允许到2步
            clamp_list = [(-1.0, 1.0), (-2.0, 2.0), (-5.0, 5.0)]
            dz_list = [0.01, 0.05, 0.1]
            rate_list = [0.5, 1.0, 2.0]
            smooth_list = [0.5, 1.0, 2.0]
        for a in ema_list:
            temporal_ops.append(f'ema:{a}')
        for k in k_list:
            temporal_ops.append(f'delay:{k}')
            temporal_ops.append(f'diff:{k}')
        for lo, hi in clamp_list:
            temporal_ops.append(f'clamp:{lo}:{hi}')
        for eps in dz_list:
            temporal_ops.append(f'deadzone:{eps}')
        for r in rate_list:
            temporal_ops.append(f'rate:{r}')
        for s in smooth_list:
            temporal_ops.append(f'smooth:{s}')
        # 合并去重（保持原有基础算子 + 时序原语）
        dsl_operators = list(dict.fromkeys(dsl_operators + temporal_ops))
        # 🔻 精简一元算子：保留 clamp、ema、diff、deadzone 以及 smooth/delay/rate
        def _is_unary_keep(op:str)->bool:
            base = op.split(':',1)[0]
            return base in ('clamp','ema','diff','deadzone','smooth','delay','rate')
        base_ops = [op for op in dsl_operators if ':' not in op]
        unary_ops = [op for op in dsl_operators if ':' in op and _is_unary_keep(op)]
        dsl_operators = base_ops + unary_ops

        # 🔧 单轴搜索包装评估函数：在评估前自动镜像扩展，补上 u_ty/u_tz/u_fz
        # 这样 MCTS 搜索只产生 u_tx 程序，但评估时会自动补全其他通道，避免无推力坠落
        def _single_axis_eval_wrapper(program):
            """单轴搜索评估包装：自动镜像扩展后评估"""
            try:
                expanded = self.evaluator._mirror_expand_single_axis_program(program)
            except Exception:
                expanded = program  # 失败则使用原程序
            return self.evaluator.evaluate_single(expanded)

        # 创建MCTS agent
        mcts = MCTS_Agent(
            evaluation_function=_single_axis_eval_wrapper,  # 使用包装函数，自动镜像扩展
            dsl_variables=dsl_variables,
            dsl_constants=[0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0],  # 扩大常数范围以产生更大动作幅度
            dsl_operators=dsl_operators,
            exploration_weight=self._exploration_weight,
            max_depth=self._max_depth,
            structure_prior_weight=self.structure_prior_weight,
            stability_prior_weight=self.stability_prior_weight
        )
        # 单轴搜索：默认仅 Roll 通道（可改为 ['u_ty'] / ['u_tz'] / ['u_fz']）
        mcts._active_channels = ['u_tx']
        
        # 🚀 悬停推力约束配置
        mcts._enforce_hover_thrust = getattr(self, '_enforce_hover_thrust', True)
        mcts._hover_thrust_value = getattr(self, '_hover_thrust_value', 0.265)
        mcts._hover_thrust_min = getattr(self, '_hover_thrust_min', 0.20)
        mcts._hover_thrust_max = getattr(self, '_hover_thrust_max', 0.35)
        mcts._hover_delta_max = getattr(self, '_hover_delta_max', 2.0)
        
        # 🌟 设置 ranking 网络用于 MCTS bias（条件启用）
        if self.enable_ranking_mcts_bias and self.use_ranking and hasattr(self, 'ranking_net') and self.ranking_net is not None:
            mcts.ranking_net = self.ranking_net
            mcts.gnn_encoder = self.nn_model
            mcts.ranking_bias_beta = self.ranking_bias_beta
            mcts.ranking_device = self.device
        else:
            # 确保关闭时 ranking_net 为 None
            mcts.ranking_net = None
        
        # 设置root（AST 内部表示）
        root_ast = to_ast_program(root_program)
        root = MCTSNode(root_ast, parent=None, depth=0)
        mcts.root = root
        
        # 🔧 优化1: GNN先验缓存 (避免重复推理) - 限制大小防止内存泄漏
        # 之前实现: 每次 mcts_search 都重新创建局部缓存 → 导致跨迭代命中率始终为 0%
        # 改进: 使用训练器级别的持久 LRU 缓存 (self._global_prior_cache) 在所有迭代之间复用
        from collections import OrderedDict
        if not hasattr(self, '_global_prior_cache'):
            self._global_prior_cache = OrderedDict()  # 首次创建
        gnn_prior_cache = self._global_prior_cache      # 引用同一对象
        MAX_CACHE_SIZE = 5000  # 扩大上限，跨迭代可积累更多结构
        
        # 注意：get_program_hash 现在是顶层函数（在文件开头定义），可以被其他模块导入
        
        def add_to_cache(prog_hash, value):
            """添加/更新缓存（LRU）。超过MAX_CACHE_SIZE时批量淘汰最久未使用的20%。"""
            # 如果已存在则更新并移动到尾部
            gnn_prior_cache[prog_hash] = value
            try:
                gnn_prior_cache.move_to_end(prog_hash, last=True)
            except Exception:
                pass
            # LRU 清理：超过限制时删除最旧的20%
            if len(gnn_prior_cache) > MAX_CACHE_SIZE:
                remove_count = max(1, int(MAX_CACHE_SIZE * 0.2))
                for _ in range(remove_count):
                    try:
                        gnn_prior_cache.popitem(last=False)
                    except Exception:
                        break
        
        # 🔧 优化2: 批量GNN推理缓冲区
        pending_gnn_nodes = []  # 收集需要GNN推理的新节点
        
        # 🔧 批量评估优化：收集待评估的leaf nodes
        pending_evals = []  # [(leaf, path, use_real_sim)]
        
        # 执行MCTS模拟（只做树扩展，延迟GNN推理）
        # Root Dirichlet 噪声一次性生成（针对编辑类型先验），仅第一轮选择阶段使用
        root_dirichlet_noise = None
        if self._root_dirichlet_eps > 0.0:
            try:
                import numpy as _np
                alpha = float(self._root_dirichlet_alpha)
                noise = _np.random.gamma(alpha, 1.0, size=len(EDIT_TYPES))
                noise = noise / max(1e-12, noise.sum())
                root_dirichlet_noise = noise
            except Exception:
                root_dirichlet_noise = None

        # 根节点自适应最小分支数：根据NN先验的累计覆盖率确定K（而非固定常数）
        # 逻辑：取最小K，使得按降序排序的先验概率累计和 ≥ tau，K 作为根节点 progressive widening 的下限
        root_min_cap_k = 2
        try:
            with torch.no_grad():
                root_graph = ast_to_pyg_graph(root.program)
                from torch_geometric.data import Batch as _PyGBatch
                _g = _PyGBatch.from_data_list([root_graph]).to(self.device)
                _logits, _, _ = self.nn_model(_g)
                _probs = F.softmax(_logits.squeeze(0), dim=-1).detach().cpu().numpy()
                # 与 Dirichlet 一致：若配置启用，按相同 eps 混入噪声（针对编辑类型）
                if root_dirichlet_noise is not None and self._root_dirichlet_eps > 0.0:
                    _probs = (1.0 - float(self._root_dirichlet_eps)) * _probs + float(self._root_dirichlet_eps) * root_dirichlet_noise
                _probs = _probs.clip(1e-12, 1.0)
                order = _probs.argsort()[::-1]
                tau = getattr(self, '_root_prior_coverage_tau', 0.80)
                csum = 0.0
                k = 0
                for idx in order:
                    csum += float(_probs[idx])
                    k += 1
                    if csum >= tau:
                        break
                # ✅ 修复2: 设置min_cap硬下限, 防止NN过度自信导致探索崩溃
                root_min_cap_k = max(5, min(k, len(EDIT_TYPES)))
        except Exception:
            root_min_cap_k = 5  # 异常时也保证最小探索宽度

        # 🚀 Leaf Parallelization: 分批评估叶节点
        leaf_batch_size = getattr(self.args, 'mcts_leaf_batch_size', 128)
        num_batches = (num_simulations + leaf_batch_size - 1) // leaf_batch_size
        
        if num_simulations > 0:
            print(f"[LeafParallel] MCTS simulations={num_simulations}, batch_size={leaf_batch_size}, num_batches={num_batches}")
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * leaf_batch_size
            batch_end = min((batch_idx + 1) * leaf_batch_size, num_simulations)
            batch_pending_evals = []  # 当前批次的待评估节点
            
            for sim_idx in range(batch_start, batch_end):
                # Selection + Expansion（使用NN先验 + Progressive Widening）
                node = root
                path = [node]
                
                if sim_idx == 0 or sim_idx == num_simulations - 1:  # DEBUG: 首次和最后一次模拟
                    print(f"[PW-DEBUG] sim={sim_idx}, root.visits={root.visits}, root.children={len(root.children)}")
                
                # Selection阶段 (考虑Progressive Widening)
                while node.children:
                    # Progressive Widening检查：是否可以扩展更多children
                    pw_c = 1.5
                    pw_alpha = 0.6
                    # Progressive Widening初始行为修正：
                    # - 以 (visits+1) 计算，避免 visits==0 时上限为0
                    # - 根节点的最小分支数由 NN 先验累计覆盖率自适应确定（不使用固定常数）
                    vis = max(0, int(node.visits))
                    num_mutations = len(node.untried_mutations) if hasattr(node, 'untried_mutations') else 0

                    if self.disable_progressive_widening:
                        # 完全放开：允许一次性扩展所有可能变异
                        max_children = num_mutations
                    else:
                        base_cap = int(pw_c * ((vis + 1) ** pw_alpha))
                        min_cap = (root_min_cap_k if node.depth == 0 else 1)
                        max_children = max(min_cap, base_cap)

                    can_expand = len(node.children) < max_children and len(node.children) < num_mutations
                    
                    if sim_idx == 0 and node.depth == 0:  # DEBUG: 只在第一次模拟打印root信息
                        print(f"[PW-DEBUG] sim_idx={sim_idx}, depth={node.depth}, visits={node.visits}, max_children={max_children}, len(children)={len(node.children)}, num_mutations={num_mutations}, can_expand={can_expand}")
                    
                    if can_expand:
                        # 可以扩展更多children，停止selection
                        break
                    
                    if node.is_fully_expanded():
                        # 完全扩展，停止selection
                        break
                    
                    # 继续向下选择
                    node = self._select_child_puct(node, root_dirichlet_noise if node.depth == 0 else None)
                    path.append(node)
                
                # Expansion阶段
                if not node.is_fully_expanded():
                    # 生成新子节点，分配NN先验
                    mcts._ensure_mutations(node)
                    
                    if node.untried_mutations and len(node.expanded_actions) < len(node.untried_mutations):
                        # 选择一个未扩展的变异
                        unexpanded_idx = [i for i in range(len(node.untried_mutations)) 
                                         if i not in node.expanded_actions][0]
                        mutation = node.untried_mutations[unexpanded_idx]
                        
                        # 克隆程序并应用变异
                        child_program = [mcts._clone_rule(r) for r in node.program]
                        mcts._apply_mutation(child_program, mutation)
                        # 变异后也转换为AST，确保内部一致
                        child_program = to_ast_program(child_program)
                        
                        # ── 调试：打印被扩展的程序摘要（仅根与其下一层，限数量）──
                        try:
                            if getattr(self.args, 'debug_programs', False) and (node.depth <= 1):
                                if not hasattr(self, '_debug_prog_count'):
                                    self._debug_prog_count = 0
                                limit = int(getattr(self.args, 'debug_programs_limit', 20))
                                if self._debug_prog_count < limit:
                                    def _summarize_rule(rule):
                                        try:
                                            if isinstance(rule, dict):
                                                if 'op' in rule:
                                                    op = rule.get('op')
                                                    var = rule.get('var')
                                                    expr = rule.get('expr')
                                                    cond = rule.get('condition')
                                                    expr_type = (expr or {}).get('type') if isinstance(expr, dict) else type(expr).__name__
                                                    has_cond = cond not in (None, False)
                                                    return f"{op}:{var}|{expr_type}|cond={has_cond}"
                                                if 'set' in rule:
                                                    s = rule.get('set')
                                                    if isinstance(s, (list, tuple)) and len(s) >= 2:
                                                        return f"set:{s[0]}|const|cond=False"
                                                    return "set:?"
                                                if 'if' in rule:
                                                    return "if:..."
                                            return str(type(rule).__name__)
                                        except Exception:
                                            return "<err>"
                                    sets = []
                                    uses_u = False
                                    has_if = False
                                    # 统计 AST 'set' 二元操作
                                    for rr in child_program:
                                        try:
                                            if isinstance(rr, dict):
                                                cond = rr.get('condition')
                                                if hasattr(cond, 'op') and getattr(cond, 'op', None) in ('if',):
                                                    has_if = True
                                                for act in rr.get('action', []) or []:
                                                    if hasattr(act, 'op') and act.op == 'set' and hasattr(act, 'left') and hasattr(act.left, 'value'):
                                                        var = str(getattr(act.left, 'value', ''))
                                                        sets.append(var)
                                                        if var.startswith('u_'):
                                                            uses_u = True
                                        except Exception:
                                            pass
                                    digest = ", ".join(_summarize_rule(r) for r in child_program[:6])
                                    print(f"[Prog] depth={node.depth+1} rules={len(child_program)} u_sets={sets} uses_u={uses_u} :: {digest}")
                                    self._debug_prog_count += 1
                        except Exception:
                            pass

                        # 创建子节点
                        child = MCTSNode(child_program, parent=node, depth=node.depth + 1)
                        edit_type = mutation[0]
                        child._edit_type = edit_type
                        
                        # 🚀 优化: 检查先验缓存（不缓存value）
                        prog_hash = get_program_hash(child_program)
                        if prog_hash in gnn_prior_cache:
                            # 命中缓存，直接使用先验 + LRU: 移动到队尾
                            child._prior_p = gnn_prior_cache[prog_hash]
                            try:
                                gnn_prior_cache.move_to_end(prog_hash, last=True)
                            except Exception:
                                pass
                            # 统计：先验（child 扩展阶段）缓存命中
                            if hasattr(self, '_mcts_stats'):
                                self._mcts_stats['prior_cached'] = self._mcts_stats.get('prior_cached', 0) + 1
                            # 可选调试: 仅前若干次命中打印（避免刷屏）
                            if getattr(self, '_debug_prior_hit_printed', 0) < 10:
                                try:
                                    print(f"[PriorCacheHit] depth={child.depth} hash={prog_hash[:10]} prior_p={child._prior_p:.4f}")
                                except Exception:
                                    pass
                                self._debug_prior_hit_printed = getattr(self, '_debug_prior_hit_printed', 0) + 1
                        else:
                            # 未命中，加入批量推理队列
                            child._prior_p = 1.0 / len(EDIT_TYPES)  # 默认先验
                            child._prog_hash = prog_hash
                            pending_gnn_nodes.append((child, edit_type))
                        
                        node.children.append(child)
                        node.expanded_actions.add(unexpanded_idx)
                        path.append(child)
                
                # 🔧 收集leaf待批量评估（不立即评估，全部使用真实仿真）
                leaf = path[-1]
                batch_pending_evals.append((leaf, path.copy()))  # 🔧 关键修复：必须使用path的副本！
                pending_evals.append((leaf, path.copy()))  # 也保留在全局列表中（用于GNN推理）
                
                # ✅ 修复1: 立即更新visits (在模拟循环内, 保证PW正确计算)
                for node in reversed(path):
                    node.visits += 1
            
            # 🚀 批量评估当前批次的叶节点
            if batch_pending_evals:
                invalid_reasons = {}
                valid_programs: List[List[Dict[str, Any]]] = []
                valid_refs: List[Tuple[MCTSNode, List[MCTSNode]]] = []
                for idx, (leaf, path) in enumerate(batch_pending_evals):
                    program = leaf.program
                    ok, reason = validate_program(program)
                    if ok:
                        valid_programs.append(program)
                        valid_refs.append((leaf, path))
                    else:
                        invalid_reasons[idx] = reason or "violates hard constraint"

                rewards_valid: List[float] = []
                if valid_programs:
                    rewards_valid = self.evaluator.evaluate_batch(valid_programs)

                valid_iter = iter(rewards_valid)
                for idx, (leaf, path) in enumerate(batch_pending_evals):
                    if idx in invalid_reasons:
                        reason = invalid_reasons[idx]
                        print(f"[HardConstraint] Reject program before sim: {reason}")
                        reward = float(HARD_CONSTRAINT_PENALTY)
                    else:
                        reward = float(next(valid_iter))
                    for node in reversed(path):
                        # visits已在模拟循环内更新, 这里只更新value_sum
                        node.value_sum += reward
        
        # 🚀 批量GNN推理阶段 (一次推理所有新节点，仅获取先验)
        if pending_gnn_nodes:
            try:
                with torch.no_grad():
                    # 批量构建图 (仅GNN路径)
                    graphs = [ast_to_pyg_graph(child.program) for child, _ in pending_gnn_nodes]
                    from torch_geometric.data import Batch
                    batch_graph = Batch.from_data_list(graphs).to(self.device)
                    policy_logits, _, _ = self.nn_model(batch_graph)  # 仅使用policy输出
                    
                    # 分配先验并缓存
                    policy_probs = F.softmax(policy_logits, dim=-1)
                    for idx, (child, edit_type) in enumerate(pending_gnn_nodes):
                        if edit_type in EDIT_TYPES:
                            type_idx = EDIT_TYPES.index(edit_type)
                            prior_p = policy_probs[idx, type_idx].item()
                        else:
                            prior_p = 1.0 / len(EDIT_TYPES)
                        
                        child._prior_p = float(prior_p)  # 转为Python原生类型避免张量引用
                        
                        # 更新缓存 - LRU淘汰策略（仅缓存先验）
                        if hasattr(child, '_prog_hash'):
                            add_to_cache(child._prog_hash, float(prior_p))
            except Exception as e:
                # 批量推理失败，使用默认值
                for child, _ in pending_gnn_nodes:
                    child._prior_p = 1.0 / len(EDIT_TYPES)
        
        # 📊 性能统计 (可选，用于调试)
        if hasattr(self, '_mcts_stats'):
            # 统计：先验GNN调用数量（child 扩展阶段）
            self._mcts_stats['prior_gnn_nodes'] = self._mcts_stats.get('prior_gnn_nodes', 0) + len(pending_gnn_nodes)
            # 统计：当前缓存大小
            self._mcts_stats['cache_size'] = len(gnn_prior_cache)
        
        # 返回root的子节点和访问分布
        if root.children:
            visit_counts = [child.visits for child in root.children]
            result_children = root.children
            result_visits = visit_counts
            
            # 🧹 MCTS内存清理：递归清除所有节点的引用,防止内存泄漏
            # ⚠️  注意：只清理深层子树(depth>=2),保护root的直接children
            def cleanup_tree(node, preserve_depth=1):
                if node is None:
                    return
                # 如果是需要保护的深度,不清理
                if node.depth < preserve_depth:
                    return
                # 递归清理子节点
                for child in node.children:
                    cleanup_tree(child, preserve_depth)
                # 清除引用(只清理深层节点)
                if node.depth >= preserve_depth:
                    node.children = []
                    node.parent = None
                    # 清除缓存的value
                    if hasattr(node, '_cached_value'):
                        delattr(node, '_cached_value')
            
            # 保存需要返回的数据后,清理深层子树(depth>=2)
            # root(depth=0)和它的直接children(depth=1)都保留
            for child in root.children:
                cleanup_tree(child, preserve_depth=2)  # 只清理depth>=2的节点
            
            return result_children, result_visits
        else:
            return [], []
    
    def _select_child_puct(self, node: MCTSNode, root_noise: Optional['np.ndarray']=None) -> MCTSNode:
        """PUCT选择（使用NN先验）"""
        if not node.children:
            return node
        
        best_score = -float('inf')
        best_child = None
        
        sqrt_n = np.sqrt(node.visits)
        c_puct = self._puct_c
        
        for idx, child in enumerate(node.children):
            q = child.value_sum / child.visits if child.visits > 0 else 0.0
            prior = getattr(child, '_prior_p', 1.0 / len(node.children))
            # 根节点混入 Dirichlet 噪声（将编辑类型映射到 children 顺序的平均）
            if root_noise is not None and node.depth == 0:
                # 若 child 有编辑类型，将噪声映射到对应 EDIT_TYPE 索引
                et = getattr(child, '_edit_type', None)
                if et in EDIT_TYPES:
                    et_idx = EDIT_TYPES.index(et)
                    prior = (1.0 - self._root_dirichlet_eps) * prior + self._root_dirichlet_eps * float(root_noise[et_idx])
            u = c_puct * prior * sqrt_n / (1 + child.visits)
            
            score = q + u
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child if best_child else node.children[0]
    
    def train_step(self):
        """单步训练（AlphaZero风格：从第一个样本就开始学习）"""
        # 空buffer无法训练
        if len(self.replay_buffer) == 0:
            return None
        
        # 采样batch（使用实际buffer大小和batch_size的较小值）
        actual_batch_size = min(self.args.batch_size, len(self.replay_buffer))
        batch = self.replay_buffer.sample(actual_batch_size)
        
        # 构建tensor（根据模式）
        # 仅保留 GNN 模式
        graph_list = [s['graph'] for s in batch]
        batch_graph = PyGBatch.from_data_list(graph_list).to(self.device)
        policy_targets = torch.stack([s['policy_target'] for s in batch]).to(self.device)
        
        # 前向传播：根据配置决定是否使用value头
        policy_logits, value_scalar, value_components = self.nn_model(batch_graph)
        
        # ===== 诊断：策略目标的质量与分布 =====
        with torch.no_grad():
            # 每个样本目标和（应≈1）
            pt_sums = policy_targets.sum(dim=-1)
            # 非零项个数
            pt_nz = (policy_targets > 1e-8).sum(dim=-1).float()
            # 目标熵（越大越分散，最大值约为 log(len(EDIT_TYPES))）
            pt_entropy = (-(policy_targets.clamp(min=1e-12) * policy_targets.clamp(min=1e-12).log()).sum(dim=-1))
            # 异常侦测：若存在和为0或NaN，记录标记
            any_zero_sum = bool((pt_sums <= 1e-8).any().item())
            any_nan_sum = bool(torch.isnan(pt_sums).any().item())
            # 预测侧诊断：正确类概率与Top-1准确率
            pred_probs = F.softmax(policy_logits, dim=-1)
            tgt_idx = torch.argmax(policy_targets, dim=-1)
            batch_indices = torch.arange(pred_probs.size(0), device=pred_probs.device)
            correct_prob = pred_probs[batch_indices, tgt_idx]
            pred_top1 = torch.argmax(pred_probs, dim=-1)
            top1_acc = (pred_top1 == tgt_idx).float()
        # 损失计算
        # 策略损失：交叉熵（MCTS访问分布作为目标）
        policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        # 预测分布熵：鼓励非零熵，避免早期塌缩
        policy_probs = F.softmax(policy_logits, dim=-1)
        policy_entropy = (-(policy_probs.clamp(min=1e-12) * policy_probs.clamp(min=1e-12).log()).sum(dim=-1)).mean()
        _ENTROPY_COEFF = 0.01  # 固定系数，NN内部正则，不暴露为外参
        
        # Value head 损失（条件启用）
        value_loss = torch.tensor(0.0, device=self.device)
        if self.enable_value_head:
            # 提取真实奖励作为value target
            reward_targets = torch.tensor([s['reward_true'] for s in batch], device=self.device, dtype=torch.float32)
            # 归一化到[-1, 1]（假设奖励范围在[-10, 0]之间）
            reward_targets_norm = torch.tanh(reward_targets / 5.0)
            # MSE loss for value scalar
            value_loss = F.mse_loss(value_scalar, reward_targets_norm)
        
        # 总损失（策略 + 熵正则 + value）
        total_loss = policy_loss - _ENTROPY_COEFF * policy_entropy
        if self.enable_value_head:
            total_loss = total_loss + value_loss
        
        # 🔧 显存优化：反向传播前清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
        self.optimizer.step()
        
        # 🔧 显存优化：训练步后立即清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 轻量参数更新监控：比较训练前后参数校验和变化百分比
        new_checksum = self._compute_param_checksum()
        delta = new_checksum - self._last_param_checksum
        rel = (delta / (abs(self._last_param_checksum) + 1e-9)) * 100.0
        # 若变化极小（<0.001%），标记提示；仅第一步或每若干步输出一次由外层调用控制，这里返回指标
        changed_flag = rel >= 0.001
        self._last_param_checksum = new_checksum

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item() if self.enable_value_head else 0.0,
            'total_loss': total_loss.item(),
            'grad_norm': float(getattr(grad_norm, 'item', lambda: grad_norm)() if hasattr(grad_norm, 'item') else float(grad_norm)),
            'param_delta': float(delta),
            'param_delta_pct': float(rel),
            'param_changed': bool(changed_flag),
            # 诊断指标（仅用于打印与定位问题）
            'pt_sum_min': float(pt_sums.min().item()),
            'pt_sum_max': float(pt_sums.max().item()),
            'pt_sum_mean': float(pt_sums.mean().item()),
            'pt_nz_mean': float(pt_nz.mean().item()),
            'pt_entropy_mean': float(pt_entropy.mean().item()),
            'pt_any_zero_sum': any_zero_sum,
            'pt_any_nan_sum': any_nan_sum,
            'pred_correct_prob_mean': float(correct_prob.mean().item()),
            'pred_top1_acc': float(top1_acc.mean().item()),
            'policy_entropy': float(policy_entropy.item()),
        }
    
    def train(self):
        """主训练循环"""
        print(f"\n{'='*80}")
        print(f"开始在线训练 - AlphaZero式程序合成")
        print(f"{'='*80}")
        print(f"总迭代数: {self.args.total_iters}")
        print(f"MCTS模拟数/迭代: {self.args.mcts_simulations}")
        print(f"NN更新频率: 每{self.args.update_freq}次迭代")
        print(f"批量大小: {self.args.batch_size}")
        print(f"🚀 GNN结构缓存: 已启用（忽略常数值，BO调参时复用结构先验）")
        
        # 零动作惩罚课程化配置
        zero_action_penalty_init = float(getattr(self.args, 'zero_action_penalty', 0.0))
        zero_action_penalty_decay = float(getattr(self.args, 'zero_action_penalty_decay', 1.0))
        zero_action_penalty_min = float(getattr(self.args, 'zero_action_penalty_min', 0.1))
        current_zero_penalty = zero_action_penalty_init
        
        if zero_action_penalty_init > 0 and zero_action_penalty_decay < 1.0:
            print(f"零动作惩罚课程化: 初始={zero_action_penalty_init:.2f}, 衰减={zero_action_penalty_decay:.3f}/轮, 下限={zero_action_penalty_min:.2f}")
        print(f"{'='*80}\n")
        
        # 初始化程序（支持从文件加载）
        if hasattr(self.args, 'warm_start') and self.args.warm_start:
            loaded_program = self._load_program_from_json(self.args.warm_start)
            if loaded_program:
                current_program = loaded_program
                print(f"[Trainer] 🔥 Warm Start: 使用预训练程序 ({len(current_program)} 条规则)")
            else:
                current_program = self._generate_random_program()
                print(f"[Trainer] ⚠️ Warm Start 失败，使用随机初始化")
        else:
            current_program = self._generate_random_program()
        
        # 🔄 初始化异步训练器（如果启用）
        if self.async_training:
            from utils.async_trainer import create_trainer
            print(f"[Trainer] 🔄 启用异步训练模式（MCTS与NN并行）")
            self.async_trainer = create_trainer(
                train_fn=lambda: self.train_step(),
                async_mode=True,
                update_interval=getattr(self.args, 'async_update_interval', 0.1),
                max_steps_per_iter=getattr(self.args, 'async_max_steps_per_iter', None)
            )
            self.async_trainer.start()
        else:
            from utils.async_trainer import create_trainer
            self.async_trainer = create_trainer(
                train_fn=lambda: self.train_step(),
                async_mode=False
            )
        
        for iter_idx in range(self.args.total_iters):
            if self.async_training and self.async_trainer is not None:
                self.async_trainer.reset_iter()
            iter_start_time = time.time()
            
            # �️ 温度退火：逐步从探索转向利用
            if iter_idx < self._policy_temperature_decay_iters:
                progress = iter_idx / self._policy_temperature_decay_iters
                self._policy_temperature = self._policy_temperature_init + \
                    (self._policy_temperature_final - self._policy_temperature_init) * progress
                if (iter_idx + 1) % 50 == 0:  # 每50轮打印一次
                    print(f"[温度退火] T={self._policy_temperature:.3f}")
            else:
                self._policy_temperature = self._policy_temperature_final
            # 🌪️ 根 Dirichlet 噪声调整：Meta-RL 动态控制 或 启发式退火
            if self.use_meta_rl and self.meta_rl_controller is not None:
                # Meta-RL 模式：根据训练指标动态调整超参数
                if iter_idx > 0:  # 跳过第一轮（没有历史数据）
                    try:
                        hyperparams = self.meta_rl_controller.predict(
                            reward_history=[s['reward'] for s in self.training_stats[-20:]],  # 最近20轮奖励
                            best_reward=self.best_reward,
                            current_iter=iter_idx
                        )
                        self._root_dirichlet_eps = hyperparams['root_dirichlet_eps']
                        self._root_dirichlet_alpha = hyperparams['root_dirichlet_alpha']
                        if (iter_idx + 1) % 50 == 0:
                            print(f"[Meta-RL] eps={self._root_dirichlet_eps:.3f}, alpha={self._root_dirichlet_alpha:.3f}")
                    except Exception as e:
                        print(f"[Meta-RL] 预测失败，使用默认值: {e}")
            else:
                # 启发式退火模式
                if iter_idx < self._root_dirichlet_decay_iters:
                    p = iter_idx / max(1, self._root_dirichlet_decay_iters)
                    self._root_dirichlet_eps = self._root_dirichlet_eps_init + (self._root_dirichlet_eps_final - self._root_dirichlet_eps_init) * p
                    self._root_dirichlet_alpha = self._root_dirichlet_alpha_init + (self._root_dirichlet_alpha_final - self._root_dirichlet_alpha_init) * p
                    if (iter_idx + 1) % 100 == 0:
                        print(f"[Dirichlet退火] eps={self._root_dirichlet_eps:.2f}, alpha={self._root_dirichlet_alpha:.2f}")
                else:
                    self._root_dirichlet_eps = self._root_dirichlet_eps_final
                    self._root_dirichlet_alpha = self._root_dirichlet_alpha_final
            
            # �🎓 零动作惩罚课程化：每轮衰减
            if iter_idx > 0 and zero_action_penalty_decay < 1.0 and current_zero_penalty > zero_action_penalty_min:
                current_zero_penalty = max(zero_action_penalty_min, current_zero_penalty * zero_action_penalty_decay)
                # 动态更新评估器的零动作惩罚
                if hasattr(self.evaluator, 'zero_action_penalty'):
                    self.evaluator.zero_action_penalty = current_zero_penalty
                if (iter_idx + 1) % 10 == 0:  # 每10轮打印一次
                    print(f"[Curriculum] 零动作惩罚衰减至: {current_zero_penalty:.3f}")
            
            penalty_info = f" | ZeroPenalty={current_zero_penalty:.2f}" if current_zero_penalty > 0 else ""
            
            # ⭐ 简化输出模式：仅每 N 轮打印一次详细信息（默认每10轮）
            verbose_interval = int(os.environ.get('TRAIN_VERBOSE_INTERVAL', '10'))
            show_iter_detail = (iter_idx + 1) % verbose_interval == 0 or iter_idx == 0 or (iter_idx + 1) == self.args.total_iters
            
            if show_iter_detail:
                print(f"\n[Iter {iter_idx+1}/{self.args.total_iters}] MCTS搜索中...{penalty_info}")
            
            # MCTS搜索（带精英根种子）
            seeded_program = current_program
            try:
                if (iter_idx + 1) >= int(getattr(self, '_elite_seed_delay', 20)) and self.elite_archive:
                    import random as _r
                    if _r.random() < float(getattr(self, '_elite_seed_prob', 0.25)):
                        k = min(int(getattr(self, '_elite_seed_topk', 5)), len(self.elite_archive))
                        cand = self.elite_archive[:k]
                        _, seeded_program, src_iter = _r.choice(cand)
                        if show_iter_detail:
                            print(f"[Seed] 使用精英根种子 (Top-{k} 内) | 来自迭代 {src_iter}")
            except Exception:
                seeded_program = current_program
            children, visit_counts = self.mcts_search(seeded_program, self.args.mcts_simulations, iter_idx)
            
            # 🧹 每次MCTS后立即清理内存
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 🌳 根节点探索多样性统计（每10轮输出）
            if show_iter_detail and children:
                total_visits = sum(visit_counts)
                entropy = 0.0
                if total_visits > 0:
                    probs = [v / total_visits for v in visit_counts]
                    entropy = -sum(p * np.log(p + 1e-12) for p in probs if p > 0)
                top3_visits = sorted(visit_counts, reverse=True)[:3]
                print(f"  [根统计] 子节点数={len(children)}, 总访问={total_visits}, 熵={entropy:.3f}, Top3访问={top3_visits}")
            
            if not children:
                if show_iter_detail:
                    print(f"[Iter {iter_idx+1}] ⚠️ 未生成子节点，跳过")
                continue
            
            # 选择访问最多的子节点
            # 依据 policy_temperature 选择根动作：T>0 进行按访问概率采样，T=0 取最大
            if children:
                if self._policy_temperature > 1e-8:
                    import numpy as _np
                    counts = _np.array(visit_counts, dtype=_np.float64)
                    # 温度缩放：p_i ∝ (N_i)^{1/T}
                    scaled = counts ** (1.0 / max(1e-6, self._policy_temperature))
                    ps = scaled / max(1e-12, scaled.sum())
                    choice = int(_np.random.choice(len(children), p=ps))
                    best_child = children[choice]
                else:
                    best_child_idx = np.argmax(visit_counts)
                    best_child = children[best_child_idx]
            else:
                best_child = None
            if best_child is None:
                print(f"[Iter {iter_idx+1}] ⚠️ 根节点无子节点，保持原程序")
                next_program = current_program
            else:
                next_program = best_child.program
            
            # 迭代诊断：变量使用与是否包含 u_* 控制键
            try:
                info = self._analyze_program(next_program)
                uses_u = False
                try:
                    if hasattr(self.evaluator, '_program_uses_u'):
                        uses_u = bool(self.evaluator._program_uses_u(next_program))  # type: ignore
                except Exception:
                    uses_u = False
                print(f"[Iter {iter_idx+1}] 诊断: variables={info.get('used_variables', [])[:8]} | rules={info.get('rule_count')} | uses_u={uses_u}")
                try:
                    prog_str = self._program_to_str(next_program, max_rules=3)
                    print(f"[Iter {iter_idx+1}] 程序: {prog_str}")
                except Exception:
                    pass
            except Exception:
                pass

            # 真实评估（每次迭代至少1次）
            # 优先使用组件级接口获取细粒度指标
            # 🔍 分离训练奖励和真实奖励
            reward_train = 0.0  # 训练信号（含惩罚）→ 用于NN和best_reward比较
            reward_true = 0.0   # 真实奖励（不含惩罚）→ 用于保存和输出
            reward_components = None

            # 🔄 单轴程序在评估前强制镜像，补上 yaw/thrust 稳定器，避免无推力坠落
            eval_program = next_program
            try:
                if hasattr(self.evaluator, '_mirror_expand_single_axis_program'):
                    # 快速检测：是否仅设置 u_tx（AST节点格式）
                    targets = set()
                    for rule in next_program or []:
                        for act in rule.get('action', []) or []:
                            if hasattr(act, 'op') and act.op == 'set' and hasattr(act, 'left') and hasattr(act.left, 'value'):
                                targets.add(str(getattr(act.left, 'value', '')))
                    
                    if targets == {'u_tx'}:
                        # 直接镜像（_mirror_expand_single_axis_program 内部能处理 AST）
                        try:
                            eval_program = self.evaluator._mirror_expand_single_axis_program(next_program)
                            print(f"[Iter {iter_idx+1}] 🔁 单轴 u_tx 已扩展: +u_ty +u_tz +u_fz")
                        except Exception as _mirror_exc:
                            print(f"  ⚠️ 镜像失败，使用原程序: {_mirror_exc}")
            except Exception as _outer_exc:
                pass  # 静默失败，使用原程序

            if hasattr(self.evaluator, 'evaluate_single_with_metrics'):
                try:
                    print(f"[Iter {iter_idx+1}] 🔍 开始评估...")
                    reward_train, reward_true, reward_components = self.evaluator.evaluate_single_with_metrics(eval_program)
                    print(f"[Iter {iter_idx+1}] ✅ 评估完成")
                    # 打印组件用于诊断
                    if reward_components:
                        state_c = reward_components.get('state_cost', 0.0)
                        action_c = reward_components.get('action_cost', 0.0)
                        print(f"[Iter {iter_idx+1}] 组件: state={state_c:.3f} | action={action_c:.3e}")
                        print(f"[Iter {iter_idx+1}] 奖励: 真实={reward_true:.4f}, 训练={reward_train:.4f}")
                except Exception as e:
                    print(f"  ⚠️  evaluate_single_with_metrics 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    reward_train = self.evaluator.evaluate_single(eval_program)
                    reward_true = reward_train
            else:
                reward_train = self.evaluator.evaluate_single(eval_program)
                reward_true = reward_true

            # 收集训练样本
            # 策略标签：将根子节点访问分布按其编辑类型聚合到 EDIT_TYPES
            total_visits = sum(visit_counts)
            policy_target = torch.zeros(len(EDIT_TYPES))
            if total_visits > 0:
                for i, child in enumerate(children):
                    prob = float(visit_counts[i]) / float(total_visits)
                    et = getattr(child, '_edit_type', None)
                    if et in EDIT_TYPES:
                        policy_target[EDIT_TYPES.index(et)] += prob
                    else:
                        # 若未知类型，等量分摊到所有维度，避免丢失概率质量
                        policy_target += prob / len(EDIT_TYPES)
                # 归一化（数值安全）
                s = float(policy_target.sum().item())
                if s > 0:
                    policy_target = policy_target / s
            else:
                # 没有访问计数时，退化为均匀分布
                policy_target += 1.0 / len(EDIT_TYPES)

                # --- NN内部平滑与探索增强（不暴露成外部超参数） ---
            # Label smoothing: 防止目标过早单一化导致 policy_loss=0
            _SMOOTH_EPS = 0.02  # 固定微小值，不作为CLI参数
            if policy_target.sum() > 0:  # 保证已归一化
                policy_target = (1.0 - _SMOOTH_EPS) * policy_target + _SMOOTH_EPS / len(EDIT_TYPES)
            # 目标熵最小正则：若熵过低(接近0)，轻微抬高非最大类
            try:
                _entropy = float((-(policy_target.clamp(min=1e-12) * policy_target.clamp(min=1e-12).log()).sum()).item())
                _H_min = 0.15  # 允许仍很尖锐，但避免绝对 one-hot
                if _entropy < _H_min:
                    # 对最大概率类做微缩，其余均匀补偿
                    _top_idx = int(policy_target.argmax().item())
                    _shrink = 0.05  # 缩减幅度
                    top_val = float(policy_target[_top_idx].item())
                    if top_val > _shrink:
                        policy_target[_top_idx] = top_val - _shrink
                        # 重新分配缩减的概率到其它维度
                        _redistrib = _shrink / (len(EDIT_TYPES) - 1)
                        for _i in range(len(EDIT_TYPES)):
                            if _i != _top_idx:
                                policy_target[_i] += _redistrib
                    # 再次归一化避免数值漂移
                    s2 = float(policy_target.sum().item())
                    if abs(s2 - 1.0) > 1e-6 and s2 > 0:
                        policy_target /= s2
            except Exception:
                pass
            
            # 构建样本（包含reward_true用于value head训练）
            sample = {
                'graph': ast_to_pyg_graph(current_program),
                'policy_target': policy_target,
                'reward_true': reward_true  # 用于训练value head
            }
            
            self.replay_buffer.push(sample)
            
            # 🔥 收集程序对到ranking buffer（若启用，整合动作特征）
            if self.use_ranking and self.ranking_buffer is not None:
                pairs_collected = 0
                
                # 🎯 强制多样化策略：如果MCTS返回的children太少，人工生成更多变异程序
                augmented_programs = []
                if len(children) < 5:  # 如果children不足5个
                    # 添加MCTS的children
                    for child in children:
                        augmented_programs.append((child.program, getattr(child, 'value_sum', 0.0) / max(1, getattr(child, 'visits', 1))))
                    
                    # 人工生成额外的变异程序
                    import copy
                    for _ in range(min(10, 15 - len(children))):  # 补足到15个
                        mutated_program = copy.deepcopy(current_program)
                        # 随机应用一个变异
                        if len(mutated_program) > 0:
                            idx = np.random.randint(0, len(mutated_program))
                            # 简单变异：调整一个规则的动作常数
                            rule = mutated_program[idx]
                            if 'action' in rule and len(rule['action']) > 0:
                                # 找到并微调一个常数
                                for action in rule['action']:
                                    if hasattr(action, 'right') and hasattr(action.right, 'value') and isinstance(action.right.value, (int, float)):
                                        action.right.value = round(float(action.right.value) * np.random.uniform(0.85, 1.15), 4)
                                        break
                            # 简化评估：使用Q值估计（基于真实奖励）
                            estimated_q = reward_true + np.random.uniform(-2.0, 2.0)  # 添加噪声
                            augmented_programs.append((mutated_program, estimated_q))
                else:
                    # children足够多，直接使用
                    for child in children:
                        augmented_programs.append((child.program, getattr(child, 'value_sum', 0.0) / max(1, getattr(child, 'visits', 1))))
                
                # 1️⃣ 当前程序 vs augmented programs
                current_reward = reward_true  # 当前根节点的真实奖励（用于ranking比较）
                current_graph = ast_to_pyg_graph(current_program)
                current_action_feat = self._quick_action_features(current_program)
                
                for prog, prog_reward in augmented_programs:
                    prog_graph = ast_to_pyg_graph(prog)
                    prog_action_feat = self._quick_action_features(prog)
                    
                    # 过滤奖励差过小的样本对，降低噪声
                    if abs(float(prog_reward) - float(current_reward)) < getattr(self, '_ranking_min_delta', 0.0):
                        continue
                    if prog_reward != current_reward:
                        if prog_reward > current_reward:
                            self.ranking_buffer.push(current_graph, prog_graph, 1.0, 
                                                    current_action_feat, prog_action_feat)
                            pairs_collected += 1
                        elif prog_reward < current_reward:
                            self.ranking_buffer.push(prog_graph, current_graph, 0.0,
                                                    prog_action_feat, current_action_feat)
                            pairs_collected += 1
                
                # 2️⃣ augmented programs之间互相比较（增加数据量）
                # 取Q值最高的top-k进行两两比较
                if len(augmented_programs) > 1:
                    top_k = min(5, len(augmented_programs))
                    top_programs = sorted(augmented_programs, key=lambda x: x[1], reverse=True)[:top_k]
                    
                    aug_pairs_before = pairs_collected
                    for i in range(len(top_programs)):
                        for j in range(i + 1, len(top_programs)):
                            prog_i, q_i = top_programs[i]
                            prog_j, q_j = top_programs[j]
                            if abs(float(q_i) - float(q_j)) < getattr(self, '_ranking_min_delta', 0.0):
                                continue
                            if q_i != q_j:
                                graph_i = ast_to_pyg_graph(prog_i)
                                graph_j = ast_to_pyg_graph(prog_j)
                                feat_i = self._quick_action_features(prog_i)
                                feat_j = self._quick_action_features(prog_j)
                                if q_i > q_j:
                                    self.ranking_buffer.push(graph_j, graph_i, 1.0, feat_j, feat_i)
                                else:
                                    self.ranking_buffer.push(graph_i, graph_j, 1.0, feat_i, feat_j)
                                pairs_collected += 1
                
                if pairs_collected > 0:
                    print(f"[Iter {iter_idx+1}] 📊 Ranking: 收集{pairs_collected}对程序 (buffer总计={len(self.ranking_buffer)}对)")
            
            # 更新NN（每N次迭代）
            nn_loss_info = ""
            if (iter_idx + 1) % self.args.update_freq == 0:
                if self.async_training:
                    # 🔄 异步模式：获取后台训练的最新 metrics
                    metrics = self.async_trainer.get_metrics()
                    if metrics:
                        v_loss_str = f", v={metrics.get('value_loss', 0.0):.4f}" if self.enable_value_head else ""
                        nn_loss_info = f" | NN Loss: {metrics.get('total_loss', 0.0):.4f} (p={metrics.get('policy_loss', 0.0):.4f}{v_loss_str})"
                        print(f"[Iter {iter_idx+1}] � 异步训练状态: {metrics.get('policy_loss', 0.0):.4f}")
                    stats = self.async_trainer.get_stats()
                    print(f"  �🔄 后台训练: {stats['total_steps']} steps, 平均 {stats.get('avg_time_per_step', 0)*1000:.1f}ms/step")
                else:
                    # 同步模式：原逻辑
                    print(f"[Iter {iter_idx+1}] 🔄 更新NN...")
                    total_policy_loss = 0.0
                    total_value_loss = 0.0
                    total_loss = 0.0
                    for step_idx in range(self.args.train_steps_per_update):
                        losses = self.train_step()
                        if losses:
                            total_policy_loss += losses['policy_loss']
                            total_value_loss += losses.get('value_loss', 0.0)
                            total_loss += losses['total_loss']
                            if step_idx == 0 or (step_idx + 1) % 10 == 0:  # 输出首次和每10步
                                # 附带策略目标分布诊断，帮助定位 policy_loss=0 的根因
                                v_str = f", v={losses.get('value_loss', 0.0):.4f}" if self.enable_value_head else ""
                                diag_msg = (
                                    f"pt_sum(mean={losses['pt_sum_mean']:.3f}, min={losses['pt_sum_min']:.3f}, max={losses['pt_sum_max']:.3f}), "
                                    f"pt_nz(mean={losses['pt_nz_mean']:.1f}), "
                                    f"pt_H(mean={losses['pt_entropy_mean']:.3f}), "
                                    f"p(correct)_mean={losses['pred_correct_prob_mean']:.3f}, "
                                    f"top1_acc={losses['pred_top1_acc']:.2f}, "
                                    f"H_pred={losses.get('policy_entropy', 0.0):.3f}"
                                )
                                if losses.get('pt_any_zero_sum') or losses.get('pt_any_nan_sum'):
                                    diag_msg += " | ALERT: target_sum_zero_or_nan"
                                print(
                                    f"  Step {step_idx+1}/{self.args.train_steps_per_update}: "
                                    f"policy={losses['policy_loss']:.4f}{v_str}, "
                                    f"total={losses['total_loss']:.4f} | " + diag_msg
                                )
                    # 平均loss
                    n_steps = self.args.train_steps_per_update
                    avg_policy = total_policy_loss / n_steps
                    avg_value = total_value_loss / n_steps
                    avg_total = total_loss / n_steps
                    v_loss_str = f", v={avg_value:.4f}" if self.enable_value_head else ""
                    nn_loss_info = f" | NN Loss: {avg_total:.4f} (p={avg_policy:.4f}{v_loss_str})"
                    print(f"  ✅ 平均Loss: policy={avg_policy:.4f}{v_loss_str}, total={avg_total:.4f}")
                
                # 🧹 定期内存清理（防止内存泄漏）
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 🔥 训练Ranking网络（若启用且buffer有足够样本）
                if self.use_ranking and self.ranking_buffer is not None:
                    buffer_size = len(self.ranking_buffer)
                    if buffer_size >= 8:
                        print(f"  🔥 训练Ranking网络 (buffer={buffer_size}对)...")
                    else:
                        print(f"  ⏸️  Ranking训练跳过 (buffer={buffer_size}对 < 8对最小值)")
                
                ranking_paused_async = False
                if self.use_ranking and self.ranking_buffer is not None and len(self.ranking_buffer) >= 8:
                    if self.async_training and self.async_trainer is not None:
                        self.async_trainer.pause_and_wait()
                        ranking_paused_async = True
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                    try:
                        ranking_loss_total = 0.0
                        ranking_acc_total = 0.0
                        ranking_steps = min(10, max(1, len(self.ranking_buffer) // 8))  # 自适应步数（降低批次大小）
                        for _ in range(ranking_steps):
                            ranking_metrics = train_ranking_step(
                                ranking_net=self.ranking_net,
                                ranking_buffer=self.ranking_buffer,
                                ranking_optimizer=self.ranking_optimizer,
                                gnn_encoder=self.nn_model,
                                device=self.device,
                                batch_size=min(8, len(self.ranking_buffer))  # 动态batch size
                            )
                            if ranking_metrics:
                                ranking_loss_total += ranking_metrics['ranking_loss']
                                ranking_acc_total += ranking_metrics['ranking_accuracy']
                        avg_ranking_loss = ranking_loss_total / ranking_steps
                        avg_ranking_acc = ranking_acc_total / ranking_steps
                        print(f"  ✅ Ranking训练完成: loss={avg_ranking_loss:.4f}, accuracy={avg_ranking_acc:.2%}")
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if ranking_paused_async and self.async_trainer is not None:
                            self.async_trainer.resume()
                    
                    # 🎓 课程学习：逐步提高ranking混合系数
                    if iter_idx < self.ranking_blend_warmup_iters:
                        progress = (iter_idx + 1) / self.ranking_blend_warmup_iters
                        self.ranking_blend_factor = self.ranking_blend_factor + progress * (self.ranking_blend_max - self.ranking_blend_factor)
                        if (iter_idx + 1) % 10 == 0:
                            print(f"  📈 Ranking混合系数: {self.ranking_blend_factor:.2f}")
            
            # 更新最佳程序
            # 🌟 使用真实奖励进行best_reward比较（避免训练惩罚项退火导致的虚假进步）
            if reward_true > self.best_reward:
                self.best_reward = reward_true
                # 保留原始单轴程序用于持久化，运行时可按需镜像
                import copy
                self.best_program = copy.deepcopy(next_program)
                # 🔒 深拷贝保护,防止cleanup_tree或GC清理（保存用原始单轴）
                self.best_program_copy = copy.deepcopy(next_program)
                # 运行时仍可使用镜像版本做快速评估/导出
                expanded_program = self.evaluator._mirror_expand_single_axis_program(next_program)
                print(f"[Iter {iter_idx+1}] 🎉 新最佳！真实奖励: {reward_true:.4f} (训练奖励: {reward_train:.4f})")
                
                # 🔐 安全检查：只有比已保存文件更优才覆盖保存（使用真实奖励比较）
                saved_reward = self._get_saved_program_reward(self.args.save_path)
                should_save = reward_true > saved_reward
                
                if should_save:
                    # 构建元数据：记录训练进度和奖励信息（保存真实奖励）
                    program_meta = {
                        'iteration': iter_idx + 1,
                        'total_iterations': self.args.total_iters,
                        'reward': float(reward_true),  # 🌟 保存真实奖励
                        'reward_train': float(reward_train),  # 附带训练奖励供参考
                        'best_reward': float(self.best_reward),  # 当前最佳真实奖励
                        'trajectory': getattr(self.args, 'traj', 'unknown'),
                        'duration': getattr(self.args, 'duration', 10),
                        'reward_profile': getattr(self.args, 'reward_profile', 'safe_control_tracking'),
                        'mcts_simulations': self.args.mcts_simulations,
                        'isaac_num_envs': self.args.isaac_num_envs,
                    }
                    
                    # 添加奖励组件详情（如果可用）
                    if reward_components:
                        program_meta['reward_components'] = {k: float(v) for k, v in reward_components.items()}
                    
                    # 添加程序结构信息
                    program_info = self._analyze_program(self.best_program)
                    if program_info:
                        program_meta.update({
                            'num_rules': program_info.get('num_rules', 0),
                            'num_variables': program_info.get('num_variables', 0),
                            'depth': program_info.get('depth', 0),
                        })
                    
                    # 保存（带元数据）
                    save_program_json(self.best_program, self.args.save_path, meta=program_meta)
                    if saved_reward == float('-inf'):
                        print(f"  💾 已保存到: {self.args.save_path} (真实奖励: {reward_true:.4f})")
                    else:
                        print(f"  💾 已保存到: {self.args.save_path} (真实奖励: {reward_true:.4f}, 超越已保存: {saved_reward:.4f})")
                    
                    # 追加程序历史（使用真实奖励）
                    self._append_program_history(iter_idx, reward_true, self.best_program)
                else:
                    print(f"  ⏸️  未保存：当前真实奖励 {reward_true:.4f} ≤ 已保存 {saved_reward:.4f}（跳过覆盖）")
            
            # 🏆 更新精英程序池 (保留Top-K最优，使用真实奖励排序)
            import copy
            self.elite_archive.append((reward_true, copy.deepcopy(next_program), iter_idx + 1))
            # 按真实reward降序排序,保留Top-K
            self.elite_archive.sort(key=lambda x: x[0], reverse=True)
            if len(self.elite_archive) > self.elite_archive_size:
                self.elite_archive = self.elite_archive[:self.elite_archive_size]
            
            # 每20轮输出精英池状态
            if (iter_idx + 1) % 20 == 0:
                top3_rewards = [r for r, _, _ in self.elite_archive[:3]]
                print(f"  🏆 精英池Top-3: {top3_rewards}")
            
            # 更新当前程序
            current_program = next_program
            
            iter_time = time.time() - iter_start_time
            
            # 📊 MCTS性能统计 (每10轮输出一次)
            mcts_info = ""
            if self._mcts_stats and (iter_idx + 1) % 10 == 0:
                prior_gnn = self._mcts_stats.get('prior_gnn_nodes', 0)
                prior_cached = self._mcts_stats.get('prior_cached', 0)
                cache_size = self._mcts_stats.get('cache_size', 0)
                # 计算prior命中率
                prior_total = prior_gnn + prior_cached
                prior_rate = (prior_cached / prior_total * 100) if prior_total > 0 else 0.0
                mcts_info = (
                    f" | PriorGNN: {prior_gnn} | PriorHit: {prior_rate:.0f}%"
                    f" | CacheSize: {cache_size}"
                )
                # 重置统计
                self._mcts_stats = {}
            
            # 🧠 内存监控（每10轮输出）
            mem_info = ""
            if (iter_idx + 1) % 10 == 0:
                import psutil
                process = psutil.Process()
                ram_mb = process.memory_info().rss / 1024 / 1024
                mem_info = f" | RAM: {ram_mb:.0f}MB"
                if torch.cuda.is_available():
                    gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_max_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    mem_info += f" | GPU: {gpu_mb:.0f}MB (峰值{gpu_max_mb:.0f}MB)"
            
            # ⭐ 简化输出：仅在指定间隔打印详细信息（使用真实奖励）
            if show_iter_detail:
                print(f"[Iter {iter_idx+1}] 完成 | 真实奖励: {reward_true:.4f} | 耗时: {iter_time:.1f}s | Buffer: {len(self.replay_buffer)}{mcts_info}{nn_loss_info}{mem_info}")
            else:
                # 简洁模式：仅显示进度百分比
                progress_pct = (iter_idx + 1) / self.args.total_iters * 100
                print(f"\r[进度 {progress_pct:.1f}%] {iter_idx+1}/{self.args.total_iters} 轮 | 真实奖励: {reward_true:.4f} | Buffer: {len(self.replay_buffer)}", end='', flush=True)
            
            # 定期保存检查点
            if (iter_idx + 1) % self.args.checkpoint_freq == 0:
                checkpoint_path = f"{self.args.save_path.replace('.json', '')}_nn_iter_{iter_idx+1}.pt"
                torch.save(self.nn_model.state_dict(), checkpoint_path)
                print(f"[Iter {iter_idx+1}] 💾 检查点已保存: {checkpoint_path}")
        
        # 🔄 停止异步训练器
        if self.async_trainer is not None:
            print(f"[Trainer] 🛑 停止异步训练器...")
            self.async_trainer.stop(wait=True)
            stats = self.async_trainer.get_stats()
            print(f"  总训练步数: {stats['total_steps']}, 总耗时: {stats.get('total_time', 0):.1f}s")
        
        print(f"\n{'='*80}")
        print(f"训练完成！最佳奖励: {self.best_reward:.4f}")
        print(f"{'='*80}\n")
        
        # 🏆 保存精英程序池
        elite_save_path = self.args.save_path.replace('.json', '_elite_archive.json')
        try:
            elite_data = []
            for reward, program, iter_num in self.elite_archive:
                # 将程序转换为可序列化形式（AST -> dict），避免直接包含Node对象
                serializable_rules = []
                try:
                    for rule in program:
                        node = rule.get('node')
                        node_ser = to_serializable_dict(node) if node is not None else None
                        serializable_rules.append({
                            'name': rule.get('name', 'rule'),
                            'multiplier': rule.get('multiplier', [1.0, 1.0, 1.0]),
                            'node': node_ser
                        })
                except Exception:
                    # 兜底：若序列化失败，保存一个简化结构，至少保留规则数量和倍增器
                    serializable_rules = [
                        {
                            'name': r.get('name', 'rule'),
                            'multiplier': r.get('multiplier', [1.0, 1.0, 1.0]),
                            'node': None
                        } for r in program
                    ]
                elite_data.append({
                    'reward': float(reward),
                    'iter': int(iter_num),
                    'program': serializable_rules
                })
            with open(elite_save_path, 'w') as f:
                json.dump(elite_data, f, indent=2, ensure_ascii=False)
            print(f"🏆 精英程序池已保存: {elite_save_path} (共{len(self.elite_archive)}个程序)")
        except Exception as e:
            print(f"⚠️  精英池保存失败: {e}")
        
        # 🔒 最终保存：使用深拷贝的best_program_copy(原始单轴)，确保不被cleanup影响
        if self.best_program_copy is not None:
            try:
                final_save_path = self.args.save_path.replace('.json', '_final.json')
                
                # 构建最终元数据
                final_meta = {
                    'final_iteration': self.args.total_iters,
                    'best_reward': float(self.best_reward),
                    'trajectory': getattr(self.args, 'traj', 'unknown'),
                    'duration': getattr(self.args, 'duration', 10),
                    'reward_profile': getattr(self.args, 'reward_profile', 'safe_control_tracking'),
                    'mcts_simulations': self.args.mcts_simulations,
                    'isaac_num_envs': self.args.isaac_num_envs,
                    'training_completed': True,
                }
                
                # 添加程序结构信息
                program_info = self._analyze_program(self.best_program_copy)
                if program_info:
                    final_meta.update({
                        'num_rules': program_info.get('num_rules', 0),
                        'num_variables': program_info.get('num_variables', 0),
                        'depth': program_info.get('depth', 0),
                    })
                
                save_program_json(self.best_program_copy, final_save_path, meta=final_meta)
                print(f"🔒 最优程序(保护副本)已保存: {final_save_path}")
                print(f"   最终奖励: {self.best_reward:.4f} | 规则数: {final_meta.get('num_rules', 'N/A')}")
            except Exception as e:
                print(f"⚠️  最优程序保存失败: {e}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='在线训练 - AlphaZero式程序合成')
    
    # 训练参数
    p.add_argument('--total-iters', type=int, default=5000, help='总迭代数')
    p.add_argument('--mcts-simulations', type=int, default=800, help='每次迭代的MCTS模拟数')
    p.add_argument('--update-freq', type=int, default=50, help='NN更新频率')
    p.add_argument('--train-steps-per-update', type=int, default=10, help='每次更新的训练步数')
    p.add_argument('--batch-size', type=int, default=128, help='批量大小（降低以节省显存）')
    p.add_argument('--replay-capacity', type=int, default=50000, help='经验回放容量')
    
    # NN参数（固定特征网络已移除，统一使用GNN v2，只训练policy）
    p.add_argument('--learning-rate', type=float, default=1e-3, help='学习率')
    
    # GNN 架构参数
    p.add_argument('--gnn-structure-hidden', type=int, default=256, help='GNN结构编码器隐藏层维度（默认256）')
    p.add_argument('--gnn-structure-layers', type=int, default=5, help='GNN结构编码器层数（默认5）')
    p.add_argument('--gnn-structure-heads', type=int, default=8, help='GNN结构编码器注意力头数（默认8）')
    p.add_argument('--gnn-feature-layers', type=int, default=3, help='GNN特征编码器层数（默认3）')
    p.add_argument('--gnn-feature-heads', type=int, default=8, help='GNN特征编码器注意力头数（默认8）')
    p.add_argument('--gnn-dropout', type=float, default=0.1, help='GNN Dropout比例（默认0.1）')
    
    # MCTS参数
    p.add_argument('--exploration-weight', type=float, default=2.5, help='UCB探索权重 (提高以增强广度探索)')
    p.add_argument('--puct-c', type=float, default=1.5, help='PUCT常数')
    p.add_argument('--max-depth', type=int, default=12, help='MCTS最大深度（降低以减少分支稀释）')
    p.add_argument('--mcts-leaf-batch-size', type=int, default=1, help='MCTS叶节点批量评估大小（>1启用并行化，推荐4-10）')
    p.add_argument('--disable-progressive-widening', action='store_true', help='禁用 Progressive Widening，节点一次性扩展所有可变异（警告：树宽可能爆炸）')
    p.add_argument('--async-training', action='store_true', help='启用异步训练模式：MCTS与NN训练并行（实验性功能）')
    p.add_argument('--async-update-interval', type=float, default=0.1, help='异步训练线程两次训练之间的最小间隔（秒）')
    p.add_argument('--async-max-steps-per-iter', type=int, default=None, help='每轮允许的异步训练步数上限（None表示不限）')
    
    # 高级优化开关
    p.add_argument('--enable-ranking-mcts-bias', action='store_true', help='启用Ranking对MCTS子节点先验加权（打破plateau）')
    p.add_argument('--ranking-bias-beta', type=float, default=0.3, help='Ranking bias强度（默认0.3）')
    p.add_argument('--enable-value-head', action='store_true', help='启用Value头辅助训练（仅用于梯度信号，不影响MCTS）')
    p.add_argument('--enable-ranking-reweight', action='store_true', help='用Ranking score重新加权policy target')
    p.add_argument('--ranking-reweight-beta', type=float, default=0.2, help='Ranking reweight强度（默认0.2）')
    
    # 注意：已移除 --real-sim-frac 和 --force-full-sim，现在全部使用真实仿真
    # AlphaZero 式探索增强
    p.add_argument('--root-dirichlet-eps', type=float, default=0.25, help='根节点先验混合 Dirichlet 噪声比例 eps (0 关闭)')
    p.add_argument('--root-dirichlet-alpha', type=float, default=0.3, help='根节点 Dirichlet 噪声 alpha 参数')
    p.add_argument('--policy-temperature', type=float, default=1.0, help='根节点根据访问计数采样的温度系数，1 为按访问计数成比例采样，0 为贪心')
    
    # Meta-RL 在线调参（可选）
    p.add_argument('--use-meta-rl', action='store_true', help='启用 Meta-RL RNN 控制器进行动态超参数调整（需要预训练模型）')
    p.add_argument('--meta-rl-checkpoint', type=str, default='meta_rl/checkpoints/meta_policy.pt', help='Meta-RL 模型检查点路径')
    
    # 启发式衰减参数（当不使用 Meta-RL 时生效）
    p.add_argument('--root-dirichlet-eps-init', type=float, default=None, help='Dirichlet eps 初始值（启发式衰减模式，None则使用--root-dirichlet-eps）')
    p.add_argument('--root-dirichlet-eps-final', type=float, default=None, help='Dirichlet eps 终止值（启发式衰减模式）')
    p.add_argument('--root-dirichlet-alpha-init', type=float, default=None, help='Dirichlet alpha 初始值（启发式衰减模式，None则使用--root-dirichlet-alpha）')
    p.add_argument('--root-dirichlet-alpha-final', type=float, default=None, help='Dirichlet alpha 终止值（启发式衰减模式）')
    p.add_argument('--heuristic-decay-window', type=int, default=200, help='启发式衰减窗口（多少轮内完成退火，默认200）')
    # 打破奖励常数死区：零动作惩罚参数化（支持课程化衰减）
    p.add_argument('--zero-action-penalty', type=float, default=0.0, help='对整集始终零动作的程序施加惩罚（初始值；0=无惩罚）')
    p.add_argument('--zero-action-penalty-decay', type=float, default=0.95, help='零动作惩罚每轮衰减因子（<1启用课程化；1=不衰减；默认0.95）')
    p.add_argument('--zero-action-penalty-min', type=float, default=0.1, help='零动作惩罚最小值（课程化下限；默认0.1）')
    p.add_argument('--action-scale-multiplier', type=float, default=1.0, help='动作输出全局缩放系数（临时用于验证是否死区；1=不缩放）')
    p.add_argument('--enable-output-mad', dest='enable_output_mad', action='store_true', help='启用输出MAD安全壳（幅值/方向/变化率约束）')
    p.add_argument('--disable-output-mad', dest='enable_output_mad', action='store_false', help='禁用输出MAD安全壳（不建议）')
    p.set_defaults(enable_output_mad=True)
    p.add_argument('--mad-min-fz', type=float, default=0.0, help='输出安全壳：u_fz 最小值（牛顿）')
    p.add_argument('--mad-max-fz', type=float, default=7.5, help='输出安全壳：u_fz 最大值（牛顿）')
    p.add_argument('--mad-max-xy', type=float, default=0.12, help='输出安全壳：横向力矩/力幅值上限')
    p.add_argument('--mad-max-yaw', type=float, default=0.04, help='输出安全壳：yaw 力矩幅值上限')
    p.add_argument('--mad-max-delta-fz', type=float, default=1.5, help='输出安全壳：相邻步 u_fz 最大变化量')
    p.add_argument('--mad-max-delta-xy', type=float, default=0.03, help='输出安全壳：相邻步横向力矩变化上限')
    p.add_argument('--mad-max-delta-yaw', type=float, default=0.02, help='输出安全壳：相邻步 yaw 力矩变化上限')
    
    # 悬停推力约束（Hover Thrust Constraint）
    p.add_argument('--enforce-hover-thrust', dest='enforce_hover_thrust', action='store_true',
                   help='启用悬停推力约束：强制 u_fz = hover_thrust + delta，确保无人机始终有最小升力')
    p.add_argument('--no-enforce-hover-thrust', dest='enforce_hover_thrust', action='store_false',
                   help='禁用悬停推力约束（允许程序输出零推力）')
    p.set_defaults(enforce_hover_thrust=True)
    p.add_argument('--hover-thrust-value', type=float, default=0.265,
                   help='悬停推力基础值（牛顿），Crazyflie 默认 0.265N = 0.027kg × 9.81m/s²')
    p.add_argument('--hover-thrust-min', type=float, default=0.20,
                   help='悬停推力搜索下限（用于 BO 优化）')
    p.add_argument('--hover-thrust-max', type=float, default=0.35,
                   help='悬停推力搜索上限（用于 BO 优化）')
    p.add_argument('--hover-delta-max', type=float, default=2.0,
                   help='u_fz 控制增量的最大幅度（相对于悬停推力的偏移量）')
    
    # Ranking Value Network参数（自适应奖励学习，打破平坦奖励困境）
    p.add_argument('--use-ranking', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True, 
                   help='启用Ranking Value Network进行自适应奖励学习（默认True）')
    p.add_argument('--ranking-lr', type=float, default=1e-3, help='Ranking网络学习率（默认1e-3）')
    p.add_argument('--ranking-blend-init', type=float, default=0.3, help='Ranking value初始混合系数（默认0.3）')
    p.add_argument('--ranking-blend-max', type=float, default=0.8, help='Ranking value最大混合系数（默认0.8）')
    p.add_argument('--ranking-blend-warmup', type=int, default=100, help='Ranking混合系数warmup轮数（默认100）')
    
    # 仿真参数（仅Isaac Gym）
    # 默认直接使用 safe-control-gym quadrotor_3D_track 对齐配置
    p.add_argument('--traj', type=str, default='figure8', choices=['hover', 'figure8', 'circle', 'helix', 'square'])
    p.add_argument('--duration', type=int, default=5, help='仿真时长（秒），默认与 safe-control-gym quadrotor_3D_track 一致')
    p.add_argument('--isaac-num-envs', type=int, default=512, help='Isaac Gym并行环境数')
    p.add_argument('--eval-replicas-per-program', type=int, default=5, help='evaluate_single 时并行副本数，取平均以提高利用率/稳定性')
    p.add_argument('--min-steps-frac', type=float, default=0.0, help='每次评估至少执行的步数比例 [0,1]，避免过早 done 退出')
    p.add_argument('--reward-reduction', type=str, default='sum', choices=['sum','mean'], help="奖励归约方式：'sum'（步次求和）或 'mean'（步次平均）")
    # 🔥 奖励权重配置：只保留 SCG 对齐版本，避免混乱
    p.add_argument('--reward-profile', type=str, default='safe_control_tracking',
                   choices=['safe_control_tracking'],
                   help='奖励权重配置文件（唯一）：safe_control_tracking，对齐 safe-control-gym quadrotor_3D_track')
    p.add_argument('--prior-profile', type=str, default='none', choices=list(PRIOR_PROFILES.keys()),
                   help='结构/稳定先验实验分组：none(A组)、structure(B组)、structure_stability(C组)')
    p.add_argument('--structure-prior-weight', type=float, default=None,
                   help='覆盖结构先验权重（默认None表示使用 profile 内置值）')
    p.add_argument('--stability-prior-weight', type=float, default=None,
                   help='覆盖稳定性先验权重（默认None表示使用 profile 内置值）')
    # AST-first pipeline switch
    p.add_argument('--ast-pipeline', action='store_true', help='启用AST优先管线：内部统一AST表示，对外序列化为dict')
    # Debug programs explored during MCTS
    p.add_argument('--debug-programs', action='store_true', help='调试：打印搜索过程中扩展的程序摘要（仅根与其下一层，限数量）')
    p.add_argument('--debug-programs-limit', type=int, default=20, help='调试程序打印条数上限（全程累积）')
    p.add_argument('--use-fast-path', action='store_true', help='启用超高性能优化路径（环境池复用+Numba JIT编译，7×加速）')
    p.add_argument('--disable-gpu-expression', action='store_true', help='关闭GPU表达式执行器，回退到CPU求值')
    p.add_argument('--prior-level', type=int, default=2, choices=[1, 2, 3], 
                   help='先验级别: 1=最高约束(单规则4通道), 2=中度(保留三轴+姿态), 3=严格(仅位置误差/速度/角速度)')
    
    # 🔥 贝叶斯优化调参（内层参数优化）
    p.add_argument('--enable-bayesian-tuning', action='store_true', help='启用贝叶斯优化对程序常数参数进行自动调优（AAAI 2024 π-Light策略）')
    p.add_argument('--bo-batch-size', type=int, default=50, help='BO每次并行评估的参数组数（利用Isaac并行环境，默认50）')
    p.add_argument('--bo-iterations', type=int, default=3, help='BO迭代次数（默认3，总评估 batch_size × iterations 组参数）')
    
    # 保存参数
    p.add_argument('--save-path', type=str, default='01_soar/results/online_best_program.json')
    p.add_argument('--checkpoint-freq', type=int, default=50, help='检查点保存频率（默认50）')
    p.add_argument('--warm-start', type=str, default=None, help='从已有程序文件开始训练（JSON 路径）')
    p.add_argument('--elite-archive-size', type=int, default=50, help='精英程序池大小,保留Top-K最优程序（默认50）')
    # 课程学习 & 程序演化日志
    p.add_argument('--curriculum-mode', type=str, default='none', choices=['none','basic'], help='课程学习模式: none=关闭, basic=三阶段变量/算子逐步解锁')
    p.add_argument('--program-history-path', type=str, default='01_soar/results/program_history.jsonl', help='保存程序演化历史(JSON Lines)，仅在出现新best时追加')
    # 调试/诊断
    p.add_argument('--debug-rewards', action='store_true', help='开启逐步奖励与零动作统计的调试日志(影响性能)')
    
    return p.parse_args(args=argv)


if __name__ == '__main__':
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 开始训练
    trainer = OnlineTrainer(args)
    trainer.train()

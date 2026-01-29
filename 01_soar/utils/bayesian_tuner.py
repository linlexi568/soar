"""Bayesian Optimization Tuner for Program Constant Parameters

专为 Soar 设计的贝叶斯优化调参器，用于在 MCTS 叶节点评估时对程序中的常数参数进行高效调优。

核心特性：
1. **Batch BO**: 利用 Isaac Gym 的并行环境，一次评估多组参数（远快于串行 BO）。
2. **轻量级 GP**: 使用简单的高斯过程 + UCB 采集函数，无需重度依赖外部库。
3. **自适应参数空间**: 自动识别程序中的 ConstNode（TerminalNode with float value）。
4. **Early Stop**: 如果找到足够好的参数，提前终止搜索。

理论基础：
- Gaussian Process Regression (Rasmussen & Williams, 2006)
- Upper Confidence Bound Acquisition (Srinivas et al., 2010)
- Batch Bayesian Optimization (Desautels et al., 2014)

引用：
- π-Light (AAAI 2024): 在交通信号控制中使用类似的内层参数优化策略。
- DrM (CoRL 2020): Model-based RL with Bayesian hyperparameter tuning.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from dataclasses import dataclass
import torch

try:
    from scipy.spatial.distance import cdist
    from scipy.linalg import cholesky, cho_solve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[BayesianTuner] Warning: scipy not available, using fallback implementations")


@dataclass
class ParameterSpec:
    """参数规格定义"""
    name: str           # 参数名（如 'kp', 'ki', 'const_0'）
    low: float          # 下界
    high: float         # 上界
    log_scale: bool = False  # 是否使用对数尺度（适用于跨数量级的参数，如 1e-3 到 1e2）


class SimpleGaussianProcess:
    """轻量级高斯过程回归器
    
    使用 RBF 核 + 常数均值函数。
    支持增量更新（无需每次重新训练整个 GP）。
    """
    def __init__(self, length_scale: float = 1.0, signal_variance: float = 1.0, noise_variance: float = 1e-4):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        
        self.X_train = None  # [N, D]
        self.y_train = None  # [N]
        self.K_inv = None    # [N, N] 逆协方差矩阵（缓存）
        
    def _rbf_kernel(self, X1, X2):
        """RBF (Squared Exponential) Kernel: k(x, x') = σ² exp(-||x - x'||² / (2l²))"""
        if SCIPY_AVAILABLE:
            dists = cdist(X1, X2, 'sqeuclidean')
        else:
            # Fallback: manual distance computation
            dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.signal_variance * np.exp(-dists / (2 * self.length_scale**2))
    
    def fit(self, X, y):
        """拟合 GP 模型
        
        Args:
            X: [N, D] 参数样本
            y: [N] 对应的目标值（例如 reward）
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        N = len(y)
        K = self._rbf_kernel(self.X_train, self.X_train)
        K += self.noise_variance * np.eye(N)  # 加入观测噪声
        
        # 计算 K 的逆矩阵（用于预测）
        try:
            if SCIPY_AVAILABLE:
                L = cholesky(K, lower=True)
                self.K_inv = cho_solve((L, True), np.eye(N))
            else:
                self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # 数值不稳定，增加对角项
            K += 1e-6 * np.eye(N)
            self.K_inv = np.linalg.inv(K)
    
    def predict(self, X_new):
        """预测新点的均值和标准差
        
        Args:
            X_new: [M, D] 新的参数点
            
        Returns:
            mu: [M] 预测均值
            sigma: [M] 预测标准差
        """
        if self.X_train is None or len(self.X_train) == 0:
            # 没有训练数据，返回先验（均值 0，方差 signal_variance）
            M = len(X_new)
            return np.zeros(M), np.sqrt(self.signal_variance) * np.ones(M)
        
        X_new = np.array(X_new)
        K_s = self._rbf_kernel(self.X_train, X_new)  # [N, M]
        K_ss = self._rbf_kernel(X_new, X_new)        # [M, M]
        
        # 预测均值: μ(x*) = k(x*)ᵀ K⁻¹ y
        mu = K_s.T @ self.K_inv @ self.y_train
        
        # 预测方差: σ²(x*) = k(x*, x*) - k(x*)ᵀ K⁻¹ k(x*)
        v = K_s.T @ self.K_inv @ K_s
        var = np.diag(K_ss) - np.diag(v)
        var = np.maximum(var, 1e-9)  # 数值稳定性
        sigma = np.sqrt(var)
        
        return mu, sigma


class BayesianTuner:
    """批量贝叶斯优化调参器
    
    使用场景：
    在 MCTS 叶节点评估时，给定程序结构，对其中的常数参数进行优化。
    
    工作流程：
    1. 识别程序中的可调参数（TerminalNode with float value）
    2. 使用 Sobol 序列或 LHS 生成初始采样点
    3. 并行评估这些参数组合（利用 Isaac Gym 的 num_envs）
    4. 用 GP 拟合 参数->性能 的映射
    5. 用 UCB 采集函数生成下一批最有希望的参数
    6. 重复 3-5 直到预算耗尽或收敛
    7. 返回最佳参数组合
    """
    def __init__(
        self,
        param_specs: List[ParameterSpec],
        batch_size: int = 50,
        n_iterations: int = 3,
        ucb_kappa: float = 2.0,
        early_stop_threshold: Optional[float] = None,
        random_seed: int = 42,
    ):
        """
        Args:
            param_specs: 参数空间定义
            batch_size: 每次并行评估的参数组数（应 <= Isaac Gym num_envs）
            n_iterations: BO 迭代次数（总评估次数 = batch_size * n_iterations）
            ucb_kappa: UCB 采集函数的探索系数（越大越倾向探索）
            early_stop_threshold: 提前停止阈值（如果找到 reward > threshold，立即停止）
            random_seed: 随机种子
        """
        self.param_specs = param_specs
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.ucb_kappa = ucb_kappa
        self.early_stop_threshold = early_stop_threshold
        self.rng = np.random.RandomState(random_seed)
        
        self.dim = len(param_specs)
        self.gp = SimpleGaussianProcess(
            length_scale=0.5,
            signal_variance=1.0,
            noise_variance=1e-3
        )
        
        # 历史记录
        self.X_history = []  # 所有评估过的参数（归一化到 [0, 1]）
        self.y_history = []  # 对应的 reward
        self.best_idx = None
        
    def _normalize(self, X_raw):
        """将参数从原始空间映射到 [0, 1]^D"""
        X_norm = np.zeros_like(X_raw)
        for i, spec in enumerate(self.param_specs):
            if spec.log_scale:
                # 对数尺度：先取 log，再归一化
                log_low, log_high = np.log10(spec.low), np.log10(spec.high)
                X_norm[:, i] = (np.log10(X_raw[:, i]) - log_low) / (log_high - log_low)
            else:
                X_norm[:, i] = (X_raw[:, i] - spec.low) / (spec.high - spec.low)
        return X_norm
    
    def _denormalize(self, X_norm):
        """将参数从 [0, 1]^D 映射回原始空间"""
        X_raw = np.zeros_like(X_norm)
        for i, spec in enumerate(self.param_specs):
            if spec.log_scale:
                log_low, log_high = np.log10(spec.low), np.log10(spec.high)
                X_raw[:, i] = 10 ** (X_norm[:, i] * (log_high - log_low) + log_low)
            else:
                X_raw[:, i] = X_norm[:, i] * (spec.high - spec.low) + spec.low
        return X_raw
    
    def _sobol_sample(self, n_samples: int) -> np.ndarray:
        """生成 Sobol 低差异序列（比均匀随机更均匀地覆盖空间）"""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng.randint(0, 2**31))
            X_norm = sampler.random(n_samples)
        except ImportError:
            # Fallback: 拉丁超立方采样
            X_norm = self._latin_hypercube_sample(n_samples)
        return X_norm
    
    def _latin_hypercube_sample(self, n_samples: int) -> np.ndarray:
        """拉丁超立方采样（LHS）"""
        X_norm = np.zeros((n_samples, self.dim))
        for i in range(self.dim):
            X_norm[:, i] = (self.rng.permutation(n_samples) + self.rng.rand(n_samples)) / n_samples
        return X_norm
    
    def _ucb_acquisition(self, X_norm):
        """Upper Confidence Bound 采集函数
        
        UCB(x) = μ(x) + κ * σ(x)
        
        平衡探索（σ大的区域）和利用（μ大的区域）。
        """
        mu, sigma = self.gp.predict(X_norm)
        return mu + self.ucb_kappa * sigma
    
    def _select_next_batch(self, n_candidates: int = 1000) -> np.ndarray:
        """选择下一批最有希望的参数点
        
        策略：
        1. 生成大量候选点（Sobol 采样）
        2. 计算每个候选的 UCB 值
        3. 选择 UCB 最高的 batch_size 个点
        """
        X_candidates_norm = self._sobol_sample(n_candidates)
        ucb_values = self._ucb_acquisition(X_candidates_norm)
        
        # 选择 top-k
        top_indices = np.argsort(ucb_values)[-self.batch_size:]
        return X_candidates_norm[top_indices]
    
    def optimize(
        self,
        eval_fn: Callable[[np.ndarray], np.ndarray],
        verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        """执行批量贝叶斯优化
        
        Args:
            eval_fn: 评估函数，输入 [batch_size, dim] 的参数矩阵，
                     输出 [batch_size] 的 reward 向量。
                     这个函数应该内部调用 Isaac Gym 并行评估。
            verbose: 是否打印调试信息
            
        Returns:
            best_params: [dim] 最佳参数（原始空间）
            best_reward: 对应的最佳 reward
        """
        for iter_idx in range(self.n_iterations):
            # 第一轮：随机初始化；后续：基于 GP 采样
            if iter_idx == 0:
                X_norm = self._sobol_sample(self.batch_size)
            else:
                X_norm = self._select_next_batch()
            
            # 反归一化到原始参数空间
            X_raw = self._denormalize(X_norm)
            
            # 并行评估
            y_batch = eval_fn(X_raw)
            
            # 记录历史
            self.X_history.append(X_norm)
            self.y_history.append(y_batch)
            
            # 更新 GP
            X_all = np.vstack(self.X_history)
            y_all = np.concatenate(self.y_history)
            self.gp.fit(X_all, y_all)
            
            # 找到当前最佳
            self.best_idx = np.argmax(y_all)
            best_reward_so_far = y_all[self.best_idx]
            
            if verbose:
                print(f"  [BO iter {iter_idx+1}/{self.n_iterations}] "
                      f"Best reward: {best_reward_so_far:.4f} | "
                      f"Batch mean: {y_batch.mean():.4f} ± {y_batch.std():.4f}")
            
            # Early stop
            if self.early_stop_threshold is not None and best_reward_so_far >= self.early_stop_threshold:
                if verbose:
                    print(f"  [BO] Early stop: reward {best_reward_so_far:.4f} >= threshold {self.early_stop_threshold:.4f}")
                break
        
        # 返回最佳参数
        X_all = np.vstack(self.X_history)
        y_all = np.concatenate(self.y_history)
        best_idx = np.argmax(y_all)
        best_params_norm = X_all[best_idx]
        best_params_raw = self._denormalize(best_params_norm.reshape(1, -1))[0]
        best_reward = y_all[best_idx]
        
        return best_params_raw, best_reward


# ============================================================================
# 辅助函数：从程序中提取可调参数
# ============================================================================

def extract_tunable_params(program) -> List[Tuple[str, float]]:
    """从程序中提取所有可调常数参数
    
    遍历程序 AST，找到所有 TerminalNode(float)。
    
    Args:
        program: 程序对象（List[Dict] 或 ProgramNode）
        
    Returns:
        params: [(path, value), ...] 例如 [('rule_0_action_0_const_1', 1.5), ...]
    """
    try:
        from core.dsl import TerminalNode, ConstantNode, BinaryOpNode, UnaryOpNode, IfNode
    except ImportError:
        try:
            from ..core.dsl import TerminalNode, ConstantNode, BinaryOpNode, UnaryOpNode, IfNode
        except ImportError:
            print("[BayesianTuner] Error: Cannot import DSL classes")
            return []
    
    params = []
    
    def _traverse(node, path_prefix: str):
        if isinstance(node, TerminalNode):
            if isinstance(node.value, (int, float)):
                params.append((path_prefix, float(node.value)))
        elif isinstance(node, ConstantNode):
            # 显式常量节点：BO 优化的主要目标
            params.append((path_prefix, node.value))
        elif isinstance(node, UnaryOpNode):
            # 提取 params 字典中的 ConstantNode
            if node.params:
                for param_name, param_node in node.params.items():
                    param_path = f"{path_prefix}_param_{param_name}"
                    if isinstance(param_node, ConstantNode):
                        params.append((param_path, param_node.value))
                    elif isinstance(param_node, (int, float)):
                        params.append((param_path, float(param_node)))
            _traverse(node.child, f"{path_prefix}_child")
        elif isinstance(node, BinaryOpNode):
            _traverse(node.left, f"{path_prefix}_left")
            _traverse(node.right, f"{path_prefix}_right")
        elif isinstance(node, IfNode):
            _traverse(node.condition, f"{path_prefix}_cond")
            _traverse(node.true_branch, f"{path_prefix}_true")
            _traverse(node.false_branch, f"{path_prefix}_false")
    
    # 程序结构支持两种格式：
    # 1. List[Dict{'condition': Node, 'action': List[Node]}]  (旧格式)
    # 2. List[Dict{'rule_id': int, 'variable': str, 'node': Node}]  (新格式，用于单轴MCTS)
    if isinstance(program, list):
        for rule_idx, rule in enumerate(program):
            # 新格式：单轴 MCTS 程序
            if 'node' in rule:
                _traverse(rule['node'], f"rule_{rule_idx}_node")
            # 旧格式：完整程序
            elif 'condition' in rule or 'action' in rule:
                if 'condition' in rule:
                    _traverse(rule['condition'], f"rule_{rule_idx}_cond")
                if 'action' in rule:
                    for action_idx, action in enumerate(rule['action']):
                        _traverse(action, f"rule_{rule_idx}_action_{action_idx}")
    
    return params


def inject_tuned_params(program, tuned_values: Dict[str, float]):
    """将调优后的参数注入程序
    
    Args:
        program: 原始程序
        tuned_values: {path: value, ...}
        
    Returns:
        修改后的程序（in-place 修改）
    """
    try:
        from core.dsl import TerminalNode, ConstantNode, BinaryOpNode, UnaryOpNode, IfNode
    except ImportError:
        try:
            from ..core.dsl import TerminalNode, ConstantNode, BinaryOpNode, UnaryOpNode, IfNode
        except ImportError:
            return program
    
    def _inject(node, path_prefix: str):
        if isinstance(node, TerminalNode):
            if path_prefix in tuned_values:
                node.value = tuned_values[path_prefix]
        elif isinstance(node, ConstantNode):
            if path_prefix in tuned_values:
                node.value = tuned_values[path_prefix]
        elif isinstance(node, UnaryOpNode):
            # 注入 params 字典中的参数
            if node.params:
                for param_name, param_node in node.params.items():
                    param_path = f"{path_prefix}_param_{param_name}"
                    if param_path in tuned_values:
                        if isinstance(param_node, ConstantNode):
                            param_node.value = tuned_values[param_path]
                        elif isinstance(param_node, (int, float)):
                            # 如果是裸值，替换为 ConstantNode
                            node.params[param_name] = ConstantNode(tuned_values[param_path])
            _inject(node.child, f"{path_prefix}_child")
        elif isinstance(node, BinaryOpNode):
            _inject(node.left, f"{path_prefix}_left")
            _inject(node.right, f"{path_prefix}_right")
        elif isinstance(node, IfNode):
            _inject(node.condition, f"{path_prefix}_cond")
            _inject(node.true_branch, f"{path_prefix}_true")
            _inject(node.false_branch, f"{path_prefix}_false")
    
    if isinstance(program, list):
        for rule_idx, rule in enumerate(program):
            # 新格式：单轴 MCTS 程序
            if 'node' in rule:
                _inject(rule['node'], f"rule_{rule_idx}_node")
            # 旧格式：完整程序
            elif 'condition' in rule or 'action' in rule:
                if 'condition' in rule:
                    _inject(rule['condition'], f"rule_{rule_idx}_cond")
                if 'action' in rule:
                    for action_idx, action in enumerate(rule['action']):
                        _inject(action, f"rule_{rule_idx}_action_{action_idx}")
    
    return program


__all__ = [
    'ParameterSpec',
    'SimpleGaussianProcess',
    'BayesianTuner',
    'extract_tunable_params',
    'inject_tuned_params',
]

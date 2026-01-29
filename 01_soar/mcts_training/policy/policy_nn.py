"""Lightweight policy network for MCTS edit-type prior (torch-optional).

- If PyTorch is available, define a tiny MLP that maps simple node/agent features
  to a weight vector over edit types.
- If PyTorch is not available, expose stubs and safe fallbacks.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING

EDIT_TYPES: List[str] = [
    'add_rule','remove_rule','mutate_action','tweak_multiplier','micro_tweak',
    'promote_rule','duplicate_rule','swap_rules','macro_triplet_tune'
]

TORCH_AVAILABLE = False
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    if TYPE_CHECKING:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]

def get_default_device(requested: str | None = None) -> str:
    if not TORCH_AVAILABLE:
        return 'cpu'
    if requested and requested.lower() == 'cuda' and hasattr(torch, 'cuda') and torch.cuda.is_available():  # type: ignore[attr-defined]
        return 'cuda'
    return 'cpu'

def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-float(x)))

class PolicyNNModel(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Tiny MLP: inputs -> hidden -> policy_logits(len(EDIT_TYPES)).
    Torch-optional: when torch is unavailable, exposes a shim.
    """
    def __init__(self, in_dim: int = 8, hidden: int = 32, out_dim: int = len(EDIT_TYPES)):
        if not TORCH_AVAILABLE:
            # shim attributes
            self.in_dim = in_dim; self.hidden = hidden; self.out_dim = out_dim
            return
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)  # type: ignore[call-arg, attr-defined]
        self.fc2 = nn.Linear(hidden, hidden)  # type: ignore[call-arg, attr-defined]
        self.out = nn.Linear(hidden, out_dim)  # type: ignore[call-arg, attr-defined]

    def forward(self, x):  # type: ignore[override]
        if not TORCH_AVAILABLE:
            # return uniform logits (zeros)
            shape0 = x.shape[0] if hasattr(x, 'shape') else 1
            return [[0.0 for _ in range(len(EDIT_TYPES))] for _ in range(shape0)]
        h = F.relu(self.fc1(x))  # type: ignore[attr-defined]
        h = F.relu(self.fc2(h))  # type: ignore[attr-defined]
        return self.out(h)

class PolicyValueNNModel(PolicyNNModel):  # type: ignore[misc]
    """Policy+Value network: shares trunk, outputs (policy_logits, value)."""
    def __init__(self, in_dim: int = 8, hidden: int = 32, out_dim: int = len(EDIT_TYPES)):
        super().__init__(in_dim=in_dim, hidden=hidden, out_dim=out_dim)  # type: ignore[misc]
        if TORCH_AVAILABLE:
            self.vhead = nn.Linear(hidden, 1)  # type: ignore[attr-defined]

    def forward(self, x):  # type: ignore[override]
        if not TORCH_AVAILABLE:
            # zeros for logits and 0.0 for value
            shape0 = x.shape[0] if hasattr(x, 'shape') else 1
            logits = [[0.0 for _ in range(len(EDIT_TYPES))] for _ in range(shape0)]
            vals = [0.0 for _ in range(shape0)]
            return logits, vals
        h = F.relu(self.fc1(x))  # type: ignore[attr-defined]
        h = F.relu(self.fc2(h))  # type: ignore[attr-defined]
        logits = self.out(h)  # type: ignore[attr-defined]
        v = self.vhead(h)  # type: ignore[attr-defined]
        return logits, v


## NOTE: PolicyValueNNLarge 已弃用并被 GNN v2 分层网络取代。
## 为保持向后兼容（旧 checkpoint 载入时避免崩溃），这里保留一个瘦身占位符。
class PolicyValueNNLarge(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):
        if TORCH_AVAILABLE:
            super().__init__()
            # 极简单层，输出恒零 (logits, value)；用于旧代码反序列化不再实际训练。
            self.dummy = nn.Parameter(torch.zeros(1))  # type: ignore[attr-defined]
        else:
            self.dummy = 0.0  # type: ignore[assignment]

    def forward(self, x):  # type: ignore[override]
        if not TORCH_AVAILABLE:
            shape0 = getattr(x, 'shape', [1])[0] if hasattr(x, 'shape') else 1
            return [[0.0 for _ in range(len(EDIT_TYPES))] for _ in range(shape0)], [0.0 for _ in range(shape0)]
        b = x.shape[0] if hasattr(x, 'shape') else 1
        logits = torch.zeros(b, len(EDIT_TYPES), device=self.dummy.device)  # type: ignore[attr-defined]
        value = torch.zeros(b, 1, device=self.dummy.device)  # type: ignore[attr-defined]
        return logits, value

def build_features(node, agent) -> List[float]:
    """Build a compact feature vector for the current node.
    Scales inputs roughly into [0,1]. No dependence on torch.
    """
    try:
        n_rules = float(len(getattr(node, 'program', []) or []))
    except Exception:
        n_rules = 0.0
    try:
        depth = float(getattr(node, 'depth', 0) or 0)
    except Exception:
        depth = 0.0
    try:
        visits = float(getattr(node, 'visits', 0) or 0)
    except Exception:
        visits = 0.0
    try:
        parent_visits = float(getattr(getattr(node, 'parent', None), 'visits', 0) or 0)
    except Exception:
        parent_visits = 0.0
    try:
        epsilon = float(getattr(agent, 'epsilon', 0.0) or 0.0)
    except Exception:
        epsilon = 0.0
    try:
        min_guard = float(getattr(agent, '_min_rules_guard_effective', getattr(agent, '_min_rules_guard', 2)))
    except Exception:
        min_guard = 2.0
    try:
        max_rules = float(getattr(agent, '_max_rules', 8))
    except Exception:
        max_rules = 8.0
    try:
        add_bias = float(getattr(agent, '_add_rule_bias_base', 2))
    except Exception:
        add_bias = 2.0
    # normalize roughly
    def nz(x): return float(max(0.0, x))
    feats = [
        nz(n_rules) / max(1.0, max_rules),
        nz(depth) / 32.0,
        nz(visits) / 64.0,
        nz(parent_visits) / 64.0,
        float(epsilon),
        nz(min_guard) / max(1.0, max_rules),
        nz(max_rules) / 64.0,
        nz(add_bias) / 8.0,
    ]
    return feats

def softmax(xs: List[float]) -> List[float]:
    import math
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(xs) for _ in xs]
    return [e / s for e in exps]

def load_model(path: str, device: str = 'cpu') -> Optional[PolicyNNModel]:
    if not TORCH_AVAILABLE:
        return None
    try:
        model = PolicyNNModel()
        sd = torch.load(path, map_location=device)  # type: ignore[attr-defined]
        model.load_state_dict(sd, strict=False)  # type: ignore[arg-type]
        model.to(device)  # type: ignore[attr-defined]
        model.eval()
        return model
    except Exception:
        return None

def load_model_value(path: str, device: str = 'cpu') -> Optional[PolicyValueNNModel]:
    """Attempt to load a PolicyValueNNModel; falls back to None on failure."""
    if not TORCH_AVAILABLE:
        return None
    try:
        model = PolicyValueNNModel()
        sd = torch.load(path, map_location=device)  # type: ignore[attr-defined]
        # allow older checkpoints without value head
        model.load_state_dict(sd, strict=False)  # type: ignore[arg-type]
        model.to(device)  # type: ignore[attr-defined]
        model.eval()
        return model
    except Exception:
        return None

def infer_prior_weights(model: PolicyNNModel | None, node, agent, available_types: List[str], device: str = 'cpu') -> Dict[str, float]:
    """Return a dict {edit_type: weight in [0,1]} over available types.
    If model is None or torch missing, returns uniform weights.
    """
    avail = [t for t in available_types if isinstance(t, str)]
    if not avail:
        return {}
    if (model is None) or (not TORCH_AVAILABLE):
        u = 1.0 / float(len(avail))
        return {t: u for t in avail}
    try:
        feats = build_features(node, agent)
        import numpy as _np
        x = torch.tensor([feats], dtype=torch.float32, device=device)  # type: ignore[attr-defined]
        with torch.no_grad():  # type: ignore[attr-defined]
            logits = model(x)
            if hasattr(logits, 'detach'):
                logits = logits.detach().cpu().numpy().reshape(-1)  # type: ignore[no-untyped-call]
            else:
                logits = _np.array(logits).reshape(-1)
        # Map logits to EDIT_TYPES then select available
        logit_map = {et: float(logits[i]) if i < len(logits) else 0.0 for i, et in enumerate(EDIT_TYPES)}
        vals = [logit_map.get(t, 0.0) for t in avail]
        probs = softmax(vals)
        # scale to [0,1]
        return {t: float(p) for t, p in zip(avail, probs)}
    except Exception:
        u = 1.0 / float(len(avail))
        return {t: u for t in avail}

def infer_p_and_v(model: Any | None, node, agent, available_types: List[str], device: str = 'cpu') -> Tuple[Dict[str, float], float]:
    """Return (policy_weights, value_estimate).
    - policy_weights: {edit_type: weight in [0,1]} for available types.
    - value_estimate: scalar in R (not bounded), NN's estimate of final reward.
    Works with PolicyValueNNModel; if only PolicyNNModel is provided or torch missing, returns value=0.0.
    """
    try:
        # policy part
        pw = infer_prior_weights(model if isinstance(model, PolicyNNModel) else None, node, agent, available_types, device=device)
        # value part
        val = 0.0
        if TORCH_AVAILABLE and (model is not None):
            try:
                # support both PV model and plain policy model
                feats = build_features(node, agent)
                import numpy as _np
                x = torch.tensor([feats], dtype=torch.float32, device=device)  # type: ignore[attr-defined]
                with torch.no_grad():  # type: ignore[attr-defined]
                    out = model(x)
                # out may be logits or (logits, v)
                if isinstance(out, tuple) and len(out) >= 2:
                    v = out[1]
                    if hasattr(v, 'detach'):
                        val = float(v.detach().cpu().numpy().reshape(-1)[0])  # type: ignore[no-untyped-call]
                    else:
                        if isinstance(v, (list, tuple)) and len(v) > 0:
                            try:
                                val = float(v[0])
                            except Exception:
                                val = 0.0
                        elif isinstance(v, (int, float)):
                            val = float(v)
                        else:
                            val = 0.0
            except Exception:
                val = 0.0
        return pw, float(val)
    except Exception:
        # robust fallback
        return infer_prior_weights(None, node, agent, available_types, device=device), 0.0

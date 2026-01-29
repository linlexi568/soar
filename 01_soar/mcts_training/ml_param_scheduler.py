from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import math


@dataclass
class MCTSContext:
    """Lightweight signals from the loop/agent for ML schedulers.

    Fields are intentionally generic to avoid tight coupling with agent internals.
    """
    iter_idx: int
    total_target: int
    progress: float                         # 0~1 logical progress
    best_reward: float                      # current best (short or gated)
    best_reward_delta: float                # improvement vs previous best (this loop)
    seconds_since_improve: float            # wall-clock seconds since last improve (if available; else -1)
    iters_since_improve: int                # iterations since last improve (if available)
    rule_count: int                         # current best program rule count
    epsilon: float                          # current epsilon
    stagnation_window: int                  # configured window (iters) for rebound


class BaseScheduler:
    """Interface for ML schedulers controlling MCTS hyper-parameters.

    Implementations should return a dict mapping attribute name -> new value or delta value.
    """
    def __init__(self, strategy: str = "delta", allowed: Optional[set[str]] = None,
                 safe_bounds: Optional[Dict[str, Tuple[float, float]]] = None, log: bool = False) -> None:
        self.strategy = strategy  # 'delta' or 'absolute'
        self.allowed = set(allowed) if allowed else set()
        self.safe_bounds = dict(safe_bounds or {})
        self.log = log

    def step(self, ctx: MCTSContext) -> Dict[str, Any]:
        raise NotImplementedError

    def _clip(self, k: str, v: float) -> float:
        b = self.safe_bounds.get(k)
        if b is None:
            return v
        lo, hi = b
        lo_f = float(lo); hi_f = float(hi)
        if lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f
        return max(lo_f, min(hi_f, float(v)))

    def _maybe(self, k: str, v: Any) -> Optional[Tuple[str, Any]]:
        if self.allowed and (k not in self.allowed):
            return None
        return (k, v)


class HeuristicScheduler(BaseScheduler):
    """A simple, fast, torch-free controller based on progress/stagnation signals.

    Rules of thumb:
    - Early stage: encourage exploration (Dirichlet noise, higher epsilon, more add_rule bias)
    - Stagnation: increase prior usage and PUCT exploration; widen PW a bit; allow more structure
    - Later stage with stability: reduce noise/prior strength; increase micro-tuning preference
    """
    def step(self, ctx: MCTSContext) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        p = max(0.0, min(1.0, ctx.progress))
        long_stall = (ctx.iters_since_improve >= max(20, ctx.stagnation_window))
        short_stall = (ctx.iters_since_improve >= max(8, ctx.stagnation_window // 3))

        # 1) Root Dirichlet noise schedule (only meaningful if PUCT is on)
        if p < 0.25:
            upd = self._maybe('_dirichlet_eps', 0.25)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
            upd = self._maybe('_dirichlet_alpha', 0.3)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        elif p < 0.60:
            upd = self._maybe('_dirichlet_eps', 0.10)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        else:
            upd = self._maybe('_dirichlet_eps', 0.0)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])

        # 2) PUCT on/off & strength
        if short_stall or long_stall:
            upd = self._maybe('_puct_enable', True)
            if upd: updates[upd[0]] = upd[1]
            target_cpuct = 1.6 if long_stall else 1.25
            upd = self._maybe('_puct_c', target_cpuct)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        else:
            # moderate PUCT usage mid-run
            upd = self._maybe('_puct_enable', True if p >= 0.15 else False)
            if upd: updates[upd[0]] = upd[1]
            upd = self._maybe('_puct_c', 1.0 if p < 0.8 else 0.8)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])

        # 3) Progressive Widening parameters
        if long_stall:
            # widen options a bit to escape local optima
            upd = self._maybe('pw_c', 1.6)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
            upd = self._maybe('pw_alpha', 0.88)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        elif short_stall:
            upd = self._maybe('pw_c', 1.35)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
            upd = self._maybe('pw_alpha', 0.85)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        else:
            # default balanced
            upd = self._maybe('pw_c', 1.1)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
            upd = self._maybe('pw_alpha', 0.82)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])

        # 4) Structure encouragement when rules are too few or we stall early
        if (ctx.rule_count < 3 and p < 0.5) or short_stall:
            upd = self._maybe('_add_rule_bias_base', 8 if long_stall else 6)
            if upd: updates[upd[0]] = int(self._clip(upd[0], upd[1]))
            upd = self._maybe('_full_action_prob', 0.65 if long_stall else 0.55)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
            # tie-break prefers more rules when almost tied
            upd = self._maybe('_prefer_more_rules_tie_delta', 0.04 if long_stall else 0.02)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
            upd = self._maybe('_prefer_fewer_rules_tie_delta', 0.0)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        else:
            # later: prefer compactness slightly if progress is high and no stall
            if p > 0.7 and (not short_stall) and (not long_stall):
                upd = self._maybe('_prefer_fewer_rules_tie_delta', 0.02)
                if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
                upd = self._maybe('_prefer_more_rules_tie_delta', 0.0)
                if upd: updates[upd[0]] = self._clip(upd[0], upd[1])

        # 5) Policy prior usage: stronger during stall, weaker when things improve well
        if long_stall:
            upd = self._maybe('_edit_prior_c', 0.6)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        elif short_stall:
            upd = self._maybe('_edit_prior_c', 0.4)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        else:
            upd = self._maybe('_edit_prior_c', 0.15 if p < 0.5 else 0.0)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])

        # 6) Value head mix: small later-stage assist
        if p > 0.6 and not long_stall:
            upd = self._maybe('_value_mix_lambda', 0.10 if p < 0.85 else 0.15)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        else:
            upd = self._maybe('_value_mix_lambda', 0.0)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])

        # 7) Exploration epsilon ceiling (outside agent's internal ramp)
        if long_stall:
            upd = self._maybe('_epsilon_max', 0.35)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])
        else:
            upd = self._maybe('_epsilon_max', 0.25 if p < 0.5 else 0.18)
            if upd: updates[upd[0]] = self._clip(upd[0], upd[1])

        return updates


# A canonical key order for NN scheduler outputs and logging/CSV headers.
# Keep this list synchronized with any consumer (training/analysis scripts).
KEY_ORDER: List[str] = [
    'pw_alpha', 'pw_c', '_puct_c', '_edit_prior_c', '_dirichlet_eps', '_full_action_prob',
    '_prefer_more_rules_tie_delta', '_prefer_fewer_rules_tie_delta', '_add_rule_bias_base',
    '_value_mix_lambda', '_epsilon_max'
]


def parse_bounds_spec(spec: Optional[str]) -> Dict[str, Tuple[float, float]]:
    """Parse a bounds spec like: "pw_alpha:0.4,1.0;pw_c:0.8,2.0;_puct_c:0.5,2.5".
    Returns a dict of name -> (lo, hi). Invalid entries are ignored.
    """
    out: Dict[str, Tuple[float, float]] = {}
    if not spec:
        return out
    for seg in str(spec).split(';'):
        seg = seg.strip()
        if not seg:
            continue
        if ':' not in seg:
            continue
        name, rng = seg.split(':', 1)
        name = name.strip()
        try:
            lo_s, hi_s = rng.split(',', 1)
            lo = float(lo_s.strip()); hi = float(hi_s.strip())
            out[name] = (lo, hi)
        except Exception:
            continue
    return out


def apply_mcts_param_updates(agent: Any, updates: Dict[str, Any],
                             strategy: str = 'delta',
                             bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                             int_keys: Optional[set[str]] = None,
                             log: bool = False) -> None:
    """Safely apply updates to agent attributes.

    - strategy 'delta': new = clip(old + delta)
      strategy 'absolute': new = clip(value)
    - bounds may specify per-key (lo, hi) clamping
    - int_keys are cast to int after clamping
    """
    b = dict(bounds or {})
    int_keys = set(int_keys or set())
    for k, v in (updates or {}).items():
        if not hasattr(agent, k):
            continue
        try:
            old = getattr(agent, k)
            # Booleans are absolute
            if isinstance(v, bool) or isinstance(old, bool):
                new_val = bool(v)
            else:
                if strategy == 'delta':
                    try:
                        base = float(old)
                    except Exception:
                        base = 0.0
                    nv = base + float(v)
                else:
                    nv = float(v)
                if k in b:
                    lo, hi = b[k]
                    lo_f = float(lo); hi_f = float(hi)
                    if lo_f > hi_f:
                        lo_f, hi_f = hi_f, lo_f
                    nv = max(lo_f, min(hi_f, float(nv)))
                new_val = int(round(nv)) if k in int_keys else float(nv)
            setattr(agent, k, new_val)
            if log:
                try:
                    print(f"[ML-Sched] {k}: {old} -> {getattr(agent, k)}")
                except Exception:
                    pass
        except Exception:
            continue


# Optional: a minimal NN wrapper (lazy Torch import). Kept simple on purpose.
class NNScheduler(BaseScheduler):
    def __init__(self, model_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        try:
            import torch  # type: ignore
            self.torch = torch
        except Exception:
            self.torch = None

    def _ensure_loaded(self):
        if self.model is not None or self.torch is None:
            return
        # Very small MLP expected; users can swap with their own checkpoint logic
        try:
            self.model = self.torch.jit.load(self.model_path, map_location=self.device)  # type: ignore[attr-defined]
            self.model.eval()
        except Exception:
            self.model = None

    def step(self, ctx: MCTSContext) -> Dict[str, Any]:
        if self.torch is None:
            # Fallback to heuristic if torch not available
            return HeuristicScheduler(strategy=self.strategy, allowed=self.allowed,
                                      safe_bounds=self.safe_bounds, log=self.log).step(ctx)
        self._ensure_loaded()
        if self.model is None:
            return {}
        import numpy as np  # lightweight dep, generally available
        x = np.array([
            ctx.progress,
            ctx.best_reward,
            ctx.best_reward_delta,
            float(ctx.seconds_since_improve if ctx.seconds_since_improve >= 0 else 0.0),
            float(ctx.iters_since_improve),
            float(ctx.rule_count),
            float(ctx.epsilon),
        ], dtype=np.float32)
        with self.torch.no_grad():  # type: ignore[attr-defined]
            t = self.torch.from_numpy(x).to(self.device)  # type: ignore[attr-defined]
            y = self.model(t)  # expects a dict-like or flat vector; keep generic
        # Interpret outputs: expect a flat vector aligned to a fixed key order
        # Use the shared KEY_ORDER constant to avoid drift across modules
        updates: Dict[str, Any] = {}
        try:
            y_np = y.detach().cpu().numpy().reshape(-1)  # type: ignore[attr-defined]
            for i, k in enumerate(KEY_ORDER):
                if i >= len(y_np):
                    break
                upd = self._maybe(k, float(y_np[i]))
                if upd:
                    # clip absolute; when strategy == 'delta', clipping applies to (old+delta) in apply()
                    updates[upd[0]] = self._clip(upd[0], upd[1])
        except Exception:
            return {}
        return updates

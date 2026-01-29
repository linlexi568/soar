"""Lightweight prior scoring utilities for program structure and stability.

These heuristics are shared between BatchEvaluator (reward shaping) and
MCTS (selection bias) so we have a single place to tweak priors.

All functions operate on the canonical segmented-program representation:
- A program is a list of rule dicts.
- Each rule may contain AST nodes (TerminalNode, BinaryOpNode, etc.) or
  serialized dict-like nodes depending on the pipeline stage.

Outputs are normalized to [0, 1] to simplify weighting.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

try:  # pragma: no cover - optional import for runtime convenience
    from core.dsl import TerminalNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore
except Exception:  # pragma: no cover
    TerminalNode = UnaryOpNode = BinaryOpNode = IfNode = None  # type: ignore

__all__ = [
    "compute_prior_scores",
    "describe_prior_scores",
    "PRIOR_PROFILES",
]

# Keywords for stability-related temporal operators in our DSL
_TEMPORAL_KEYWORDS = (
    "ema",
    "smooth",
    "delay",
    "diff",
    "rate",
    "clamp",
    "deadzone",
    "rate_limit",
    "smoothstep",
)
_SATURATION_KEYWORDS = ("clamp", "deadzone", "rate", "smoothstep")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _child_nodes(node: Any) -> Iterable[Any]:
    if node is None:
        return []
    children: List[Any] = []
    if UnaryOpNode is not None and isinstance(node, UnaryOpNode):
        children.append(node.child)
    elif BinaryOpNode is not None and isinstance(node, BinaryOpNode):
        children.extend([node.left, node.right])
    elif IfNode is not None and isinstance(node, IfNode):
        children.extend([node.condition, node.then_branch, node.else_branch])
    elif isinstance(node, dict):
        for key in ("child", "left", "right", "condition", "then_branch", "else_branch", "true_branch", "false_branch"):
            if key in node:
                children.append(node[key])
    return [c for c in children if c is not None]


def _iter_nodes(root: Any) -> Iterable[Any]:
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        yield node
        stack.extend(_child_nodes(node))


def _terminal_variable(node: Any) -> Optional[str]:
    value = _get_attr(node, "value")
    if isinstance(value, str):
        return value
    if isinstance(node, dict):
        if isinstance(node.get("var"), str):
            return node["var"]
        if node.get("type") == "variable":
            name = node.get("name")
            if isinstance(name, str):
                return name
    return None


def _terminal_constant(node: Any) -> Optional[float]:
    value = _get_attr(node, "value")
    if _is_number(value):
        return float(value)
    if isinstance(node, dict) and node.get("type") == "const":
        val = node.get("value")
        if _is_number(val):
            return float(val)
    return None


def _base_name(op: Optional[str]) -> Optional[str]:
    if not isinstance(op, str):
        return None
    return op.split(":", 1)[0]


def _estimate_condition_narrowness(program: Sequence[Dict[str, Any]]) -> float:
    total = 0.0
    count = 0
    for rule in program or []:
        cond = rule.get("condition")
        if cond is None:
            continue
        op = _get_attr(cond, "op")
        base = _base_name(op)
        if base not in ("<", ">"):
            continue
        right = _get_attr(cond, "right")
        threshold = _terminal_constant(right)
        if threshold is None:
            continue
        left = _get_attr(cond, "left")
        var_name = _terminal_variable(left)
        if base == "<":
            cap = 1.0
            if var_name in ("pos_err_z", "pos_err_xy", "ang_vel_mag", "rpy_err_mag"):
                cap = {"pos_err_z": 1.0, "pos_err_xy": 1.6, "ang_vel_mag": 2.5, "rpy_err_mag": 1.8}[var_name]
            score = 1.0 - min(1.0, max(0.0, threshold) / max(1e-6, cap))
        else:  # '>'
            ref = 0.2
            if var_name in ("pos_err_z", "pos_err_xy"):
                ref = 0.15
            score = min(1.0, max(0.0, threshold) / (3.0 * ref))
        total += max(0.0, min(1.0, score))
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _collect_variables(program: Sequence[Dict[str, Any]]) -> List[str]:
    vars_found: set[str] = set()
    for rule in program or []:
        cond = rule.get("condition")
        for node in _iter_nodes(cond):
            var_name = _terminal_variable(node)
            if var_name:
                vars_found.add(var_name)
        for action in rule.get("action", []) or []:
            # action usually BinaryOpNode('set', TerminalNode(key), expr)
            expr = _get_attr(action, "right", None)
            if expr is None and isinstance(action, dict):
                expr = action.get("expr")
            for node in _iter_nodes(expr):
                var_name = _terminal_variable(node)
                if var_name:
                    vars_found.add(var_name)
    return sorted(vars_found)


def _count_action_ops(program: Sequence[Dict[str, Any]]) -> Tuple[int, int, int]:
    total = 0
    temporal = 0
    saturation = 0
    for rule in program or []:
        for action in rule.get("action", []) or []:
            expr = _get_attr(action, "right", None)
            if expr is None and isinstance(action, dict):
                expr = action.get("expr")
            for node in _iter_nodes(expr):
                op = _get_attr(node, "op")
                base = _base_name(op)
                if base is None:
                    continue
                total += 1
                if base in _TEMPORAL_KEYWORDS:
                    temporal += 1
                if base in _SATURATION_KEYWORDS:
                    saturation += 1
    return total, temporal, saturation


def compute_prior_scores(program: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Return normalized structure/stability scores for a program."""
    rules = len(program or [])
    rule_score = min(1.0, rules / 4.0)
    variables = _collect_variables(program)
    var_score = min(1.0, len(variables) / 6.0)
    narrow_score = _estimate_condition_narrowness(program)
    structure = 0.4 * rule_score + 0.35 * var_score + 0.25 * narrow_score

    total_ops, temporal_ops, saturation_ops = _count_action_ops(program)
    if total_ops <= 0:
        stability = 0.0
    else:
        stability = 0.6 * (temporal_ops / total_ops) + 0.4 * (saturation_ops / total_ops)

    return {
        "structure": float(max(0.0, min(1.0, structure))),
        "stability": float(max(0.0, min(1.0, stability))),
        "rule_score": float(rule_score),
        "variable_score": float(var_score),
        "narrow_score": float(narrow_score),
        "temporal_ratio": float(temporal_ops / total_ops) if total_ops > 0 else 0.0,
        "saturation_ratio": float(saturation_ops / total_ops) if total_ops > 0 else 0.0,
    }


def describe_prior_scores(program: Sequence[Dict[str, Any]]) -> str:
    scores = compute_prior_scores(program)
    lines = ["Prior scores:"]
    for key in ("structure", "stability", "rule_score", "variable_score", "narrow_score", "temporal_ratio", "saturation_ratio"):
        lines.append(f"  {key}: {scores[key]:.4f}")
    return "\n".join(lines)


# Preset combinations for quick experiments / CLI profiles
PRIOR_PROFILES: Dict[str, Tuple[float, float]] = {
    "none": (0.0, 0.0),
    "structure": (0.35, 0.0),
    "structure_stability": (0.35, 0.20),
}

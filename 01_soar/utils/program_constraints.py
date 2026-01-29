"""Hard control-theoretic constraints for DSL programs.

This module encodes channel-specific variable whitelists and helpers that
can be reused across the MCTS search as well as the evaluator to make sure
unphysical control laws never get simulated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

try:
    from ..core.dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode
except Exception:  # pragma: no cover - fallback when executed as script
    from core.dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode  # type: ignore

# Outputs that appear on the left-hand side of set operations
CONTROL_OUTPUTS = {"u_fz", "u_tx", "u_ty", "u_tz"}

# Penalty used when a program violates the hard constraints.
HARD_CONSTRAINT_PENALTY = -1e6

# Base feature groups to keep definitions compact
POSITION_ERRS = {
    "pos_err_x",
    "pos_err_y",
    "pos_err_z",
    "pos_err_xy",
    "pos_err_z_abs",
}
VELOCITIES = {
    "vel_x",
    "vel_y",
    "vel_z",
    "vel_err",
}
ANGULAR_VELS = {
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
    "ang_vel",
    "ang_vel_mag",
}
ATTITUDE_ERRS = {
    "err_p_roll",
    "err_p_pitch",
    "err_p_yaw",
    "rpy_err_mag",
}
INTEGRALS = {
    "err_i_x",
    "err_i_y",
    "err_i_z",
    "err_i_roll",
    "err_i_pitch",
    "err_i_yaw",
}
DERIVATIVES = {
    "err_d_x",
    "err_d_y",
    "err_d_z",
    "err_d_roll",
    "err_d_pitch",
    "err_d_yaw",
}

# Channel-specific whitelists. Each set enumerates the state variables a given
# actuator is allowed to reference.
# NOTE: Keep these tight to avoid pathological cross-couplings. In particular,
#       thrust should only depend on z-axis errors and (optionally) roll/pitch
#       attitude terms for leveling. Do NOT allow xy position/velocity here.
CHANNEL_ALLOWED_INPUTS: Dict[str, Set[str]] = {
    # 独立通道模式 (Independent Channel Mode)
    # 每个通道使用对应轴的物理变量
    "u_tx": {  # Roll torque
        "err_p_roll", "ang_vel_x", "err_i_roll", "err_d_roll",
        # 允许外环位置/速度误差（用于轨迹跟踪前馈/反馈）
        "pos_err_y", "vel_y"
    },
    "u_ty": {  # Pitch torque
        "err_p_pitch", "ang_vel_y", "err_i_pitch", "err_d_pitch",
        # 允许外环位置/速度误差（与u_tx镜像）
        "pos_err_x", "vel_x"
    },
    "u_tz": {  # Yaw torque
        "err_p_yaw", "ang_vel_z", "err_i_yaw", "err_d_yaw"
    },
    "u_fz": {  # Vertical thrust
        "pos_err_z", "vel_z", "vel_err", "err_i_z", "err_d_z"
    }
}



def _collect_state_variables(node: ProgramNode, bucket: Set[str]) -> None:
    """Recursively collect state variable names referenced by an AST node."""
    if node is None:
        return
    if isinstance(node, TerminalNode):
        val = node.value
        if isinstance(val, str) and val not in CONTROL_OUTPUTS:
            bucket.add(val)
        return
    if isinstance(node, UnaryOpNode):
        _collect_state_variables(node.child, bucket)
        return
    if isinstance(node, BinaryOpNode):
        _collect_state_variables(node.left, bucket)
        _collect_state_variables(node.right, bucket)
        return
    if isinstance(node, IfNode):
        _collect_state_variables(node.condition, bucket)
        _collect_state_variables(node.then_branch, bucket)
        _collect_state_variables(node.else_branch, bucket)


def allowed_variables_for_channel(channel: str, available: List[str]) -> List[str]:
    """Return the subset of DSL variables that a channel is allowed to use."""
    allowed = CHANNEL_ALLOWED_INPUTS.get(channel)
    if not allowed:
        return list(available)
    subset = [v for v in available if v in allowed]
    return subset


def validate_action_channel(action_node: ProgramNode) -> Tuple[bool, str]:
    """Validate a single Binary(set, u_*, expr) node against the whitelist."""
    if not isinstance(action_node, BinaryOpNode) or action_node.op != 'set':
        return True, ""
    left = action_node.left
    if not isinstance(left, TerminalNode) or not isinstance(left.value, str):
        return False, "action missing output terminal"
    channel = left.value
    if channel not in CONTROL_OUTPUTS:
        return False, f"unknown actuator '{channel}'"
    expr = action_node.right
    used_vars: Set[str] = set()
    _collect_state_variables(expr, used_vars)
    allowed = CHANNEL_ALLOWED_INPUTS.get(channel)
    if allowed is None:
        return True, ""
    illegal = sorted(v for v in used_vars if v not in allowed)
    if illegal:
        return False, f"{channel} references disallowed inputs: {illegal}"
    return True, ""


def validate_program(program: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Check whether every action in the program satisfies hard constraints."""
    # 收集程序中所有设置的通道
    all_channels: Set[str] = set()
    # 跨规则检测重复通道
    global_channels: Set[str] = set()
    
    for rule_idx, rule in enumerate(program or []):
        # 新格式：{'variable': 'u_tx', 'node': ...}
        if isinstance(rule, dict) and isinstance(rule.get('variable'), str):
            ch = rule['variable']
            if ch in global_channels and ch != 'u_generic':
                return False, f"rule#{rule_idx}: channel '{ch}' already set by previous rule"
            global_channels.add(ch)
            all_channels.add(ch)

        actions = rule.get('action', []) if isinstance(rule, dict) else []
        # Disallow multiple set operations to the same channel in a single rule
        seen_channels: Set[str] = set()
        for action_idx, action in enumerate(actions):
            ok, reason = validate_action_channel(action)
            if not ok:
                return False, f"rule#{rule_idx}/action#{action_idx}: {reason}"
            # track duplicates within rule
            if isinstance(action, BinaryOpNode) and action.op == 'set':
                left = action.left
                if isinstance(left, TerminalNode) and isinstance(left.value, str):
                    ch = left.value
                    if ch in seen_channels:
                        return False, f"rule#{rule_idx}: duplicate set for channel '{ch}'"
                    # 检测跨规则重复：不同规则不能设置同一通道（会导致累加/覆盖混乱）
                    # EXCEPTION: u_generic 允许跨规则重复设置（支持条件分支逻辑）
                    if ch in global_channels and ch != 'u_generic':
                        return False, f"rule#{rule_idx}: channel '{ch}' already set by previous rule"
                    seen_channels.add(ch)
                    global_channels.add(ch)
                    all_channels.add(ch)
    
    # 强制要求：程序必须设置所有必需的通道
    # 模式 A: 通用控制律发现 (Generalist Control Discovery) -> 只需要 u_generic
    # 模式 B: 完整 MIMO 控制 -> 需要 u_tx, u_ty (u_fz, u_tz 可选或固定)
    
    if "u_generic" in all_channels:
        # 如果是通用模式，只允许 u_generic
        if len(all_channels) > 1:
            return False, "u_generic cannot be mixed with other channels"
    else:
        # 传统模式：允许单轴或多轴；至少需要任意一个 u_* 控制通道
        required_any = {'u_tx', 'u_ty', 'u_tz', 'u_fz'}
        if not (all_channels & required_any):
            return False, "program must set at least one control channel"
    
    return True, ""

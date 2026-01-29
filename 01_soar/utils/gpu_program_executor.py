"""GPU 表达式执行器：支持 SOAR DSL 的 Level 1/2/3 GPU 求值。

本实现分三个层级：
- Level 1：无状态表达式（Terminal/Constant + Unary/Binary），无条件。
- Level 2：包含状态型一元算子（ema/delay/diff/rate/rate_limit）。
- Level 3：包含条件/比较/If 节点。

全部级别默认走 GPU IR 执行，若编译失败则自动回退至旧版递归求值（CPU 兜底）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import json

import os
import torch
from torch import Tensor

try:  # pragma: no cover
    from core.dsl import (  # type: ignore
        TerminalNode,
        ConstantNode,
        UnaryOpNode,
        BinaryOpNode,
        IfNode,
        SAFE_VALUE_MIN,
        SAFE_VALUE_MAX,
        MIN_EMA_ALPHA,
        MAX_EMA_ALPHA,
        MAX_DELAY_STEPS,
        MAX_DIFF_STEPS,
        MAX_RATE_LIMIT,
    )
except Exception:  # pragma: no cover
    from ..core.dsl import (  # type: ignore
        TerminalNode,
        ConstantNode,
        UnaryOpNode,
        BinaryOpNode,
        IfNode,
        SAFE_VALUE_MIN,
        SAFE_VALUE_MAX,
        MIN_EMA_ALPHA,
        MAX_EMA_ALPHA,
        MAX_DELAY_STEPS,
        MAX_DIFF_STEPS,
        MAX_RATE_LIMIT,
    )

OUTPUT_KEYS = ("u_fz", "u_tx", "u_ty", "u_tz")
STATEFUL_UNARY_OPS = {"ema", "delay", "diff", "rate", "rate_limit"}


@dataclass
class Instruction:
    kind: str
    op: Optional[str] = None
    args: Tuple[int, ...] = ()
    value: float = 0.0
    name: Optional[str] = None
    params: Dict[str, float] = field(default_factory=dict)
    state_slot: Optional[int] = None


@dataclass
class StateSlot:
    kind: str
    size: int = 1
    global_idx: Optional[int] = None


@dataclass
class CompiledProgram:
    instructions: List[Instruction]
    outputs: Dict[str, int]
    level: int
    state_slots: List[StateSlot]
    uses_condition: bool
    fingerprint: str


class ProgramCompileError(RuntimeError):
    """Raised when program cannot be compiled into GPU IR."""


class ProgramCompiler:
    """Compile AST-based DSL程序到简单的命令序列。"""

    def __init__(self) -> None:
        self.instructions: List[Instruction] = []
        self.cache: Dict[int, int] = {}
        self.level = 1
        self.uses_condition = False
        self.state_slots: List[StateSlot] = []
        self._zero_idx: Optional[int] = None

    def compile(self, program: List[Dict[str, Any]]) -> CompiledProgram:
        outputs: Dict[str, Optional[int]] = {k: None for k in OUTPUT_KEYS}
        for rule in program or []:
            cond_idx = None
            condition_node = rule.get("condition")
            if condition_node is not None and not self._is_trivial_condition(condition_node):
                cond_idx = self._compile_node(condition_node)
                self.level = max(self.level, 3)
                self.uses_condition = True
            for action in rule.get("action", []) or []:
                if not isinstance(action, BinaryOpNode) or action.op != "set":
                    continue
                left = getattr(action, "left", None)
                if not isinstance(left, TerminalNode):
                    continue
                key = str(getattr(left, "value", ""))
                if key not in OUTPUT_KEYS:
                    continue
                expr_idx = self._compile_node(getattr(action, "right", None))
                if cond_idx is not None:
                    expr_idx = self._emit_cond_gate(cond_idx, expr_idx)
                # 改为覆盖而非累加：后面的规则覆盖前面的规则
                # 注意：validate_program 已禁止跨规则重复通道，这里理论上不会触发覆盖
                outputs[key] = expr_idx
        finalized: Dict[str, int] = {}
        for key in OUTPUT_KEYS:
            if outputs[key] is None:
                outputs[key] = self._emit_const(0.0)
            finalized[key] = int(outputs[key])
        fingerprint = self._build_fingerprint(finalized)
        return CompiledProgram(
            instructions=self.instructions,
            outputs=finalized,
            level=self.level,
            state_slots=self.state_slots,
            uses_condition=self.uses_condition,
            fingerprint=fingerprint,
        )

    def _build_fingerprint(self, outputs: Dict[str, int]) -> str:
        payload = {
            "instructions": [
                {
                    "kind": inst.kind,
                    "op": inst.op,
                    "args": list(inst.args),
                    "value": inst.value,
                    "name": inst.name,
                    "params": inst.params,
                    "state_slot": inst.state_slot,
                }
                for inst in self.instructions
            ],
            "outputs": outputs,
        }
        data = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(data.encode("utf-8")).hexdigest()

    def _compile_node(self, node: Any) -> int:
        if node is None:
            return self._emit_const(0.0)
        node_id = id(node)
        if node_id in self.cache:
            return self.cache[node_id]
        idx: int
        if isinstance(node, ConstantNode):
            idx = self._emit_const(float(node.value))
        elif isinstance(node, (int, float)):
            idx = self._emit_const(float(node))
        elif isinstance(node, TerminalNode):
            value = getattr(node, "value", 0.0)
            if isinstance(value, str):
                idx = self._emit_state(value)
            else:
                idx = self._emit_const(float(value))
        elif isinstance(node, UnaryOpNode):
            op_name = str(getattr(node, "op", ""))
            base_name = op_name.split(":")[0]
            params = self.extract_params(node, base_name)
            child_idx = self._compile_node(node.child)
            if base_name in STATEFUL_UNARY_OPS:
                slot_idx = self._allocate_state_slot(base_name, params)
                idx = self._emit_stateful_unary(base_name, child_idx, params, slot_idx)
                self.level = max(self.level, 2)
            else:
                idx = self._emit_unary(base_name, child_idx, params)
        elif isinstance(node, BinaryOpNode):
            left_idx = self._compile_node(node.left)
            right_idx = self._compile_node(node.right)
            idx = self._emit_binary(str(node.op), left_idx, right_idx)
            if node.op in ("<", ">", "==", "!="):
                self.level = max(self.level, 3)
        elif isinstance(node, IfNode):
            cond_idx = self._compile_node(node.condition)
            then_idx = self._compile_node(node.then_branch)
            else_idx = self._compile_node(node.else_branch)
            idx = self._emit_if(cond_idx, then_idx, else_idx)
            self.level = max(self.level, 3)
        else:
            raise ProgramCompileError(f"Unsupported node type: {type(node)}")
        self.cache[node_id] = idx
        return idx

    def _emit_state(self, name: str) -> int:
        inst = Instruction(kind="state", name=name)
        self.instructions.append(inst)
        return len(self.instructions) - 1

    def _emit_const(self, value: float) -> int:
        if value == 0.0 and self._zero_idx is not None:
            return self._zero_idx
        inst = Instruction(kind="const", value=float(value))
        self.instructions.append(inst)
        idx = len(self.instructions) - 1
        if value == 0.0 and self._zero_idx is None:
            self._zero_idx = idx
        return idx

    def _emit_unary(self, op: str, arg_idx: int, params: Dict[str, float]) -> int:
        inst = Instruction(kind="unary", op=op, args=(arg_idx,), params=params)
        self.instructions.append(inst)
        return len(self.instructions) - 1

    def _emit_stateful_unary(self, op: str, arg_idx: int, params: Dict[str, float], slot_idx: int) -> int:
        inst = Instruction(kind="stateful_unary", op=op, args=(arg_idx,), params=params, state_slot=slot_idx)
        self.instructions.append(inst)
        return len(self.instructions) - 1

    def _emit_binary(self, op: str, left_idx: int, right_idx: int) -> int:
        inst = Instruction(kind="binary", op=op, args=(left_idx, right_idx))
        self.instructions.append(inst)
        return len(self.instructions) - 1

    def _emit_if(self, cond_idx: int, then_idx: int, else_idx: int) -> int:
        inst = Instruction(kind="if", args=(cond_idx, then_idx, else_idx))
        self.instructions.append(inst)
        return len(self.instructions) - 1

    def _emit_cond_gate(self, cond_idx: int, value_idx: int) -> int:
        inst = Instruction(kind="cond_gate", args=(cond_idx, value_idx))
        self.instructions.append(inst)
        return len(self.instructions) - 1

    def _allocate_state_slot(self, kind: str, params: Dict[str, float]) -> int:
        size = int(params.get("k", 1)) if kind in ("delay", "diff") else 1
        slot = StateSlot(kind=kind, size=size)
        self.state_slots.append(slot)
        return len(self.state_slots) - 1

    @staticmethod
    def extract_params(node: UnaryOpNode, base_name: str) -> Dict[str, float]:
        params: Dict[str, float] = {}
        if base_name == "ema":
            params["alpha"] = float(node.get_param("alpha", 0.2, MIN_EMA_ALPHA, MAX_EMA_ALPHA))
        elif base_name in ("delay", "diff"):
            limit = MAX_DELAY_STEPS if base_name == "delay" else MAX_DIFF_STEPS
            k = int(node.get_param("k", 1, 1, limit))
            params["k"] = float(max(1, min(limit, k)))
        elif base_name in ("rate", "rate_limit"):
            params["r"] = float(node.get_param("r", 1.0, 0.01, MAX_RATE_LIMIT))
        elif base_name == "clamp":
            params["lo"] = float(node.get_param("lo", SAFE_VALUE_MIN, SAFE_VALUE_MIN, SAFE_VALUE_MAX))
            params["hi"] = float(node.get_param("hi", SAFE_VALUE_MAX, SAFE_VALUE_MIN, SAFE_VALUE_MAX))
        elif base_name == "deadzone":
            params["eps"] = float(node.get_param("eps", 0.01, 0.0, 1.0))
        elif base_name in ("smooth", "smoothstep"):
            params["s"] = float(node.get_param("s", 1.0, 1e-3, 2.0))
        return params

    @staticmethod
    def _is_trivial_condition(node: Any) -> bool:
        if node is None:
            return True
        if isinstance(node, TerminalNode):
            val = getattr(node, "value", None)
            if isinstance(val, (int, float)):
                return float(val) >= 1.0
        return False


class GPUProgramExecutor:
    """GPU 版 DSL 执行器，支持 Level1/2/3 GPU IR + CPU fallback."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        self._batches: Dict[int, Dict[str, Any]] = {}
        self._next_batch_token = 1
        self._zero = torch.tensor(0.0, device=self.device)
        self._one = torch.tensor(1.0, device=self.device)

    def build_state_tensors(
        self,
        pos: Tensor,
        vel: Tensor,
        omega: Tensor,
        quat: Tensor,
        target: Tensor,
        integral_states: List[Dict[str, float]],
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        batch_size = pos.shape[0]
        if target.dim() == 1:
            target = target.view(1, -1).expand(batch_size, -1)
        pos_err = target - pos
        pos_err_xy = torch.linalg.norm(pos_err[:, :2], dim=1)
        pos_err_mag = torch.linalg.norm(pos_err, dim=1)
        vel_err = torch.linalg.norm(vel, dim=1)
        ang_vel_mag = torch.linalg.norm(omega, dim=1)
        rpy = self.quat_to_rpy_gpu(quat)
        rpy_err_mag = torch.linalg.norm(rpy, dim=1)

        integral_tensor = torch.zeros((batch_size, 6), device=self.device)
        for idx in range(batch_size):
            buf = integral_states[idx]
            integral_tensor[idx, 0] = float(buf.get("err_i_x", 0.0))
            integral_tensor[idx, 1] = float(buf.get("err_i_y", 0.0))
            integral_tensor[idx, 2] = float(buf.get("err_i_z", 0.0))
            integral_tensor[idx, 3] = float(buf.get("err_i_roll", 0.0))
            integral_tensor[idx, 4] = float(buf.get("err_i_pitch", 0.0))
            integral_tensor[idx, 5] = float(buf.get("err_i_yaw", 0.0))

        state_tensors = {
            "pos_err_x": pos_err[:, 0],
            "pos_err_y": pos_err[:, 1],
            "pos_err_z": pos_err[:, 2],
            "pos_err": pos_err_mag,
            "pos_err_xy": pos_err_xy,
            "pos_err_z_abs": torch.abs(pos_err[:, 2]),
            "vel_x": vel[:, 0],
            "vel_y": vel[:, 1],
            "vel_z": vel[:, 2],
            "vel_err": vel_err,
            "err_p_roll": rpy[:, 0],
            "err_p_pitch": rpy[:, 1],
            "err_p_yaw": rpy[:, 2],
            "ang_err": rpy_err_mag,
            "rpy_err_mag": rpy_err_mag,
            "ang_vel_x": omega[:, 0],
            "ang_vel_y": omega[:, 1],
            "ang_vel_z": omega[:, 2],
            "ang_vel": ang_vel_mag,
            "ang_vel_mag": ang_vel_mag,
            "err_i_x": integral_tensor[:, 0],
            "err_i_y": integral_tensor[:, 1],
            "err_i_z": integral_tensor[:, 2],
            "err_i_roll": integral_tensor[:, 3],
            "err_i_pitch": integral_tensor[:, 4],
            "err_i_yaw": integral_tensor[:, 5],
            "err_d_x": -vel[:, 0],
            "err_d_y": -vel[:, 1],
            "err_d_z": -vel[:, 2],
            "err_d_roll": -omega[:, 0],
            "err_d_pitch": -omega[:, 1],
            "err_d_yaw": -omega[:, 2],
        }
        return state_tensors, pos_err, rpy

    def evaluate_from_raw_obs(
        self,
        token: int,
        pos: Tensor,
        vel: Tensor,
        omega: Tensor,
        quat: Tensor,
        target: Tensor,
        integral_states: List[Dict[str, float]],
        use_u_mask: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        *,
        force_cpu: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        state_tensors, pos_err, rpy = self.build_state_tensors(pos, vel, omega, quat, target, integral_states)
        outputs = self.evaluate(token, state_tensors, use_u_mask, active_mask, force_cpu=force_cpu)
        return outputs, pos_err, rpy

    def prepare_batch(self, programs: List[List[Dict[str, Any]]]) -> int:
        compiled_list: List[Optional[CompiledProgram]] = []
        level_counts = {1: 0, 2: 0, 3: 0, "fallback": 0}
        compiled_cache: Dict[str, CompiledProgram] = {}
        unique_compiled_order: List[CompiledProgram] = []
        for prog in programs:
            try:
                comp = ProgramCompiler().compile(prog)
                cached = compiled_cache.get(comp.fingerprint)
                if cached is None:
                    compiled_cache[comp.fingerprint] = comp
                    unique_compiled_order.append(comp)
                    compiled_entry = comp
                else:
                    compiled_entry = cached
                compiled_list.append(compiled_entry)
                level_counts[compiled_entry.level] += 1
            except Exception:
                compiled_list.append(None)
                level_counts["fallback"] += 1
        state_slots: List[StateSlot] = []
        for comp in unique_compiled_order:
            for slot in comp.state_slots:
                if slot.global_idx is None:
                    slot.global_idx = len(state_slots)
                    state_slots.append(slot)
            for inst in comp.instructions:
                if inst.state_slot is not None and comp.state_slots:
                    inst.state_slot = comp.state_slots[inst.state_slot].global_idx
        state_buffers = self._allocate_state_buffers(len(programs), state_slots)
        token = self._next_batch_token
        self._next_batch_token += 1
        self._batches[token] = {
            "programs": programs,
            "compiled": compiled_list,
            "state_buffers": state_buffers,
            "node_states": [{} for _ in programs],
        }
        if os.getenv("DEBUG_GPU_EXECUTOR", "0") in ("1", "true", "True"):
            print(
                f"[GPUExecutor] batch prepared | level1={level_counts[1]} "
                f"level2={level_counts[2]} level3={level_counts[3]} fallback={level_counts['fallback']}"
            )
        return token

    def release_batch(self, token: int) -> None:
        self._batches.pop(token, None)

    def reset_state(self) -> None:
        self._batches.clear()
        self._next_batch_token = 1

    def evaluate(
        self,
        token: int,
        state_tensors: Dict[str, Tensor],
        use_u_mask: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        *,
        force_cpu: bool = False,
    ) -> Tensor:
        batch = self._batches.get(token)
        if batch is None:
            raise RuntimeError("GPUProgramExecutor batch token 无效")
        programs = batch["programs"]
        compiled_list = batch["compiled"]
        node_states = batch["node_states"]
        state_buffers = batch["state_buffers"]
        batch_size = len(programs)
        outputs = torch.zeros((batch_size, 4), device=self.device)
        use_u_mask = use_u_mask.to(self.device).bool()
        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        else:
            active_mask = active_mask.to(self.device).bool()
        grouped: Dict[str, Dict[str, Any]] = {}
        fallback_indices: List[int] = []
        for idx, program in enumerate(programs):
            if not use_u_mask[idx] or not active_mask[idx]:
                continue
            compiled = compiled_list[idx]
            if compiled is None or force_cpu:
                fallback_indices.append(idx)
                continue
            key = compiled.fingerprint
            bucket = grouped.setdefault(key, {"compiled": compiled, "indices": []})
            bucket["indices"].append(idx)

        for key, bucket in grouped.items():
            compiled: CompiledProgram = bucket["compiled"]
            indices: List[int] = bucket["indices"]
            if not indices:
                continue
            group_outputs = self._run_compiled_group(
                compiled,
                indices,
                state_tensors,
                state_buffers,
                active_mask,
            )
            idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
            # u_fz (推力) 必须非负，上限约束在合理范围内 (Crazyflie 质量 0.027kg * 重力 ≈ 0.265N，加裕度)
            outputs[idx_tensor, 0] = torch.clamp(group_outputs[:, 0], 0.0, 1.5)
            # u_tx, u_ty (roll/pitch torque) - Crazyflie 力矩范围约 ±0.002 N·m，但 DSL 输出是归一化值
            # 扩大到 ±1.0 允许更大的控制增益
            outputs[idx_tensor, 1] = torch.clamp(group_outputs[:, 1], -1.0, 1.0)
            outputs[idx_tensor, 2] = torch.clamp(group_outputs[:, 2], -1.0, 1.0)
            # u_tz (yaw torque) 保持合理范围
            outputs[idx_tensor, 3] = torch.clamp(group_outputs[:, 3], -0.5, 0.5)

        for idx in fallback_indices:
            vec = self._run_fallback(programs[idx], state_tensors, idx, node_states[idx])
            # u_fz 必须非负 (推力只能向上)
            outputs[idx, 0] = torch.clamp(vec[0], 0.0, 1.5)
            outputs[idx, 1] = torch.clamp(vec[1], -1.0, 1.0)
            outputs[idx, 2] = torch.clamp(vec[2], -1.0, 1.0)
            outputs[idx, 3] = torch.clamp(vec[3], -0.5, 0.5)
        return outputs

    def _run_compiled(
        self,
        compiled: CompiledProgram,
        state_tensors: Dict[str, Tensor],
        program_idx: int,
        state_buffers: List[Dict[str, Tensor]],
        active_mask: torch.Tensor,
    ) -> Tensor:
        values: List[Tensor] = []
        for inst in compiled.instructions:
            if inst.kind == "state":
                tensor = state_tensors.get(inst.name or "")
                values.append(tensor[program_idx] if tensor is not None else self._zero)
            elif inst.kind == "const":
                values.append(torch.tensor(inst.value, device=self.device))
            elif inst.kind == "unary":
                operand = values[inst.args[0]]
                values.append(self._apply_unary(inst.op or "", operand, inst.params))
            elif inst.kind == "binary":
                left = values[inst.args[0]]
                right = values[inst.args[1]]
                values.append(self._apply_binary(inst.op or "", left, right))
            elif inst.kind == "stateful_unary":
                operand = values[inst.args[0]]
                values.append(
                    self._apply_stateful_unary(
                        inst.op or "",
                        operand,
                        inst.params,
                        inst.state_slot,
                        program_idx,
                        state_buffers,
                        active_mask,
                    )
                )
            elif inst.kind == "if":
                cond = values[inst.args[0]]
                then_val = values[inst.args[1]]
                else_val = values[inst.args[2]]
                values.append(torch.where(cond > 0.0, then_val, else_val))
            elif inst.kind == "cond_gate":
                cond = values[inst.args[0]]
                expr = values[inst.args[1]]
                values.append(torch.where(cond > 0.0, expr, torch.zeros_like(expr)))
            else:
                values.append(self._zero)
        fz = values[compiled.outputs["u_fz"]]
        tx = values[compiled.outputs["u_tx"]]
        ty = values[compiled.outputs["u_ty"]]
        tz = values[compiled.outputs["u_tz"]]
        return torch.stack([fz, tx, ty, tz])

    def _run_compiled_group(
        self,
        compiled: CompiledProgram,
        program_indices: List[int],
        state_tensors: Dict[str, Tensor],
        state_buffers: List[Dict[str, Tensor]],
        active_mask: torch.Tensor,
    ) -> Tensor:
        group_size = len(program_indices)
        if group_size == 0:
            return torch.zeros((0, 4), device=self.device)
        idx_tensor = torch.tensor(program_indices, device=self.device, dtype=torch.long)
        values: List[Tensor] = []
        zero_vec = torch.zeros(group_size, device=self.device)
        for inst in compiled.instructions:
            if inst.kind == "state":
                tensor = state_tensors.get(inst.name or "")
                values.append(tensor[idx_tensor] if tensor is not None else zero_vec)
            elif inst.kind == "const":
                if inst.value == 0.0:
                    values.append(zero_vec)
                else:
                    values.append(torch.full((group_size,), inst.value, device=self.device))
            elif inst.kind == "unary":
                operand = values[inst.args[0]]
                values.append(self._apply_unary(inst.op or "", operand, inst.params))
            elif inst.kind == "binary":
                left = values[inst.args[0]]
                right = values[inst.args[1]]
                values.append(self._apply_binary(inst.op or "", left, right))
            elif inst.kind == "stateful_unary":
                operand = values[inst.args[0]]
                values.append(
                    self._apply_stateful_unary_batch(
                        inst.op or "",
                        operand,
                        inst.params,
                        inst.state_slot,
                        idx_tensor,
                        state_buffers,
                        active_mask,
                    )
                )
            elif inst.kind == "if":
                cond = values[inst.args[0]]
                then_val = values[inst.args[1]]
                else_val = values[inst.args[2]]
                values.append(torch.where(cond > 0.0, then_val, else_val))
            elif inst.kind == "cond_gate":
                cond = values[inst.args[0]]
                expr = values[inst.args[1]]
                values.append(torch.where(cond > 0.0, expr, torch.zeros_like(expr)))
            else:
                values.append(zero_vec)
        fz = values[compiled.outputs["u_fz"]]
        tx = values[compiled.outputs["u_tx"]]
        ty = values[compiled.outputs["u_ty"]]
        tz = values[compiled.outputs["u_tz"]]
        return torch.stack([fz, tx, ty, tz], dim=1)

    def _apply_stateful_unary_batch(
        self,
        op: str,
        value: Tensor,
        params: Dict[str, float],
        state_slot: Optional[int],
        program_indices: Tensor,
        state_buffers: List[Dict[str, Tensor]],
        active_mask: torch.Tensor,
    ) -> Tensor:
        if state_slot is None or state_slot >= len(state_buffers):
            return value
        slot = state_buffers[state_slot]
        batch_mask = active_mask[program_indices]
        if op == "ema":
            alpha = params.get("alpha", 0.2)
            values_buf = slot.setdefault("values", torch.zeros(active_mask.shape[0], device=self.device))
            prev = values_buf[program_indices]
            result = (1.0 - alpha) * prev + alpha * value
            if batch_mask.any():
                updated = torch.where(batch_mask, result, prev)
                values_buf[program_indices] = updated
            return result
        if op in ("rate", "rate_limit"):
            r = params.get("r", 1.0)
            values_buf = slot.setdefault("values", torch.zeros(active_mask.shape[0], device=self.device))
            prev = values_buf[program_indices]
            lo = prev - r
            hi = prev + r
            result = torch.clamp(value, lo, hi)
            if batch_mask.any():
                updated = torch.where(batch_mask, result, prev)
                values_buf[program_indices] = updated
            return result
        if op in ("delay", "diff"):
            k = max(1, min(int(params.get("k", 1)), MAX_DELAY_STEPS if op == "delay" else MAX_DIFF_STEPS))
            buffer = slot.setdefault("buffer", torch.zeros(active_mask.shape[0], k, device=self.device))
            head = slot.setdefault("head", torch.zeros(active_mask.shape[0], dtype=torch.long, device=self.device))
            count = slot.setdefault("count", torch.zeros(active_mask.shape[0], dtype=torch.long, device=self.device))
            head_vals = head[program_indices]
            count_vals = count[program_indices]
            gathered = buffer[program_indices, head_vals.clamp(0, k - 1)]
            if op == "delay":
                result = torch.where(count_vals >= k, gathered, torch.zeros_like(value))
            else:
                prev = torch.where(count_vals >= k, gathered, value)
                result = value - prev
            if batch_mask.any():
                active_rows = program_indices[batch_mask]
                active_cols = head_vals[batch_mask] % k
                buffer.index_put_((active_rows, active_cols), value[batch_mask])
                head_next = (head_vals + 1) % k
                count_next = torch.clamp(count_vals + 1, max=k)
                head.index_put_((active_rows,), head_next[batch_mask])
                count.index_put_((active_rows,), count_next[batch_mask])
            return result
        return value

    def _apply_unary(self, op: str, value: Tensor, params: Dict[str, float]) -> Tensor:
        if op == "abs":
            return torch.abs(value)
        if op == "sign":
            return torch.sign(value)
        if op == "sin":
            return torch.sin(value)
        if op == "cos":
            return torch.cos(value)
        if op == "tan":
            return torch.clamp(torch.tan(value), -10.0, 10.0)
        if op == "log1p":
            return torch.log1p(torch.abs(value))
        if op == "sqrt":
            return torch.sqrt(torch.abs(value))
        if op == "clamp":
            lo = params.get("lo", SAFE_VALUE_MIN)
            hi = params.get("hi", SAFE_VALUE_MAX)
            if lo > hi:
                lo, hi = hi, lo
            return torch.clamp(value, lo, hi)
        if op == "deadzone":
            eps = params.get("eps", 0.01)
            return torch.where(torch.abs(value) <= eps, self._zero, value - torch.sign(value) * eps)
        if op in ("smooth", "smoothstep"):
            scale = max(params.get("s", 1.0), 1e-6)
            return scale * torch.tanh(value / scale)
        return value

    def _apply_stateful_unary(
        self,
        op: str,
        value: Tensor,
        params: Dict[str, float],
        state_slot: Optional[int],
        program_idx: int,
        state_buffers: List[Dict[str, Tensor]],
        active_mask: torch.Tensor,
    ) -> Tensor:
        if state_slot is None or state_slot >= len(state_buffers):
            return value
        slot = state_buffers[state_slot]
        is_active = bool(active_mask[program_idx].item())
        batch = active_mask.shape[0]
        if op == "ema":
            alpha = params.get("alpha", 0.2)
            values = slot.setdefault("values", torch.zeros(batch, device=self.device))
            prev = values[program_idx]
            result = (1.0 - alpha) * prev + alpha * value
            if is_active:
                values[program_idx] = result
            return result
        if op in ("rate", "rate_limit"):
            r = params.get("r", 1.0)
            values = slot.setdefault("values", torch.zeros(batch, device=self.device))
            prev = values[program_idx]
            result = torch.clamp(value, prev - r, prev + r)
            if is_active:
                values[program_idx] = result
            return result
        if op == "delay":
            k = max(1, min(int(params.get("k", 1)), MAX_DELAY_STEPS))
            buffer = slot.setdefault("buffer", torch.zeros(batch, k, device=self.device))
            head = slot.setdefault("head", torch.zeros(batch, dtype=torch.long, device=self.device))
            count = slot.setdefault("count", torch.zeros(batch, dtype=torch.long, device=self.device))
            idx_tensor = head[program_idx]
            idx = int(idx_tensor.item())
            count_val = int(count[program_idx].item())
            out = buffer[program_idx, idx] if count_val >= k else self._zero
            if is_active:
                buffer[program_idx, idx] = value
                head[program_idx] = (idx + 1) % k
                new_count = min(count_val + 1, k)
                count[program_idx] = new_count
            return out
        if op == "diff":
            k = max(1, min(int(params.get("k", 1)), MAX_DIFF_STEPS))
            buffer = slot.setdefault("buffer", torch.zeros(batch, k, device=self.device))
            head = slot.setdefault("head", torch.zeros(batch, dtype=torch.long, device=self.device))
            count = slot.setdefault("count", torch.zeros(batch, dtype=torch.long, device=self.device))
            idx_tensor = head[program_idx]
            idx = int(idx_tensor.item())
            count_val = int(count[program_idx].item())
            prev = buffer[program_idx, idx] if count_val >= k else value
            result = value - prev
            if is_active:
                buffer[program_idx, idx] = value
                head[program_idx] = (idx + 1) % k
                new_count = min(count_val + 1, k)
                count[program_idx] = new_count
            return result
        return value

    def _run_fallback(
        self,
        program: List[Dict[str, Any]],
        state_tensors: Dict[str, Tensor],
        program_idx: int,
        cache: Dict[int, Any],
    ) -> Tensor:
        fz = self._zero
        tx = self._zero
        ty = self._zero
        tz = self._zero
        for rule in program or []:
            cond = rule.get("condition")
            cond_mask = self._one
            if cond is not None:
                cond_val = self._eval_node(cond, state_tensors, program_idx, cache)
                cond_mask = torch.where(cond_val > 0.0, self._one, self._zero)
            for action in rule.get("action", []) or []:
                if not isinstance(action, BinaryOpNode) or action.op != "set":
                    continue
                left = getattr(action, "left", None)
                if not isinstance(left, TerminalNode):
                    continue
                key = str(getattr(left, "value", ""))
                val = self._eval_node(getattr(action, "right", None), state_tensors, program_idx, cache) * cond_mask
                if key == "u_fz":
                    fz = fz + val
                elif key == "u_tx":
                    tx = tx + val
                elif key == "u_ty":
                    ty = ty + val
                elif key == "u_tz":
                    tz = tz + val
        return torch.stack([fz, tx, ty, tz])

    def _eval_node(
        self,
        node: Any,
        state_tensors: Dict[str, Tensor],
        idx: int,
        cache: Dict[int, Any],
    ) -> Tensor:
        if node is None:
            return self._zero
        node_id = id(node)
        if node_id in cache:
            return cache[node_id]
        should_cache = True
        if isinstance(node, (int, float)):
            val = torch.tensor(float(node), device=self.device)
        elif isinstance(node, TerminalNode):
            value = getattr(node, "value", 0.0)
            if isinstance(value, str):
                tensor = state_tensors.get(value)
                val = tensor[idx] if tensor is not None else self._zero
            else:
                val = torch.tensor(float(value), device=self.device)
        elif isinstance(node, ConstantNode):
            val = torch.tensor(float(node.value), device=self.device)
        elif isinstance(node, UnaryOpNode):
            operand = self._eval_node(node.child, state_tensors, idx, cache)
            base = str(getattr(node, "op", "")).split(":")[0]
            params = ProgramCompiler.extract_params(node, base)
            val = self._legacy_eval_unary(base, operand, node_id, cache, params)
            if base in STATEFUL_UNARY_OPS:
                should_cache = False
        elif isinstance(node, BinaryOpNode):
            left = self._eval_node(node.left, state_tensors, idx, cache)
            right = self._eval_node(node.right, state_tensors, idx, cache)
            val = self._apply_binary(str(node.op), left, right)
        elif isinstance(node, IfNode):
            cond = self._eval_node(node.condition, state_tensors, idx, cache)
            then_val = self._eval_node(node.then_branch, state_tensors, idx, cache)
            else_val = self._eval_node(node.else_branch, state_tensors, idx, cache)
            val = torch.where(cond > 0.0, then_val, else_val)
        else:
            val = self._zero
        if should_cache:
            cache[node_id] = val
        return val

    def _apply_binary(self, op: str, left: Tensor, right: Tensor) -> Tensor:
        eps = 1e-6
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            denom = torch.where(torch.abs(right) > eps, right, torch.where(right >= 0, torch.tensor(eps, device=self.device), torch.tensor(-eps, device=self.device)))
            return left / denom
        if op == "max":
            return torch.maximum(left, right)
        if op == "min":
            return torch.minimum(left, right)
        if op == ">":
            return torch.where(left > right, self._one, self._zero)
        if op == "<":
            return torch.where(left < right, self._one, self._zero)
        if op == "==":
            return torch.where(torch.abs(left - right) < eps, self._one, self._zero)
        if op == "!=":
            return torch.where(torch.abs(left - right) >= eps, self._one, self._zero)
        return self._zero

    def _legacy_eval_unary(
        self,
        op: str,
        value: Tensor,
        node_id: int,
        cache: Dict[int, Any],
        params: Dict[str, float],
    ) -> Tensor:
        if op not in STATEFUL_UNARY_OPS:
            return self._apply_unary(op, value, params)
        key = (node_id, op)
        if op == "ema":
            alpha = params.get("alpha", 0.2)
            prev = cache.get(key, self._zero)
            result = (1.0 - alpha) * prev + alpha * value
            cache[key] = result
            return result
        if op in ("rate", "rate_limit"):
            r = params.get("r", 1.0)
            prev = cache.get(key, self._zero)
            lo = prev - r
            hi = prev + r
            result = torch.clamp(value, lo, hi)
            cache[key] = result
            return result
        if op == "delay":
            k = max(1, min(int(params.get("k", 1)), MAX_DELAY_STEPS))
            buf = cache.get(key)
            if not isinstance(buf, deque) or buf.maxlen != k:
                buf = deque(maxlen=k)
            out = buf[0] if len(buf) == k else self._zero
            buf.appendleft(value)
            cache[key] = buf
            return out
        if op == "diff":
            k = max(1, min(int(params.get("k", 1)), MAX_DIFF_STEPS))
            buf = cache.get(key)
            if not isinstance(buf, deque) or buf.maxlen != k:
                buf = deque(maxlen=k)
            prev = buf[0] if len(buf) == k else value
            buf.appendleft(value)
            cache[key] = buf
            return value - prev
        return value

    def _allocate_state_buffers(self, batch_size: int, slots: List[StateSlot]) -> List[Dict[str, Tensor]]:
        buffers: List[Dict[str, Tensor]] = []
        for slot in slots:
            if slot.kind in ("ema", "rate", "rate_limit"):
                buffers.append({"values": torch.zeros(batch_size, device=self.device)})
            elif slot.kind in ("delay", "diff"):
                k = max(1, int(slot.size))
                buffers.append(
                    {
                        "buffer": torch.zeros(batch_size, k, device=self.device),
                        "head": torch.zeros(batch_size, dtype=torch.long, device=self.device),
                        "count": torch.zeros(batch_size, dtype=torch.long, device=self.device),
                    }
                )
            else:
                buffers.append({})
        return buffers

    def quat_to_rpy_gpu(self, quat: Tensor) -> Tensor:
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=1)


__all__ = ["GPUProgramExecutor"]

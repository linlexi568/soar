"""Serialization utilities for SOAR DSL programs.

This module converts AST nodes (TerminalNode, UnaryOpNode, BinaryOpNode, IfNode,
and advanced nodes) to and from JSON-serializable dictionaries so that
searched programs can be persisted and later reloaded for evaluation or
dataset collection.
"""
from __future__ import annotations
from typing import Any, Dict, List
from .dsl import (
    ProgramNode,
    TerminalNode,
    ConstantNode,
    UnaryOpNode,
    BinaryOpNode,
    IfNode,
)

def _is_trivial_condition(node: ProgramNode | None) -> bool:
    if node is None:
        return True
    if isinstance(node, TerminalNode):
        val = node.value
        if isinstance(val, (int, float)):
            try:
                return float(val) == 1.0
            except Exception:
                return False
    return False

def _always_true_condition() -> TerminalNode:
    return TerminalNode(1.0)

ASTDict = Dict[str, Any]

def serialize_ast(node: ProgramNode) -> ASTDict:
    if isinstance(node, TerminalNode):
        return {"type": "Terminal", "value": node.value}
    
    if isinstance(node, ConstantNode):
        result = {"type": "Constant", "value": node.value}
        if node.name:
            result["name"] = node.name
        if node.min_val is not None:
            result["min_val"] = node.min_val
        if node.max_val is not None:
            result["max_val"] = node.max_val
        return result
    
    if isinstance(node, UnaryOpNode):
        result = {"type": "Unary", "op": node.op, "child": serialize_ast(node.child)}
        # 序列化参数字典（新格式）
        if node.params:
            params_serialized = {}
            for key, val in node.params.items():
                if isinstance(val, ConstantNode):
                    params_serialized[key] = serialize_ast(val)
                elif isinstance(val, (int, float)):
                    params_serialized[key] = {"type": "Constant", "value": float(val)}
            if params_serialized:
                result["params"] = params_serialized
        return result
    
    if isinstance(node, BinaryOpNode):
        return {"type": "Binary", "op": node.op, "left": serialize_ast(node.left), "right": serialize_ast(node.right)}
    if isinstance(node, IfNode):
        return {"type": "If", "condition": serialize_ast(node.condition), "then": serialize_ast(node.then_branch), "else": serialize_ast(node.else_branch)}
    raise TypeError(f"Cannot serialize unknown node type: {type(node)}")

def deserialize_ast(obj: ASTDict) -> ProgramNode:
    t = obj.get("type")
    if t == "Terminal":
        return TerminalNode(obj["value"])
    
    if t == "Constant":
        return ConstantNode(
            value=obj["value"],
            name=obj.get("name"),
            min_val=obj.get("min_val"),
            max_val=obj.get("max_val")
        )
    
    if t == "Unary":
        child = deserialize_ast(obj["child"])
        params = None
        if "params" in obj:
            params = {}
            for key, val_dict in obj["params"].items():
                params[key] = deserialize_ast(val_dict)
        return UnaryOpNode(obj["op"], child, params)
    
    if t == "Binary":
        return BinaryOpNode(obj["op"], deserialize_ast(obj["left"]), deserialize_ast(obj["right"]))
    if t == "If":
        return IfNode(deserialize_ast(obj["condition"]), deserialize_ast(obj["then"]), deserialize_ast(obj["else"]))
    raise ValueError(f"Unknown AST dict type: {t}")

def serialize_program(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    serial_rules: List[Dict[str, Any]] = []
    for r in rules:
        action_list = r.get('action', [])
        cond_payload = r.get('condition')
        entry = {
            'action': [serialize_ast(a) for a in action_list]
        }
        if not _is_trivial_condition(cond_payload):
            entry['condition'] = serialize_ast(cond_payload)
        serial_rules.append(entry)
    return {"rules": serial_rules}

def deserialize_program(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules_out: List[Dict[str, Any]] = []
    for r in obj.get('rules', []):
        cond_ast = None
        if 'condition' in r:
            cond_ast = deserialize_ast(r['condition'])
        if _is_trivial_condition(cond_ast):
            cond_ast = _always_true_condition()
        action_asts = [deserialize_ast(a) for a in r.get('action', [])]
        rules_out.append({'condition': cond_ast, 'action': action_asts})
    return rules_out

def save_program_json(rules: List[Dict[str, Any]], path: str, meta: Dict[str, Any] | None = None):
    import json, os, time
    payload = serialize_program(rules)
    if meta:
        payload['meta'] = meta
    payload.setdefault('meta', {})['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_program_json(path: str) -> List[Dict[str, Any]]:
    import json
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return deserialize_program(data)

def save_search_history(history: List[Dict[str, Any]], path: str):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'history': history}, f, ensure_ascii=False, indent=2)

__all__ = [
    'serialize_ast','deserialize_ast','serialize_program','deserialize_program',
    'save_program_json','load_program_json','save_search_history'
]

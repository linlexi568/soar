#!/usr/bin/env python3
"""迁移脚本：将旧格式程序转换为新的参数化格式

旧格式：UnaryOpNode(op='ema:0.2', child=...)
新格式：UnaryOpNode(op='ema', child=..., params={'alpha': ConstantNode(0.2)})

保留所有元数据（性能指标、时间戳等）。
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from 01_soar.core.dsl import TerminalNode, ConstantNode, UnaryOpNode, BinaryOpNode, IfNode
from 01_soar.core.serialization import (
    serialize_program, deserialize_program,
    save_program_json, load_program_json
)


# 参数映射规则
PARAM_SPECS = {
    'ema': {'alpha': {'default': 0.2, 'min': 0.05, 'max': 0.8}},
    'delay': {'k': {'default': 1, 'min': 1, 'max': 3}},
    'diff': {'k': {'default': 1, 'min': 1, 'max': 3}},
    'clamp': {'lo': {'default': -5.0, 'min': -6.0, 'max': 6.0},
              'hi': {'default': 5.0, 'min': -6.0, 'max': 6.0}},
    'deadzone': {'eps': {'default': 0.01, 'min': 0.0, 'max': 1.0}},
    'rate': {'r': {'default': 1.0, 'min': 0.01, 'max': 1.0}},
    'rate_limit': {'r': {'default': 1.0, 'min': 0.01, 'max': 1.0}},
    'smooth': {'s': {'default': 1.0, 'min': 0.001, 'max': 2.0}},
    'smoothstep': {'s': {'default': 1.0, 'min': 0.001, 'max': 2.0}},
}


def upgrade_node(node):
    """将单个节点从旧格式升级为新格式"""
    if isinstance(node, UnaryOpNode):
        # 检查是否为旧格式 (op 包含 ':')
        if ':' in node.op and not node.params:
            parts = node.op.split(':')
            op_name = parts[0]
            param_values = parts[1:]
            
            # 构建新格式的 params 字典
            if op_name in PARAM_SPECS:
                params = {}
                param_names = list(PARAM_SPECS[op_name].keys())
                
                for i, param_name in enumerate(param_names):
                    if i < len(param_values):
                        value = float(param_values[i])
                    else:
                        value = PARAM_SPECS[op_name][param_name]['default']
                    
                    # 获取参数范围
                    spec = PARAM_SPECS[op_name][param_name]
                    min_val = spec.get('min')
                    max_val = spec.get('max')
                    
                    # 创建 ConstantNode
                    params[param_name] = ConstantNode(
                        value=value,
                        name=f"{op_name}_{param_name}",
                        min_val=min_val,
                        max_val=max_val
                    )
                
                # 创建新的 UnaryOpNode
                upgraded_child = upgrade_node(node.child)
                return UnaryOpNode(op_name, upgraded_child, params)
        
        # 递归处理子节点
        node.child = upgrade_node(node.child)
    
    elif isinstance(node, BinaryOpNode):
        node.left = upgrade_node(node.left)
        node.right = upgrade_node(node.right)
    
    elif isinstance(node, IfNode):
        node.condition = upgrade_node(node.condition)
        node.then_branch = upgrade_node(node.then_branch)
        node.else_branch = upgrade_node(node.else_branch)
    
    return node


def upgrade_program(program):
    """升级整个程序"""
    upgraded_rules = []
    
    for rule in program:
        upgraded_rule = {}
        
        if 'condition' in rule:
            upgraded_rule['condition'] = upgrade_node(rule['condition'])
        
        if 'action' in rule:
            upgraded_rule['action'] = [upgrade_node(action) for action in rule['action']]
        
        upgraded_rules.append(upgraded_rule)
    
    return upgraded_rules


def migrate_file(input_path: str, output_path: str = None, backup: bool = True):
    """迁移单个文件"""
    if output_path is None:
        output_path = input_path
    
    # 备份原文件
    if backup and input_path == output_path:
        backup_path = input_path + '.v1.backup'
        if os.path.exists(input_path):
            import shutil
            shutil.copy2(input_path, backup_path)
            print(f"✓ 备份: {backup_path}")
    
    # 加载旧格式
    try:
        program = load_program_json(input_path)
    except Exception as e:
        print(f"✗ 加载失败 {input_path}: {e}")
        return False
    
    # 升级
    try:
        upgraded = upgrade_program(program)
    except Exception as e:
        print(f"✗ 升级失败 {input_path}: {e}")
        return False
    
    # 保存新格式
    try:
        # 读取原始元数据
        with open(input_path, 'r') as f:
            original_data = json.load(f)
        meta = original_data.get('meta', {})
        meta['migrated_to_v2'] = True
        
        save_program_json(upgraded, output_path, meta=meta)
        print(f"✓ 迁移成功: {input_path} → {output_path}")
        return True
    except Exception as e:
        print(f"✗ 保存失败 {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将旧格式程序迁移到 V2 参数化格式')
    parser.add_argument('--input', '-i', required=True, help='输入文件或目录')
    parser.add_argument('--output', '-o', help='输出文件或目录（默认覆盖原文件）')
    parser.add_argument('--no-backup', action='store_true', help='不创建备份')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归处理目录')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    files_to_process = []
    
    # 收集文件
    if input_path.is_file():
        files_to_process.append((str(input_path), 
                                 str(output_path) if output_path else str(input_path)))
    elif input_path.is_dir():
        pattern = '**/*.json' if args.recursive else '*.json'
        for json_file in input_path.glob(pattern):
            if '.backup' in json_file.name:
                continue
            
            if output_path:
                rel_path = json_file.relative_to(input_path)
                out_file = output_path / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_file = json_file
            
            files_to_process.append((str(json_file), str(out_file)))
    
    # 处理文件
    print(f"找到 {len(files_to_process)} 个文件")
    success_count = 0
    
    for in_path, out_path in files_to_process:
        if migrate_file(in_path, out_path, backup=not args.no_backup):
            success_count += 1
    
    print(f"\n完成: {success_count}/{len(files_to_process)} 个文件成功迁移")


if __name__ == '__main__':
    main()

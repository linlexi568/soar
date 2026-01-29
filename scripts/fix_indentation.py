#!/usr/bin/env python3
"""修复batch_evaluation.py的缩进问题"""

import re

file_path = '01_soar/utils/batch_evaluation.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修复1247行附近的缩进问题
# 查找问题行
for i in range(1240, min(1260, len(lines))):
    line = lines[i]
    if 'if not hasattr(self, \'_compiled_forces\'):' in line:
        print(f"找到行{i+1}: {line.rstrip()}")
        # 检查下一行
        if i+1 < len(lines):
            next_line = lines[i+1]
            print(f"下一行{i+2}: {next_line.rstrip()}")
            # 如果下一行缩进不足，修复它
            if '# 仅当所有程序' in next_line and not next_line.startswith(' ' * 32):
                lines[i+1] = ' ' * 32 + next_line.lstrip()
                print(f"  → 修复为: {lines[i+1].rstrip()}")
        
        # 检查再下一行
        if i+2 < len(lines):
            line3 = lines[i+2]
            print(f"第三行{i+3}: {line3.rstrip()}")
            if 'if self._all_programs_const' in line3 and not line3.startswith(' ' * 32):
                lines[i+2] = ' ' * 32 + line3.lstrip()
                print(f"  → 修复为: {lines[i+2].rstrip()}")

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\n✅ 修复完成")

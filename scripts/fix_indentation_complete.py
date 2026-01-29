#!/usr/bin/env python3
"""完整修复batch_evaluation.py的缩进问题"""

file_path = '01_soar/utils/batch_evaluation.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 找到问题区域并完整替换
old_block = """                        try:
                            if not hasattr(self, '_compiled_forces'):
                                # 仅当所有程序皆为"无条件常量 set u_*"时，才启用 UltraFast
                                if self._all_programs_const(batch_programs):
                                self._compiled_forces = self._ultra_executor.compile_programs(batch_programs)
                                print(f"[UltraFast] ✅ 预编译{len(batch_programs)}程序 → 缓存{len(self._ultra_executor.program_cache)}个唯一程序")
                                # 若全部常量结果几乎为零，且严格无先验，则放弃 UltraFast 以避免长期零动作退化
                                try:
                                    import numpy as _np
                                    if _np.all(_np.abs(self._compiled_forces) < 1e-8) and self.strict_no_prior:
                                        print("[UltraFast] ⚠️ 全常量为零，禁用UltraFast以避免零动作退化")
                                        self._ultra_executor = None
                                        if hasattr(self, '_compiled_forces'):
                                            delattr(self, '_compiled_forces')
                                except Exception:
                                    pass
                            else:
                                # 存在条件/非常量表达式：禁用 UltraFast，回退到逐步AST评估，确保动作依赖状态
                                self._ultra_executor = None
                    except Exception as e:
                        print(f"[UltraFast] ⚠️ 预编译失败: {e}, 回退到标准快速路径")
                        self._ultra_executor = None"""

new_block = """                        try:
                            if not hasattr(self, '_compiled_forces'):
                                # 仅当所有程序皆为"无条件常量 set u_*"时，才启用 UltraFast
                                if self._all_programs_const(batch_programs):
                                    self._compiled_forces = self._ultra_executor.compile_programs(batch_programs)
                                    print(f"[UltraFast CPU] ✅ 预编译{len(batch_programs)}程序 → 缓存{len(self._ultra_executor.program_cache)}个唯一程序")
                                    # 若全部常量结果几乎为零，且严格无先验，则放弃 UltraFast 以避免长期零动作退化
                                    try:
                                        import numpy as _np
                                        if _np.all(_np.abs(self._compiled_forces) < 1e-8) and self.strict_no_prior:
                                            print("[UltraFast CPU] ⚠️ 全常量为零，禁用UltraFast以避免零动作退化")
                                            self._ultra_executor = None
                                            if hasattr(self, '_compiled_forces'):
                                                delattr(self, '_compiled_forces')
                                    except Exception:
                                        pass
                                else:
                                    # 存在条件/非常量表达式：禁用 UltraFast，回退到逐步AST评估，确保动作依赖状态
                                    self._ultra_executor = None
                        except Exception as e:
                            print(f"[UltraFast CPU] ⚠️ 预编译失败: {e}, 回退到标准快速路径")
                            self._ultra_executor = None"""

if old_block in content:
    content = content.replace(old_block, new_block)
    print("✅ 找到并修复缩进问题")
else:
    print("⚠️  未找到完整匹配，尝试部分修复...")
    # 尝试逐行修复
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'self._compiled_forces = self._ultra_executor' in line:
            # 检查缩进 (应该是36个空格)
            current_indent = len(line) - len(line.lstrip())
            if current_indent != 36:
                lines[i] = ' ' * 36 + line.lstrip()
                print(f"修复行{i+1}:  缩进 {current_indent} → 36")
        
        if 'print(f"[UltraFast]' in line and i > 1245 and i < 1252:
            current_indent = len(line) - len(line.lstrip())
            if current_indent != 36:
                lines[i] = ' ' * 36 + line.lstrip()
                # 同时更新标签
                lines[i] = lines[i].replace('[UltraFast]', '[UltraFast CPU]')
                print(f"修复行{i+1}: 缩进 {current_indent} → 36")
    
    content = '\n'.join(lines)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ 修复完成，验证语法...")

import subprocess
result = subprocess.run(
    ['/home/linlexi/桌面/soar/.venv/bin/python', '-m', 'py_compile', file_path],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("✅ 语法正确!")
else:
    print(f"❌ 仍有语法错误:\n{result.stderr}")

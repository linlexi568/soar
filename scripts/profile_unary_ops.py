#!/usr/bin/env python
import os, sys, math, random, time
from pathlib import Path
sys.path.insert(0, str(Path('01_soar').resolve()))
from core.dsl import TerminalNode, UnaryOpNode, ConstantNode, get_op_profile

"""简单算子性能剖析脚本
设置 OP_PROFILE=1 后统计 evaluate 内各一元算子的平均耗时 (微秒)、调用次数、总耗时(ms)。
不依赖完整环境执行器，只对算子本身进行微基准。
"""

# 若外部未设定，内部强制开启
os.environ['OP_PROFILE'] = '1'

# 构造一组典型参数化算子节点
child = TerminalNode('vel_err')

def c(v, name=None):
    return ConstantNode(v, name=name)

nodes = [
    UnaryOpNode('abs', child),
    UnaryOpNode('sign', child),
    UnaryOpNode('sin', child),
    UnaryOpNode('cos', child),
    UnaryOpNode('tan', child),
    UnaryOpNode('log1p', child),
    UnaryOpNode('sqrt', child),
    UnaryOpNode('ema', child, params={'alpha': c(0.25,'alpha')}),
    UnaryOpNode('delay', child, params={'k': c(3,'k')}),
    UnaryOpNode('diff', child, params={'k': c(3,'k')}),
    UnaryOpNode('clamp', child, params={'lo': c(-2.0,'lo'), 'hi': c(2.0,'hi')}),
    UnaryOpNode('deadzone', child, params={'eps': c(0.05,'eps')}),
    UnaryOpNode('rate', child, params={'r': c(0.5,'r')}),
    UnaryOpNode('rate_limit', child, params={'r': c(0.5,'r')}),
    UnaryOpNode('smooth', child, params={'s': c(1.0,'s')}),
    UnaryOpNode('smoothstep', child, params={'s': c(1.0,'s')}),
]

N_ITERS = 20000  # 适度次数，快速又有统计意义
state = {}

for i in range(N_ITERS):
    # 随机生成 vel_err，模拟真实波动
    state['vel_err'] = random.uniform(-3.0, 3.0)
    for node in nodes:
        _ = node.evaluate(state)

profile = get_op_profile(reset=False)

# 排序按平均耗时
items = sorted(profile.items(), key=lambda kv: kv[1]['avg_us'], reverse=True)

print(f"== Unary Operator Micro Benchmark (N_ITERS={N_ITERS}) ==")
print(f"op\tavg_us\tcount\ttotal_ms")
for op, stats in items:
    print(f"{op}\t{stats['avg_us']:.3f}\t{stats['count']}\t{stats['total_ms']:.2f}")

slow = items[:5]
print("\nTop 5 slow (avg_us):")
for op, stats in slow:
    print(f"  {op}: {stats['avg_us']:.3f} us (total {stats['total_ms']:.2f} ms, count {stats['count']})")

print("\n提示: delay/diff/ema 含状态访问, tan/log1p/sqrt 含数学调用; clamp/deadzone/abs/sign 较轻量。")

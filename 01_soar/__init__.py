"""soar package

Programmatic flight controller search via DSL + MCTS. This is the canonical
package going forward (renamed from 01_pi_light). It exposes the same API.

This module is resilient to internal folder reorganization. It will try to
import symbols from both legacy flat layout and the new split layout:
  - mcts:   .mcts or .mcts_training.mcts
  - nn:     (not exported here; training utilities live under .nn_training)
  - cma-es: (not exported here; tooling can live under .cma_training)
"""
from .core.dsl import ProgramNode, TerminalNode, UnaryOpNode, BinaryOpNode, IfNode

# 为了兼容 Python 3.8（避免在导入时触发不兼容的类型注解），延迟导入 MCTS_Agent。
def _load_mcts_agent():
    try:
        from .mcts_training.mcts import MCTS_Agent  # type: ignore
        return MCTS_Agent
    except Exception:
        try:
            from .mcts import MCTS_Agent  # type: ignore
            return MCTS_Agent
        except Exception as e:
            raise ImportError(f"Unable to load MCTS_Agent: {e}")

__all__ = [
    'ProgramNode','TerminalNode','UnaryOpNode','BinaryOpNode','IfNode',
    '_load_mcts_agent'
]

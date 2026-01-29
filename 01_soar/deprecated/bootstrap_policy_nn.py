"""Shim to mcts_training/policy/bootstrap_policy_nn.py"""
from __future__ import annotations

from .mcts_training.policy.bootstrap_policy_nn import *  # type: ignore

if __name__ == '__main__':
    from .mcts_training.policy.bootstrap_policy_nn import main  # type: ignore
    main()

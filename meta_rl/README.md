# Meta-RL Controller (Hot-Swappable)

This package hosts an experimental meta-RL / recurrent policy module that learns to adjust Soar's MCTS and evaluator hyper-parameters online. The design goals are:

1. **Real meta-RL policy** – a recurrent controller consumes recent training telemetry (reward stats, zero-action ratio, exploration entropy, etc.) and outputs deltas for key knobs (Dirichlet noise, policy temperature, zero-action penalty, replicas per program…).
2. **Hot-plug friendly** – the controller can be enabled/disabled without touching `01_soar/train_online.py`. Overrides are written to a JSON file; the trainer only needs to read it before each meta-epoch. When the file is missing, default behavior is unchanged.
3. **Train offline, deploy online** – you can pretrain the RNN on archived sweep logs (`results/mcts_tune/*.csv`) and later fine-tune it in the loop using meta-losses (reward improvements, constraint satisfaction).

## Folder layout

```
meta_rl/
├── README.md                   # This document
├── __init__.py                 # Convenience exports
├── config.py                   # Dataclasses describing inputs/outputs and IO helpers
├── rnn_meta_policy.py          # PyTorch implementation of the recurrent controller
├── controller.py               # High-level MetaRLController coordinating history + policy
├── hot_swap.py                 # File-based protocol for safely enabling/disabling overrides
├── run_meta_pretrain.py        # CLI script to pretrain the RNN using logged CSV traces
└── tests/
    └── test_meta_controller.py # Smoke tests to ensure the pieces wire together
```

## Quick start

1. **Collect traces** (single machine tuning):
   ```bash
   python scripts/sweep_mcts.py --summary-out results/mcts_tune/summary.csv
   ```
   Each row should include `iter_idx`, `reward_mean`, `reward_std`, `zero_action_frac`, `entropy`, etc.

2. **Pretrain the controller**:
   - One-liner convenience script (auto-resolves paths):
     ```bash
     ./run_meta_rl.sh -s results/mcts_tune/summary.csv -o meta_rl/checkpoints/meta_policy.pt
     ```
   - Or call the Python entry directly:
   ```bash
   python meta_rl/run_meta_pretrain.py \
     --summary-csv results/mcts_tune/summary.csv \
     --output-checkpoint meta_rl/checkpoints/meta_policy.pt
   ```

3. **Hot-plug into the trainer**:
   - Start a background watcher or add a tiny hook before each meta-epoch to call:
     ```python
     from meta_rl.hot_swap import maybe_apply_overrides
     args = maybe_apply_overrides(args, iter_idx=current_iter, metrics=latest_metrics)
     ```
   - Drop `meta_rl/configs/current_override.json` whenever you want to change behavior; remove the file to revert instantly.

4. **Fine-tune online** (optional): configure `MetaRLController` with `online_update=True`, feed it live telemetry, and periodically call `controller.update()` + `controller.save_state()`.

## Why RNN/meta-RL?

- The controller observes trends (reward slope, entropy collapse, evaluator penalties) rather than a single snapshot, which fits naturally into an RNN hidden state.
- We can encode constraints (e.g., keep `replicas_per_program ≤ 8`) by bounding the controller outputs before applying them.
- With JSON-based overrides you can “hot unplug” by simply deleting the file – the trainer reverts to its compiled defaults without restarting.

## Next steps

- Wire `meta_rl.hot_swap.maybe_apply_overrides` into `train_online.py` (likely near the start of each iteration).
- Extend the dataset loader to parse richer telemetry (curriculum stage, ranking blend, etc.).
- Add reinforcement style losses (e.g., policy gradient on meta-parameters) for fully online adaptation when multiple machines feed data simultaneously.

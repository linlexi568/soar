# Isaac Gym Baseline Evaluation

This folder hosts utilities that evaluate safe-control-gym style PPO/SAC
policies inside the Soar Isaac Gym environment while reusing the
Stepwise reward profiles. The primary entry points are:

- `eval_safecontrol_baselines.py`: load a Stable-Baselines3 policy (PPO/SAC)
  trained via safe-control-gym configs and evaluate it across hundreds of
  Isaac Gym replicas.
- `isaac_vec_env.py`: thin VecEnv wrapper exposing the native
  `IsaacGymDroneEnv` to Stable-Baselines3.
- `eval_utils.py`: shared rollout helper returning aggregated metrics.

Typical usage:

```bash
python scripts/baselines/eval_safecontrol_baselines.py \
    --policy-type ppo \
    --model-path path/to/checkpoint \
    --task figure8 \
    --reward-profile balanced \
    --episodes 128 \
    --device cuda:0 \
    --output compare/ppo_baseline_metrics.json
```

The command prints batched statistics and stores a JSON payload that can be fed
into downstream comparison scripts (e.g., juxtaposed with `utilities/verify_program.py`
outputs). PPO/SAC models benefit from Isaac Gym's massive parallelism by setting
`--isaac-num-envs` to match the training setup (default 512).

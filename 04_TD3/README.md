# TD3 Baseline for Soar Comparison

## Overview

TD3 (Twin Delayed DDPG) is a **deterministic off-policy** RL algorithm that serves as a complementary baseline to SAC. While SAC uses stochastic policies with entropy maximization, TD3 uses deterministic policies with explicit exploration noise.

## Anti-Early-Death Strategy (Inherited from SAC)

TD3 automatically benefits from the same crash-prevention mechanisms via `scg_vec_env.py`:

1. **Reduced Action Scale**: `[10.0, 5.0, 5.0, 5.0]` (50% reduction)
2. **Crash Penalty**: `-1000` for premature crashes
3. **TimeLimit.truncated**: Distinguishes timeout vs crash for proper bootstrapping
4. **Reward Normalization**: `VecNormalize(norm_reward=True, clip_reward=10.0)`

## TD3 vs SAC Key Differences

| Feature | TD3 | SAC |
|---------|-----|-----|
| **Policy Type** | Deterministic | Stochastic |
| **Exploration** | Action noise (Gaussian) | Entropy maximization |
| **Q-Networks** | Twin critics | Twin critics |
| **Policy Update** | Delayed (every 2 steps) | Every step |
| **Entropy Regularization** | ❌ None | ✅ Auto-tuned |
| **Complexity** | Simpler | More complex |

## Training Configuration

- **Task**: figure8 (5s trajectory)
- **Parallel Envs**: 16,384
- **Total Steps**: 100M
- **Learning Rate**: 3e-4
- **Network**: 512-512-256 (actor & critic)
- **Buffer Size**: 2M
- **Action Noise**: σ=0.1
- **Policy Noise**: 0.2 (for target smoothing)

## Expected Performance

Based on literature and SAC results:
- **SAC**: reward = -5.41 (figure8, 240 steps)
- **TD3**: Expected similar or slightly worse (~-6 to -8)
- **PPO**: Expected worse due to on-policy inefficiency (~-10 to -15)

TD3's deterministic policy may be less robust than SAC's stochastic exploration but typically trains faster.

## Usage

### Start Training
```bash
bash 04_TD3/run_td3.sh
```

### Monitor Progress
```bash
tensorboard --logdir results/td3/figure8/tb
```

### Key Metrics to Watch
- `rollout/ep_rew_mean`: Episode reward (target: < -10)
- `rollout/ep_len_mean`: Episode length (target: 240)
- `train/actor_loss`: Policy gradient magnitude
- `train/critic_loss`: TD error

## Why TD3?

1. **Complementary to SAC**: Tests deterministic vs stochastic policies
2. **Simpler**: No entropy tuning, easier to debug
3. **Literature Standard**: Widely used RL baseline
4. **Performance**: Often competitive with SAC on continuous control

## Comparison with Soar

| Method | Type | Reward | Sample Efficiency |
|--------|------|--------|------------------|
| **Soar** | Program Search | **-31.98** | High (MCTS guided) |
| **SAC** | Stochastic RL | -5.41 | Medium (40M steps) |
| **TD3** | Deterministic RL | TBD | Medium (expected) |
| **PPO** | On-policy RL | TBD | Low (needs more steps) |
| CascadedPID | Classical | -45.07 | N/A |

**Key Insight**: All RL methods (SAC/TD3/PPO) far outperform classical baselines, but Soar's program search achieves the best reward by finding control structures beyond what neural networks can express with the same sample budget.

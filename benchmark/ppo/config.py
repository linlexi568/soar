"""PPO Configuration for Benchmark"""
import torch

# 轨迹配置
TRAJECTORIES = {
    'hover': {
        'height': 1.0,
    },
    'circle': {
        'period': 5.0,
        'radius': 0.9,
        'center': [0.0, 0.0, 1.0],
    },
    'figure8': {
        'period': 5.0,
        'scale': 0.8,
        'plane': 'xy',
        'center': [0.0, 0.0, 1.0],
    },
    'square': {
        'period': 5.0,
        'side_length': 0.8,
        'center': [0.0, 0.0, 1.0],
    },
    'helix': {
        'period': 8.0,
        'radius': 0.7,
        'pitch_per_rev': 0.1,
        'center': [0.0, 0.0, 1.0],
    },
}

# PPO 超参数
PPO_CONFIG = {
    'learning_rate_start': 5e-4,
    'learning_rate_end': 5e-5,
    'n_steps': 256,
    'batch_size': 1024,
    'n_epochs': 3,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'net_arch': [dict(pi=[512, 512, 256], vf=[512, 512, 256])],
    'activation_fn': torch.nn.Tanh,
}

# 训练配置
TRAIN_CONFIG = {
    'num_envs': 8196,
    'max_steps': 500_000_000,
    'seed': 42,
    'duration': 5.0,
}

# SCG 奖励配置
REWARD_CONFIG = {
    'use_pure_scg': True,
}
